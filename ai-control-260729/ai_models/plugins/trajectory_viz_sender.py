#!/usr/bin/env python3
"""Docker-aware, non-blocking trajectory visualization sender.

Place this file at:
    ai_models/plugins/trajectory_viz_sender.py

The docking plugin can keep its current call unchanged:
    send_velocity_trajectory(
        ema,
        step_dt=STEP_DT,
        max_steps=min(n, HORIZON),
        endpoint="tcp://127.0.0.1:5555",
    )

To see the whole candidate set instead of only the executed plan, send a
bundle -- every one of the diffusion policy's n_samples trajectories plus the
one actually being driven:
    send_trajectory_bundle(
        candidates=plan.samples,      # [M,H,2] (vx,wz) per sample
        selected=ema,                 # [H,2] the plan actually executed
        selected_index=pick,          # which candidate, or -1 for mean/medoid
        step_dt=STEP_DT,
        max_steps=min(n, HORIZON),
    )

When running inside Docker, 127.0.0.1 is automatically rewritten to the
Docker host gateway so that a viz_server running on the Linux host receives it.
"""

from __future__ import annotations

import logging
import os
import socket as socket_lib
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"

_CONTEXT: Optional[Any] = None
_SOCKET: Optional[Any] = None
_CONNECTED_ENDPOINT: Optional[str] = None
_LOGGED_FAILURES: set[str] = set()
_RESOLVED: dict[str, str] = {}


def _inside_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        text = open("/proc/1/cgroup", "r", encoding="utf-8").read()
        return "docker" in text or "containerd" in text or "kubepods" in text
    except OSError:
        return False


def _docker_gateway() -> Optional[str]:
    """Return Linux Docker's default gateway using /proc/net/route."""
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as route_file:
            next(route_file, None)
            for line in route_file:
                fields = line.strip().split()
                if len(fields) < 4 or fields[1] != "00000000":
                    continue
                flags = int(fields[3], 16)
                if not flags & 0x2:
                    continue
                gateway_hex = fields[2]
                raw = bytes.fromhex(gateway_hex)
                return socket_lib.inet_ntoa(raw[::-1])
    except Exception:
        return None
    return None


def _resolve_endpoint(requested: str) -> str:
    """Rewrite loopback to the host gateway when called inside Docker.

    Memoized per requested endpoint: this runs on the control path once per
    inference, and the container branch does a DNS lookup for
    host.docker.internal that must not repeat at inference rate.
    """
    cached = _RESOLVED.get(requested)
    if cached is not None:
        return cached
    resolved = _resolve_endpoint_uncached(requested)
    _RESOLVED[requested] = resolved
    return resolved


def _resolve_endpoint_uncached(requested: str) -> str:
    env_endpoint = os.getenv("TRAJECTORY_VIZ_ENDPOINT", "").strip()
    if env_endpoint:
        return env_endpoint

    if not _inside_container():
        return requested

    if "127.0.0.1" not in requested and "localhost" not in requested:
        return requested

    # Prefer Docker's standard host alias when Compose configured extra_hosts.
    try:
        socket_lib.gethostbyname("host.docker.internal")
        host = "host.docker.internal"
    except socket_lib.gaierror:
        host = _docker_gateway()

    if not host:
        raise RuntimeError(
            "Docker host address could not be resolved. Add "
            "extra_hosts: ['host.docker.internal:host-gateway'] to Compose, "
            "or set TRAJECTORY_VIZ_ENDPOINT=tcp://HOST_IP:5555."
        )

    return requested.replace("127.0.0.1", host).replace("localhost", host)


def _get_socket(endpoint: str):
    global _CONTEXT, _SOCKET, _CONNECTED_ENDPOINT

    import zmq

    resolved = _resolve_endpoint(endpoint)
    if _SOCKET is not None and _CONNECTED_ENDPOINT == resolved:
        return _SOCKET, resolved

    if _SOCKET is not None:
        try:
            _SOCKET.close(linger=0)
        except Exception:
            pass

    _CONTEXT = zmq.Context.instance()
    _SOCKET = _CONTEXT.socket(zmq.PUSH)
    _SOCKET.setsockopt(zmq.LINGER, 0)
    _SOCKET.setsockopt(zmq.SNDHWM, 1)
    _SOCKET.setsockopt(zmq.IMMEDIATE, 1)
    _SOCKET.connect(resolved)
    _CONNECTED_ENDPOINT = resolved

    logger.warning("trajectory visualization endpoint: %s", resolved)
    return _SOCKET, resolved


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def velocity_batch_to_xy(
    velocity: Any,
    step_dt: float = 1.0 / 30.0,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """Integrate an [M,H,2+] batch of vx,wz plans into [M,N,2] XY paths.

    Unicycle integration in the world frame, identical step-for-step to the
    single-trajectory path: at step i the robot advances along the heading it
    accumulated over steps 0..i-1, then applies wz[i].
    """
    arr = _as_numpy(velocity)
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"expected velocity shape (M,H,2+), got {arr.shape}")
    if not np.isfinite(arr[:, :, :2]).all():
        raise ValueError("velocity includes NaN or Inf")
    if step_dt <= 0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")

    horizon = arr.shape[1]
    count = horizon if max_steps is None else min(horizon, int(max_steps))
    if count <= 0 or arr.shape[0] == 0:
        return np.empty((arr.shape[0], 0, 2), dtype=np.float32)

    vx = arr[:, :count, 0].astype(np.float64)
    wz = arr[:, :count, 1].astype(np.float64)

    # heading BEFORE step i = dt * sum(wz[:i]); leading 0 makes step 0 use theta=0
    theta = np.zeros_like(wz)
    theta[:, 1:] = np.cumsum(wz, axis=1)[:, :-1] * step_dt

    xs = np.cumsum(vx * np.cos(theta) * step_dt, axis=1)
    ys = np.cumsum(vx * np.sin(theta) * step_dt, axis=1)
    return np.stack((xs, ys), axis=-1).astype(np.float32)


def velocity_to_xy(
    velocity: Any,
    step_dt: float = 1.0 / 30.0,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """Integrate an [N,2+] vx,wz array into a display-only XY path."""
    arr = _as_numpy(velocity)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"expected velocity shape (N,2+), got {arr.shape}")
    return velocity_batch_to_xy(arr[None], step_dt, max_steps)[0]


def send_xy_trajectory(
    trajectory_xy: Any,
    endpoint: str = DEFAULT_ENDPOINT,
) -> bool:
    """Non-blocking send_pyobj of an [N,2] float32 trajectory."""
    try:
        import zmq

        arr = np.asarray(trajectory_xy, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"expected trajectory shape (N,2), got {arr.shape}")
        if len(arr) == 0:
            return False
        if not np.isfinite(arr).all():
            raise ValueError("trajectory includes NaN or Inf")

        zmq_socket, resolved = _get_socket(endpoint)
        zmq_socket.send_pyobj(arr, flags=zmq.NOBLOCK)
        return True

    except Exception as exc:
        # Never interrupt robot control because visualization is unavailable.
        _log_once("trajectory visualization disabled for now", exc)
        return False


def send_velocity_trajectory(
    velocity: Any,
    step_dt: float = 1.0 / 30.0,
    max_steps: Optional[int] = None,
    endpoint: str = DEFAULT_ENDPOINT,
) -> bool:
    """Convert model vx,wz output into XY and send it to viz_server."""
    try:
        trajectory = velocity_to_xy(velocity, step_dt, max_steps)
        return send_xy_trajectory(trajectory, endpoint)
    except Exception as exc:
        _log_once("trajectory visualization conversion failed", exc)
        return False


def send_trajectory_bundle(
    candidates: Any = None,
    selected: Any = None,
    selected_index: int = -1,
    step_dt: float = 1.0 / 30.0,
    max_steps: Optional[int] = None,
    endpoint: str = DEFAULT_ENDPOINT,
) -> bool:
    """Send every sampled candidate plus the executed plan in one message.

    `candidates` is the raw [M,H,2] (vx,wz) sample stack straight off the
    diffusion policy (RolloutPlan.samples) -- all M of them, so the viewer can
    show the full spread the aggregator or the geometry selector chose from.
    `selected` is the [H,2] plan actually being driven; it is NOT necessarily
    candidates[selected_index] because continuity smoothing runs after the
    pick, and because mean/medoid aggregation produces a plan that is no
    sample at all (selected_index = -1 in that case).

    Both are integrated to XY here so the server stays a dumb renderer.
    """
    try:
        import zmq

        payload: dict[str, Any] = {
            "type": "trajectory_bundle",
            "version": 1,
            "step_dt": float(step_dt),
            "selected_index": int(selected_index),
        }

        if candidates is not None:
            cand_xy = velocity_batch_to_xy(candidates, step_dt, max_steps)
            if cand_xy.shape[0] and cand_xy.shape[1]:
                payload["candidates"] = cand_xy
        if selected is not None:
            sel_xy = velocity_to_xy(selected, step_dt, max_steps)
            if len(sel_xy):
                payload["selected"] = sel_xy

        if "candidates" not in payload and "selected" not in payload:
            return False

        zmq_socket, _resolved = _get_socket(endpoint)
        zmq_socket.send_pyobj(payload, flags=zmq.NOBLOCK)
        return True

    except Exception as exc:
        # Never interrupt robot control because visualization is unavailable.
        _log_once("trajectory visualization disabled for now", exc)
        return False


def _log_once(prefix: str, exc: BaseException) -> None:
    """Log a viz failure at most once per distinct message; ignore backpressure.

    zmq.Again just means the viewer is absent or slower than inference, which
    is the expected steady state when nobody is watching.
    """
    if type(exc).__name__ == "Again":
        return
    key = f"{type(exc).__name__}: {exc}"
    if key not in _LOGGED_FAILURES:
        _LOGGED_FAILURES.add(key)
        logger.warning("%s: %s", prefix, key)


def close_viz_sender() -> None:
    global _CONTEXT, _SOCKET, _CONNECTED_ENDPOINT
    if _SOCKET is not None:
        try:
            _SOCKET.close(linger=0)
        except Exception:
            pass
    _SOCKET = None
    _CONTEXT = None
    _CONNECTED_ENDPOINT = None
