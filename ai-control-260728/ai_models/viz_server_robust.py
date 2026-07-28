#!/usr/bin/env python3
"""Live trajectory visualization server.

Receives trajectory data through a ZeroMQ PULL socket on TCP port 5555 and
draws EVERY candidate the diffusion policy sampled, with the plan actually
being executed highlighted on top.

Supported payloads:
  1. socket.send_pyobj(dict)   <- the rich form, see BUNDLE below
  2. socket.send_pyobj(numpy_array)  ..(N,2) single path (legacy)
  3. socket.send_json([[x, y], ...])
  4. UTF-8 JSON bytes, optionally {"trajectory"|"points"|"data": [[x,y],...]}
  5. NumPy .npy byte stream
  6. Raw float32/float64 bytes containing x,y pairs

BUNDLE (what ai_models/plugins/trajectory_viz_sender.send_trajectory_bundle
sends):
    {"type": "trajectory_bundle",
     "candidates":     (M, N, 2) float  all M sampled plans, already in XY
     "selected":       (N, 2)    float  the plan being driven
     "selected_index": int              which candidate, -1 if aggregated}

Interaction (the axes are fully zoomable; nothing snaps back on new data):
    scroll            zoom about the cursor, both axes
    shift + scroll    zoom X only        ctrl + scroll   zoom Y only
    p / o             matplotlib pan tool / zoom-to-rect tool
    h                 home (matplotlib)
    f                 fit once to the current data
    a                 toggle continuous auto-fit (default on)
    r                 reset to the default +/-AXIS_LIMIT box
    e                 toggle equal aspect <-> independent X/Y scaling
    i                 toggle candidate index labels
"""

import argparse
import io
import json
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zmq
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def _select_backend() -> str:
    """Pick an interactive backend, falling back to Agg when there is no GUI.

    matplotlib.use() only records the name -- the backend module is imported
    lazily at the first figure, so a missing tkinter would surface as a crash
    deep inside plt.subplots() and never reach a try/except here.
    switch_backend() imports eagerly, which is what makes the fallback real.
    """
    env = os.environ.get("MPLBACKEND", "").strip()
    for name in ([env] if env else []) + ["TkAgg", "QtAgg", "Agg"]:
        try:
            plt.switch_backend(name)
            return name
        except Exception:
            continue
    return matplotlib.get_backend()


def _is_interactive(name: str) -> bool:
    """True only for backends that can actually open a window and take events."""
    try:
        from matplotlib.backends.registry import BackendFilter, backend_registry

        return name.lower() in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    except Exception:                                   # matplotlib < 3.9
        return name.lower() not in ("agg", "svg", "pdf", "ps", "cairo", "pgf", "template")


BACKEND = _select_backend()
INTERACTIVE = _is_interactive(BACKEND)

# 'f' is matplotlib's fullscreen key by default; we want it for fit-to-data.
plt.rcParams["keymap.fullscreen"] = []

DEFAULT_PORT = 5555
AXIS_LIMIT = 0.075
ZOOM_STEP = 1.25          # scroll-wheel zoom factor per notch
FIT_MARGIN = 0.15         # fraction of the data span added as padding

CAND_CMAP = "viridis"
EXEC_COLOR = "#e6194b"    # executed plan: thick crimson
PICK_COLOR = "#ff8c1a"    # the candidate that was picked, before smoothing


@dataclass
class Bundle:
    """One frame's worth of paths, all in the robot body frame at t=0."""

    candidates: Optional[np.ndarray] = None   # (M, N, 2)
    selected: Optional[np.ndarray] = None     # (N, 2)
    selected_index: int = -1

    def all_points(self) -> np.ndarray:
        chunks = [np.zeros((1, 2))]           # always include the robot origin
        if self.candidates is not None and len(self.candidates):
            chunks.append(self.candidates.reshape(-1, 2))
        if self.selected is not None and len(self.selected):
            chunks.append(self.selected)
        return np.concatenate(chunks, axis=0)

    def describe(self) -> str:
        n_cand = 0 if self.candidates is None else len(self.candidates)
        n_pts = 0
        if self.selected is not None:
            n_pts = len(self.selected)
        elif self.candidates is not None and len(self.candidates):
            n_pts = self.candidates.shape[1]
        pick = "mean/medoid" if self.selected_index < 0 else f"#{self.selected_index}"
        return f"{n_cand} candidates x {n_pts} steps | executing {pick}"


# ------------------------------------------------------------------ decoding
def decode_payload(message: bytes) -> Bundle:
    """Decode any supported ZeroMQ payload into a Bundle."""
    return to_bundle(loads_any(message))


def loads_any(message: bytes) -> Any:
    """Deserialize the wire format without interpreting the contents."""
    errors = []

    # 1. Python pickle, including send_pyobj(dict) and send_pyobj(np.ndarray)
    try:
        return pickle.loads(message)
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    # 2. UTF-8 JSON, e.g. [[0.0, 0.0], [0.01, 0.02]]
    try:
        return json.loads(message.decode("utf-8"))
    except Exception as exc:
        errors.append(f"json: {exc}")

    # 3. NumPy .npy byte stream
    try:
        return np.load(io.BytesIO(message), allow_pickle=False)
    except Exception as exc:
        errors.append(f"npy: {exc}")

    # 4. Raw float buffers. Prefer float32, then float64.
    for dtype in (np.float32, np.float64):
        try:
            item_size = np.dtype(dtype).itemsize
            if len(message) > 0 and len(message) % (2 * item_size) == 0:
                arr = np.frombuffer(message, dtype=dtype).reshape(-1, 2)
                if np.isfinite(arr).all():
                    return arr.astype(float)
        except Exception as exc:
            errors.append(f"raw {dtype.__name__}: {exc}")

    preview = repr(message[:80])
    raise ValueError(
        "Unsupported trajectory payload. "
        f"bytes={len(message)}, preview={preview}, attempts={' | '.join(errors)}"
    )


def to_bundle(obj: Any) -> Bundle:
    """Interpret a decoded object as candidates + selected path."""
    if isinstance(obj, dict):
        candidates = obj.get("candidates")
        selected = obj.get("selected")
        if selected is None:
            for key in ("trajectory", "points", "data"):
                if key in obj:
                    selected = obj[key]
                    break
        if candidates is None and selected is None:
            raise ValueError(f"dict payload has no path keys: {sorted(obj)[:8]}")

        bundle = Bundle(
            candidates=None if candidates is None else validate_batch(candidates),
            selected=None if selected is None else validate_trajectory(selected),
            selected_index=int(obj.get("selected_index", -1)),
        )
    else:
        bundle = Bundle(selected=validate_trajectory(obj))

    n_cand = 0 if bundle.candidates is None else len(bundle.candidates)
    if not -1 <= bundle.selected_index < max(n_cand, 1):
        bundle.selected_index = -1
    return bundle


def validate_trajectory(arr: Any) -> np.ndarray:
    """Validate and normalize a single trajectory to exactly two columns."""
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"1-D payload has odd length: {arr.size}")
        arr = arr.reshape(-1, 2)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"expected shape (N, 2+), received {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("trajectory is empty")

    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        raise ValueError("trajectory contains NaN or Inf")

    return arr


def validate_batch(arr: Any) -> np.ndarray:
    """Validate a candidate stack down to (M, N, 2)."""
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 2:                                  # a lone path, promote it
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"expected candidates (M, N, 2+), received {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("candidate set is empty")

    arr = arr[:, :, :2]
    if not np.isfinite(arr).all():
        raise ValueError("candidates contain NaN or Inf")

    return arr


# ----------------------------------------------------------------- rendering
class TrajectoryVisualizer:
    """Persistent-artist renderer.

    Artists are created once and updated in place rather than redrawing from a
    cleared axes, so a view the user zoomed or panned into survives every
    incoming frame -- clearing the axes would silently snap the limits back.
    """

    def __init__(self, axis_limit: float = AXIS_LIMIT, autofit: bool = True):
        plt.ion()
        self.axis_limit = float(axis_limit)
        self.autofit = bool(autofit)
        self.equal_aspect = True
        self.show_labels = True
        self.bundle: Optional[Bundle] = None
        self._cand_labels: list = []

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title("AI trajectory candidates")

        self.ax.set_xlabel("X [m]  (robot forward)")
        self.ax.set_ylabel("Y [m]  (robot left)")
        self.ax.grid(True, linestyle=":", alpha=0.35)
        self.ax.set_aspect("equal", adjustable="box")
        self.reset_view()

        # All M sampled plans: thin, colour-coded by sample index.
        self.cand_lines = LineCollection([], linewidths=1.3, zorder=3)
        self.ax.add_collection(self.cand_lines)
        self.cand_ends = self.ax.scatter(
            [], [], s=18, edgecolors="none", zorder=4
        )

        # The candidate the selector picked, pre continuity-smoothing.
        self.pick_line, = self.ax.plot(
            [], [], color=PICK_COLOR, lw=2.2, ls="--", zorder=5
        )

        # The plan actually being driven.
        self.exec_line, = self.ax.plot(
            [], [], color=EXEC_COLOR, lw=3.4, solid_capstyle="round", zorder=6
        )
        self.exec_pts, = self.ax.plot(
            [], [], ls="none", marker="o", ms=4.5,
            mfc=EXEC_COLOR, mec="white", mew=0.7, zorder=7
        )
        self.exec_end, = self.ax.plot(
            [], [], ls="none", marker="*", ms=18,
            mfc=EXEC_COLOR, mec="white", mew=0.8, zorder=8
        )

        self.ax.plot(
            0, 0, marker="s", ms=9, color="black", ls="none", zorder=9
        )

        self.ax.legend(
            handles=[
                Line2D([], [], color=matplotlib.colormaps[CAND_CMAP](0.5), lw=1.3,
                       label="candidates (all samples)"),
                Line2D([], [], color=PICK_COLOR, lw=2.2, ls="--", label="picked candidate"),
                Line2D([], [], color=EXEC_COLOR, lw=3.4, label="executed plan"),
                Line2D([], [], color="black", marker="s", ms=8, ls="none", label="robot (now)"),
            ],
            loc="upper right", fontsize=9, framealpha=0.9,
        )

        self.fig.text(
            0.01, 0.012,
            "scroll=zoom  shift/ctrl+scroll=X/Y  p=pan  o=zoom box  h=home  f=fit  r=reset",
            fontsize=7.5, family="monospace", color="#777777", ha="left",
        )
        self.status = self.fig.text(
            0.99, 0.012, "", fontsize=7.5, family="monospace",
            color="#555555", ha="right",
        )
        self.set_title("Waiting for trajectory data...")
        self.fig.tight_layout(rect=(0, 0.035, 1, 1))

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.show(block=False)
        self._refresh_status()
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

    # -- view control -------------------------------------------------------
    def reset_view(self):
        self.ax.set_xlim(-self.axis_limit, self.axis_limit)
        self.ax.set_ylim(-self.axis_limit, self.axis_limit)

    def fit_view(self):
        """Frame the current data with a margin; keep it square when aspect is equal."""
        if self.bundle is None:
            self.reset_view()
            return

        points = self.bundle.all_points()
        x_lo, y_lo = points.min(axis=0)
        x_hi, y_hi = points.max(axis=0)
        x_mid, y_mid = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
        x_half = max((x_hi - x_lo) / 2.0, 1e-4)
        y_half = max((y_hi - y_lo) / 2.0, 1e-4)

        if self.equal_aspect:
            # A shared half-span keeps the axes box square instead of letting
            # matplotlib squash it to satisfy the aspect ratio.
            x_half = y_half = max(x_half, y_half)

        self.ax.set_xlim(x_mid - x_half * (1 + FIT_MARGIN), x_mid + x_half * (1 + FIT_MARGIN))
        self.ax.set_ylim(y_mid - y_half * (1 + FIT_MARGIN), y_mid + y_half * (1 + FIT_MARGIN))

    def set_aspect(self, equal: bool):
        self.equal_aspect = bool(equal)
        self.ax.set_aspect("equal" if self.equal_aspect else "auto", adjustable="box")

    def set_title(self, text: str):
        self.ax.set_title(text, fontsize=11)

    def _refresh_status(self):
        self.status.set_text(
            f"a autofit:{'on' if self.autofit else 'OFF'}   "
            f"e aspect:{'equal' if self.equal_aspect else 'free'}   "
            f"i labels:{'on' if self.show_labels else 'off'}"
        )

    # -- events -------------------------------------------------------------
    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return

        notches = getattr(event, "step", 0) or (1.0 if event.button == "up" else -1.0)
        factor = ZOOM_STEP ** (-notches)

        key = event.key or ""
        zoom_x = "control" not in key
        zoom_y = "shift" not in key
        if not (zoom_x and zoom_y):
            # A one-axis zoom is meaningless while the aspect is locked.
            self.set_aspect(False)

        if zoom_x:
            lo, hi = self.ax.get_xlim()
            self.ax.set_xlim(
                event.xdata + (lo - event.xdata) * factor,
                event.xdata + (hi - event.xdata) * factor,
            )
        if zoom_y:
            lo, hi = self.ax.get_ylim()
            self.ax.set_ylim(
                event.ydata + (lo - event.ydata) * factor,
                event.ydata + (hi - event.ydata) * factor,
            )

        self.autofit = False
        self._refresh_status()
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        # Reaching for the pan/zoom tool means the user wants manual control.
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if self.autofit and getattr(toolbar, "mode", ""):
            self.autofit = False
            self._refresh_status()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key == "f":
            self.fit_view()
        elif key == "a":
            self.autofit = not self.autofit
            if self.autofit:
                self.fit_view()
        elif key == "r":
            self.autofit = False
            self.set_aspect(True)
            self.reset_view()
        elif key == "e":
            self.set_aspect(not self.equal_aspect)
            if self.autofit:
                self.fit_view()
        elif key == "i":
            self.show_labels = not self.show_labels
            self._draw_labels()
        else:
            return

        self._refresh_status()
        self.fig.canvas.draw_idle()

    # -- drawing ------------------------------------------------------------
    def _draw_labels(self):
        """Number each candidate at its endpoint so a pick can be traced."""
        bundle = self.bundle
        candidates = None if bundle is None else bundle.candidates
        count = 0 if (candidates is None or not self.show_labels) else len(candidates)

        while len(self._cand_labels) < count:
            self._cand_labels.append(
                self.ax.text(0, 0, "", fontsize=7, ha="center", va="bottom", zorder=10)
            )

        for i, text in enumerate(self._cand_labels):
            if i >= count:
                text.set_visible(False)
                continue
            end = candidates[i, -1]
            text.set_position((float(end[0]), float(end[1])))
            text.set_text(str(i))
            text.set_color(EXEC_COLOR if i == bundle.selected_index else "#444444")
            text.set_fontweight("bold" if i == bundle.selected_index else "normal")
            text.set_visible(True)

    def update(self, bundle: Bundle):
        self.bundle = bundle

        candidates = bundle.candidates
        if candidates is not None and len(candidates):
            count = len(candidates)
            spread = np.linspace(0.12, 0.92, count) if count > 1 else np.array([0.5])
            colors = matplotlib.colormaps[CAND_CMAP](spread)
            colors[:, 3] = 0.55                      # translucent so overlaps read
            self.cand_lines.set_segments(list(candidates))
            self.cand_lines.set_color(colors)
            self.cand_ends.set_offsets(candidates[:, -1, :])
            self.cand_ends.set_color(colors)
        else:
            self.cand_lines.set_segments([])
            self.cand_ends.set_offsets(np.empty((0, 2)))

        pick = bundle.selected_index
        if candidates is not None and 0 <= pick < len(candidates):
            self.pick_line.set_data(candidates[pick, :, 0], candidates[pick, :, 1])
        else:
            self.pick_line.set_data([], [])

        selected = bundle.selected
        if selected is not None and len(selected):
            self.exec_line.set_data(selected[:, 0], selected[:, 1])
            self.exec_pts.set_data(selected[:, 0], selected[:, 1])
            self.exec_end.set_data(selected[-1:, 0], selected[-1:, 1])
        else:
            self.exec_line.set_data([], [])
            self.exec_pts.set_data([], [])
            self.exec_end.set_data([], [])

        self._draw_labels()
        if self.autofit:
            self.fit_view()

        self.set_title(f"Live AI trajectory - {bundle.describe()}")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


# -------------------------------------------------------------------- server
def run_viz_server(port: int = DEFAULT_PORT, axis_limit: float = AXIS_LIMIT,
                   autofit: bool = True):
    host = f"tcp://*:{port}"
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(host)

    viz = TrajectoryVisualizer(axis_limit=axis_limit, autofit=autofit)
    running = True

    def stop_server(_signum=None, _frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    print(f"Visualization Server Ready: {host}")
    print(f"Matplotlib backend: {BACKEND}")
    if not INTERACTIVE:
        forced = os.environ.get("MPLBACKEND", "").strip()
        why = f"MPLBACKEND={forced} forces it" if forced else "no GUI toolkit was importable"
        print(
            f"WARNING: backend '{BACKEND}' is not interactive ({why}), so no window "
            "opens and the zoom/pan controls do nothing. Unset MPLBACKEND, install a "
            "toolkit (apt install python3-tk, or pip install PyQt6), and set DISPLAY.",
            file=sys.stderr,
        )
    print("Waiting for bundle, pickle, JSON, NPY, or raw float trajectory data...")

    last_log = 0.0
    try:
        while running and plt.fignum_exists(viz.fig.number):
            if socket.poll(timeout=20, flags=zmq.POLLIN):
                message = socket.recv()
                try:
                    bundle = decode_payload(message)
                    viz.update(bundle)
                    now = time.monotonic()
                    if now - last_log >= 1.0:          # one line/sec, not one/frame
                        last_log = now
                        points = bundle.all_points()
                        print(
                            f"{len(message)} bytes -> {bundle.describe()}, "
                            f"x=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}], "
                            f"y=[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]"
                        )
                except Exception as exc:
                    print(f"Decode error: {exc}", file=sys.stderr)

            # Keep the X11 GUI responsive even while no data is arriving.
            plt.pause(0.01)
    finally:
        socket.close()
        context.term()
        plt.close("all")
        print("Visualization server stopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"ZeroMQ PULL bind port (default {DEFAULT_PORT})")
    parser.add_argument("--limit", type=float, default=AXIS_LIMIT,
                        help=f"half-width [m] of the default/reset view (default {AXIS_LIMIT})")
    parser.add_argument("--no-autofit", action="store_true",
                        help="start with the fixed default view instead of fitting to data")
    args = parser.parse_args()
    run_viz_server(port=args.port, axis_limit=args.limit, autofit=not args.no_autofit)


if __name__ == "__main__":
    main()
