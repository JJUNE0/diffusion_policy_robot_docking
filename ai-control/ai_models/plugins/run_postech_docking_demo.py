"""Docking demo plugin (2026-07-13) — flow_goal_scratch20 checkpoint.

Runs the current-generation diffusion policy (rectified flow + goal image +
raw-lidar branch + aux head, single camera room2) inside the ai-control P2
process:

    window_snapshot -> (velocity 30, room2 DINO 5, goal DINO 1, lidar crop 256)
    -> ContinuousRectifiedFlow.sample (euler, 20 steps, 8 samples, EMA weights)
    -> medoid sample (test/mpc_rank.py: beats mean by ~20%)
    -> trajectory-EMA smoothing across calls (alpha 0.3)
    -> integrate (vx, wz)@30Hz -> cumulative CommandStep(dx, dy, dtheta, dt)

Site checklist (README_DEMO.md):
  * replace ai_models/goal_image.jpg with a photo of THIS site's dock taken
    from the docked position by the room2 camera (the shipped file is only an
    example from the test set).
  * verify which LiveKit track is the room2 (=image_bottom) camera by comparing
    ai_models/reference_room2.jpg with the live streams; set VIDEO_STREAM below
    (or config video assignment) accordingly.
  * config.yml must have require_lidar: true and run_plugin: run_postech_docking_demo.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

_AI_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AI_MODELS_DIR not in sys.path:
    sys.path.insert(0, _AI_MODELS_DIR)   # makes bundled `cleandiffuser`, `dino_detector` importable

from node_sdk import CommandStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants
CKPT = os.path.join(_AI_MODELS_DIR, "scratch20_checkpoint_step_8460.pt")
GOAL_IMAGE = os.path.join(_AI_MODELS_DIR, "goal_image.jpg")
VIDEO_STREAM = "video2"          # which snapshot stream is the room2 (image_bottom) camera
OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
STEP_DT = 1.0 / 30.0
N_SAMPLES = 8
SAMPLE_STEPS = 20
TRAJ_EMA_ALPHA = 0.3
TRAJ_EMA_RESET_SEC = 0.5         # stale smoothing state is worse than none
LIDAR_CROP_R = 0.8               # utils/preprocessing.py nearest-cluster crop
LIDAR_MAX_POINTS = 256
# architecture (= flow_goal_scratch20 config.yaml)
D_MODEL, N_HEADS, DEPTH, DROPOUT = 384, 6, 12, 0.1

_STATE: Dict[str, Any] = {}      # lazy singletons: model, dino, goal feat, traj-EMA


# ---------------------------------------------------------------- init
def _init():
    if "model" in _STATE:
        return _STATE
    import torch
    from cleandiffuser.diffusion import ContinuousRectifiedFlow
    from cleandiffuser.nn_diffusion import DiT1d
    from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
    from dino_detector import DinoBatchDetector

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nn_condition = SensorFusionConditionNetwork(
        state_dim=2, obs_horizon=OBS_HORIZON, vision_horizon=5,
        d_model=D_MODEL, nhead=N_HEADS, num_layers=2, dropout=DROPOUT,
        num_image_latents=16, velocity_dim=2, velocity_dropout_prob=0.0,
        use_goal=True, num_goal_latents=16,
        use_lidar_points=True, num_lidar_latents=16,
        use_aux_pose=True, use_room1=False,
    ).to(device)
    nn_diffusion_model = DiT1d(in_dim=2, emb_dim=D_MODEL, d_model=D_MODEL,
                               n_heads=N_HEADS, depth=DEPTH, dropout=0.0).to(device)
    model = ContinuousRectifiedFlow(nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
                                    ema_rate=0.999, device=device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model_ema.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    dino = DinoBatchDetector(device=device)

    import cv2
    goal_bgr = cv2.imread(GOAL_IMAGE)
    if goal_bgr is None:
        raise FileNotFoundError(f"goal image missing: {GOAL_IMAGE} (site checklist!)")
    goal_t = _frames_to_tensor([goal_bgr], device)                     # [1,3,H,W]
    with torch.no_grad():
        gf, _, _ = dino.get_heatmap(goal_t)
    _STATE.update(dict(
        model=model, dino=dino, device=device,
        act_min=np.asarray(ckpt["action_min"], np.float32),
        act_scale=np.asarray(ckpt["action_scale"], np.float32),
        goal_feat=gf.view(1, 1, 196, 768).repeat(N_SAMPLES, 1, 1, 1),
        prev_ema=None, last_call=0.0,
    ))
    logger.info("docking demo model loaded: %s (device=%s)", os.path.basename(CKPT), device)
    return _STATE


def _frames_to_tensor(bgr_frames, device):
    """list[BGR HxWx3 uint8] -> [T,3,H,W] float RGB in [0,1] (training convention)."""
    import torch
    arr = np.stack([f[:, :, ::-1] for f in bgr_frames]).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()).to(device)


# ---------------------------------------------------------------- inputs
def _crop_scan(points_xy: np.ndarray) -> Tuple[np.ndarray, int]:
    """utils/preprocessing.py nearest-cluster crop: keep points within
    LIDAR_CROP_R of the nearest scan point, closest LIDAR_MAX_POINTS, zero-pad."""
    out = np.zeros((LIDAR_MAX_POINTS, 2), np.float32)
    pts = points_xy[np.isfinite(points_xy).all(axis=1)]
    if len(pts) == 0:
        return out, 0
    dn = np.hypot(pts[:, 0], pts[:, 1])
    anchor = pts[dn.argmin()]
    dk = np.hypot(*(pts - anchor).T)
    keep = pts[dk < LIDAR_CROP_R]
    if len(keep) > LIDAR_MAX_POINTS:
        dk2 = np.hypot(*(keep - anchor).T)
        keep = keep[np.argsort(dk2)[:LIDAR_MAX_POINTS]]
    out[:len(keep)] = keep
    return out, len(keep)


def _zeros(config) -> List[CommandStep]:
    n = int(getattr(config, "inference_size", 16))
    return [CommandStep(0.0, 0.0, 0.0, (i + 1) * STEP_DT) for i in range(n)]


# ---------------------------------------------------------------- inference
def inference_fn(window_snapshot, latency_marks, config) -> List[CommandStep]:
    try:
        return _infer(window_snapshot, config)
    except Exception:
        logger.exception("docking demo inference failed -> zero horizon")
        return _zeros(config)


def _infer(snap, config) -> List[CommandStep]:
    import torch
    st = _init()
    device = st["device"]

    frames = snap.get(VIDEO_STREAM, ([], []))[0]
    enc = snap.get("encoder", ([], []))[0]
    lidar = snap.get("lidar", ([], []))[0]
    if len(frames) < OBS_HORIZON or len(enc) < OBS_HORIZON or len(lidar) < 1:
        logger.warning("window not ready (video=%d enc=%d lidar=%d) -> zeros",
                       len(frames), len(enc), len(lidar))
        return _zeros(config)

    # velocity history: last 30 (vx, wz), normalized like training
    vel = np.array([[float(e["vx"]), float(e["wz"])] for e in enc[-OBS_HORIZON:]], np.float32)
    vel_n = 2.0 * (vel - st["act_min"]) / st["act_scale"] - 1.0

    # vision: the exact strided frames training used (t-29+{0,6,12,18,24})
    hist = frames[-OBS_HORIZON:]
    sparse = [hist[i] for i in range(0, OBS_HORIZON, VISION_STRIDE)]
    with torch.no_grad():
        vf, _, _ = st["dino"].get_heatmap(_frames_to_tensor(sparse, device))
    dino_feat2 = vf.view(1, len(sparse), 196, 768).repeat(N_SAMPLES, 1, 1, 1)

    # lidar: latest scan -> nearest-cluster crop
    pts = np.array([[p["x"], p["y"]] for p in lidar[-1].get("points", [])], np.float32)
    crop, npts = _crop_scan(pts if pts.size else np.zeros((0, 2), np.float32))

    context = {
        "velocity": torch.from_numpy(vel_n).unsqueeze(0).to(device).repeat(N_SAMPLES, 1, 1),
        "dino_feat2": dino_feat2,
        "goal_feat2": st["goal_feat"],
        "goal_mask": torch.ones(N_SAMPLES, device=device),
        "lidar_points": torch.from_numpy(crop).unsqueeze(0).to(device).repeat(N_SAMPLES, 1, 1),
        "lidar_npoints": torch.tensor([npts], device=device).repeat(N_SAMPLES),
    }
    with torch.no_grad():
        prior = torch.randn(N_SAMPLES, HORIZON, 2, device=device)
        out = st["model"].sample(solver="euler", w_cfg=1, prior=prior, condition_cfg=context,
                                 n_samples=N_SAMPLES, sample_steps=SAMPLE_STEPS, use_ema=True)
        out = out[0] if isinstance(out, tuple) else out
        res = ((out.cpu().numpy() + 1.0) / 2.0 * st["act_scale"] + st["act_min"])  # [N,H,2]

    # medoid sample (robust aggregation, see test/mpc_rank.py)
    d2 = ((res[:, None] - res[None]) ** 2).sum(axis=(2, 3))
    cur = res[d2.sum(axis=1).argmin()]                                  # [H,2]

    # trajectory-EMA smoothing across calls (reset after a gap)
    now = time.monotonic()
    if st["prev_ema"] is None or now - st["last_call"] > TRAJ_EMA_RESET_SEC:
        ema = cur
    else:
        ema = TRAJ_EMA_ALPHA * cur + (1 - TRAJ_EMA_ALPHA) * st["prev_ema"]
    st["prev_ema"], st["last_call"] = ema.copy(), now

    # integrate (vx, wz) -> cumulative robot-frame pose steps
    n = int(getattr(config, "inference_size", 16))
    steps, x, y, th = [], 0.0, 0.0, 0.0
    for i in range(min(n, HORIZON)):
        vx, wz = float(ema[i, 0]), float(ema[i, 1])
        x += vx * math.cos(th) * STEP_DT
        y += vx * math.sin(th) * STEP_DT
        th += wz * STEP_DT
        steps.append(CommandStep(x, y, th, (i + 1) * STEP_DT))
    return steps
