"""Docking demo plugin (2026-07-13, extended 2026-07-22) — generic across the
flow_goal_scratch20 (07-13 demo) AND the graft6/wave2/wave3/auxfb checkpoint
generations (07-17 onward).

Camera naming (2026-07-22): this file names the two cameras by their physical
identity -- **usb0** (config.yml's `livekit_tracks_video1: usb-0`, the
optional 2nd camera some checkpoints don't use) and **orbbec0**
(`livekit_tracks_video2: orbbec-0`, the always-required base camera) -- NOT
"room1"/"room2" (the old naming, now retired from config/log/variable names
here). The one place "room1" still appears is the `use_room1=` constructor
kwarg on SensorFusionConditionNetwork itself (cleandiffuser/nn_condition/
sensor_fusion_condition.py, shared with the training pipeline) -- that name
belongs to the model's fixed interface, out of scope for a deployment-only
rename. Likewise the condition dict keys the model requires (`dino_feat1`,
`dino_feat2`, `goal_feat1`, `goal_feat2`) are dictated by that same forward()
contract and are NOT renamed; this plugin maps usb0 -> the *1 keys and
orbbec0 -> the *2 keys when building context.

Runs the sensor-fusion diffusion policy inside the ai-control P2 process:

    window_snapshot -> (velocity 30, orbbec0 DINO 5, [usb0 DINO 5], goal DINO 1,
                         lidar crop 256, [goal-lidar crop 256])
    -> model.sample (euler|dpmsolver++_2M per demo_backbone, N samples, EMA)
    -> medoid|mean sample (test/mpc_rank.py: medoid beats mean by ~20%)
    -> trajectory-EMA / chunk+warm-start (shared cleandiffuser/rollout_core.py)
    -> integrate (vx, wz)@30Hz -> cumulative CommandStep(dx, dy, dtheta, dt)

Sensor architecture (goal/lidar/goal-lidar/aux/aux-feedback/usb0-camera) is
ALL auto-detected from the demo_ckpt's weight keys -- swap demo_ckpt to any
checkpoint generation freely, no code change needed for that part.

`demo_backbone` is the ONE thing that can't be auto-detected (rectified_flow
and ddpm share identical parameter names by design -- see
utils/setups.py _select_backbone's docstring in the main repo) and MUST be
set to match whatever demo_ckpt was actually trained with:
  * 07-13 scratch20 demo checkpoint          -> demo_backbone: rectified_flow
  * graft6/wave2/wave3/auxfb (07-17 onward)  -> demo_backbone: ddpm

Site checklist (README_DEMO.md):
  * set demo_goal_image_orbbec0 to a photo of THIS site's dock taken from
    the docked position by the orbbec0 camera (the bundled goal_ref image is
    only an example from the test set).
  * verify which LiveKit track is orbbec0 (=image_bottom) by comparing
    ai_models/reference_room2.jpg with the live streams; set demo_video2 in
    config.yml accordingly.
  * 2-camera checkpoints require demo_goal_image_usb0 (a docked-position
    photo from the USB camera) and demo_video1 set to the USB LiveKit stream.
    Orbbec-only checkpoints do not load or require either USB input. Set
    demo_camera_mode to match the checkpoint; _init() rejects mismatches.
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

from .trajectory_viz_sender import send_trajectory_bundle

_AI_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AI_MODELS_DIR not in sys.path:
    sys.path.insert(0, _AI_MODELS_DIR)   # makes bundled `cleandiffuser`, `dino_detector` importable

from node_sdk import CommandStep

logger = logging.getLogger(__name__)

from ai_control_node.csv_debug import CsvLogger  # noqa: E402
# 모델(diffusion) raw 출력 velocity 그 자체 진단 CSV. AI_CONTROL_CSV_LOG=1 일 때만.
_MODEL_CSV = CsvLogger("model_output.csv", ["t", "infer", "step", "vx_raw", "wz_raw"])
_INFER_IDX = 0

# ── exponential continuity smoothing (ai-control-old 이식) ─────────────────
# 실행하는 첫 스텝들을 이전 궤적(1스텝 shift)에 앵커해 실효 속도의 급변(급가속)을 억제.
# beta[i]=1-exp(-γ·i) (초반≈0→후반=1): 초반=이전궤적, 후반=현재. config demo_continuity_blend
# (γ, 기본 0.5; <=0이면 off). 0.5s 이상 공백이면 stale prev 리셋.
_PREV_SMOOTHED = None
_LAST_SMOOTH_T = 0.0
_SMOOTH_RESET_SEC = 0.5


def _shift_prev(p):
    s = np.zeros_like(p)
    s[:-1] = p[1:]
    s[-1] = p[-1]
    return s


def _blend_schedule(h, gamma):
    idx = np.arange(h, dtype=np.float32)
    beta = 1.0 - np.exp(-gamma * idx)
    if h > 1:
        beta = beta / max(float(beta[-1]), 1e-8)
    return np.clip(beta, 0.0, 1.0)


def _continuity_smooth(ema, config):
    """ema[H,2](raw)에 continuity smoothing 적용 → [H,2] numpy 반환. blend<=0이면 off."""
    global _PREV_SMOOTHED, _LAST_SMOOTH_T
    arr = ema.detach().cpu().numpy() if hasattr(ema, "detach") else np.asarray(ema, dtype=np.float32)
    arr = arr.astype(np.float32)
    blend = float(getattr(config, "demo_continuity_blend", 0.5))
    if blend <= 0.0:
        _PREV_SMOOTHED = None
        return arr
    # old 충실: 시간 리셋 없음(prev 항상 유지). shape 바뀔 때만 초기화.
    if _PREV_SMOOTHED is None or _PREV_SMOOTHED.shape != arr.shape:
        out = arr
    else:
        beta = _blend_schedule(arr.shape[0], blend)[:, None]
        out = beta * arr + (1.0 - beta) * _shift_prev(_PREV_SMOOTHED)
    _PREV_SMOOTHED = out
    return out

# ---------------------------------------------------------------- constants
# Runtime knobs now live in config.yml (single source of truth) and reach this
# plugin via the AIControlConfig passed into inference_fn -> _init(config). See
# the `demo_*` fields in ai_control_node/config.py. To re-tune on site, edit
# config.yml (volume-mounted into the container) and restart — no code edit, no
# env. Defaults below (config dataclass) reproduce the validated 2026-07-13 demo.
#   demo_ckpt      checkpoint path relative to ai_models/ (swap the model)
#   demo_agg       sample aggregation: "medoid" (default) | "mean"
#   demo_ema       trajectory-EMA alpha 0..1 (higher = less smoothing/more
#                  decisive; 1.0 = no smoothing). default 0.3
#   demo_nsamples  number of trajectory samples (default 8; lower if too slow)
#   demo_steps     denoising sampling steps (default 20; 50-100 = slower but
#                  higher trajectory quality, like the old 100-step baseline)
#   demo_use_ema   use EMA weights (default true; 0.999-gen checkpoints only)
#   demo_camera_mode  "auto" | "orbbec" | "orbbec_usb"; explicit modes are
#                     checked against the selected checkpoint's sensor spec
#   demo_video2    which snapshot stream is the orbbec0 (base) camera (default video2)
#   demo_video1    which snapshot stream is the usb0 (2nd) camera -- only 2-camera
#                  checkpoints need this
#   demo_goal_image_orbbec0  docked-position photo, orbbec0 camera
#   demo_goal_image_usb0     docked-position photo, usb0 camera (2-camera goal ckpts)
#   demo_goal_lidar          docked-position scan .npy for glidar checkpoints
GOAL_IMAGE_ORBBEC0_DEFAULT = os.path.join(
    _AI_MODELS_DIR, "goal_ref", "goal_image_orbbec0.jpg")
GOAL_IMAGE_USB0_DEFAULT = os.path.join(
    _AI_MODELS_DIR, "goal_ref", "goal_image_usb0.jpg")
GOAL_LIDAR_DEFAULT = os.path.join(_AI_MODELS_DIR, "goal_lidar_scan.npy")        # use_goal_lidar=True ckpts only
OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
# R-NoGoal/R-Goal/R-Geo (token-sequence + cross-attention) checkpoints use a
# DIFFERENT observation window than the legacy models above: 60-frame wheel
# history and RGB sampled at stride 12 over that 60 (vs 30 / stride 6). Feeding
# the legacy window would silently change the temporal spacing of the 5 vision
# frames (0.2s vs 0.4s apart) and quietly degrade the policy, so these are read
# per-checkpoint from its saved config rather than hardcoded. See
# docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md.
STEP_DT = 1.0 / 30.0
TRAJ_EMA_RESET_SEC = 0.5         # stale smoothing state is worse than none
LIDAR_CROP_R = 0.8               # utils/preprocessing.py nearest-cluster crop
LIDAR_MAX_POINTS = 256
# architecture (= flow_goal_scratch20 config.yaml)
D_MODEL, N_HEADS, DEPTH, DROPOUT = 384, 6, 12, 0.1

_STATE: Dict[str, Any] = {}      # lazy singletons: model, dino, goal feat, traj-EMA


def _validate_camera_mode(config, use_usb0: bool) -> str:
    """Validate config intent against the checkpoint's fixed sensor graph."""
    raw = str(getattr(config, "demo_camera_mode", "auto") or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "orbbec": "orbbec",
        "orbbec_only": "orbbec",
        "1cam": "orbbec",
        "orbbec_usb": "orbbec_usb",
        "usb_orbbec": "orbbec_usb",
        "2cam": "orbbec_usb",
    }
    if raw not in aliases:
        raise ValueError(
            f"unknown demo_camera_mode={raw!r}; expected auto|orbbec|orbbec_usb")
    requested = aliases[raw]
    trained = "orbbec_usb" if use_usb0 else "orbbec"
    if requested == "auto":
        return trained
    if requested != trained:
        raise ValueError(
            f"demo_camera_mode={requested!r}, but checkpoint sensors require "
            f"{trained!r}. Select a matching demo_ckpt; camera branches cannot be "
            f"added to or removed from trained weights at deployment time.")
    return requested


def _token_camera_plan(sensors: Dict[str, dict]) -> Dict[str, bool]:
    """Derive live-camera and goal-image requirements from sensor specs."""
    visual_encoders = {"dino_image", "reloc3r_relation", "reloc3r_goal_pair"}
    goal_encoders = {"reloc3r_relation", "reloc3r_goal_pair"}
    visual_items = [(name, spec) for name, spec in sensors.items()
                    if spec.get("encoder") in visual_encoders]
    goal_items = [(name, spec) for name, spec in sensors.items()
                  if spec.get("encoder") in goal_encoders]

    def is_usb(name, spec):
        source = str(spec.get("source", name))
        return source.endswith("_top") or name.endswith("_top")

    return {
        "use_orbbec": ("goal" in sensors or "geometry" in sensors
                       or any(not is_usb(name, spec)
                              for name, spec in visual_items)),
        "use_usb": any(is_usb(name, spec) for name, spec in visual_items),
        "need_orbbec_goal": ("goal" in sensors or "geometry" in sensors
                             or any(not is_usb(name, spec)
                                    for name, spec in goal_items)),
        "need_usb_goal": any(is_usb(name, spec) for name, spec in goal_items),
    }


def _resolve_ai_models_path(configured_path, default_path):
    """Resolve a config asset path without depending on the process cwd.

    Relative paths are based at ``ai_models/``. Keep accepting the older
    ``ai_models/...`` spelling used by existing config.yml files, and pass
    absolute container paths through unchanged.
    """
    raw = str(configured_path or "").strip()
    if not raw:
        return os.path.normpath(default_path)
    path = os.path.expanduser(raw)
    if os.path.isabs(path):
        return os.path.normpath(path)

    path = os.path.normpath(path)
    ai_models_name = os.path.basename(_AI_MODELS_DIR)
    if path == ai_models_name or path.startswith(ai_models_name + os.sep):
        return os.path.normpath(os.path.join(os.path.dirname(_AI_MODELS_DIR), path))
    return os.path.normpath(os.path.join(_AI_MODELS_DIR, path))


# ---------------------------------------------------------------- init
def _init(config):
    if "model" in _STATE:
        return _STATE
    import torch
    from cleandiffuser.diffusion import ContinuousDiffusionSDE, ContinuousRectifiedFlow
    from cleandiffuser.nn_diffusion import DiT1d
    from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
    from dino_detector import DinoBatchDetector

    # resolve runtime knobs from config.yml (single source of truth)
    ckpt_path = _resolve_ai_models_path(
        getattr(config, "demo_ckpt", ""),
        os.path.join(_AI_MODELS_DIR, "scratch20_checkpoint_step_8460.pt"))
    n_samples = int(getattr(config, "demo_nsamples", 8))
    sample_steps = int(getattr(config, "demo_steps", 20))
    agg = str(getattr(config, "demo_agg", "medoid")).lower()   # "medoid" | "mean"
    video_stream_orbbec0 = getattr(config, "demo_video2", "video2")   # base camera stream
    ema_alpha = float(getattr(config, "demo_ema", 0.3))        # higher = decisive (less smoothing); 1.0 = off
    # Backbone (2026-07-22): orthogonal to the sensor architecture below, so it
    # CANNOT be auto-detected from weight keys (utils/setups.py
    # _select_backbone's own comment: "one config flag, orthogonal to the
    # sensors" -- both backbones share the identical DiT1d/condition-net
    # param names). Must be set explicitly per checkpoint generation, same as
    # demo_ckpt itself. The 07-13 scratch20 demo checkpoint is rectified_flow;
    # all graft6/wave2/auxfb checkpoints (2026-07-17 onward) are ddpm.
    backbone = str(getattr(config, "demo_backbone", "rectified_flow")).lower()
    if backbone in ("rectified_flow", "flow", "rf"):
        Backbone, solver = ContinuousRectifiedFlow, "euler"
    elif backbone in ("ddpm", "diffusion", "sde"):
        Backbone, solver = ContinuousDiffusionSDE, "ode_dpmsolver++_2M"
    else:
        raise ValueError(f"unknown demo_backbone '{backbone}' (expected rectified_flow|ddpm)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    # Sensor architecture must match the checkpoint. ALL of it is auto-detected
    # from the weight keys themselves (same evidence-from-weights approach
    # docs/ablation_study_2026-07.md §2.12 used for use_room1) rather than
    # hardcoded, because the graft6/wave2/auxfb checkpoint generations vary
    # every one of these flags independently (e.g. graft_goallidar has
    # use_goal=False/use_aux_pose=False; graft_auxfb_full has both True) --
    # hardcoding any of them silently builds the WRONG architecture and
    # load_state_dict (strict=True below) hard-fails with a missing/unexpected
    # key error. demo_ckpt can now swap across ANY of these generations freely.
    # --- R-NoGoal/R-Goal/R-Geo (token-sequence + cross-attention DiT) ---------
    # These checkpoints have a completely different module layout
    # (condition.encoders.* / condition.tfm.* / diffusion.blocks.*) from the
    # legacy SensorFusionConditionNetwork below, so the legacy builder's
    # strict load_state_dict hard-fails on them. Detect and branch.
    if any(k.startswith("condition.encoders.") for k in sd):
        return _init_token_sequence(config, ckpt, sd, ckpt_path, device, Backbone,
                                    n_samples, sample_steps, agg, ema_alpha,
                                    video_stream_orbbec0)

    use_room1 = any(k.startswith("condition.room1_resampler") for k in sd)
    camera_mode = _validate_camera_mode(config, use_room1)
    use_goal = any(k.startswith("condition.goal_resampler2") or k == "condition.goal_slot_emb" for k in sd)
    use_lidar_points = any(k.startswith("condition.lidar_resampler") for k in sd)
    use_aux_pose = any(k.startswith("condition.aux_pose_head") or k.startswith("condition.aux_head") for k in sd)
    use_goal_lidar = any(k.startswith("condition.goal_lidar") for k in sd)
    use_aux_feedback = any(k.startswith("condition.aux_feedback_proj") for k in sd)
    nn_condition = SensorFusionConditionNetwork(
        state_dim=2, obs_horizon=OBS_HORIZON, vision_horizon=5,
        d_model=D_MODEL, nhead=N_HEADS, num_layers=2, dropout=DROPOUT,
        num_image_latents=16, velocity_dim=2, velocity_dropout_prob=0.0,
        use_goal=use_goal, num_goal_latents=16,
        use_lidar_points=use_lidar_points, num_lidar_latents=16,
        use_aux_pose=use_aux_pose, use_room1=use_room1,
        use_goal_lidar=use_goal_lidar,
        use_aux_feedback=use_aux_feedback,
    ).to(device)
    # Runtime on/off for aux-pose feedback (2026-07-22, config-controlled): only
    # meaningful when the checkpoint actually HAS the feedback projection
    # (use_aux_feedback above); this just decides whether it's exercised.
    # Independent of demo_ckpt swaps -- a non-auxfb checkpoint simply has
    # nothing to toggle (aux_feedback_enabled has no effect if use_aux_feedback
    # is False, see sensor_fusion_condition.py forward()).
    nn_condition.aux_feedback_enabled = bool(getattr(config, "demo_aux_feedback", True))
    nn_diffusion_model = DiT1d(in_dim=2, emb_dim=D_MODEL, d_model=D_MODEL,
                               n_heads=N_HEADS, depth=DEPTH, dropout=0.0).to(device)
    # x_min/x_max=-1/1 → 각 denoising 스텝에서 x0를 [-1,1]로 clip (clip_prediction).
    # ai-control-old의 per-step `x_theta.clamp(-1,1)`과 동일 — 정규화 범위 초과(→ 과속/급가속) 방지.
    model = Backbone(nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
                     ema_rate=0.999, device=device, x_min=-1.0, x_max=1.0)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model_ema.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    # All three goal-reference paths are config-overridable (fall back to the
    # ai_models/ default filename if unset) -- so a new site, or a second dock
    # geometry, only needs a config.yml edit, never a code change.
    goal_image_orbbec0_path = _resolve_ai_models_path(
        getattr(config, "demo_goal_image_orbbec0", ""), GOAL_IMAGE_ORBBEC0_DEFAULT)
    goal_image_usb0_path = _resolve_ai_models_path(
        getattr(config, "demo_goal_image_usb0", ""), GOAL_IMAGE_USB0_DEFAULT)
    gl_path = _resolve_ai_models_path(
        getattr(config, "demo_goal_lidar", ""), GOAL_LIDAR_DEFAULT)

    # glidar models need the DOCKED-position scan as a goal-lidar input (analog
    # of the goal image). Missing -> hard error: running a glidar model
    # without it silently drops a trained-on modality.
    goal_lidar = None
    if use_goal_lidar:
        if not os.path.exists(gl_path):
            raise FileNotFoundError(
                f"glidar model needs a docked-position scan but {gl_path} is missing. "
                f"Capture one at the dock (crop like utils/preprocessing.py: nearest-cluster "
                f"0.8m, <=256 pts) and save as .npy (path configurable via demo_goal_lidar), "
                f"or use a non-glidar demo_ckpt.")
        pts = np.load(gl_path).astype(np.float32)
        crop, npts = _crop_scan(pts)
        goal_lidar = (torch.from_numpy(crop).unsqueeze(0).to(device).repeat(n_samples, 1, 1),
                      torch.tensor([npts], device=device).repeat(n_samples))

    dino = DinoBatchDetector(device=device)

    import cv2
    goal_bgr = cv2.imread(goal_image_orbbec0_path)
    if goal_bgr is None:
        raise FileNotFoundError(f"goal image missing: {goal_image_orbbec0_path} (site checklist! "
                                f"path configurable via demo_goal_image_orbbec0)")
    goal_t = _frames_to_tensor([goal_bgr], device)                     # [1,3,H,W]
    with torch.no_grad():
        gf, _, _ = dino.get_heatmap(goal_t)
    goal_feat_orbbec0 = gf.view(1, 1, 196, 768).repeat(n_samples, 1, 1, 1)

    # usb0 (2nd camera) goal image + stream id -- ALL graft6/wave2/auxfb
    # checkpoints (2026-07-17 onward) use it (auto-detected above as
    # use_room1), unlike the 07-13 scratch20 demo checkpoint this plugin was
    # originally written for. SITE CHECKLIST: a photo of the docked position
    # from the usb0 camera must be captured at this site -- there is no
    # generic default, same as the orbbec0 goal image itself.
    goal_feat_usb0, video_stream_usb0 = None, None
    if use_room1:
        video_stream_usb0 = getattr(config, "demo_video1", "video1")
        goal1_bgr = cv2.imread(goal_image_usb0_path)
        if goal1_bgr is None:
            raise FileNotFoundError(
                f"this checkpoint uses the usb0 camera (2-camera) but {goal_image_usb0_path} is "
                f"missing. Capture a photo of the docked position from the usb0/{video_stream_usb0} "
                f"camera and save it there (path configurable via demo_goal_image_usb0), or use "
                f"a demo_ckpt that doesn't need the usb0 camera (e.g. the 07-13 scratch20 checkpoint).")
        goal1_t = _frames_to_tensor([goal1_bgr], device)
        with torch.no_grad():
            gf1, _, _ = dino.get_heatmap(goal1_t)
        goal_feat_usb0 = gf1.view(1, 1, 196, 768).repeat(n_samples, 1, 1, 1)
    # EMA is trustworthy only for ema_rate<=0.999 checkpoints (the 0.9999 gen is
    # contaminated). All demo-bundled models are 0.999, so default True; set
    # demo_use_ema: false if ever loading an old checkpoint.
    use_ema = bool(getattr(config, "demo_use_ema", True))
    # Shared rollout core (single source of truth with the repo's
    # test/eval_openloop_metrics.py + scripts/inference_ema_v2.py). This plugin
    # is called once per node cycle and emits a whole trajectory; the controller
    # holds the cross-call trajectory-EMA state (with wall-clock gap reset) and
    # does sampling + medoid/mean aggregation + EMA exactly as the old inline
    # block did. Chunking is a NODE-level concern here (inference_size /
    # action_horizon), so chunk_k stays 1 -- warm-start off by default; both are
    # opt-in via config (demo_chunk_warm) once the K sweep picks a value.
    from cleandiffuser.rollout_core import RolloutController
    warm_start = bool(getattr(config, "demo_warm_start", False))
    warm_level = float(getattr(config, "demo_warm_level", 0.3))
    ctrl = RolloutController(
        model, solver=solver, sample_steps=sample_steps, use_ema=use_ema,
        n_samples=n_samples, horizon=HORIZON, w_cfg=1, agg=agg, traj_ema_alpha=ema_alpha,
        warm_start=warm_start, warm_level=warm_level, ema_reset_sec=TRAJ_EMA_RESET_SEC,
        device=device)
    _STATE.update(dict(
        model=model, dino=dino, device=device, use_ema=use_ema, ctrl=ctrl,
        use_goal_lidar=use_goal_lidar, goal_lidar=goal_lidar,
        use_usb0=use_room1, video_stream_usb0=video_stream_usb0, goal_feat_usb0=goal_feat_usb0,
        act_min=np.asarray(ckpt["action_min"], np.float32),
        act_scale=np.asarray(ckpt["action_scale"], np.float32),
        goal_feat_orbbec0=goal_feat_orbbec0,
        n_samples=n_samples, sample_steps=sample_steps, agg=agg,
        video_stream_orbbec0=video_stream_orbbec0, ema_alpha=ema_alpha,
    ))
    logger.info("docking demo loaded: %s | backbone=%s solver=%s use_usb0=%s(video1=%s) "
                "goal=%s goal_lidar=%s aux_feedback=%s(enabled=%s) agg=%s ema_alpha=%.2f "
                "n_samples=%d steps=%d use_ema=%s video2=%s warm_start=%s(level=%.2f)",
                os.path.basename(ckpt_path), backbone, solver, use_room1, video_stream_usb0,
                use_goal, use_goal_lidar, use_aux_feedback, nn_condition.aux_feedback_enabled,
                agg, ema_alpha, n_samples, sample_steps, use_ema, video_stream_orbbec0, warm_start, warm_level)
    return _STATE


def _init_token_sequence(config, ckpt, sd, ckpt_path, device, Backbone,
                         n_samples, sample_steps, agg, ema_alpha, video_stream_orbbec0):
    """Build an R-NoGoal/R-Goal/R-Geo checkpoint (TokenSequenceFusionCondition +
    DiTCrossAttn1d). Unlike the legacy branch, the architecture here is driven by
    the `sensors:` dict, which is NOT recoverable from the weights alone (window
    lengths and strides are data-side), so the run's saved config.yaml must sit
    next to the .pt as `<ckpt_stem>.config.yaml` or `config.yaml`.
    """
    import torch
    from omegaconf import OmegaConf
    from cleandiffuser.nn_condition.token_sequence_condition import TokenSequenceFusionCondition
    from cleandiffuser.nn_diffusion.dit import DiTCrossAttn1d
    from cleandiffuser.rollout_core import RolloutController

    # `${hydra:runtime.cwd}` appears in the saved sensors[*].file paths. Those
    # point at training h5 caches that do not exist (and are not needed) on the
    # robot -- the plugin feeds live sensor data -- but OmegaConf still has to
    # resolve the interpolation to read the rest of the spec.
    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver("hydra", lambda _k: ".")
    stem = os.path.splitext(ckpt_path)[0]
    cfg_path = next((p for p in (stem + ".config.yaml", stem + "_config.yaml",
                                 os.path.join(os.path.dirname(ckpt_path), "config.yaml"))
                     if os.path.exists(p)), None)
    if cfg_path is None:
        raise FileNotFoundError(
            f"{os.path.basename(ckpt_path)} is a token-sequence (R-NoGoal/R-Goal/R-Geo) "
            f"checkpoint, whose architecture is defined by its `sensors:` spec. Copy the "
            f"run's config.yaml next to the .pt as {os.path.basename(stem)}.config.yaml.")
    cfg = OmegaConf.load(cfg_path)
    sensors = OmegaConf.to_container(cfg.sensors, resolve=True)

    nn_condition = TokenSequenceFusionCondition(
        sensors=sensors, d_model=int(cfg.d_model), nhead=int(cfg.n_heads),
        num_layers=int(cfg.get("condition_num_layers", 4)), dropout=float(cfg.dropout),
    ).to(device)
    nn_diffusion_model = DiTCrossAttn1d(
        in_dim=2, emb_dim=int(cfg.d_model), d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads), depth=int(cfg.depth), dropout=0.0).to(device)
    model = Backbone(nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
                     ema_rate=float(cfg.get("ema_rate", 0.999)), device=device,
                     x_min=-1.0, x_max=1.0)
    model.model.load_state_dict(sd)
    model.model_ema.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    dino = None

    # Observation window: defaults come from the checkpoint's own sensors spec
    # (the values it was trained on), but each is overridable from config.yml so
    # a window can be re-tuned on site without editing code or retraining.
    # Overriding changes what the policy sees vs training -- log loudly.
    wheel = sensors.get("wheel", {})
    relation_specs = {name: spec for name, spec in sensors.items()
                      if spec.get("encoder") == "reloc3r_relation"}
    pair_specs = {name: spec for name, spec in sensors.items()
                  if spec.get("encoder") == "reloc3r_goal_pair"}
    rgb_specs = {name: spec for name, spec in sensors.items()
                 if spec.get("encoder") == "dino_image"}
    lidar_specs = {name: spec for name, spec in sensors.items()
                   if spec.get("encoder") == "pointcloud"}
    visual_specs = (list(rgb_specs.values()) + list(relation_specs.values())
                    + list(pair_specs.values()))
    if rgb_specs or "goal" in sensors:
        from dino_detector import DinoBatchDetector
        dino = DinoBatchDetector(device=device)
    obs_h = int(wheel.get("horizon", cfg.get("obs_horizon", 60)))
    vis_stride = int(visual_specs[0].get("stride", 12)) if visual_specs else 12
    vis_n = int(visual_specs[0].get("horizon", 5)) if visual_specs else 5
    for key, name, trained in (("demo_obs_horizon", "obs_h", obs_h),
                               ("demo_vision_stride", "vis_stride", vis_stride),
                               ("demo_vision_frames", "vis_n", vis_n)):
        ov = getattr(config, key, 0)
        if ov:
            logger.warning("config.yml %s=%d OVERRIDES the checkpoint's trained %s=%d",
                           key, int(ov), name, trained)
            if name == "obs_h":
                obs_h = int(ov)
            elif name == "vis_stride":
                vis_stride = int(ov)
            else:
                vis_n = int(ov)
    def _is_top(name, spec):
        return str(spec.get("source", name)).endswith("_top") or name.endswith("_top")

    camera_plan = _token_camera_plan(sensors)
    use_usb0 = camera_plan["use_usb"]
    camera_mode = _validate_camera_mode(config, use_usb0)
    need_bottom_goal = camera_plan["need_orbbec_goal"]
    need_top_goal = camera_plan["need_usb_goal"]

    goal_feat_orbbec0 = goal_feat_usb0 = None
    goal_bgr = goal_top_bgr = None
    if need_bottom_goal or need_top_goal:
        import cv2
    if need_bottom_goal:
        gp = _resolve_ai_models_path(
            getattr(config, "demo_goal_image_orbbec0", ""), GOAL_IMAGE_ORBBEC0_DEFAULT)
        goal_bgr = cv2.imread(gp)
        if goal_bgr is None:
            raise FileNotFoundError(
                f"goal image missing: {gp} "
                f"(config: demo_goal_image_orbbec0={getattr(config, 'demo_goal_image_orbbec0', '')!r})")
    if need_top_goal:
        gp_top = _resolve_ai_models_path(
            getattr(config, "demo_goal_image_usb0", ""), GOAL_IMAGE_USB0_DEFAULT)
        goal_top_bgr = cv2.imread(gp_top)
        if goal_top_bgr is None:
            raise FileNotFoundError(
                f"top-camera goal image missing: {gp_top} "
                f"(config: demo_goal_image_usb0={getattr(config, 'demo_goal_image_usb0', '')!r})")
    if "goal" in sensors:
        with torch.no_grad():
            gf, _, _ = dino.get_heatmap(_frames_to_tensor([goal_bgr], device))
        goal_feat_orbbec0 = gf.view(1, 196, 768).repeat(n_samples, 1, 1)

    # Live Reloc3r geometry token. Both the goal DINO feature above and this use
    # the SAME goal image (the bottom/orbbec0 camera's docked-position photo),
    # matching training, where `goal` and `geometry` both derive from
    # dino_bottom / reloc3r_bottom. ~61 ms per inference call (measured), i.e.
    # ~19% of the 1067 ms period at inference_size=32.
    geometry = None
    if "geometry" in sensors:
        from reloc3r_geometry import Reloc3rGeometry
        geometry = Reloc3rGeometry(
            goal_bgr, device=device,
            calib_path=(getattr(config, "demo_reloc3r_calib", "") or None))

    relation_extractors = {}
    if relation_specs:
        from reloc3r_geometry import Reloc3rRelationFeatures
        by_camera = {"bottom": [], "top": []}
        for name, spec in relation_specs.items():
            camera = "top" if _is_top(name, spec) else "bottom"
            by_camera[camera].append((name, spec))
            source = str(spec.get("source", name))
            if "dec1" not in source and "dec2" not in source:
                raise ValueError(
                    f"relation sensor {name!r} source must contain dec1 or dec2, got {source!r}")
        for camera, entries in by_camera.items():
            if not entries:
                continue
            temporal = {(int(s.get("horizon", 5)), int(s.get("stride", 12)))
                        for _, s in entries}
            if len(temporal) != 1:
                raise ValueError(
                    f"all {camera} relation streams must share horizon/stride, got {temporal}")
            goal_image = goal_top_bgr if camera == "top" else goal_bgr
            relation_extractors[camera] = Reloc3rRelationFeatures(
                goal_image, device=device)

    pair_extractors = {}
    if pair_specs:
        from reloc3r_geometry import Reloc3rGoalPairFeatures
        for name, spec in pair_specs.items():
            camera = "top" if _is_top(name, spec) else "bottom"
            temporal = (int(spec.get("horizon", 5)), int(spec.get("stride", 12)))
            if temporal != (vis_n, vis_stride):
                raise ValueError(
                    f"goal-pair sensor {name!r} uses horizon/stride={temporal}, "
                    f"but the live visual window is {(vis_n, vis_stride)}")
            goal_image = goal_top_bgr if camera == "top" else goal_bgr
            pair_extractors[name] = (
                camera,
                Reloc3rGoalPairFeatures(
                    goal_image,
                    checkpoint=str(spec.get(
                        "reloc3r_checkpoint", "siyan824/reloc3r-224")),
                    device=device, n_patch=int(spec.get("n_patch", 196)),
                    feat_dim=int(spec.get("feat_dim", 1024)),
                ),
            )

    use_ema = float(cfg.get("ema_rate", 0.999)) <= 0.999
    solver = "euler" if str(cfg.get("diffusion_backbone", "")) == "rectified_flow" else "ode_dpmsolver++_2M"
    horizon = int(cfg.get("horizon", HORIZON))
    ctrl = RolloutController(
        model, solver=solver, sample_steps=sample_steps, use_ema=use_ema,
        n_samples=n_samples, horizon=horizon, w_cfg=1, agg=agg, traj_ema_alpha=ema_alpha,
        warm_start=bool(getattr(config, "demo_warm_start", False)),
        warm_level=float(getattr(config, "demo_warm_level", 0.3)),
        ema_reset_sec=TRAJ_EMA_RESET_SEC, device=device)
    _STATE.update(dict(
        model=model, dino=dino, device=device, use_ema=use_ema, ctrl=ctrl,
        token_sequence=True, sensors=sensors, obs_horizon=obs_h,
        vision_stride=vis_stride, vision_n=vis_n, horizon=horizon, geometry=geometry,
        relation_specs=relation_specs, relation_extractors=relation_extractors,
        pair_specs=pair_specs, pair_extractors=pair_extractors,
        rgb_specs=rgb_specs, need_lidar=bool(lidar_specs),
        use_goal_lidar=False, goal_lidar=None,
        camera_mode=camera_mode, use_orbbec=camera_plan["use_orbbec"],
        use_usb0=use_usb0,
        video_stream_usb0=getattr(config, "demo_video1", "video1"),
        goal_feat_usb0=goal_feat_usb0, goal_feat_orbbec0=goal_feat_orbbec0,
        act_min=np.asarray(ckpt["action_min"], np.float32),
        act_scale=np.asarray(ckpt["action_scale"], np.float32),
        n_samples=n_samples, sample_steps=sample_steps, agg=agg,
        video_stream_orbbec0=video_stream_orbbec0, ema_alpha=ema_alpha,
    ))
    logger.info("docking demo loaded (TOKEN-SEQUENCE): %s | sensors=%s | camera_mode=%s "
                "obs_h=%d vis=%dx stride%d | solver=%s use_ema=%s horizon=%d "
                "n_samples=%d steps=%d",
                os.path.basename(ckpt_path), list(sensors), camera_mode, obs_h,
                vis_n, vis_stride, solver, use_ema, horizon, n_samples, sample_steps)
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
    st = _init(config)
    device = st["device"]
    n_samples = st["n_samples"]

    frames_orbbec0 = snap.get(st["video_stream_orbbec0"], ([], []))[0]
    enc = snap.get("encoder", ([], []))[0]
    lidar = snap.get("lidar", ([], []))[0]
    frames_usb0 = snap.get(st["video_stream_usb0"], ([], []))[0] if st["use_usb0"] else []
    obs_h = int(st.get("obs_horizon", OBS_HORIZON))

    if st.get("token_sequence"):
        sensors = st["sensors"]
        relation_specs = st.get("relation_specs", {})
        pair_specs = st.get("pair_specs", {})
        rgb_specs = st.get("rgb_specs", {})
        need_bottom = bool(st.get("use_orbbec", True))
        not_ready = (len(enc) < obs_h or
                     (need_bottom and len(frames_orbbec0) < obs_h) or
                     (st["use_usb0"] and len(frames_usb0) < obs_h) or
                     (st.get("need_lidar") and len(lidar) < 1))
    else:
        not_ready = (len(frames_orbbec0) < obs_h or len(enc) < obs_h or
                     len(lidar) < 1 or
                     (st["use_usb0"] and len(frames_usb0) < obs_h))
    if not_ready:
        logger.warning(
            "window not ready (orbbec0=%d usb0=%s enc=%d lidar=%d, need %d) -> zeros",
            len(frames_orbbec0), len(frames_usb0) if st["use_usb0"] else "n/a",
            len(enc), len(lidar), obs_h)
        return _zeros(config)

    vis_stride = int(st.get("vision_stride", VISION_STRIDE))
    vis_n = int(st.get("vision_n", OBS_HORIZON // VISION_STRIDE))

    def _sparse(frames):
        hist = frames[-obs_h:]
        idx = list(range(0, obs_h, vis_stride))[-vis_n:]
        if len(idx) != vis_n:
            raise ValueError(
                f"cannot sample {vis_n} visual frames at stride {vis_stride} from {obs_h}")
        return [hist[i] for i in idx]

    vel = np.array([[float(e["vx"]), float(e["wz"])] for e in enc[-obs_h:]], np.float32)
    vel_n = 2.0 * (vel - st["act_min"]) / st["act_scale"] - 1.0
    vel_t = torch.from_numpy(vel_n).unsqueeze(0).to(device).repeat(n_samples, 1, 1)

    if st.get("token_sequence"):
        context = {}
        if "wheel" in sensors:
            context["wheel"] = vel_t

        sparse_by_camera = {}
        if need_bottom:
            sparse_by_camera["bottom"] = _sparse(frames_orbbec0)
        if st["use_usb0"]:
            sparse_by_camera["top"] = _sparse(frames_usb0)

        dino_by_camera = {}
        for name, spec in rgb_specs.items():
            camera = "top" if str(spec.get("source", name)).endswith("_top") else "bottom"
            if camera not in dino_by_camera:
                with torch.no_grad():
                    vf, _, _ = st["dino"].get_heatmap(
                        _frames_to_tensor(sparse_by_camera[camera], device))
                dino_by_camera[camera] = vf.view(
                    1, len(sparse_by_camera[camera]), 196, 768).repeat(n_samples, 1, 1, 1)
            context[name] = dino_by_camera[camera]

        relation_by_camera = {}
        for camera, extractor in st.get("relation_extractors", {}).items():
            dec1, dec2 = extractor(sparse_by_camera[camera])
            relation_by_camera[camera] = (
                dec1.unsqueeze(0).repeat(n_samples, 1, 1, 1),
                dec2.unsqueeze(0).repeat(n_samples, 1, 1, 1))
        for name, spec in relation_specs.items():
            source = str(spec.get("source", name))
            camera = "top" if source.endswith("_top") else "bottom"
            stream = 0 if "dec1" in source else 1
            context[name] = relation_by_camera[camera][stream]

        for name, (camera, extractor) in st.get("pair_extractors", {}).items():
            pair = extractor(sparse_by_camera[camera])
            context[name] = pair.unsqueeze(0).repeat(n_samples, 1, 1, 1, 1)

        if st.get("need_lidar"):
            pts = np.array([[p["x"], p["y"]] for p in lidar[-1].get("points", [])], np.float32)
            crop, npts = _crop_scan(pts if pts.size else np.zeros((0, 2), np.float32))
            lidar_t = torch.from_numpy(crop).unsqueeze(0).to(device).repeat(n_samples, 1, 1)
            npts_t = torch.tensor([npts], device=device).repeat(n_samples)
            for name, spec in sensors.items():
                if spec.get("encoder") == "pointcloud":
                    context[name] = lidar_t
                    context[f"{name}_npoints"] = npts_t
        if "goal" in sensors:
            context["goal"] = st["goal_feat_orbbec0"]
        if st.get("geometry") is not None:
            g4 = st["geometry"](frames_orbbec0[-1])
            context["geometry"] = torch.from_numpy(g4).unsqueeze(0).to(device).repeat(n_samples, 1)
    else:
        sparse_orbbec0 = _sparse(frames_orbbec0)
        with torch.no_grad():
            vf, _, _ = st["dino"].get_heatmap(_frames_to_tensor(sparse_orbbec0, device))
        dino_feat_orbbec0 = vf.view(
            1, len(sparse_orbbec0), 196, 768).repeat(n_samples, 1, 1, 1)
        dino_feat_usb0 = None
        if st["use_usb0"]:
            sparse_usb0 = _sparse(frames_usb0)
            with torch.no_grad():
                vf1, _, _ = st["dino"].get_heatmap(_frames_to_tensor(sparse_usb0, device))
            dino_feat_usb0 = vf1.view(
                1, len(sparse_usb0), 196, 768).repeat(n_samples, 1, 1, 1)

        pts = np.array([[p["x"], p["y"]] for p in lidar[-1].get("points", [])], np.float32)
        crop, npts = _crop_scan(pts if pts.size else np.zeros((0, 2), np.float32))
        lidar_t = torch.from_numpy(crop).unsqueeze(0).to(device).repeat(n_samples, 1, 1)
        npts_t = torch.tensor([npts], device=device).repeat(n_samples)
        context = {
            "velocity": vel_t,
            "dino_feat2": dino_feat_orbbec0,
            "goal_feat2": st["goal_feat_orbbec0"],
            "goal_mask": torch.ones(n_samples, device=device),
            "lidar_points": lidar_t,
            "lidar_npoints": npts_t,
        }
        if st["use_goal_lidar"]:
            context["goal_lidar_points"], context["goal_lidar_npoints"] = st["goal_lidar"]
        if st["use_usb0"]:
            context["dino_feat1"] = dino_feat_usb0
            context["goal_feat1"] = st["goal_feat_usb0"]

    # sampling + medoid/mean aggregation + trajectory-EMA (with gap reset) all
    # live in the shared RolloutController (cleandiffuser.rollout_core), the one
    # copy exercised by the repo's eval/inference scripts too.
    plan = st["ctrl"].plan(context, st["act_min"], st["act_scale"], now=time.monotonic())
    # ★ OLD 동일성(D1): OLD(run_postech_dino_rtc_trt_fast)는 궤적 프레임간 EMA를 쓰지 않는다
    #   (apply_trajectory_ema는 정의만 있고 postech_inference에서 호출 안 함). 따라서 여기서도
    #   plan.ema(=demo_ema 프레임간 EMA)가 아니라 plan.current(집계 직후 pre-EMA)를 실행한다.
    #   → OLD의 selected_action(=n_samples=1이라 samples[0])과 동일한 입력이 continuity로 들어감.
    #   (demo_ema는 이 경로에선 미사용 — 이중 프레임간 필터가 "이상한 출력"의 원인이었음.)
    ema = plan.current                                                  # [H,2] pre-EMA (OLD 동일)
    sel_idx = -1        # which of plan.samples is executed; -1 = aggregated (viz only)

    # --- geometry-cost candidate selection (opt-in: demo_select: geometry) ----
    # Aggregating the M samples (mean/medoid) blends diverse candidates into a
    # mediocre one. Measured offline on r2cam_geo@60k over 480 near-dock frames
    # (test/selector_compare.py): picking the candidate whose commanded net
    # rotation best closes the Reloc3r yaw-to-goal beats mean-aggregation
    #   mean-agg   1.942 deg   (what runs today)
    #   geometry   0.981 deg   <- 67% of the way to the ICP oracle, and better
    #                             than the demonstrations' own 1.028 deg
    #   ICP oracle 0.584 deg   (upper bound, not deployable)
    # corr(psi_geometry, theta_icp) = +0.893, i.e. the token is a sound stand-in
    # for the dock yaw the robot cannot measure. Uses ONLY signals the policy
    # already receives -- no ICP at run time.
    if str(getattr(config, "demo_select", "")).lower() == "geometry":
        g = context.get("geometry")
        if g is None:
            logger.warning("demo_select=geometry but this checkpoint has no geometry token; "
                           "falling back to aggregation")
        else:
            samples = plan.samples                                      # [M,H,2] denormalized
            psi = float(np.arctan2(float(g[0, 2]), float(g[0, 3])))     # yaw current->goal, body frame
            dpsi = samples[:, :, 1].sum(axis=1) * STEP_DT
            resid = np.abs(np.arctan2(np.sin(psi - dpsi), np.cos(psi - dpsi)))
            pick = int(resid.argmin())
            ema = samples[pick]
            sel_idx = pick
            logger.debug("geometry-select: psi=%.3f rad, picked cand %d/%d (resid %.3f rad)",
                         psi, pick, len(samples), float(resid[pick]))

    # --- geometry-cost + LATERAL bearing selection (opt-in: demo_select: geometry_xy) ---
    # The geometry token is [dx_body, dy_body, sin(psi), cos(psi)]: (dx,dy) is
    # the unit BEARING to the goal (Reloc3r translation is scale-ambiguous, so
    # direction only), psi is the relative YAW needed to square up. The block
    # above (demo_select=geometry) only used psi. Lateral (dy) misalignment is
    # the failure mode the field notes call out as dominant (peg-in-hole offset,
    # not depth), so this variant ALSO scores each candidate's own SE(2) travel
    # direction against the goal bearing and adds that to the yaw cost:
    #   beta         = atan2(dy_body, dx_body)      goal bearing, body frame
    #   heading(tau) = atan2(py(tau), px(tau))       candidate's own net travel direction
    #   J = wrap(psi - sum(w)*dt)^2 + wrap(beta - heading(tau))^2
    # Still ICP-free -- both terms come from the geometry token plus the
    # candidate's own commanded (v, w), nothing the robot lacks at run time.
    # See test/selector_compare.py for the offline validation of this cost.
    elif str(getattr(config, "demo_select", "")).lower() == "geometry_xy":
        g = context.get("geometry")
        if g is None:
            logger.warning("demo_select=geometry_xy but this checkpoint has no geometry token; "
                           "falling back to aggregation")
        else:
            samples = plan.samples                                      # [M,H,2] denormalized
            psi = float(np.arctan2(float(g[0, 2]), float(g[0, 3])))      # yaw-to-goal
            beta = float(np.arctan2(float(g[0, 1]), float(g[0, 0])))     # lateral bearing-to-goal
            dpsi = samples[:, :, 1].sum(axis=1) * STEP_DT
            J_yaw = np.arctan2(np.sin(psi - dpsi), np.cos(psi - dpsi)) ** 2
            # integrate each candidate's own (v, w) to its SE(2) endpoint, then
            # compare its net travel direction against the goal bearing
            px = np.zeros(len(samples)); py = np.zeros(len(samples))
            for _m in range(len(samples)):
                _x = _y = _th = 0.0
                for _k in range(samples.shape[1]):
                    _v, _w = float(samples[_m, _k, 0]), float(samples[_m, _k, 1])
                    _x += _v * np.cos(_th) * STEP_DT
                    _y += _v * np.sin(_th) * STEP_DT
                    _th += _w * STEP_DT
                px[_m], py[_m] = _x, _y
            heading = np.arctan2(py, px)
            J_lat = np.arctan2(np.sin(beta - heading), np.cos(beta - heading)) ** 2
            pick = int((J_yaw + J_lat).argmin())
            ema = samples[pick]
            sel_idx = pick
            logger.debug("geometry_xy-select: psi=%.3f beta=%.3f, picked cand %d/%d "
                         "(yaw_resid %.3f, lat_resid %.3f)",
                         psi, beta, pick, len(samples), float(J_yaw[pick] ** 0.5), float(J_lat[pick] ** 0.5))

    # 진단: 모델 raw 출력(정규화 해제된 velocity, 집계 직후·continuity 이전) 그 자체를 저장
    global _INFER_IDX
    _INFER_IDX += 1
    _t_now = time.time()
    for _i in range(len(ema)):
        _MODEL_CSV.write([round(_t_now, 4), _INFER_IDX, _i,
                          round(float(ema[_i, 0]), 5), round(float(ema[_i, 1]), 5)])

    # continuity smoothing (old 이식) — raw(CSV 기록 후)에 적용해 실효 속도 급변 억제
    ema = _continuity_smooth(ema, config)
    n = int(getattr(config, "inference_size", 16))
    # Visualization (fire-and-forget; drops silently when no viewer is up).
    # Ship ALL demo_nsamples candidates, not just the executed plan: the spread
    # is the diagnostic -- a tight fan means the policy is confident, a fan that
    # straddles the dock means the aggregator is averaging away a good candidate,
    # which is exactly what demo_select=geometry* exists to avoid. selected is
    # post-continuity-smoothing, so it can differ from candidates[sel_idx].
    send_trajectory_bundle(
        candidates=plan.samples,
        selected=ema,
        selected_index=sel_idx,
        step_dt=STEP_DT,
        max_steps=min(n, HORIZON),
    )

    # integrate (vx, wz) -> 누적 로봇-전방 거리 + 누적 heading (ai-control-old과 동일).
    # ※ 기존 world-frame 회전누적(x+=v·cos θ, y+=v·sin θ)은 runner의 steps_to_velocity가
    #    dx만 차분(dy 무시)해 vx=v·cos θ 로 복원 → 회전 시 속도 왜곡/급가속. old 방식으로 복원.
    steps, dx, dth = [], 0.0, 0.0
    for i in range(min(n, HORIZON)):
        vx, wz = float(ema[i, 0]), float(ema[i, 1])
        dx += vx * STEP_DT       # 로봇 전방거리 누적 (world 회전 X)
        dth += wz * STEP_DT      # heading 누적
        steps.append(CommandStep(dx, 0.0, dth, (i + 1) * STEP_DT))
    return steps
