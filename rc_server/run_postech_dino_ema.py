import logging
import os
import sys
import time
from types import SimpleNamespace

_ai_control_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ai_control_dir not in sys.path:
    sys.path.insert(0, _ai_control_dir)

import numpy as np
import torch
import yaml

from ai_control.config import AIControlConfig
from ai_control.filters.registry import get_registry
from ai_control.runner import AIControlRunner
from ai_control.command_types import CommandStep
from ai_control.dataset.docking_dataset import denormalize
from ai_control.cleandiffuser.diffusion import ContinuousDiffusionSDE
from ai_control.cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
from ai_control.cleandiffuser.nn_diffusion import DiT1d
from ai_control.dino_detector import DinoBatchDetector
from ai_control.orb_feature_extractor import VisualGoalChecker

logger = logging.getLogger(__name__)

import tensorrt as trt
import cv2
import zmq

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- 플러그인 run 설정 ---
config_overrides = {
    "log_snapshot": False,
    "no_publish_ai_command": True,
    "enable_webrtc_video": True,
    "require_marker_pose": False,
    "require_lidar": False,
    "require_video": True,
    "inference_fps": 30,
    "inference_size": 16,
    "action_horizon": 4,
}

# --- Postech inference 글로벌 (lazy load) ---
_NN_DIFFUSION = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ACT_MIN = 0.0
_ACT_SCALE = 0.0

_HYDRA_CFG = None
_DETECTOR = None
_GOAL_DETECTOR = None

_FINISH_FLAG = False

_PREV_ACTION_EMA = None

_ZMQ_SOCKET = None
_ZMQ_SOCKET_IMG = None


def get_zmq_socket():
    global _ZMQ_SOCKET
    if _ZMQ_SOCKET is None:
        try:
            context = zmq.Context()
            _ZMQ_SOCKET = context.socket(zmq.PUSH)
            _ZMQ_SOCKET.setsockopt(zmq.SNDHWM, 2)
            _ZMQ_SOCKET.connect("tcp://172.17.0.1:5558")
            logger.info("ZMQ 궤적 소켓 연결 완료 (tcp://172.17.0.1:5558)")
        except Exception as e:
            logger.error(f"ZMQ 소켓 연결 실패: {e}")
    return _ZMQ_SOCKET


def get_zmq_socket_img():
    global _ZMQ_SOCKET_IMG
    if _ZMQ_SOCKET_IMG is None:
        try:
            context = zmq.Context()
            _ZMQ_SOCKET_IMG = context.socket(zmq.PUSH)
            _ZMQ_SOCKET_IMG.setsockopt(zmq.SNDHWM, 2)
            _ZMQ_SOCKET_IMG.connect("tcp://172.17.0.1:5559")
            logger.info("ZMQ 이미지 소켓 연결 완료 (tcp://172.17.0.1:5559)")
        except Exception as e:
            logger.error(f"ZMQ 소켓 연결 실패: {e}")
    return _ZMQ_SOCKET_IMG


# ===== [CHANGED] =====
# RTC helpers removed.
# Trajectory EMA helper added.
def apply_trajectory_ema(current_action, prev_ema_action=None, ema_alpha=0.3):
    """
    current_action: [horizon, 2]
    prev_ema_action: [horizon, 2] or None

    EMA over the whole predicted velocity trajectory.
    """
    if prev_ema_action is None:
        return current_action
    return ema_alpha * current_action + (1.0 - ema_alpha) * prev_ema_action


# ---------------- geometry / preprocessing helpers ----------------
def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _normalize_action(action: np.ndarray) -> np.ndarray:
    act_scale = np.asarray(_ACT_SCALE, dtype=np.float32)
    act_min = np.asarray(_ACT_MIN, dtype=np.float32)
    return 2.0 * (action - act_min) / act_scale - 1.0


def _extract_recent_encoder_history(window_snapshot, obs_horizon: int) -> np.ndarray:
    encoder_data, _ = window_snapshot.get("encoder", ([], []))

    if encoder_data and len(encoder_data) >= obs_horizon:
        recent_encoder = encoder_data[-obs_horizon:]
    elif encoder_data:
        recent_encoder = encoder_data[:]
    else:
        recent_encoder = []

    if recent_encoder:
        states = []
        for item in recent_encoder:
            v = float(item.get("vx", 0.0))
            w = float(item.get("wz", 0.0))
            states.append([v, w])
        pad_len = obs_horizon - len(states)
        if pad_len > 0:
            states = [states[0]] * pad_len + states
        return np.asarray(states, dtype=np.float32)

    return np.zeros((obs_horizon, 2), dtype=np.float32)


def _extract_recent_image_history(window_snapshot, stream_name: str, obs_horizon: int, device: torch.device):
    img_list, _ = window_snapshot.get(stream_name, ([], []))

    current_image = img_list[-1]

    if img_list and len(img_list) >= obs_horizon:
        recent_imgs = img_list[-obs_horizon:]
    elif img_list:
        recent_imgs = img_list[:]
    else:
        recent_imgs = []

    processed = []
    latest_rgb = None
    for img in recent_imgs:
        img_arr = img.copy()
        if img_arr.shape[:2] != (240, 320):
            img_arr = cv2.resize(img_arr, (320, 240), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        latest_rgb = img_rgb
        img_tensor = torch.from_numpy(img_rgb).to(device).float().permute(2, 0, 1) / 255.0
        processed.append(img_tensor)

    if not processed:
        zero_img = torch.zeros((3, 240, 320), device=device)
        processed = [zero_img for _ in range(obs_horizon)]
        latest_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    else:
        pad_len = obs_horizon - len(processed)
        if pad_len > 0:
            processed = [processed[0].clone() for _ in range(pad_len)] + processed

    history = torch.stack(processed, dim=0).unsqueeze(0)  # [1, T, 3, H, W]
    return current_image, history, latest_rgb


def _load_postech_config():
    """postech_config.yaml을 CWD와 무관하게 ai_control 디렉터리에서 직접 로드."""
    path = os.path.join(_ai_control_dir, "postech_config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"postech_config.yaml을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw.pop("hydra", None)
    return SimpleNamespace(**raw)

def _preprocess_snapshot(window_snapshot, device):
    # CHANGED: sparse two-camera preprocessing for velocity-output model
    obs_horizon = int(getattr(_HYDRA_CFG, "obs_horizon", 30))
    vision_stride = int(getattr(_HYDRA_CFG, "vision_stride", 6))

    velocity_seq_raw = _extract_recent_encoder_history(window_snapshot, obs_horizon)
    velocity_seq_norm = _normalize_action(velocity_seq_raw).astype(np.float32)

    _, room1_history_full, latest_room1_rgb = _extract_recent_image_history(window_snapshot, "video1", obs_horizon, device)
    current_image, room2_history_full, latest_room2_rgb = _extract_recent_image_history(window_snapshot, "video2", obs_horizon, device)

    _FINISH_FLAG, _  = _GOAL_DETECTOR.check_match(current_image)

    # CHANGED: sparse sampling on full 30-step history
    room1_history = room1_history_full[:, ::vision_stride]
    room2_history = room2_history_full[:, ::vision_stride]

    B, T_vis, C, H, W = room1_history.shape
    image_room1_flat = room1_history.reshape(B * T_vis, C, H, W)
    image_room2_flat = room2_history.reshape(B * T_vis, C, H, W)

    with torch.no_grad():
        # CHANGED: room1 + room2 both processed
        dino_feat1, sim_map1, _ = _DETECTOR.get_heatmap(image_room1_flat)
        dino_feat2, sim_map2, _ = _DETECTOR.get_heatmap(image_room2_flat)

    dino_feat1 = dino_feat1.view(B, T_vis, 196, 768)
    dino_feat2 = dino_feat2.view(B, T_vis, 196, 768)
    sim_map1 = sim_map1.view(B, T_vis, 240, 320)
    sim_map2 = sim_map2.view(B, T_vis, 240, 320)

    velocity_tensor = torch.from_numpy(velocity_seq_norm).to(device).unsqueeze(0)
    encoder_tensor = torch.from_numpy(velocity_seq_raw).to(device).unsqueeze(0)

    latest_hm1 = sim_map1[:, -1]
    latest_hm2 = sim_map2[:, -1]

    return {
        "encoder": encoder_tensor,          # [1, T, 2] raw
        "velocity": velocity_tensor,        # [1, T, 2] normalized
        "dino_feat1": dino_feat1,           # [1, Tv, 196, 768]
        "dino_feat2": dino_feat2,           # [1, Tv, 196, 768]
        "image1": latest_room1_rgb,
        "image2": latest_room2_rgb,
        "hm1": latest_hm1,
        "hm2": latest_hm2,
    }


def _load_model_once():
    global _NN_DIFFUSION, _HYDRA_CFG, _DETECTOR, _ACT_MIN, _ACT_SCALE
    if _NN_DIFFUSION is not None:
        return _HYDRA_CFG

    _HYDRA_CFG = _load_postech_config()
    args = _HYDRA_CFG

    _GOAL_DETECTOR = VisualGoalChecker('goal_image.jpg')
    _DETECTOR = DinoBatchDetector(device=_DEVICE, master_vector_path="master_vector.pt")

    # CHANGED: two-camera + velocity condition network signature
    obs_horizon = int(getattr(args, "obs_horizon", 30))
    vision_stride = int(getattr(args, "vision_stride", 6))
    vision_horizon = len(range(0, obs_horizon, vision_stride))

    nn_condition = SensorFusionConditionNetwork(
        state_dim=args.state_dim,
        obs_horizon=obs_horizon,
        vision_horizon=vision_horizon,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=getattr(args, "condition_num_layers", 2),
        dropout=args.dropout,
        num_image_latents=getattr(args, "num_image_latents", 16),
        velocity_dim=getattr(args, "velocity_dim", 2),
        velocity_dropout_prob=0.0,
    ).to(_DEVICE)

    nn_diffusion_model = DiT1d(
        in_dim=2,  # velocity-output model
        emb_dim=args.d_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
    ).to(_DEVICE)

    _NN_DIFFUSION = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion_model,
        nn_condition=nn_condition,
        device=_DEVICE,
    )

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_file_dir)
    checkpoint_path = os.path.join(parent_dir, f"checkpoint_step_{args.checkpoint_step}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=False)

    _ACT_SCALE = np.array(ckpt["action_scale"], dtype=np.float32)
    _ACT_MIN = np.array(ckpt["action_min"], dtype=np.float32)
    logger.info(f"Act min: {_ACT_MIN}, Act scale: {_ACT_SCALE}")

    _NN_DIFFUSION.model.load_state_dict(ckpt["model_state_dict"])
    _NN_DIFFUSION.model_ema.load_state_dict(ckpt["ema_state_dict"])
    _NN_DIFFUSION.eval()
    logger.info("======== Postech AI 모델 로드 완료 ========")
    return _HYDRA_CFG


def postech_inference(window_snapshot, latency_marks, polaris_config: AIControlConfig):
    global _PREV_ACTION_EMA

    logger.info("======== Inference with POSTECH Codes ========")
    args = _load_model_once()
    dt = 1.0 / max(1, polaris_config.inference_fps)
    n_samples = int(getattr(args, "num_samples", 8))

    traj_ema_alpha = float(getattr(args, "traj_ema_alpha", 0.3))

    with torch.no_grad():
        obs = _preprocess_snapshot(window_snapshot, _DEVICE)

        # CHANGED: new condition keys
        dino_feat1 = obs["dino_feat1"]
        dino_feat2 = obs["dino_feat2"]
        velocity = obs["velocity"]
        img1_np = obs["image1"]
        img2_np = obs["image2"]
        hm1_np = obs["hm1"].squeeze(0).detach().cpu().numpy()
        hm2_np = obs["hm2"].squeeze(0).detach().cpu().numpy()

        context = {
            "dino_feat1": dino_feat1.repeat(n_samples, 1, 1, 1),
            "dino_feat2": dino_feat2.repeat(n_samples, 1, 1, 1),
            "velocity": velocity.repeat(n_samples, 1, 1),
        }

        prior = torch.randn(n_samples, args.horizon, 2, device=_DEVICE)
        sample_out = _NN_DIFFUSION.sample(
            solver=args.solver,
            w_cfg=args.w_cfg,
            prior=prior,
            condition_cfg=context,
            n_samples=n_samples,
            sample_steps=args.inference_sampling_steps,
            use_ema=True,
        )
        if isinstance(sample_out, tuple):
            sample_out = sample_out[0]
        res_norm = sample_out.detach().cpu().numpy()
        samples = denormalize(res_norm, _ACT_SCALE, _ACT_MIN)

    if n_samples == 1:
        current_action = samples[0]
    else:
        current_action = samples.mean(axis=0)

    selected_action = apply_trajectory_ema(
        current_action=current_action,
        prev_ema_action=_PREV_ACTION_EMA,
        ema_alpha=traj_ema_alpha,
    )
    _PREV_ACTION_EMA = selected_action.copy()

    # velocity-output model: selected_action is still velocity trajectory
    velocities = selected_action
    horizon = velocities.shape[0]
    trajectory = np.zeros((horizon + 1, 3), dtype=np.float32)
    curr_q = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def f(q, v, w):
        return np.array([v * np.cos(q[2]), v * np.sin(q[2]), w], dtype=np.float32)

    for i in range(horizon):
        v, w = velocities[i]
        k1 = f(curr_q, v, w)
        k2 = f(curr_q + 0.5 * dt * k1, v, w)
        k3 = f(curr_q + 0.5 * dt * k2, v, w)
        k4 = f(curr_q + dt * k3, v, w)
        curr_q += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        curr_q[2] = _wrap_to_pi(curr_q[2])
        trajectory[i + 1] = curr_q

    try:
        sock = get_zmq_socket()
        if sock is not None:
            sock.send_pyobj(trajectory, flags=zmq.NOBLOCK)

        sock_img = get_zmq_socket_img()
        if sock_img is not None:
            sock_img.send_pyobj((img1_np, img2_np, hm1_np, hm2_np), flags=zmq.NOBLOCK)
    except zmq.Again:
        pass
    except Exception as e:
        logger.debug(f"ZMQ 전송 에러 (무시됨): {e}")

    if _FINISH_FLAG:
        return [CommandStep(0.0, 0.0, 0.0, (i + 1) * dt) for i in range(polaris_config.inference_size)]

    else:
        num_steps = min(velocities.shape[0], polaris_config.inference_size)
        command_steps = []
        acc_dx = 0.0
        acc_dtheta = 0.0
        for i in range(num_steps):
            v, w = float(velocities[i][0]), float(velocities[i][1])
            acc_dx += v * dt
            acc_dtheta += w * dt
            accumulated_dt = (i + 1) * dt
            command_steps.append(
                CommandStep(
                    dx=acc_dx,
                    dy=0.0,
                    dtheta=acc_dtheta,
                    dt=accumulated_dt,
                )
            )
        return command_steps

inference_fn = postech_inference

def run(config: AIControlConfig) -> None:
    filters = get_registry()
    runner = AIControlRunner(config, inference_fn=postech_inference, filters=filters)
    runner.start()
    try:
        fps = max(1, config.inference_fps)
        period = 1.0 / fps
        publish_msg = (
            "ai-command 발행함"
            if not getattr(config, "no_publish_ai_command", False)
            else "ai-command 미발행"
        )
        logger.info(
            "run_postech: postech_inference, %s, WebRTC 비디오 수신. "
            "비디오가 안 오면 시그널링 서버(%s:%s) 및 room1/room2 스트리머 확인. Ctrl+C 종료.",
            publish_msg,
            getattr(config, "signaling_host", "localhost"),
            getattr(config, "signaling_port", 4732),
        )
        while True:
            runner.tick()
            time.sleep(period)
    except KeyboardInterrupt:
        logger.info("Ctrl+C, 종료")
    finally:
        runner.stop()