"""
ai-control RC Hub config_schema 선언 (config-schema.md 매핑).

전체 config.yml 파라미터를 UI에 노출하되, **AI 입출력 계약에 영향 큰 항목(window_size/inference_size/
inference_fps/require_*)은 USER_MANAGEMENT + read_only**로 보호하고, 운영 튜닝(패딩/지연/색 등)은
ROBOT_CONTROL로 둔다. run_plugin options는 ai_models/plugins/*.py 자동발견(import 없이 파일명만).
"""
import re
from pathlib import Path
from typing import Any, Dict, List

_RC = "ROBOT_CONTROL"
_UM = "USER_MANAGEMENT"
_RM = "ROBOT_MONITORING"


# 모듈 레벨 inference_fn (대입 또는 def) 보유 여부 — torch import 없이 소스만 스캔.
_HAS_INFERENCE_FN = re.compile(r"^(inference_fn\s*=|def\s+inference_fn\b)", re.M)


def _discover_run_plugins() -> List[str]:
    """ai_models/plugins/ 중 모듈 레벨 inference_fn 정의 파일만 (P1에서 torch import 없이 소스 스캔).
    유틸/미완 모듈(record_control 등 inference_fn 없음)은 select에서 제외. 없으면 ['default']."""
    p = Path(__file__).resolve().parent.parent.parent / "ai_models" / "plugins"
    if not p.is_dir():
        return ["default"]
    names: List[str] = []
    for f in sorted(p.glob("*.py")):
        if f.stem.startswith("_") or f.stem == "base":
            continue
        try:
            if _HAS_INFERENCE_FN.search(f.read_text(encoding="utf-8")):
                names.append(f.stem)
        except OSError:
            continue
    return names or ["default"]


def _f(key, ko, en, typ, default, perm=None, ro=False, **extra) -> Dict[str, Any]:
    d = {
        "key": key,
        "title": {"ko": ko, "en": en},
        "type": typ,
        "default": default,
        "required_permission": perm,
        "read_only": ro,
    }
    d.update(extra)
    return d


def ai_control_config_schema(config) -> List[Dict[str, Any]]:
    obs_pad = config.observation_send_padding_sec
    act_h = config.action_horizon
    return [
        # ── 운영 토글 ──
        _f("enable_send", "MQTT 발행/제어권", "Enable Publish", "boolean", config.enable_send, _RC),
        _f("run_plugin", "추론 모델", "Model", "select", config.run_plugin, _UM, options=_discover_run_plugins()),
        _f("log_level", "로그 레벨", "Log Level", "select", config.log_level, _RM,
           options=["DEBUG", "INFO", "WARNING", "ERROR"]),
        _f("log_snapshot", "스냅샷 로그", "Log Snapshot", "boolean", config.log_snapshot, _RM),
        # ── 전송/지연 보정 튜닝 ──
        _f("prediction_delay_sec", "예측 지연 보정(s)", "Prediction Delay", "number", config.prediction_delay_sec, _RC,
           min=-0.5, max=0.5, step=0.05),
        _f("observation_send_padding_sec", "옵저베이션 패딩(s, 음수=동적)", "Obs Padding", "number",
           obs_pad if obs_pad is not None else -1.0, _RC, min=-1.0, max=2.0, step=0.05),
        _f("send_padding_cap_interval_multiplier", "패딩 상한 배수", "Pad Cap x", "number",
           config.send_padding_cap_interval_multiplier, _RC, min=1.0, max=10.0, step=0.5),
        _f("send_padding_reflect_small_latency", "소지연 반영", "Reflect Small Latency", "boolean",
           config.send_padding_reflect_small_latency, _RC),
        _f("initial_send_padding_sec", "초기 패딩(s)", "Initial Padding", "number", config.initial_send_padding_sec, _RC,
           min=0.0, max=3.0, step=0.1),
        _f("data_latency_threshold_sec", "지연 임계(s)", "Latency Threshold", "number", config.data_latency_threshold_sec, _RC,
           min=0.05, max=1.0, step=0.05),
        _f("data_unreliable_slow_tail_size", "slow tail", "Slow Tail", "integer", config.data_unreliable_slow_tail_size, _RC,
           min=0, max=60, step=1),
        _f("window_sync_mode", "윈도우 sync", "Window Sync", "select", config.window_sync_mode, _RC,
           options=["none", "fps", "track"]),
        _f("send_interval_sec", "전송 주기(s)", "Send Interval", "number", config.send_interval_sec, _RC,
           min=0.05, max=1.0, step=0.05),
        _f("action_horizon", "액션 호라이즌(0=auto)", "Action Horizon", "integer", act_h if act_h is not None else 0, _RC,
           min=0, max=150, step=1),
        # ── overlay ──
        _f("trajectory_color", "경로 색", "Path Color", "string", config.trajectory_color, _RC),
        _f("v_max", "최대 속도(m/s)", "v_max", "number", config.v_max, _RC, min=0.1, max=5.0, step=0.1),
        _f("trajectory_ttl_ms", "overlay TTL(ms)", "Traj TTL", "integer", config.trajectory_ttl_ms, _RC,
           min=100, max=2000, step=100),
        # ── AI 입출력 계약 (read_only — 변경 시 모델 깨짐, 재시작 필요) ──
        _f("window_size", "윈도우 크기", "Window Size", "integer", config.window_size, _UM, ro=True, min=10, max=300, step=1),
        _f("inference_size", "추론 horizon", "Inference Size", "integer", config.inference_size, _UM, ro=True, min=1, max=150, step=1),
        _f("inference_fps", "추론 FPS", "Inference FPS", "integer", config.inference_fps, _UM, ro=True, min=1, max=60, step=1),
        _f("require_video", "영상 필수", "Require Video", "boolean", config.require_video, _UM, ro=True),
        _f("require_lidar", "라이다 필수", "Require Lidar", "boolean", config.require_lidar, _UM, ro=True),
        _f("require_encoder", "엔코더 필수", "Require Encoder", "boolean", config.require_encoder, _UM, ro=True),
        _f("require_marker_pose", "마커 필수", "Require Marker Pose", "boolean", config.require_marker_pose, _UM, ro=True),
    ]
