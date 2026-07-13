"""
AIControlConfig — BasePluginConfig(common) + ai-control 전용 필드.

공통 필드(윈도우/추론/전송/영상/NTP 등)는 BasePluginConfig에 있고, 여기엔 ai-control 고유 필드만:
  - enable_send: MQTT 발행/robot_control 점유 토글 (구버전 no_publish_ai_command 의미 반전)
  - ai_command_topic: 발행 토픽 (robot/{id}/ai-command)
  - trajectory_color / v_max / trajectory_ttl_ms: pointcloud_trajectory overlay
  - 테스트 플러그인용(phase_sec, episode_dir, replay_csv_path 등)
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from node_sdk.config import BasePluginConfig, load_config


@dataclass
class AIControlConfig(BasePluginConfig):
    # 출력 토글 (구버전 no_publish_ai_command 반전: True=발행+robot_control 점유)
    enable_send: bool = False
    ai_command_topic: str = ""               # __post_init__: robot/{id}/ai-command

    # pointcloud_trajectory overlay
    trajectory_color: str = "#00cfff"
    v_max: float = 1.0                        # waypoint v 정규화 분모 (m/s)
    trajectory_ttl_ms: int = 400

    # 테스트 플러그인용 (test_80cm_rotate, test_csv_replay 등) — AI 개발자 일반 모델엔 무관
    phase_sec: float = 5.0
    forward_m: float = 0.8
    horizon_sec: float = 5.0
    pause_sec: float = 5.0
    episode_dir: str = "saved_data/episode_000"
    replay_ticks: int = 50
    replay_csv_path: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.ai_command_topic:
            self.ai_command_topic = f"robot/{self.robot_id}/ai-command"   # 신버전: leading slash 없음


def load_ai_control_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AIControlConfig:
    return load_config(config_path, overrides, config_cls=AIControlConfig)
