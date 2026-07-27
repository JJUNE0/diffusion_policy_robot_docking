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

    # ── 추론 knob (구버전 DEMO_* env를 config.yml로 단일화; run_postech_docking_demo.py) ──
    demo_ckpt: str = "scratch20_checkpoint_step_8460.pt"  # ai_models/ 기준 상대경로
    demo_backbone: str = "rectified_flow"     # "rectified_flow"(euler) | "ddpm"(dpmsolver++_2M).
    #   ★자동감지 불가(두 백본이 파라미터명 동일) — 체크포인트 학습과 반드시 일치시켜야 함.
    #   틀리면 같은 가중치라도 완전히 다른(급가속/이상) 출력. (없으면 rectified_flow로 빠지던 버그 수정)
    demo_video: str = "video2"                # room2(image_bottom) 카메라 스트림
    demo_agg: str = "medoid"                  # 샘플 집계: "medoid" | "mean"
    demo_nsamples: int = 8                    # 트레젝토리 샘플 수 (느리면 4)
    demo_steps: int = 20                      # denoising 스텝 (50~100 = 느리지만 고품질)
    demo_ema: float = 0.3                     # 궤적 EMA 알파 (높을수록 덜 평활/결단적; 1.0=끔)
    demo_use_ema: bool = True                 # EMA 가중치 사용 (0.999-gen 체크포인트만)
    demo_continuity_blend: float = 0.5        # exponential continuity 스무딩 γ (old 이식; 급가속 억제). 0=끔
    demo_goal_lidar: str = ""                 # glidar 체크포인트용 docked-pose scan .npy (빈값=기본경로)

    # ── 종단(마지막 몇 cm) knob (2026-07-25, 07-23 실기 진단 반영) ──────────────
    # 07-23 15회 중 5회가 도크면 474.6±5.1mm에서 멈춰 실패(성공은 450.0±3.4mm).
    # 두 knob 모두 기본 OFF — 켜기 전에 test/viz_dock_shift.py로 현장 값 재측정할 것.
    #
    # (1) near-dock progress gate = 추론 시점 AWR. 도크 근처에서 샘플 평균 대신
    #     "전진하는 샘플" 쪽으로 지수 가중. 학습 불필요.
    demo_progress_beta: float = 0.0           # >0이면 켜짐. 0.5~1.0 권장(작을수록 공격적)
    demo_progress_gate_far: float = 0.60      # m. 이보다 멀면 gate=0 (기존 mean 그대로)
    demo_progress_gate_near: float = 0.47     # m. 이보다 가까우면 gate=1 (최대 가중)
    demo_progress_max_vx: float = 0.03        # m/s. 가중 결과 속도 상한. 해당 구간 시연이
                                              #   ~15mm/s라 2배. 과전류 위험 때문에 낮게 잡음.
    demo_progress_lateral: float = 0.010      # m. ⚠안전 인터록: 측면 오차가 이보다 크면 부스트
                                              #   거부(peg가 홈 턱에 걸린 상태에서 밀면 과전류).
                                              #   07-23 실측: 성공 5.6±3.4mm / 걸림 16.9±2.8mm
    #
    # (2) arrival latch = 도킹 완료 감지 후 정지 래치. 00780(성공→미인지→후진→실패) 대응.
    demo_arrive_dist: float = 0.0             # m. >0이면 켜짐. 07-23 기준 0.458 권장
    demo_arrive_lateral: float = 0.012        # m. 도크면 중심 좌우 오차 허용치.
                                              #   07-23 성공 |lat| 5.6±3.4mm, 최악 성공 9.0mm
    demo_arrive_frames: int = 3               # 연속 몇 프레임 만족해야 래치
    demo_release_dist: float = 0.52           # m. 이보다 멀어지면 래치 해제(히스테리시스)
    demo_release_frames: int = 5              # 연속 몇 프레임 만족해야 해제

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
