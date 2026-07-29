"""
CommandStep horizon → `pointcloud_trajectory` ui_overlay data 변환 (ai-control 전용).

전송하는 velocity_steps와 별개로, UI 시각화를 위해 horizon의 누적 위치를 waypoints로 푼다.
모델 출력이 누적 위치(dx,dy=절대 변위)라고 가정하고, 속도(v)는 인접 step 차분으로 추정한다.
(속도 인코딩 모델이면 pos가 작게 나오지만 시각화 용도라 치명적이지 않음 — 필요 시 모델별 조정.)
"""
import math
from typing import List

from node_sdk import CommandStep


def build_pointcloud_trajectory_data(
    horizon: List[CommandStep],
    start_index: int,
    action_horizon: int,
    step_dt: float,
    color: str = "#00cfff",
    v_max: float = 1.0,
    label: str = "AI 경로",
) -> dict:
    """horizon[start_index : start_index+action_horizon] → pointcloud_trajectory data dict (spec §8.4)."""
    end = min(start_index + action_horizon, len(horizon))
    waypoints = []
    for i in range(start_index, end):
        step = horizon[i]
        # 누적 위치 (anchor 기준 절대 변위)
        x, y = float(step.dx), float(step.dy)
        # 속도 추정: 인접 step 차분 / dt 간격
        if i > start_index:
            prev = horizon[i - 1]
            ddt = step.dt - prev.dt
            if ddt > 1e-9:
                vx = (step.dx - prev.dx) / ddt
                vy = (step.dy - prev.dy) / ddt
                v = math.hypot(vx, vy)
            else:
                v = 0.0
        else:
            v = abs(step.dx) / step.dt if step.dt > 1e-9 else 0.0
        v_norm = min(1.0, v / max(0.01, v_max))
        waypoints.append({
            "pos": [x, y, 0.0],
            "v": float(v_norm),
            "confidence": 1.0,
            "label": "시작" if i == start_index else ("목표" if i == end - 1 else ""),
        })
    return {
        "label": label,
        "color": color,
        "animated": True,
        "waypoints": waypoints,
        "eta_sec": (end - start_index) * step_dt,
    }
