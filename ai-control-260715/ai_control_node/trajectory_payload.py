"""
ai-control 전용 — ai-command MQTT 페이로드 빌더.

구버전 ``ai_control/command_types.py``에서 `CommandStep`(공통 추론 출력 타입)만 제외하고 그대로 이식한다.
로봇에 보내는 ai-command 페이로드의 **byte 단위 동등성**을 보존하는 것이 유일한 책임 — 함수 이름·
시그니처·로직·반환 구조를 변경하지 말 것. (ai-world는 MQTT를 안 보내므로 이 모듈을 쓰지 않는다.)
"""
import json
from dataclasses import dataclass, field
from typing import List, Tuple

from node_sdk import CommandStep   # 공통 추론 출력 타입

# 프로토콜 상수 (구버전 config.py와 동일 값)
PROTOCOL_VERSION = 1
TRAJ_MAX_STEPS = 150
IS_FINAL_FLAG = 0x01


@dataclass
class AICommand:
    """AI command 메시지 (로봇에 보낼 포맷). anchor_time = 로봇 시점 clock time, steps는 anchor 기준 오프셋."""
    version: int = PROTOCOL_VERSION
    num_steps: int = 0
    seq_num: int = 0
    flags: int = 0
    anchor_time: float = 0.0
    steps: List[CommandStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "num_steps": self.num_steps,
            "seq_num": self.seq_num,
            "flags": self.flags,
            "anchor_time": self.anchor_time,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_steps(
        cls,
        steps: List[CommandStep],
        seq_num: int = 0,
        is_final: bool = False,
        anchor_time: float = 0.0,
    ) -> "AICommand":
        n = min(len(steps), TRAJ_MAX_STEPS)
        return cls(
            version=PROTOCOL_VERSION,
            num_steps=n,
            seq_num=seq_num & 0xFFFF,
            flags=IS_FINAL_FLAG if is_final else 0,
            anchor_time=anchor_time,
            steps=steps[:n],
        )


def make_stop_command(seq_num: int = 0, anchor_time: float = 0.0) -> AICommand:
    """로봇 정지: num_steps=1, step 모두 0. anchor_time은 로봇 시점."""
    return AICommand(
        version=PROTOCOL_VERSION,
        num_steps=1,
        seq_num=seq_num & 0xFFFF,
        flags=IS_FINAL_FLAG,
        anchor_time=anchor_time,
        steps=[CommandStep(0.0, 0.0, 0.0, 0.1)],
    )


def steps_to_velocity(horizon: List[CommandStep], index: int, step_dt: float) -> Tuple[float, float]:
    """
    horizon[index] 구간의 속도 (vx, wz) 계산.
    - step이 velocity 인코딩(dx=vx*dt, dtheta=wz*dt, 동일 dt)이면 step.dx/dt, step.dtheta/dt 그대로 반환.
    - step이 누적 위치(absolute)이면 index 0은 dx/dt, index>0은 (step-prev)/dt.
    """
    if not horizon or index < 0 or index >= len(horizon):
        return (0.0, 0.0)
    step = horizon[index]
    t = step.dt if step.dt > 1e-9 else step_dt
    if t < 1e-9:
        return (0.0, 0.0)
    if index == 0:
        return (step.dx / t, step.dtheta / t)
    prev = horizon[index - 1]
    dt_delta = step.dt - prev.dt
    if abs(dt_delta) < 1e-9:
        return (step.dx / t, step.dtheta / t)
    return ((step.dx - prev.dx) / dt_delta, (step.dtheta - prev.dtheta) / dt_delta)


def steps_to_velocity_steps(
    horizon: List[CommandStep],
    step_dt: float,
    start_index: int = 0,
    count: int | None = None,
) -> List[dict]:
    """
    horizon의 [start_index, start_index+count) 구간을 전송용 스텝 리스트로 변환.
    각 스텝은 { "vx", "wz", "dt" } (선속도, 각속도, 구간 길이 [초]).
    """
    if not horizon or start_index < 0 or start_index >= len(horizon):
        return []
    end = start_index + (count if count is not None else len(horizon) - start_index)
    end = min(end, len(horizon))
    out = []
    for i in range(start_index, end):
        vx, wz = steps_to_velocity(horizon, i, step_dt)
        step = horizon[i]
        dt_sec = step.dt if i == 0 else (step.dt - horizon[i - 1].dt)
        if dt_sec < 1e-9:
            dt_sec = step_dt
        out.append({"vx": vx, "wz": wz, "dt": dt_sec})
    return out


def ai_command_velocity_steps_dict(
    anchor_time: float,
    steps_list: List[dict],
    seq_num: int = 0,
    is_final: bool = False,
) -> dict:
    """ai-command 메시지 딕셔너리. steps_list는 [{ vx, wz, dt }, ...] 형식."""
    n = min(len(steps_list), TRAJ_MAX_STEPS)
    return {
        "version": PROTOCOL_VERSION,
        "num_steps": n,
        "seq_num": seq_num & 0xFFFF,
        "flags": IS_FINAL_FLAG if is_final else 0,
        "anchor_time": anchor_time,
        "steps": steps_list[:n],
    }
