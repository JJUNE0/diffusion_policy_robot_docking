"""
AIControlRunner — RunnerBase(공통 골격) + 트레젝토리 _send_tick 보정 (ai-control 전용).

구버전 ``ai_control/runner.py``의 `_send_tick`(187-307)을 `_build_send_tick_outputs`에 **byte 단위로
보존** 이식한다. 변경점은 딱 셋:
  1. inference는 P2가 수행 → `consume_result()`로 horizon을 받아 `_steps_buffer` 갱신
     (구버전 `_maybe_run_inference`의 버퍼 반영 부분).
  2. NTP는 `self.robot_time_sec()` (RunnerBase, 미동기 시 server monotonic fallback).
  3. `publisher.publish_ai_command(...)` → `outputs`(mqtt + ui_overlay) 반환 → `hub.send_output`.
  no_publish_ai_command → `enable_send` 반전. 보정 수식(anchor/start_index/+0.1/stride/initial_pad/
  is_final/seq 동결 규칙)은 구버전 라인을 그대로 둔다.
"""
import logging
import os
import time
from typing import Any, List, Optional

from node_sdk import CommandStep
from node_sdk.runner_base import RunnerBase

from .config import AIControlConfig
from .config_schema import ai_control_config_schema
from .trajectory_overlay import build_pointcloud_trajectory_data
from .trajectory_payload import ai_command_velocity_steps_dict, steps_to_velocity_steps

logger = logging.getLogger(__name__)


class AIControlRunner(RunnerBase):
    def __init__(self, config: AIControlConfig, hub, control_q, result_q, shm_ring=None, livekit_receiver=None):
        super().__init__(config, hub, control_q, result_q, shm_ring, livekit_receiver)
        # 트레젝토리 보정 상태 (구버전 runner.__init__)
        self._steps_buffer: List[CommandStep] = []
        self._steps_index: int = 0
        # 재시작에도 seq 단조 유지 — 로봇이 seq로 최신 궤적 판단(0 리셋 시 옛 패킷으로 드롭).
        # 벽시계/전송주기 기반 시드로 "카운터가 멈춘 적 없는 것처럼" 이어감 (16비트 wrap은 로봇이 처리).
        self._seq_num: int = int(time.time() / max(0.01, config.send_interval_sec)) & 0xFFFF
        self._trajectory_anchor: float = 0.0      # 호라이즌당 1개 고정 anchor
        self._robot_send_count: int = 0
        self._has_new_inference: bool = False
        self._last_observation_time: float = 0.0

    def _anchor_now(self) -> float:
        """구버전 _get_anchor_time(): robot_time_sec(prediction_delay)."""
        return self.robot_time_sec(self.config.prediction_delay_sec)

    # ── register ──
    def get_register_payload(self) -> dict:
        cfg = self.config
        rid = cfg.robot_id
        return {
            "type": "register",
            "plugin_id": cfg.plugin_id or "ai-control",
            "instance_id": cfg.instance_id or (cfg.plugin_id or "ai-control"),
            "robot_id": rid,
            "robot_scope": "single",
            "run_mode": "on_demand",   # 서버 start 명령 시 RUNNING 전환 (docs §5.1). always_on은 APPROVED 즉시 RUNNING → 서버 제어 불가.
            "scale_mode": "fan_out",
            # 센서(lidar/encoder/marker_pose) 연속 수신은 stream + required_data.sensors (docs §6.2).
            # on_events(reactive)에만 넣으면 키 allowed_subscribe 거부 시 무로그 누락 → required_data는 sensor_subscribe_denied 로그로 진단됨.
            "trigger": {"type": "stream"},
            "required_data": {"sensors": ["lidar", "encoder", "marker_pose"]},
            "output_mode": ["mqtt", "ui_overlay"],
            # enable_send=false(검증)면 robot_control 미점유 — 제어권만 잡고 명령 안 보내는 상태 회피
            "managed_resources": ["robot_control"] if cfg.enable_send else [],
            "heartbeat_interval_sec": 5,
            "mqtt_topics": [f"robot/{rid}/ai-command"] if rid else ["robot/{robot_id}/ai-command"],
            "ui_overlay": {
                "overlay_type": "pointcloud_trajectory",
                "target_component": "pointcloud_map",
                "layer_id": "ai-control-path",
            },
            "livekit_tracks": [
                {"track_name": cfg.livekit_tracks_video1, "label": cfg.livekit_tracks_video1, "allowed_modes": ["subscribe"], "default_mode": "subscribe"},
                {"track_name": cfg.livekit_tracks_video2, "label": cfg.livekit_tracks_video2, "allowed_modes": ["subscribe"], "default_mode": "subscribe"},
            ],
            "config_schema": ai_control_config_schema(cfg),
        }

    # ── send_tick (RunnerBase가 send_interval마다 호출) ──
    async def on_send_tick(self) -> None:
        # 1) P2 추론 결과 반영 (구버전 _maybe_run_inference의 버퍼 갱신부)
        res = self.consume_result()
        if res is not None:
            horizon = [CommandStep.from_dict(s) for s in res.get("horizon", [])][: self.config.inference_size]
            if horizon:
                self._steps_buffer = horizon
                self._steps_index = 0
                obs = res.get("observation_time")
                self._last_observation_time = obs if obs is not None else self.robot_time_sec(0.0)
                self._has_new_inference = True
        # 2) 트레젝토리 보정 + outputs
        outputs = self._build_send_tick_outputs()
        if outputs:
            await self.hub.send_output(outputs)

    def _make_outputs(
        self, ts: float, velocity_steps: List[dict], seq_num: int, is_final: bool,
        horizon: List[CommandStep], start_index: int, with_overlay: bool,
    ) -> List[dict]:
        cfg = self.config
        outs: List[dict] = []
        if cfg.enable_send and velocity_steps:
            outs.append({
                "delivery": "mqtt",
                "topic": cfg.ai_command_topic,
                "payload": ai_command_velocity_steps_dict(ts, velocity_steps, seq_num, is_final),
            })
        if with_overlay and horizon:
            step_dt = 1.0 / max(1, cfg.inference_fps)
            data = build_pointcloud_trajectory_data(
                horizon, 0, len(horizon), step_dt,
                color=cfg.trajectory_color, v_max=cfg.v_max,
            )
            outs.append({
                "delivery": "ui_overlay",
                "overlay_type": "pointcloud_trajectory",
                "target_component": "pointcloud_map",
                "layer_id": "ai-control-path",
                "robot_id": self.robot_id,
                "ttl_ms": cfg.trajectory_ttl_ms,
                "data": data,
            })
        return outs

    def _build_send_tick_outputs(self) -> List[dict]:
        """구버전 _send_tick(runner.py:187-307) byte 보존. publish → outputs 반환."""
        cfg = self.config
        send_interval = max(0.01, cfg.send_interval_sec)
        fps = max(1, cfg.inference_fps)
        step_dt = 1.0 / fps
        action_horizon = getattr(cfg, "action_horizon", None)
        if action_horizon is None or action_horizon < 1:
            action_horizon = max(1, round(send_interval * fps))
        stride_steps = max(1, round(send_interval * fps))

        now_robot = self.robot_time_sec(0.0)

        horizon = list(self._steps_buffer) if self._steps_buffer else []
        has_new = self._has_new_inference
        last_obs = self._last_observation_time
        idx = self._steps_index

        if not horizon:
            if getattr(cfg, "no_send_when_no_horizon", False):
                return []
            ts = self._anchor_now()
            zero_steps = [{"vx": 0.0, "wz": 0.0, "dt": step_dt} for _ in range(action_horizon)]
            outs = self._make_outputs(ts, zero_steps, self._seq_num, True, horizon, 0, with_overlay=False)
            self._seq_num = (self._seq_num + 1) & 0xFFFF   # 구버전: 호라이즌 없음 경로는 seq++ 항상
            return outs

        if has_new:
            fixed_pad = getattr(cfg, "observation_send_padding_sec", None)
            if fixed_pad is not None:
                padding_sec = max(0.0, float(fixed_pad))
            else:
                padding_sec = max(0.0, now_robot - last_obs + 0.1)
            cap_mult = max(0.0, float(getattr(cfg, "send_padding_cap_interval_multiplier", 2.0)))
            padding_sec = min(padding_sec, send_interval * cap_mult)
            reflect_small = bool(getattr(cfg, "send_padding_reflect_small_latency", False))
            if padding_sec < send_interval and not reflect_small:
                padding_steps = 0
            else:
                padding_steps = min(int(round(padding_sec / step_dt)), len(horizon) - 1)
                padding_steps = max(0, padding_steps)
            start_index = padding_steps
        else:
            start_index = idx

        # 항상 action_horizon개 보내도록 상한
        start_index = min(start_index, max(0, len(horizon) - action_horizon))

        if start_index >= len(horizon):
            self._last_observation_time = now_robot
            ts = self._anchor_now()
            zero_steps = [{"vx": 0.0, "wz": 0.0, "dt": step_dt} for _ in range(action_horizon)]
            outs = self._make_outputs(ts, zero_steps, self._seq_num, True, horizon, start_index, with_overlay=False)
            self._seq_num = (self._seq_num + 1) & 0xFFFF   # 구버전: 버퍼 소진 경로도 seq++ 항상
            return outs

        # 다음 틱 인덱스 갱신 (캡된 start_index 사용, has_new/재사용 공통)
        self._has_new_inference = False
        self._steps_index = start_index + stride_steps

        logger.info(
            "send_tick: %s horizon_idx=%d/%d stride=%d",
            "new_infer" if has_new else "stride",
            start_index, len(horizon), stride_steps,
        )

        # ── 진단(임시): latency 보정이 0으로 죽는 원인 추적 — now_robot vs last_obs 시계 스케일 ──
        try:
            _spans = self.windows.snapshot_time_spans()
            _maxts = {
                k: (round(v["max_ts"], 3) if v.get("max_ts") is not None else None)
                for k, v in _spans.items()
            }
        except Exception:  # noqa: BLE001
            _maxts = {}
        logger.info(
            "clock_diag: now_robot=%.3f last_obs=%.3f diff=%.3f raw_pad=%.3f | stream_max_ts=%s",
            now_robot, last_obs, now_robot - last_obs, now_robot - last_obs + 0.1, _maxts,
        )

        velocity_steps = steps_to_velocity_steps(horizon, step_dt, start_index, action_horizon)
        while len(velocity_steps) < action_horizon:
            velocity_steps.append({"vx": 0.0, "wz": 0.0, "dt": step_dt})

        init_pad = getattr(cfg, "initial_send_padding_sec", 0.0)
        if init_pad > 0:
            remaining_sec = max(0.0, init_pad - (self._robot_send_count * send_interval))
            if remaining_sec > 0:
                pad_steps = min(int(round(remaining_sec / step_dt)), action_horizon - 1)
                if pad_steps > 0:
                    zero_pad = [{"vx": 0.0, "wz": 0.0, "dt": step_dt} for _ in range(pad_steps)]
                    velocity_steps = zero_pad + velocity_steps[: action_horizon - pad_steps]

        all_zero = all(s.get("vx", 0.0) == 0.0 and s.get("wz", 0.0) == 0.0 for s in velocity_steps)
        next_index = start_index + stride_steps
        buffer_exhausted_after_this = next_index >= len(horizon)
        is_final = all_zero or buffer_exhausted_after_this

        # 궤적 기준 시각: 호라이즌당 하나의 _trajectory_anchor (메시지마다 재계산하면 0.1s씩 밀림)
        if has_new:
            self._trajectory_anchor = self._anchor_now()
        anchor_offset_sec = start_index * step_dt
        ts = self._trajectory_anchor + anchor_offset_sec

        outs = self._make_outputs(ts, velocity_steps, self._seq_num, is_final, horizon, start_index, with_overlay=True)
        # 구버전 메인 경로: seq++/send_count++는 publish(enable_send) 시에만 (overlay만 보낼 땐 동결)
        if cfg.enable_send and velocity_steps:
            self._seq_num = (self._seq_num + 1) & 0xFFFF
            self._robot_send_count += 1
        return outs
