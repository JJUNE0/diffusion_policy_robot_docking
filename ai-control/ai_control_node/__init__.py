"""ai-control 노드 — 도킹 제어를 rc 노드 그래프로.

옛 ``ai_control_node``(AIControlRunner·trajectory_payload·ai_process·ai_models)을 **그대로
재사용**하고, 전송계층만 ``node_sdk.NodeHub``로 바꾼다. ``_build_send_tick_outputs``의
seq/anchor/stride/padding 보정은 byte 동등 보존.

흐름:
  - 시작 시 ``rc.control.acquire``(source=ai)로 제어권 lease 획득 + keepalive.
  - 추론 horizon → 멀티스텝 속도 궤적 → ``ai_command`` 포트(버스 ``robot.{id}.ai_command``)
    → connection 노드가 MQTT ``robot/{id}/ai-command`` 로 브리지(legacy 도킹 펌웨어 동등).
  - 상시 도킹: compose env ``ENABLE_SEND=true``.
"""

import os

from ai_control_node.ai_process import ai_main
from ai_control_node.config import load_ai_control_config
from ai_control_node.runner import AIControlRunner

from node_sdk import NodeSpec, run_node_main


def build_spec() -> NodeSpec:
    return NodeSpec(
        name="ai-control",
        label="AI Control (도킹)",
        tags=["ai", "control"],
        sensor_inputs=[
            ("lidar", "robot.*.lidar", "LidarScan@1", "lidar"),
            ("encoder", "robot.*.encoder", "Encoder@1", "encoder"),
            ("marker_pose", "robot.*.marker_pose", "MarkerPose@1", "marker_pose"),
        ],
        outputs=[
            {
                "name": "ai_command",
                "topic": "robot.{robot_id}.ai_command",
                "type": "AiCommand@1",
                "requires_keys": ["robot_id"],
            },
        ],
        control_lease=True,
        control_owner_key="ai-control",
    )


def main() -> None:
    config = load_ai_control_config(config_path=os.environ.get("CONFIG_PATH", "config.yml"))
    run_node_main(
        runner_factory=lambda c, cq, rq, ring: AIControlRunner(
            c, hub=None, control_q=cq, result_q=rq, shm_ring=ring
        ),
        config=config,
        ai_entry=ai_main,
        spec=build_spec(),
    )
