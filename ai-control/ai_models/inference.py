"""
AI 인퍼런스 인터페이스 (AI 개발자용 계약).

구버전 ai_control/inference.py 이식. 변경점: CommandStep을 node_sdk(공통)에서 import,
config 타입은 Any(ai_models는 플러그인 코어에 의존하지 않음 — 느슨하게).

  inference_fn(window_snapshot, latency_marks, config) -> List[CommandStep]
    window_snapshot[stream] = (data_list, slow_list)
      video1/video2 = [BGR ndarray (H,W,3)], lidar/encoder/marker_pose/command = [dict]
    config: inference_size / inference_fps / send_interval_sec 등 (AIControlConfig)
    반환: CommandStep(dx,dy,dtheta,dt) × config.inference_size. dt = anchor 기준 경과 초.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from node_sdk import CommandStep

logger = logging.getLogger(__name__)

InferenceFn = Callable[
    [Dict[str, Tuple[List[Any], List[bool]]], Dict[str, List[bool]], Any],
    List[CommandStep],
]


def placeholder_inference(
    window_snapshot: Dict[str, Tuple[List[Any], List[bool]]],
    latency_marks: Dict[str, List[bool]],
    config: Any,
) -> List[CommandStep]:
    """항상 정지(0) inference_size개 반환. 실제 모델 미연동 시 기본값."""
    n = int(getattr(config, "inference_size", 16))
    step_dt = 1.0 / max(1, int(getattr(config, "inference_fps", 30)))
    return [CommandStep(0.0, 0.0, 0.0, (i + 1) * step_dt) for i in range(n)]


def run_inference(
    window_snapshot: Dict[str, Tuple[List[Any], List[bool]]],
    latency_marks: Dict[str, List[bool]],
    config: Any,
    inference_fn: Optional[InferenceFn] = None,
) -> List[CommandStep]:
    """inference_fn 실행. None이면 placeholder."""
    fn = inference_fn or placeholder_inference
    return fn(window_snapshot, latency_marks, config)
