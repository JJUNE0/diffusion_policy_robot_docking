"""ai-control P2 진입점. 공통 run_ai_main(node_sdk.ai_runtime) + horizon build_result."""
from node_sdk.ai_runtime import run_ai_main

from .config import AIControlConfig


def make_build_result(inference_fn):
    """inference_fn(List[CommandStep]) → result {"horizon": [step.to_dict()]}."""
    def build_result(snapshot, latency_marks, config):
        horizon = inference_fn(snapshot, latency_marks, config)
        return {"horizon": [s.to_dict() for s in horizon]}
    return build_result


def ai_main(config_dict, control_q, result_q, ring_meta, lock):
    run_ai_main(AIControlConfig(**config_dict), control_q, result_q, ring_meta, lock, make_build_result)
