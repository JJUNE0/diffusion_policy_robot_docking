"""ai-control AI 모델 코드 (AI 개발자 영역).

AI 개발자는 `plugins/run_*.py`에 `inference_fn(window_snapshot, latency_marks, config) -> List[CommandStep]`을
정의하면 된다(자동발견). 서버/통신/트레젝토리 보정은 ai_control_node(코어)가 처리하므로 건드리지 않는다.

weight(.pt / TensorRT 엔진 / goal_image 등)는 이미지에 넣지 않고 host volume(/models)에서 mount한다.
"""
