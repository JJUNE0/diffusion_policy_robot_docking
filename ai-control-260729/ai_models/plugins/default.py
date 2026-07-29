"""기본 모델 플러그인: placeholder(정지). 실제 모델 미연동 / 드라이런 검증용.

AI 개발자는 이 파일을 복사해 run_mymodel.py를 만들고 inference_fn만 자기 모델로 교체하면 된다.
신버전은 run() 진입점이 필요 없다 — ai_process(코어)가 inference_fn만 자동발견해 호출한다.
"""
from ai_models.inference import placeholder_inference

# 모델 플러그인 규약: 모듈 레벨 inference_fn 노출
inference_fn = placeholder_inference
