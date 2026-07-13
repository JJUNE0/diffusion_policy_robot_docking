# ai_models — AI 모델 (AI 개발자 영역)

각 `plugins/run_*.py`는 모듈 레벨에 `inference_fn(window_snapshot, latency_marks, config) -> List[CommandStep]`을
노출한다. `ai_process`(P2 코어)가 `ai_models.plugins.{run_plugin}` 을 import해 자동발견·호출한다.
**서버/통신/트레젝토리 보정은 코어(`ai_control_node`)가 처리하므로 건드리지 않는다.**

## import 매핑 (구버전 → 신버전)
- `from ai_control.command_types import CommandStep` → `from node_sdk import CommandStep`
- `from ai_control.config import AIControlConfig` → `from ai_control_node.config import AIControlConfig`
- `from ai_control.runner import AIControlRunner` → `from ai_control_node.runner import AIControlRunner` (구버전 run()용 — 신버전은 미사용)
- `from ai_control.{inference,dino_detector,orb_feature_extractor,cleandiffuser,dataset,filters} ...` → `from ai_models. ...`

## 복사하지 않은 것 (volume mount / 구버전 참조)
- **weight**: `checkpoint_*.pt`, `master_vector.pt`, `goal_image.jpg` → host `/models` 에 두고 volume mount.
- **TensorRT 엔진**: `tensorrt/` (1.2GB, 빌드된 엔진) → host `/models/tensorrt` volume mount. 모델 코드가 참조하는 경로는 config/env로 지정.
- **학습 데이터셋(원본)**: 구버전 `~/rc_server/ai_control/dataset/` 및 `saved_data/`. 여기 `dataset/`은 런타임 코드(`docking_dataset.py`의 `denormalize` 등)만 복사했고, 대용량 학습 데이터는 복사하지 않음.

## cleandiffuser
vendored 라이브러리(19MB). 내부에서 `cleandiffuser`를 top-level 이름으로 import하므로, 컨테이너에서 `ai_models/`가
sys.path(WORKDIR)에 있어야 한다. (구버전 run_postech의 sys.path 처리 로직을 그대로 유지.)

## 검증
torch/tensorrt 의존이라 로컬(.venv)에서는 `default`(placeholder)만 로드 검증되고, 실모델(run_postech*)은
**arm64 Docker 컨테이너**에서 통합 검증한다.
