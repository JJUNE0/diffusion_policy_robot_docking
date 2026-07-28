# run 모드 플러그인

config의 `run_plugin`(또는 환경변수 `AI_CONTROL_RUN_PLUGIN`)으로 어떤 run 모드를 쓸지 선택합니다.

## 내장 플러그인

| 이름 | 설명 |
|------|------|
| `default` | 프로덕션. config 그대로, placeholder inference, tick 루프 + 전송 스레드. |
| `run_ai_control` | default와 동일 (별칭). |
| `run_postech` | WebRTC 영상 + postech_inference (실제 사용하는 inference 통합). |
| `test_80cm_rotate` | 전진→정지→후진→정지→360°회전→정지 무한 반복 테스트. |
| `test_80cm_rotate2` | 5초 호라이즌 배치 전송 → 대기 반복 테스트. |
| `test_csv_replay` | CSV 에피소드 리플레이로 윈도우 주입 후 tick 테스트. |

## 사용

- **config.yml:** `run_plugin: "default"` 또는 `run_plugin: "run_postech"`
- **Docker:** `AI_CONTROL_RUN_PLUGIN=run_postech`
- **진입점:** `python -m ai_control.main` 하나만 사용. config의 `run_plugin` 또는 env `AI_CONTROL_RUN_PLUGIN`으로 플러그인 선택. (쇼트컷 스크립트 없음)

## 새 플러그인 추가

1. `ai_control/plugins/` 아래에 모듈 추가 (예: `my_mode.py`).
2. **필수:** `run(config: AIControlConfig) -> None` 함수 정의.  
   내부에서 러너 생성·start, tick 루프(또는 자체 루프), KeyboardInterrupt 시 stop.
3. **선택:** `config_overrides: dict` 로 해당 모드에서만 쓸 설정 오버레이.
4. `plugins/__init__.py`의 `_register_builtin()` 에서 `register("my_mode", my_mode)` 호출 추가.
5. config.yml 또는 env에 `run_plugin: "my_mode"` 지정.

예시 (최소):

```python
# my_mode.py
def run(config):
    from ai_control.runner import AIControlRunner
    runner = AIControlRunner(config, inference_fn=my_inference, filters=[])
    runner.start()
    try:
        while True:
            runner.tick()
            time.sleep(1.0 / config.inference_fps)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()

config_overrides = {"require_marker_pose": False}  # 선택
```
