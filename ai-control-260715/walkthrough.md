# 🚀 시연 실행 가이드 (2026-07-13)

## 1. Dry Run (명령만 계산, 로봇 미동작)

> [!TIP]
> **먼저 반드시 dry-run으로 궤적이 정상 생성되는지 UI 오버레이로 확인한 후 실제 전송을 켜세요.**

```bash
# 추론 서버에서 (ai-control 디렉토리)
cd ai-control
ENABLE_SEND=false docker compose up -d --build
```

또는 `.env` 파일을 수정:
```bash
# ai-control/.env
ROBOT_ID=7023
ENABLE_SEND=false
```
```bash
docker compose up -d --build
```

**확인 방법**: 로그에서 추론 결과가 나오고, UI 오버레이에 궤적이 그려지지만 **로봇이 움직이지 않아야** 합니다.

```bash
# 로그 실시간 확인
docker compose logs -f ai-control
```

---

## 2. 실제 작동 (로봇 도킹 명령 전송)

> [!CAUTION]
> 반드시 dry-run으로 궤적이 dock 방향으로 정상 생성되는 것을 **육안 확인** 후 전환하세요!

```bash
# 방법 A: env로 바로 켜기
cd ai-control
ENABLE_SEND=true docker compose up -d --build
```

```bash
# 방법 B: .env 파일 수정 후
# ai-control/.env
ROBOT_ID=7023
ENABLE_SEND=true
```
```bash
docker compose up -d --build
```

```bash
# 방법 C: dry-run 중이었다면 env만 바꿔서 재시작
ENABLE_SEND=true docker compose up -d
```

**컨테이너 중지:**
```bash
docker compose down
```

---

## 3. 배포 폴더 (`additional-nodes/ai-control/`)

[additional-nodes/ai-control/](file:///home/polaris3d/ws/additional-nodes/ai-control) 폴더에 추론 서버에 업로드해야 할 파일이 모두 있습니다 (체크포인트 포함).

### 포함된 파일

| 디렉토리/파일 | 설명 |
|---|---|
| `ai_control_node/` | 노드 런타임 (runner, config, trajectory 등) |
| `ai_models/` | 시연용 모델 코드 + goal 이미지 + 참조 이미지 |
| `ai_models/plugins/run_postech_docking_demo.py` | **핵심 시연 플러그인** |
| `ai_models/cleandiffuser/` | diffusion 모델 라이브러리 (추론에 필요한 부분만) |
| `ai_models/dino_detector.py` | DINO 특징 추출기 |
| `ai_models/master_vector.pt` | 마스터 벡터 |
| `ai_models/goal_image.jpg` | ⚠️ **현장에서 교체 필요** |
| `ai_models/reference_room2.jpg` | 카메라 방향 확인용 참조 이미지 |
| `config.yml` | 추론 설정 |
| `docker-compose.yml` | Docker 구성 |
| `.env` | 환경변수 (ROBOT_ID) |

### 업로드 위치

추론 서버의 기존 `ai-control/` 위치에 **덮어씌워서** 배포:

```bash
# 이 서버에서 → 추론 서버로 전송
scp -r /home/polaris3d/ws/additional-nodes/ai-control/ <추론서버>:<ai-control이_있는_경로>/ai-control/
```

또는 rsync:
```bash
rsync -avz --delete /home/polaris3d/ws/additional-nodes/ai-control/ <추론서버>:<ai-control이_있는_경로>/ai-control/
```

> [!IMPORTANT]
> 추론 서버에 `ai-control/`이 이미 있으므로, **해당 위치에 그대로 덮어쓰면** 됩니다.

### ✅ 체크포인트 파일

> [!NOTE]
> `scratch20_checkpoint_step_8460.pt` 파일은 `ai_models/demo_20260713/` 에 존재하며,
> `ai_models/scratch20_checkpoint_step_8460.pt` 심볼릭 링크로 플러그인이 접근합니다.

플러그인이 찾는 경로: `ai_models/scratch20_checkpoint_step_8460.pt` → `demo_20260713/scratch20_checkpoint_step_8460.pt` (629MB)

---

## 4. 현장 체크리스트

> [!IMPORTANT]
> 시연 전 반드시 확인!

- [ ] **체크포인트**: `ai_models/scratch20_checkpoint_step_8460.pt` 존재 확인
- [ ] **Goal 이미지 교체**: 로봇을 도킹 완료 위치에 놓고 room2 카메라로 촬영 → `ai_models/goal_image.jpg`로 저장
- [ ] **카메라 확인**: `reference_room2.jpg`와 라이브 스트림 비교 → room2가 `video2`(orbbec-0)가 맞는지 확인. 아니면 플러그인 상단 `VIDEO_STREAM = "video2"` → `"video1"`로 수정
- [ ] **rc 스택 확인**: nats/livekit/connection이 이미 떠 있어야 함
- [ ] **Dry-run 먼저**: `ENABLE_SEND=false`로 시작 → 궤적 확인 → `ENABLE_SEND=true`로 전환

### 성능 튜닝
- GPU에서 추론이 **300ms 초과**시: 플러그인의 `N_SAMPLES = 8` → `4`로 변경
- 정밀도보다 접근이 문제면: 현재 scratch20 유지
- 접근은 되는데 최종 정밀만 문제면: auxw2 체크포인트(4.6mm 정밀도)로 교체 고려
