# 시연 번들 2026-07-15

`ai-control`이 다른 서버에서 돌아가므로, 이 폴더를 통째로 그 서버의
`ai-control/ai_models/demo_20260715/` 로 복사하면 된다. (체크포인트 포함, 추가 다운로드 없음)

## 들어있는 모델

| 파일 | 모델 | 특징 | held-out |
|---|---|---|---|
| `scratch20.pt` | flow_goal_scratch20 | **기본 추천**. 종합 1위, 안정판 | 정밀 5.18mm / FDE 11.1 |
| `flow_adv_speed.pt` | flow_goal_adv (속도 AWR) | **더 빠르고 결단적**. 오른쪽 우회전 도킹 개선 확인용 | 정밀 5.30mm / FDE 12.6 |
| `flow_glidar_abs.pt` | flow_goal_glidar_abs | goal LiDAR 정합 조건. **추가 입력 필요**(아래) | 정밀 4.95mm / FDE 10.0 |

플러그인이 체크포인트의 아키텍처를 **자동 감지**한다 (goal-lidar 브랜치 유무). `DEMO_CKPT`만 바꾸면 됨.

> ⚠️ **glidar_abs만의 추가 요구**: 이 모델은 "도킹 완료 시점의 LiDAR 스캔"을 입력으로 받는다
> (goal_image.jpg의 LiDAR 버전). 번들의 `goal_lidar_scan.npy`(같은 dock의 도킹 스캔)를 기본으로
> 쓰지만, **현장 dock 위치/형상이 바뀌었으면** 도킹 위치에서 스캔을 떠서 `[M,2]` robot-frame
> 포인트 `.npy`로 저장 후 `DEMO_GOAL_LIDAR=경로`로 지정해야 정확하다.
> 참고: ablation에서 glidar_abs는 scratch20 대비 **통계적으로 유의한 개선 없음**(짝지은 비교).
> 정밀도 숫자(4.95mm)만 근소히 좋아 비교 시연용으로 넣은 것.

## 실행 — 환경변수로 모델/집계/EMA를 코드 수정 없이 바꾼다

플러그인(`ai_models/plugins/run_postech_docking_demo.py`)이 아래 env를 읽는다.
`docker-compose.yml`의 `environment:` 블록이나 `docker run -e KEY=VAL` 로 준다.

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `DEMO_CKPT` | `scratch20_checkpoint_step_8460.pt` | 쓸 체크포인트 (ai_models/ 기준 상대경로) |
| `DEMO_AGG` | `medoid` | 샘플 집계: **`medoid`**(결단적, 급회전 유지) \| **`mean`**(부드럽지만 급회전 뭉갬) |
| `DEMO_EMA` | `0.3` | 궤적 EMA 알파. **높을수록 덜 부드럽고 더 결단적**. `1.0`=평활화 끔 |
| `DEMO_NSAMPLES` | `8` | 샘플 수 (추론 느리면 4로) |
| `DEMO_USE_EMA` | `1` | EMA 가중치 사용 (이 번들 모델은 전부 1로 두면 됨) |
| `DEMO_VIDEO` | `video2` | room2(image_bottom) 카메라 스트림 |

> ⚠️ `DEMO_CKPT`는 이 번들에선 `demo_20260715/scratch20.pt` 처럼 하위폴더 경로로 줘야 한다
> (플러그인은 `ai_models/` 를 기준으로 찾음).

### 예시

**기본(안정판, 지난번과 동일):**
```yaml
# docker-compose.yml environment:
DEMO_CKPT: demo_20260715/scratch20.pt
```

**속도 AWR + 더 결단적 (오른쪽 도킹 개선 시도):**
```yaml
DEMO_CKPT: demo_20260715/flow_adv_speed.pt
DEMO_AGG: medoid
DEMO_EMA: "0.6"          # 0.3 -> 0.6, 급회전을 덜 뭉갬
```

**mean 집계로 되돌려 비교 (예전 동작 재현):**
```yaml
DEMO_AGG: mean
DEMO_EMA: "0.3"
```

`docker run` 이면:
```bash
docker run ... -e DEMO_CKPT=demo_20260715/flow_adv_speed.pt -e DEMO_AGG=medoid -e DEMO_EMA=0.6 ...
```

바꾼 뒤엔 컨테이너 **재시작**해야 반영된다 (`docker compose up -d`). 로그 첫 줄에
`docking demo loaded: ... agg=medoid ema_alpha=0.60 ...` 로 실제 적용값이 찍히니 확인.

## 오른쪽 우회전 도킹이 안 되던 문제
분석: 데이터는 좌우 균형(좌69/우67)이지만, **급격한 우회전(최대 0.73 rad/s)** 이 필요한데
`mean` 집계 + 낮은 EMA(0.3)가 그 결단을 뭉갠다. 대응 두 가지:
1. `flow_adv_speed.pt` (결단적 구간을 더 학습한 모델)
2. `DEMO_AGG=medoid` + `DEMO_EMA=0.6` (집계·평활화를 덜 뭉개게)
둘을 조합해 시연에서 우회전 도킹이 나아지는지 관찰 요망 — 실기 검증의 첫 데이터.

## 현장 체크리스트
1. `ai_models/goal_image.jpg` = 현장 dock 도킹뷰 사진인지 확인 (dock 위치 바뀌었으면 재촬영)
2. `ai_models/reference_room2.jpg` 와 라이브 비교해 room2가 video2 맞는지 (아니면 `DEMO_VIDEO=video1`)
3. `ENABLE_SEND=false` dry-run → UI 궤적 확인 → true
4. 추론 300ms↑면 `DEMO_NSAMPLES=4`
