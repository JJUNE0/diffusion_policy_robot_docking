# 시연 준비 (2026-07-13 13:00)

## 모델 — ⚠️ 07-13 오전 변경: scratch20 사용
**`scratch20_checkpoint_step_8460.pt` = flow_goal_scratch20** (+ `scratch20_config.yaml`)

07-12 밤 held-out(미학습 10 에피소드) 재평가에서 순위가 뒤집혔다:

| 모델 | train FDE | **held-out FDE** | held-out 정밀도 |
|---|---|---|---|
| auxw2 (기존 1순위) | 4.0 cm | 15.0 cm (과적합) | 4.6 mm |
| **scratch20 (새 1순위)** | 6.7 cm | **11.1 cm** | 5.2 mm |

시연장은 처음 보는 조건이므로 **held-out 성적이 예측치** → scratch20 사용.
아키텍처는 auxw2와 동일 (rectified flow + goal + lidar + aux, use_room1=false)
→ 아래 추론 설정/입력 규격 전부 그대로 적용.

(`checkpoint_step_8460.pt` = 구 auxw2 번들, 백업으로 유지 — 정밀도는 이쪽이 4.6mm로 최고이므로
접근이 아니라 최종 정밀만 문제가 되면 교체 고려)

## 추론 설정 (반드시 이대로)
| 항목 | 값 | 비고 |
|---|---|---|
| solver | Euler | rectified flow |
| sample_steps | 20 | |
| num_samples | 8, 평균(mean) | medoid 선택이 가능하면 +20% 이득 (test/mpc_rank.py 참고) |
| w_cfg | 1 | 이 모델은 goal_mask_prob 0.99 학습이라 w_cfg>1 금지 |
| use_ema | **true 가능** | 이 체크포인트는 ema_rate 0.999로 EMA 정상. 구세대(7/9) ckpt는 반드시 false |
| denormalize | ckpt 안의 action_min / action_scale 사용 | |

## 입력 규격 (학습과 동일해야 함)
- velocity: 최근 30스텝 (vx, wz) @30 Hz, action min/max 정규화
- vision: room2(=image_bottom) 최근 30프레임에서 stride 6 → 5프레임 → DINO 인코딩 [5,196,768]
- lidar: 현재 스캔을 **최근접점 기준 0.8 m 크롭, 최대 256점, zero-pad** (utils/preprocessing.py 크롭과 동일해야 aux/lidar 브랜치가 정상 동작)
- goal: **현장 dock 완료 시점의 room2 이미지 1장** → DINO 인코딩 [1,196,768] + goal_mask=1

## ⚠️ 시연 최대 리스크: goal_feat 배선
docs/ai_server_robot_pipeline.md §6 기준, ai-control 플러그인 추론 경로에 goal_feat가 **미배선**.
이 모델은 99% goal-조건부로 학습되어 goal 없이 돌리면 분포 밖 동작이다.

**현장 절차 (시연 전 30분 확보):**
1. 로봇을 수동으로 도킹 위치에 놓고 room2 프레임 1장 캡처
2. DINO 인코딩: `DinoBatchDetector.get_heatmap(img)` → [1,196,768] → context["goal_feat2"], context["goal_mask"]=1
3. 플러그인 inference_fn context에 위 두 키 추가 (조건 네트워크는 키가 있으면 자동으로 사용)

goal 배선이 도저히 안 되면: goal 키를 아예 빼고 실행(브랜치 자동 스킵)은 되지만 성능 저하 감수.

## ai-control 배포 (07-13 오전 준비 완료)
`ai-control/`에 시연용 플러그인을 만들어뒀다 — **goal 배선 문제는 이걸로 해결됨**:
- `ai_models/plugins/run_postech_docking_demo.py` — scratch20 로드, goal 이미지 자동 인코딩,
  LiDAR nearest-cluster 크롭(학습과 동일), medoid 샘플 선택, CommandStep 변환까지 전부 내장
- `config.yml` 수정됨: `run_plugin: run_postech_docking_demo`, `require_lidar: true`
- 체크포인트/의존 파일 동봉: `ai_models/scratch20_checkpoint_step_8460.pt`, `master_vector.pt`
- 오프라인 검증: `test/smoke_plugin_demo.py` 통과 (호출당 158 ms @H100, 콜드 로드 ~1분)

**현장에서 할 일 두 가지:**
1. `ai_models/goal_image.jpg`를 **현장 dock 도킹 완료 시점의 room2 카메라 사진**으로 교체
   (동봉된 파일은 test set 예시일 뿐)
2. 카메라 확인: `ai_models/reference_room2.jpg`(학습 room2 뷰)와 라이브 두 스트림을 비교해서
   room2가 video2(orbbec-0)가 맞는지 확인. 아니면 플러그인 상단 `VIDEO_STREAM = "video2"`를
   "video1"로 수정

현장 GPU에서 추론이 300ms를 넘으면: 플러그인 상단 `N_SAMPLES = 8` → 4로 낮출 것.

## 안전 절차
1. **dry-run 먼저**: 플러그인 enable_send=false → 명령 계산만, UI 오버레이로 궤적 확인
2. 궤적이 dock 방향으로 정상 생성되는지 육안 확인 후 send 활성화
3. 참고: 시연은 closed-loop(매 프레임 재계획)이므로 open-loop FDE 4 cm보다 잘 붙는 게 정상

## 백업
- 2순위 모델: flow_goal_auxw (checkpoint_step_4230.pt, outputs/train/flow_goal_auxw/2026-07-10_15-36-56/) — 같은 아키텍처, 약간 낮은 성능
