# 실험 로드맵 — 진행 중 + 예정 (2026-07-14 기준)

> 완료된 실험들의 결과/해석은 [ablation_study_2026-07.md](ablation_study_2026-07.md),
> 평가 지표의 정의와 한계는 [offline_metrics.md](offline_metrics.md) 참고.
> 이 문서는 "지금 GPU에서 돌고 있는 것"과 "다음에 돌릴 것"의 설계 의도를 기록한다.

## 0. 현재 좌표 — 우리가 어디까지 왔고 무엇이 남았나

두 성능 축 기준 현재 상태 (held-out):

| 축 | 현재 최고 | 상태 |
|---|---|---|
| 정밀도 (aux dock-pose) | ~5.0mm | **포화** — ICP 선생의 노이즈 바닥(5.7mm)에 도달. 조건화 변경으로는 더 안 내려감 (glidar/glidar_abs로 확인) |
| 골 도달 (open-loop FDE) | ~10-11cm | **미포화 + 측정도 어려움** — train↔held-out 격차 크고(암기), FDE(n=10)의 통계력도 부족 |
| (미측정) 도달 시간 | — | 시연 자체가 느려서(54%가 <2cm/s) BC로는 원리적으로 개선 불가 |

따라서 로드맵의 무게중심은 **"조건화 개선" → "행동 개선"으로 전환**됐다. 시연의 상한(느림, 정밀도)을
넘으려면 모방이 아니라 리워드가 필요하다.

---

## 1. 진행 중 (GPU1 큐, `test/queue_gpu1.sh`)

### 1.1 `flow_goal_adv` — 오프라인 residual 1단계: advantage-weighted BC (AWR)
**설정**: `adv_weight=true`, from-scratch 20ep (scratch20과 동일 조건 + 가중만 추가).
**메커니즘**: 각 학습 샘플의 denoising loss에 `exp(advantage/β)` 가중.
advantage = 그 horizon(2초) 동안 dock 거리가 줄어든 속도(z-score). 즉 **"빠르게 접근한 시연 구간을
더 세게 배우고, 멈칫거린 구간은 약하게 배운다."** ICP 라벨이 모든 프레임에 있어서 리워드가 공짜라는
이 레포의 이점을 활용한 것.
**확인하는 질문**: 시연 분포의 "빠른 꼬리"쪽으로 정책을 밀 수 있는가 — 시연 느림 문제(07-13 시연에서
체감)의 오프라인 처방이 통하는가.
**성공 기준**: velRMSE/ADE가 크게 나빠지지 않으면서, open-loop 롤아웃의 평균 전진 속도가 GT 대비
유의하게 올라가는 것. (주의: FDE만으로 판단 금지 — [offline_metrics.md](offline_metrics.md) §4)
**한계 (미리 인지)**: AWR은 시연 분포 안에서의 재가중이라 **시연 최고 속도를 넘을 수는 없다.**
그 이상은 §2.2 온라인 RL의 몫.

### 1.2 `ddpm_goal_scratch20` — DDPM 깨끗한 기준선
**설정**: `diffusion_backbone=ddpm`, from-scratch 20ep (= scratch20의 ddpm 버전).
**확인하는 질문**: 주말 큐의 ddpm_goal_auxw는 warm-start 계보(구세대 가중치)를 물려받아 flow와의
비교가 오염돼 있었다. 같은 조건 from-scratch로 **backbone 비교를 깨끗하게** 다시 한다.
flow(scratch20) held-out 5.2mm/5.2/11.1과 비교.

### 1.3 `ddpm_goal_adv` — AWR이 backbone에 무관하게 통하는가
**설정**: ddpm + `adv_weight=true`, from-scratch 20ep.
**확인하는 질문**: 1.1의 효과가 flow 특유가 아니라 일반적인지 (2×2 factorial 완성:
{flow, ddpm} × {uniform, AWR}).

각 런은 학습 종료 후 train/held-out 평가가 자동으로 붙는다 (`test/eval_run.py`,
결과는 `test/out/weekend/<exp>{,_heldout}.json`).

---

## 2. 예정 (우선순위 순)

### 2.1 AWR 하이퍼파라미터 스윕 (1.1이 긍정적일 때만)
`adv_beta ∈ {0.5, 2.0}` — 온도가 낮을수록 빠른 구간에 공격적으로 쏠림. β=1(기본)에서 효과가
보이면 어느 쪽이 더 나은지, 효과가 없으면 β를 낮춰 한 번 더 확인 후 이 갈래를 닫는다.

### 2.2 경량 SE(2) 시뮬레이터 + 온라인 residual RL — "시연을 넘어서기"의 본체
**왜**: AWR의 상한(시연 분포)을 넘으려면 정책이 만든 행동의 **결과를 되먹임받는 환경**이 필요.
**구성**: 로봇은 차동구동(vx, wz)이라 다이나믹스는 수십 줄이면 충분. 관측 합성이 문제인데 —
LiDAR는 dock 템플릿(`endgame/assets/`)을 pose에 따라 투영해 합성 가능, **카메라는 합성 불가**하므로
정밀 구간(<0.6m)에서는 실데이터의 근접 프레임을 재생하거나 vision 브랜치를 dropout하는 절충이 필요.
**정책 구조**: base(frozen diffusion policy) + 작은 residual 헤드 Δ(obs). 리워드
`r = -(α·dock거리 + β·|각도|) - γ·|Δ|²`, 정밀 구간에서만 활성.
**성공 기준**: 시뮬 내 도달 시간 단축 + 최종 pose 오차 유지/개선 → 실기 검증.
**전제**: extrinsic(센서→로봇) 캘리브레이션 확정, ICP 합성 스캔의 충실도 검증.

### 2.3 접근 축 일반화 — 데이터/guidance
- **시연 추가 수집**이 가장 확실 (held-out 격차의 근본 원인이 145개 에피소드의 커버리지 부족).
- **w_cfg 스윕** (cfg07 체크포인트로, 학습 불필요): goal 조건을 추론 때 증폭. cfg07이 존재하는
  이유가 이 실험의 전제 조건이었음.
- **medoid/top-k 합의 + dock 점수 랭킹** (학습 불필요): mpc_rank의 교훈(§ mean<medoid,
  탐욕적 dock 랭킹은 해악)을 절충한 2단 선택.

### 2.4 closed-loop 평가 인프라 — 지표의 근본 해결
open-loop 지표의 한계([offline_metrics.md](offline_metrics.md) §3)를 넘으려면:
- **실기**: ai-control 플러그인 + ICP 그림자 모드(제어는 정책, 채점은 ICP) → 도달 시간·최종 오차·
  성공률을 직접 측정. 시연(07-13)에서 파이프라인은 검증됨.
- **시뮬**: 2.2의 시뮬레이터가 생기면 closed-loop 평가도 공짜로 따라옴.

### 2.5 held-out 확대 (10 → 30+ 에피소드)
FDE 통계력 문제(ablation §4.1)의 직접 처방. 원본 데이터가 있는 머신에서 에피소드를 추가 export해
`after_0328_test.h5`를 확장 — 재라벨링 포함 반나절 작업으로 추정.

---

## 3. 닫힌 갈래 (다시 열려면 새 근거 필요)

| 갈래 | 닫은 근거 |
|---|---|
| goal-lidar 조건화 | glidar_abs에서 이득 무검출 (ablation §2.10) |
| 상대 pose aux 타깃 | ICP 잡음 2배 합성으로 정밀도 2배 악화, 확증 (ablation §2.9→2.10) |
| aux 하이퍼 스윕 (w2/p4류) | 정밀도가 선생 바닥에서 포화 — 조건/손실 조정으로는 못 내려감 |
| warm-start 계보 | scratch가 일반화 우위 (ablation §2.6) — 새 실험은 from-scratch 기본 |
| EMA rate 0.9999 | 짧은 런에서 EMA 오염 (memory: ema-undertrained-4230-steps) |
