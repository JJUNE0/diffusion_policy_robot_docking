# Ablation 스터디 (2026-07-10 ~ 07-14) — 각 실험이 확인하는 것

> 기준일: 2026-07-14 (glidar_abs 결과 + FDE 통계력 재해석 반영)
> 근거: `test/weekend_queue.sh`, `outputs/weekend_queue.log`, `test/out/weekend/*.json`
> 성능을 **두 축**으로만 본다: **정밀도**(aux dock-pose 오차, mm) + **골 도달**(open-loop ADE/FDE, cm).
> 평가는 두 세트로 한다: **train**(외운 문제, 145 에피소드) vs **held-out**(처음 보는 10 에피소드,
> `dataset/after_0328_test.h5`). **결론은 반드시 held-out 기준으로 낸다** — train 순위는 07-12 밤
> 재평가에서 뒤집혔다 (§4). 총 10개 실험: 주말 큐 8개(§2.1~2.8) + goal-lidar 정합 2종(§2.9~2.10).
> **⚠️ §4.1의 통계력 분석을 먼저 읽을 것** — FDE(n=10)로는 1~2cm 차이를 판별할 수 없어서, 이 문서의
> FDE 기반 순위 서술은 "큰 효과"만 신뢰해야 한다. 진행 중/예정 실험은 [experiment_roadmap.md](experiment_roadmap.md),
> 메트릭 자체의 설명과 한계는 [offline_metrics.md](offline_metrics.md) 참고.

---

## 0. 왜 이 순서인가 — 큐 설계 논리

GPU 시간이 주말 이틀로 제한돼 있어서, **"확실히 좋을 것 같은 실험부터"** 배치했다
(`test/weekend_queue.sh` 주석 참고). 그래서 두 가지 학습 전략이 섞여 있다:

- **warm-start (fine-tune)**: 이미 수렴한 `flow_goal_auxw` 체크포인트에서 이어 학습, lr을
  낮춰(3e-5) 빠르게 여러 하이퍼파라미터를 저비용으로 스캔. `auxw2`, `cfg07`, `auxw_w2`, `p4`가 이 방식.
- **from-scratch**: 처음부터 새로 학습(lr 1e-4). 아키텍처가 바뀌거나(`nolidar`, `nogoal`) 깨끗한
  비교 기준이 필요할 때(`scratch20`) 사용.

### ⚠️ 0.1 이 큐의 설계 결함 — §2.1~2.8은 단일 변수 ablation이 **아니다** (07-14 인정, 재실행 착수)

이 주말 큐는 **깨끗한 ablation이 아니다.** 두 가지 confound가 있다:

1. **warm-start 계보 오염** (`cfg07`, `auxw_w2`, `p4`, `auxw2`):
   `flow_goal_auxw`에서 이어받았는데, 그 가중치는 **구세대 균일 aux loss**로 학습된 것이고,
   실질 학습량도 다르다(base 4230 + auxw 4230 + 자기 자신 vs scratch20의 8460).
   → "knob 하나를 바꾼 효과"가 "다른 loss로 사전학습된 효과"와 뒤섞여 있다.
2. **에포크 불일치** (`nolidar`, `nogoal`): 10 epoch인데 비교 대상 `scratch20`은 20 epoch.
   → "모달리티 제거 효과"와 "학습량 절반"이 뒤섞여 있다.

**결과적으로 §2.1~2.8의 held-out 순위는 knob의 효과로 읽으면 안 된다.**
→ **07-14, 전 항목을 `scratch20` 기준 from-scratch 20 epoch으로 재실행 중**
(`test/queue_gpu0_clean_ablation.sh`: `s20_nolidar`, `s20_nogoal`, `s20_cfg07`, `s20_w2`, `s20_p4`).
결과가 나오면 §4 표를 이 클린 런들로 교체한다. (§2.9~2.11의 glidar/glidar_abs/adv는 처음부터
from-scratch 20ep이라 이 문제가 없다.)

**정정 (07-14)**: 이전 판본은 warm-start 런이 "EMA 버그 세대의 가중치를 물려받았다"고 썼는데
**부정확하다.** `init_from`은 `model_state_dict`(멀쩡한 raw 가중치)를 로드하고 EMA도 그것으로
재시드하므로, 오염된 EMA 사본은 전파되지 않았다. EMA 버그의 실제 피해는 **가중치가 아니라
평가/추론**이었다 — §0.2 참조. 위 1번의 confound는 EMA가 아니라 **loss 함수가 다른 가중치에서
출발했다는 것**이다.

### 0.2 EMA 버그란 무엇이었나 (07-09~07-10)

`ema_rate: 0.9999`의 시간상수 τ=1/(1−0.9999)=**1만 스텝**인데 학습은 **4230 스텝**뿐이었다.
EMA는 초기 랜덤 가중치의 사본에서 출발하므로, 학습 종료 시점에도
`0.9999^4230 ≈ 0.655` — **EMA 사본에 랜덤 초기값이 65% 남아 있었다.**

- **오염된 것**: `ema_state_dict`(EMA 사본)뿐. **학습된 raw 가중치(`model_state_dict`)는 멀쩡했다.**
- **실제 피해**: 추론 기본값이 `use_ema: true`였으므로, 그 시기의 **모든 추론/평가가 반쯤 랜덤인
  모델을 평가**하고 있었다 — aux 정밀도 17mm(raw) vs **148mm(EMA)**, open-loop 종점 오차
  10.5cm(raw) vs **404cm(EMA, 6m를 헤매는 랜덤워크)**.
- **수정**: `ema_rate: 0.999`(τ=1천 스텝 → 4230스텝 후 초기값 잔량 ~1.5%)로 변경.
  구세대 체크포인트는 반드시 `use_ema: false`로 평가해야 한다.
  (`test/eval_run.py`가 `ema_rate ≤ 0.999`일 때만 EMA를 쓰도록 자동 판정한다.)

---

## 1. 계보 (lineage)

```
flow_goal (07-09, EMA 버그 세대: ema_rate=0.9999, uniform aux loss)
  │
  ▼ warm-start, 새 loss(거리 가중 aux) + ema_rate=0.999로 교체
flow_goal_auxw (07-10)                                    ← 이 문서 실험들의 공통 조상
  │
  ├─▶ warm-start, +10~20 epoch, lr=3e-5 ─┬─ auxw2      (그냥 더 학습)
  │                                      ├─ cfg07      (goal_mask_prob 0.99→0.7)
  │                                      ├─ auxw_w2    (aux_weight 1→2)
  │                                      └─ p4         (aux_dist_power 2→4)
  │
  └─(비교 기준점, 독립 런) ─────────────── scratch20   (from-scratch 20ep, 같은 최종 loss 설정)
                                          nolidar     (from-scratch 10ep, LiDAR 브랜치 제거)
                                          nogoal      (from-scratch 10ep, goal 조건 제거)

ddpm_goal_auxw: flow_goal_auxw와 같은 세대의 ddpm(구 diffusion) 버전 — backbone 비교용
```

---

## 2. 실험별 설명 — 무엇을 바꿨고, 무엇을 확인하는가

### 2.1 `ddpm_goal_auxw` — backbone 비교 (rectified flow vs DDPM)
**바꾼 것**: `diffusion_backbone: ddpm` (나머지 전부 flow_goal_auxw와 동일 조건, ddpm_goal 체크포인트에서 warm-start).
**확인하는 질문**: 궤적 생성 backbone으로 rectified flow 대신 고전 DDPM을 쓰면 어떤가?
**왜 필요한가**: DDPM은 검증된 표준이지만 샘플링이 느리다(원래 100+ step). rectified flow가
"빠르면서 정밀도 손해 없음"을 주장하려면 같은 조건에서 이겨야 한다.

### 2.2 `flow_goal_auxw2` — 추가 학습량 효과
**바꾼 것**: `flow_goal_auxw`에서 warm-start, 나머지 설정 동일, epoch만 10→20.
**확인하는 질문**: 이 loss 설정으로 더 오래 학습하면 계속 좋아지는가, 아니면 포화/과적합하는가?
**결과 미리보기**: train에서는 계속 좋아졌지만(§4), held-out에서는 **가장 크게 악화** — 과적합의
교과서적 사례.

### 2.3 `flow_goal_cfg07` — Classifier-Free Guidance 준비
**바꾼 것**: `goal_mask_prob: 0.99 → 0.7` (골 조건이 "켜진" 샘플 비율을 99%→70%로 낮춰, 무조건부
학습 비중을 늘림).
**확인하는 질문**: 무조건부 branch를 더 많이 학습시키면(NoMaD식 CFG 준비), goal-conditioning이
더 강하게/유연하게 작동하는가? (참고: 0.99 그대로면 `w_cfg > 1` guidance가 사실상 무의미 — 무조건부
분포가 학습이 안 돼 있어서. cfg07은 이후 `w_cfg` 스윕을 가능하게 하는 전제 작업이기도 함.)

### 2.4 `flow_goal_auxw_w2` — aux loss 비중
**바꾼 것**: `aux_weight: 1.0 → 2.0` (전체 loss = denoising + aux_weight × aux_loss에서 정밀도 항 비중↑).
**확인하는 질문**: aux(정밀도) 항의 가중치를 올리면 정밀도가 더 좋아지는 대신 궤적 생성(denoising)이
희생되는가? — "정밀도 vs 접근"의 직접적인 트레이드오프 실험.

### 2.5 `flow_goal_p4` — 근거리 가중을 더 날카롭게
**바꾼 것**: `aux_dist_power: 2.0 → 4.0`. aux loss 가중치 = `(aux_dist_ref / 거리)^power`
(`aux_dist_ref=0.6m`)이므로, power를 올리면 도킹 직전 프레임에 가중이 더 쏠린다
(0.55m 프레임이 1.1m 프레임 대비: power=2면 4배, power=4면 16배).
**확인하는 질문**: 근거리 집중을 더 세게 하면 근거리 정밀도(mm)가 더 좋아지는가, 아니면 이미
포화(선생 ICP 노이즈 바닥에 근접, [[aux-head-vs-icp-noise-floor]])했는가?

### 2.6 `flow_goal_scratch20` — 깨끗한 기준선 ⭐ (held-out 1위)
**바꾼 것**: 없음(warm-start 안 함) — `flow_goal_auxw`와 동일한 최종 설정(거리 가중 aux, ema 0.999)을 , goal image : O , goal lidar : x
**처음부터** 20 epoch 학습.
**확인하는 질문**: warm-start 계보(구세대 uniform-loss 가중치를 물려받음)가 실험 결과를 오염시키고
있진 않은가? 즉 "auxw 계열이 좋아 보이는 게 새 loss 덕분인지, 단지 사전학습된 가중치 덕분인지"를
분리하는 대조군.
**결과**: held-out 전체 1위. warm-start 없이 시작한 게 오히려 **일반화가 더 잘 됐다** — 구세대
가중치 계보가 근소하게 일반화를 해치고 있었을 가능성.

### 2.7 `flow_goal_nolidar` — LiDAR 브랜치의 존재 이유
**바꾼 것**: `use_lidar_points: false` (LiDAR 포인트 브랜치 + aux head 전체 제거). from-scratch 10 epoch.
**확인하는 질문**: **"접근에는 카메라+엔코더만으로 충분하고, LiDAR는 정밀 구간에서만 필요하다"**는
핵심 가설의 절반. LiDAR를 빼도 접근(ADE/FDE)이 유지되고 정밀도(near_mm — 이 런은 aux head가 없으므로
사실 측정 불가에 가깝다는 점 주의)만 떨어진다면 가설 지지.
**결과**: held-out ADE/FDE가 오히려 scratch20과 비슷하거나 나음(단, 10ep vs 20ep 비교라 절반만
학습했는데도 안 밀린 것 — §0의 캐비엇 참고, 오히려 가설에 유리한 방향), **near_mm은 7.3mm(train)/
6.9mm(held-out)로 전 런 중 최악** → LiDAR 제거가 정밀도만 확실히 깎는다는 가설 지지.

### 2.8 `flow_goal_nogoal` — goal 조건의 존재 이유
**바꾼 것**: `use_goal: false` (goal 이미지 조건 전체 제거). from-scratch 10 epoch.
**확인하는 질문**: goal(도킹 완료 프레임) 조건이 실제로 "어디로 가야 하는지"를 알려주는 역할을
하는가, 아니면 장식인가?
**결과**: held-out ADE 6.9 / FDE 15.0 — **접근 축이 전 런 중 가장 크게 악화**. goal 조건이 접근
성능에 실질적으로 기여한다는 증거. (단, 이 런도 10 epoch — scratch20과의 직접 비교엔 같은 캐비엇 적용)

### 2.9 `flow_goal_glidar` — goal-lidar 정합 조건화 (07-13, 새 아키텍처)
**바꾼 것**: `use_goal_lidar: true` + `aux_relative: true` (커밋 `8bb7c74`). 두 가지가 동시에 켜진다:
1. **조건 입력 추가**: goal(도킹 완료) 프레임의 LiDAR 스캔을 현재 스캔과 **같은 point encoder**로
   토큰화해 별도 modality로 fusion transformer에 투입 — 현재 스캔과 골 스캔 토큰이 서로 attention해
   "정합 상태"를 내부적으로 표현하도록 유도.
2. **aux 타깃 교체**: 절대 dock pose 대신 **현재→골 SE(2) 상대 pose**(골까지 남은 거리+각도)를
   회귀하도록 aux head의 지도신호를 바꿈.
`scratch20`과 동일하게 from-scratch 20 epoch, 나머지 설정(거리 가중, ema_rate 0.999)은 동일.

**확인하는 질문**: §2.7(nolidar)·§2.8(nogoal)이 지지한 가설 — "LiDAR=정밀, goal=접근" — 을 이어받아,
**"정밀 담당(LiDAR)과 접근 담당(goal)을 하나의 정합 표현으로 합치면 둘 다 더 좋아지는가"**를
직접 검증하는, 이 스터디의 novelty 본체 실험.

**⚠️ 평가 하네스 버그 (07-13, 발견 즉시 수정)**: 처음 돌린 평가에서 근거리 정밀도가 **60.8mm**로
나와, 다른 모든 런(4.6~7.3mm)과 10배 이상 차이가 나는 게 이상해 원인을 추적했다. 원인은 모델이
아니라 평가 스크립트: `test/eval_run.py`/`test/dist_binned_error.py`가 (1) `aux_relative` 타깃을
반영하지 않고 **항상 절대 dock pose 정규화 통계로 오차를 계산**하고 있었고 (2) **goal-lidar 입력
자체를 평가 컨텍스트에 안 넣고 있었다** — 즉 학습 때 항상 존재하던 입력 modality 하나가 통째로
빠진 채로, 그것도 모델이 예측하지 않는 값과 비교하고 있었다. `test/dist_binned_error.py`(H5Batcher에
`aux_relative` 지원 + goal-lidar 항상 배선), `test/eval_run.py`, `test/eval_openloop_metrics.py`를
수정해 재평가했다 — scratch20 등 기존(절대 타깃) 런들은 재평가해도 수치가 그대로임을 회귀 테스트로
확인(5.665 vs 5.663mm, 부동소수점 오차 수준).

**결과 (수정된 하네스, held-out)**: 정밀도 **9.7mm** — scratch20(5.2mm) 대비 오히려 악화, 전 런 중
최악. 궤적(ADE 5.9 / FDE 11.4)은 scratch20(5.2 / 11.1)과 비슷한 수준, 개선 없음.

**해석 (가설, 미검증)**: 절대 dock pose는 ICP 잡음원이 하나(현재 프레임 추정)인데, 상대 pose
타깃(`_relative_to_goal`)은 **현재 프레임 추정과 골 프레임 추정(에피소드 기준점 g) 두 ICP 잡음원을
합성**한다. 두 잡음이 독립에 가깝다면 분산이 대략 더해져, 유효 노이즈 바닥이 `√2 × 5.7mm ≈ 8mm`
근방으로 올라갈 수 있다 — 관측된 9.7mm와 근접. 즉 **아이디어(정합 조건화) 자체가 아니라, "상대
pose로 타깃을 바꾼 것"이 노이즈를 두 배로 만들었을 가능성**이 크다. `use_goal_lidar`와
`aux_relative`는 코드상 독립 플래그이므로, 다음 시도는 **goal-lidar 조건 입력은 유지하되 aux
타깃은 절대 pose로 되돌리는** 조합(`use_goal_lidar=true, aux_relative=false`)으로 두 변경의 효과를
분리하는 것을 권장 (§6).

### 2.10 `flow_goal_glidar_abs` — 정합 조건화와 타깃 설계의 분리 (07-14)
**바꾼 것**: `use_goal_lidar: true` + **`aux_relative: false`** — §2.9의 두 변경 중 골-스캔 조건
입력만 유지하고, aux 타깃은 절대 dock pose(다른 모든 런과 동일)로 되돌림. from-scratch 20 epoch.
**확인하는 질문**: §2.9의 9.7mm 악화가 (a) 정합 조건화 아이디어 자체의 문제인가,
(b) 상대 pose 타깃의 잡음 배가 문제인가 — 두 요인의 분리 실험.

**glidar(_abs)와 scratch20의 실제 구조적 차이** (config diff + state_dict 비교로 확인, 07-14):
config는 `use_goal_lidar`/`aux_relative` 두 줄만 다르지만, 이게 아키텍처에 실질적인 변화를
만든다 — "그 외 조건 전부 동일"은 부정확한 표현이라 정정한다.

| 항목 | scratch20 | glidar(_abs) |
|---|---|---|
| condition net 파라미터 | 41,152,134 | 42,349,830 (**+1.20M, +2.9%**) |
| 신규 모듈 | — | `goal_lidar_resampler` (lidar_resampler와 동일 구조의 독립 PerceiverResampler) + slot/modality/null 임베딩 3종 |
| fusion transformer 입력 시퀀스 길이 | 기본 토큰 수 | **+16 토큰** (goal-lidar latent) → self-attention 연산량↑ |
| 체크포인트 크기 | 659MB | 678MB |

즉 glidar는 "같은 모델에 조건 하나 추가"가 아니라 **새 서브네트워크(2.9% 더 큰 모델)를 처음부터
같이 학습**시킨 것이다. 이게 갖는 함의: (1) 파라미터가 늘었는데 20 epoch·145 에피소드라는 같은
데이터/스텝 예산을 나눠 써야 해서 미세한 용량 경쟁이 있을 수 있고, (2) 골 스캔이 매 스텝 20-step
샘플링 루프 안에서 매번 새로 인코딩되므로(현재 스캔과 달리 값이 매 스텝 불변인데도) 추론 연산량도
약간 늘어난다(최적화 여지 — 골 스캔 latent를 에피소드당 1회만 계산해 캐싱 가능). 성능 차이(§2.10
결과)가 크지 않았던 것과 별개로, 이 구조 차이 자체가 "정합 조건화 자체는 무효과"라는 결론에
힘을 보탠다 — 파라미터를 더 쓰고도 이득이 없었다는 뜻이므로.

**결과 (held-out)**: 정밀도 **4.95mm** — glidar(9.72mm)에서 정상 복귀, scratch20(5.18mm)과 동급.
궤적 ADE 6.5 / FDE 10.0.

**결론 두 가지**:
1. **(b) 확증** — `aux_relative` 플래그 하나로 정밀도가 9.72↔4.95mm, 2배 차이. 예측한 √2 잡음
   합성(현재 ICP + 골 ICP) 메커니즘과 일치하며, 효과가 커서 노이즈로 설명 불가. **상대 pose 타깃은
   현 라벨 품질에서는 쓰지 말 것.**
2. **그러나 goal-lidar 조건화 자체의 이득은 검출되지 않음** — FDE median(10.0 vs 11.1)만 보면
   이긴 것 같지만, **에피소드별 짝지은 비교**에서는 10개 중 3개만 승리, 평균 차이 +1.1cm(오히려
   나쁨), 부트스트랩 95% CI [-1.7, +3.6]으로 유의하지 않다 (§4.1). 정밀도 4.95 vs 5.18mm 차이도
   전 런이 몰려 있는 4.6~5.3mm 스프레드 안의 잡음. **정직한 결론: 골 스캔을 조건으로 더 주는 것은
   현 데이터/지표에서는 도움도 해도 안 됨.**

---

### 2.11 `flow_goal_adv` / `ddpm_goal_adv` — 오프라인 리워드 학습 (AWR, 07-14 진행 중)

**바꾼 것**: `adv_weight: true` (+ `adv_beta`, `adv_clip`). 나머지는 `scratch20`과 완전히 동일
(from-scratch 20ep, rectified flow) — 즉 **AWR 하나만의 효과를 분리**하는 대조 실험.
ddpm backbone 버전(`ddpm_goal_adv`)도 함께 돌려 backbone 독립성을 확인한다.

**확인하는 질문**: 시연을 그대로 흉내 내는 BC의 한계 —
데이터를 보면 **시연 프레임의 54%가 2cm/s 미만, 28%는 사실상 정지**(중앙값 1.75cm/s)다.
BC는 `p(행동|관측)`을 시연 분포에서 복사하므로 **시연에 없는 빠른 속도는 원리적으로 낼 수 없다**
(07-13 시연에서 로봇이 느렸던 근본 원인). 리워드로 "빠르게 접근한 시연 구간"을 더 세게 학습시키면
정책이 시연 분포의 **빠른 꼬리** 쪽으로 이동하는가?

자세한 알고리즘은 §7 참조.

### 2.12 `checkpoint_step_100000.pt` — 구 baseline (07-15, 사후 추가 — 실기 실패 후 소급 평가)

**배경**: 07-15 현장 시연에서 §2.6·§2.10·§2.11의 세 모델(scratch20, glidar_abs, flow_goal_adv)이
**전부 실패**했다 — 특히 flow_goal_adv는 정면 도킹조차 못 함. 반면 이전부터 배포돼 있던
`outputs/checkpoint_step_100000.pt`("느리지만 도킹이 잘 됨")는 held-out 지표 체계에 한 번도 편입된
적이 없었다. §4의 held-out 순위가 실전을 예측하지 못했다는 뜻이므로, 이 구 모델을 **처음으로**
같은 지표 체계에 넣어 직접 비교했다. (학습 없음 — 순수 추론 평가. `test/eval_old_baseline.py`)

**아키텍처 (`.pt`의 `model_state_dict` 키로 직접 확인 — config 파일 없이도 증명 가능)**:

```
room1 (2번째 카메라)   O   <- §2.6~2.11 신규 모델은 전부 X (room2 단일 카메라)
room2 (카메라)         O
goal 조건화            X   <- §2.8 nogoal 과 동일 상태
goal-lidar             X
lidar 브랜치            X   <- §2.7 nolidar 와 동일 상태
aux pose head          X   <- 인지 정밀도(near_mm) 측정 불가 (아래 참고)
```

`condition.room1_resampler.*` 키의 유무가 결정적 증거다 — `SensorFusionConditionNetwork`가
`use_room1=True`일 때만 이 파라미터를 생성하므로([sensor_fusion_condition.py:144-146](../cleandiffuser/nn_condition/sensor_fusion_condition.py#L144-L146)),
가중치만 보고 room1 사용 여부를 100% 확정할 수 있다. backbone은 DDPM(`ContinuousDiffusionSDE`,
`ema_rate=0.9999`, `solver=ode_dpmsolver++_2M`, `inference_sampling_steps=100`) — `postech_config.yaml`
(git 최초 커밋 `dca73b2`)의 명세와 일치. **ema_rate 0.9999라도 100,000 스텝이면
`0.9999^100000 ≈ 4.5e-5`로 EMA가 충분히 수렴** — [[ema-undertrained-4230-steps]]의 4230-스텝 버그
케이스와 달리 이 체크포인트의 `use_ema=True`는 신뢰할 수 있다.

**학습량 비교 (배치 크기 반영 — 스텝 수 단순 비교는 오독)**:

| | 배치 × 스텝 | 실제 데이터 노출량 | epoch 상당 |
|---|---|---|---|
| 구 baseline | 16 × 100,000 | 1,600,000 샘플 | ~7 epoch |
| flow_goal_scratch20 | 512 × 8,460 | 4,331,520 샘플 | 20 epoch |

**신규 모델이 오히려 2.7배 더 많이 학습했다** — "구 모델이 더 오래 학습해서 좋다"는 가설은
데이터로 기각된다. 스텝 수만 보고 "8,460은 미수렴"이라 판단한 것은 배치 크기를 무시한 오류였다
(사용자 지적, 07-15).

**측정 지표**: near_mm(인지 정밀도)은 aux head가 없어 측정 불가 — 비교할 예측값 자체가 없다.
대신 §7의 counterfactual align/xpos 지표(ICP 라벨 + 정책 자신의 롤아웃만 필요, aux head 불필요)로
제어 정밀도를 근사했다. ADE/FDE는 기존과 동일한 open-loop 프로토콜, 단 DDPM 100-step 샘플링 +
room1/room2 라이브 DINO 인코딩(캐시 없음 — room1 캐시가 애초에 없었음)이라 rectified-flow
20-step 모델보다 호출당 훨씬 느리다(homogeneous 비교 위해 에피소드 수는 축소: train 3개/
held-out 3개, MAX_STEPS 250).

**결과**:

| | ADE (cm) | FDE (cm) | velRMSE | align (°) | xpos (mm) | near_mm |
|---|---|---|---|---|---|---|
| **train** | **1.0** | **2.8** | 0.0145 | 1.67 | 13.8 | 측정불가(aux無) |
| **held-out** | **4.9** | **9.6** | 0.0264 | 2.10 | 12.2 | 측정불가(aux無) |

§4 표의 신규 모델(held-out)과 나란히 놓으면:

| 모델 | held-out ADE | held-out FDE | 비고 |
|---|---|---|---|
| **구 baseline (100k)** | **4.9** | **9.6** | 2카메라, goal/lidar/aux 없음, DDPM@100 |
| flow_goal_scratch20 | 5.2 | 11.1 | held-out 신규 1위였던 모델 |
| flow_goal_glidar_abs | 6.5 | 10.0 | |
| flow_goal_adv | 6.8 | 12.6 | 실기 최악 |

**해석 — 두 갈래**:
1. **held-out ADE/FDE만 보면 구 모델이 근소 우위**(9.6 vs 11.1)지만, §4.1의 통계력 한계(n=3~10
   에피소드로는 1~2cm 차이 판별 불가)를 고려하면 **"압도적으로 낫다"고 결론 내릴 근거는 약하다.**
   즉 07-15 실기에서 구 모델은 되고 신규는 실패한 그 격차를, **이 오프라인 지표 자체는 거의 못
   담아낸다** — open-loop + 동일 h5 전처리라는 방법론적 한계(§7.2, closed-loop 발산을 원리적으로
   못 봄)가 실기 처참한 차이를 예측하지 못한 정황과 일치한다.
2. **train/held-out 격차(ADE 1.0→4.9, FDE 2.8→9.6)는 구 모델도 신규 모델과 비슷한 비율로
   벌어진다** — "일반화 격차"는 구 모델도 갖고 있다. 즉 오프라인 지표가 안 잡는 실패 모드는
   신규 모델만의 문제가 아니라 **이 평가 방법론 전체의 사각지대**일 가능성이 크다.

**아직 미확정 — 다음 검증 필요**:
- 진짜 원인 후보 1순위는 **카메라 대수**(2카메라 vs 1카메라, room1 DINO 캐시 부재로 아직 재현
  실험 못 함), 2순위는 **샘플링 스텝**(100 vs 20 — `DEMO_STEPS` 노브로 재학습 없이 현장 검증
  가능), 3순위는 **배치 크기**(sharp-minima 가설, 256 재학습 필요).
- 이 세 가설을 분리하는 ablation(§6 갱신) 전까지, **오늘 결론은 "구 모델이 미신 아니게
  실측으로 더 낫다"는 것과 "그 이유를 이 문서의 held-out 지표로는 아직 설명 못 한다"는 것
  둘 다**다.

---

## 3. 공통 평가 방법

- **정밀도**: aux head가 예측한 dock pose와 ICP 라벨의 오차(mm), dock 거리 <0.6m 구간만
  (`test/dist_binned_error.py`의 거리 구간화 로직, `test/eval_run.py`가 재사용).
- **골 도달**: open-loop rollout — 실제 관측을 그대로 넣고 모델이 낸 속도를 **피드백 없이 적분**해
  ADE(경로 평균 오차)/FDE(최종 지점 오차)를 계산 (`test/eval_openloop_metrics.py`). 에피소드
  0/50/110(train) 또는 0~9(held-out) 기준.
- **주의**: open-loop는 실배포(closed-loop, 매 프레임 재계획)보다 항상 나쁘게 나오는 스트레스
  테스트다. 절대치가 아니라 **런 간 상대 비교**로만 쓴다.

---

## 4. 결과 종합표 — train vs held-out (근거: `test/out/weekend/*.json`)

정밀(mm) / ADE(cm) / FDE(cm) / **velRMSE**(×1000, 정규화 행동공간 [-1,1] 기준 — 절대 속도 단위
아님, 런 간 상대 비교용). velRMSE는 지금까지 측정만 하고 표에 누락돼 있었다 (07-14 보정).
`vel_progress_rmse`(vx 비대칭)·`speedup_frac`은 §4.2 참고 — AWR 실험 전용이라 이 표에는 아직
포함하지 않음(재현/암기 성적표에 시연-초과 지표를 섞으면 해석이 꼬인다, [[offline_metrics]] §3).

| 실험 | 학습 방식 | train (mm/ADE/FDE/velRMSE) | **held-out (mm/ADE/FDE/velRMSE)** |
|---|---|---|---|
| ddpm_goal_auxw | warm-start(ddpm) | 6.0 / 4.3 / 7.1 / 34.7 | 5.3 / 8.3 / 15.3 / 46.2 |
| flow_goal_auxw | warm-start(base) | 5.9 / 4.5 / 4.8 / 21.0 | 5.2 / 5.7 / 11.3 / 25.1 |
| flow_goal_auxw2 | warm-start +10ep | 5.4 / 2.7 / **4.0** / 22.6 | 4.6 / 7.7 / 15.0 / 29.3 (과적합) |
| flow_goal_cfg07 | warm-start | 5.6 / 2.6 / 5.8 / 21.1 | 4.8 / 8.3 / 12.3 / 26.6 |
| flow_goal_auxw_w2 | warm-start | 5.7 / 3.2 / 5.6 / 21.1 | 4.8 / 7.3 / 12.4 / 25.8 |
| flow_goal_p4 | warm-start | 5.3 / 2.8 / 6.3 / 21.3 | 4.9 / 8.6 / 12.8 / 26.4 |
| **flow_goal_scratch20** | **from-scratch 20ep** | 5.7 / 2.3 / 6.7 / **19.1** | **5.2 / 5.2 / 11.1** / 25.5 ⭐ |
| flow_goal_nolidar | from-scratch 10ep | **7.3** / 4.3 / 9.6 / 21.0 | **6.9** / 5.0 / 8.6 / 28.4 |
| flow_goal_nogoal | from-scratch 10ep | 6.6 / 6.3 / 11.0 / 27.3 | 6.2 / **6.9** / **15.0** / 29.2 |
| flow_goal_glidar | from-scratch 20ep | 6.9 / 3.1 / 4.7 / 19.2 | **9.7** / 5.9 / 11.4 / 27.6 |
| flow_goal_glidar_abs | from-scratch 20ep | 5.6 / 2.7 / 5.4 / **18.9** | 5.0 / 6.5 / 10.0 / 28.3 |
| **구 baseline (100k)** | 배치16, 100k step | 측정불가 / **1.0** / **2.8** / 14.5 | 측정불가 / 4.9 / 9.6 / 26.4 |
| `graft_g5_full` | graft, 구 baseline+풀스택 | 5.9 / 2.6 / 2.9 / 15.9 | 5.6 / 4.9 / 8.9 / 21.6 |
| `graft_goalimg_lidar` | graft, +goal-img+goal-lidar | 측정불가 / 0.7 / 2.0 / 7.7 | 측정불가 / 2.1 / 3.6 / 9.9 |
| `graft_goalimg` | graft, +goal-img만 | 측정불가 / 0.9 / 1.7 / 11.0 | 측정불가 / 1.9 / 4.0 / 10.9 |
| `graft_g0_awr` | graft, +AWR(precision)만 | 측정불가 / 1.2 / 2.4 / 16.4 | 측정불가 / 3.8 / 6.2 / 22.9 |
| `graft_g0_control` | graft, 구 baseline 배치/에폭만 변경 | 측정불가 / 1.5 / 4.3 / 8.4 | 측정불가 / 2.4 / 4.5 / 10.4 |
| `graft_goallidar` | graft, +lidar+goal-lidar만 | 측정불가 / 2.4 / 6.8 / 7.4 | 측정불가 / 2.1 / 3.9 / 9.1 |

> **graft6 행(07-17) 상세는 §9 참고** — 배치 128, 10 epoch, 구 baseline(100k)에서 warm-start.
> `graft_g5_full` 외 5개는 aux head가 없어(`use_aux_pose=false`) mm 열 측정불가는 §2.7/§2.8의
> nolidar/nogoal과 같은 이유(빈칸 아님, "다른 지표"). `speedup_frac` 등 AWR 전용 비대칭 지표는
> 이 표에 포함하지 않음(위 §4 서두 원칙과 동일) — §9 표에서 확인.
>
> 구 baseline은 아키텍처가 달라(2카메라, aux head 없음) 표의 다른 행과 직접 비교 시 §2.12를
> 반드시 함께 읽을 것 — mm 열은 aux head 부재로 측정 자체가 불가능(빈칸 아님, "다른 지표"），
> ADE/FDE는 표기상 낮아 보이지만 실기(07-15)에서는 이 모델만 도킹에 성공했다.

**velRMSE로 본 추가 통찰**: scratch20과 glidar_abs가 train velRMSE 최저(18.9~19.1)인데, 이 둘이
FDE 기준으로도 상위권이던 것과 방향이 일치 — velRMSE는 누적이 없어(§4.1) FDE보다 믿을 만한
신호다. nogoal은 train velRMSE부터 이미 최악(27.3)이라, 접근 축 악화가 "적분 오차의 우연"이
아니라 매 스텝 행동 자체가 부정확해서임을 확인해준다 (goal 조건이 진짜 행동 정확도에 기여).
ddpm_goal_auxw도 train부터 velRMSE가 압도적으로 나쁨(34.7, 2위와 1.6배 차이) — backbone
선택이 FDE 노이즈에 기대지 않고도 flow 우위로 판정 가능한 사례.

**읽는 법**: train 열만 보면 auxw2가 압도적 1위(FDE 4.0cm)지만, held-out에서는 **전 런 중 가장 나쁜
축에 속한다(FDE 15.0)** — 전형적 과적합. 이래서 "학습을 더 오래"가 무조건 답이 아니고, 판단은 반드시
held-out으로 해야 한다는 게 이번 스터디의 메타 결론이다. glidar도 train만 보면 정밀 6.9mm로 준수해
보이지만, held-out에서 **정밀도가 전 런 중 최악(9.7mm)으로 뒤집힌다** — auxw2와는 다른 경로(과적합이
아니라 §2.9의 타깃 잡음 배가, §2.10에서 확증)로 같은 교훈("train 수치를 믿지 말 것")을 한 번 더
확인시켜준 사례.

### 4.1 ⚠️ FDE의 통계력 — 이 표의 순위를 곧이곧대로 읽으면 안 되는 이유 (07-14 분석)

held-out FDE는 **에피소드 10개의 median**이다. 에피소드별 짝지은 차이(같은 에피소드에서 모델 A−B)의
표준편차를 재보면 **4~5cm** — 그런데 위 표에서 모델 간 차이는 대부분 1~2cm다. 부트스트랩 95% CI로
확인한 결과:

- glidar_abs vs scratch20: 평균 차이 +1.1cm, CI **[-1.7, +3.6]** → 유의하지 않음 (3/10 에피소드 승)
- nolidar vs scratch20: 평균 차이 +0.3cm, CI **[-2.6, +3.3]** → 유의하지 않음 (5/10 승)

즉 **FDE 순위표의 1~2cm 차이는 대부분 잡음**이다. 이 스터디에서 신뢰할 수 있는 결론은 효과가 큰
것들뿐: ① aux_relative의 해악(정밀 9.7↔5.0mm, 2배), ② LiDAR 제거 → 정밀도 악화(6.9 vs ~5.0mm,
수천 프레임 기준이라 통계력 충분), ③ auxw2/nogoal/ddpm의 FDE 15+ (차이 4cm급). scratch20 vs
auxw/glidar_abs 사이의 우열(11.1 vs 11.3 vs 10.0)은 **판별 불가**가 정직한 결론이다.
대응책은 [offline_metrics.md](offline_metrics.md) §4 (에피소드 수 확대, 짝지은 검정 상시화, 분산
작은 지표 우선).

---

## 5. 가설 검증 결론

1. **정밀도는 LiDAR가, 접근은 카메라+엔코더+goal이 담당한다** (사용자 가설) — held-out 데이터로
   지지됨: nolidar는 정밀도만 최악(6.9mm)이고 접근은 안 밀림; nogoal은 접근만 최악(FDE 15.0)이고
   정밀도는 상대적으로 덜 나쁨(6.2mm).
2. **모든 모델이 held-out 정밀도 4.6~5.3mm 구간에 수렴** — 이는 ICP 선생의 노이즈 바닥
   (~5.7mm, [[aux-head-vs-icp-noise-floor]])과 사실상 같은 수준. 정밀도 축은 이미 포화 —
   더 큰 개선은 선생(ICP 라벨) 품질 자체를 높이거나, 완전히 다른 정보원(예: goal-lidar 정합)이
   필요하다.
3. **접근(궤적 모방) 축은 아직 일반화 격차가 크다** (scratch20 기준 held-out FDE 11.1cm, train
   6.7cm) — 145개 시연으로는 부분 암기 상태. 다음 개선 대상.
4. **학습을 오래 하는 것과 일반화는 다른 문제** — auxw2가 이를 명확히 보여줌. 향후 모든 런은
   held-out 평가 없이 "최종"으로 채택하지 않는다.
5. **goal-lidar 정합: 상대 타깃은 확증적으로 해악, 조건화 자체는 무효과** (§2.9→§2.10 분리 실험
   완료) — `aux_relative`가 정밀도를 2배 악화시키는 범인임은 확증(9.7↔5.0mm). 그러나 타깃을
   되돌린 뒤에도 goal-lidar 조건 입력의 이득은 검출되지 않았다(짝지은 비교 유의성 없음, §4.1).
   **novelty 방향으로서의 "골 스캔 조건화"는 현 형태로는 기각** — 골 스캔이 주는 정보(도킹 완료
   시점의 dock 기하)가 이미 현재 스캔+goal 이미지로 충분히 커버되는 것으로 보인다. 남는 경로는
   조건화가 아니라 **행동 개선**(리워드 가중, residual RL)이다.
6. **FDE(n=10)는 1~2cm 차이를 판별할 통계력이 없다** (§4.1) — 이 문서와 후속 실험의 순위 판단은
   짝지은 비교 + CI를 상시 첨부하고, 큰 효과(≥4cm 또는 정밀도 mm 배수 차이)만 결론으로 채택한다.
7. **⚠️ 이 held-out 지표 체계 전체가 실기(closed-loop) 성능을 예측하지 못한다는 게 07-15 확인됐다**
   (§2.12). scratch20·glidar_abs·flow_goal_adv는 held-out 순위표에서 상위권이었지만 실제 시연에서
   전부 실패했고, 반대로 이 체계에 한 번도 들어온 적 없는 구 baseline(2카메라, aux 無)이 유일하게
   성공했다. 원인은 아직 미확정(§2.12 "미확정" 항 참고: 카메라 대수/샘플링 스텝/배치크기 3가지
   후보, 분리 안 됨) — 결론 6번의 "판별 불가"보다 더 근본적인 문제로, **이 문서의 결론 1~6은
   모두 held-out 지표가 실기를 대변한다는 전제 위에 있었고 그 전제 자체가 흔들렸다.** 새 결론을
   내리기 전 §6의 재현 ablation이 선행돼야 한다.

---

## 6. 이 스터디가 가리키는 다음 단계

**⚠️ 07-15 갱신 — 아래 항목들은 §2.12(실기 실패 후 구 baseline 대조) 이전에 쓰여, 지표 체계
자체의 신뢰성이 흔들리기 전의 우선순위다. 지금 최우선은 아래가 아니라 §2.12 "미확정" 3가설
(카메라 대수 / 샘플링 스텝 / 배치 크기) 분리이며, 그 결과가 나오기 전까지 재학습 착수를 보류한다
(사용자 지시, 07-15).**

- ~~goal-lidar 조건화 재시도 (타깃 분리)~~ → **§2.10으로 완료.** 타깃 분리 결과 상대-타깃 해악은
  확증, 조건화 이득은 무검출. 이 갈래는 종료. (단, §2.12로 이 novelty 갈래 자체의 우선순위가
  재검토 대상 — 접근/정밀 축 최적화가 실기 무관하다면 무의미하다.)
- **행동 개선 축으로 전환**: 시연이 느리다는 한계(프레임 54%가 <2cm/s)를 BC는 복사할 수밖에 없으므로,
  advantage-weighted BC(오프라인 residual, 진행 중) → 온라인 residual RL(시뮬레이터 필요) 순서로
  "시연을 넘어서는" 경로를 검증한다. 상세는 [experiment_roadmap.md](experiment_roadmap.md).
- **평가 통계력 확보**: held-out 에피소드 확대(10→30+), 짝지은 검정 상시화, closed-loop 지표 도입.
  상세는 [offline_metrics.md](offline_metrics.md) §4.
- **(보류) 상대 타깃 재시도 조건**: 골 기준점 `g`를 마지막 N개 reliable 프레임의 median으로 잡아
  잡음 배가를 줄일 수 있다면 재시도 가치가 있으나, §2.10에서 조건화 이득 자체가 없었으므로
  우선순위 낮음.

---

## 7. 강화학습 파트 — AWR (Advantage-Weighted Regression) 핵심 알고리즘

> 07-14 진행 중인 `flow_goal_adv` / `ddpm_goal_adv`가 쓰는 방법.
> 코드: `scripts/train.py`(가중 적용), `utils/docking_dataset.py`(advantage 계산),
> `cleandiffuser/diffusion/{rectifiedflow,diffusionsde}.py`(`sample_weight` 인자).

### 7.1 왜 RL이 필요한가 — BC의 원리적 한계

BC(모방학습)는 시연의 조건부 행동분포 `p(a|s)`를 그대로 복사한다. 따라서:

- 시연이 느리면 **정책도 느리다**. 시연에 존재하지 않는 빠른 궤적은 출력 자체가 불가능하다.
- 시연이 도달한 정밀도가 상한이다. ICP 선생의 노이즈 바닥(~5.7mm)에 이미 붙어 있으므로
  (§5-2), BC를 더 돌려도 정밀도는 안 내려간다.

**"시연보다 더 잘하려면" 리워드를 직접 최대화해야 한다** — 그게 RL이다.

### 7.2 그런데 진짜 RL은 지금 불가능하다 (중요한 제약)

RL은 "행동 → 결과 관측 → 보정"의 **closed-loop 환경**이 필수다. 지금 우리에겐
시뮬레이터도, 온라인 로봇 루프도 없다. 정책이 만든 행동이 실제로 어디로 데려가는지
되먹임받을 수단이 없으므로 PPO/SAC 같은 온라인 RL은 돌릴 수 없다.

그래서 **오프라인에서 가능한 리워드 학습**을 택했다 = AWR.
(진짜 residual RL은 시뮬레이터 구축 후 — [experiment_roadmap.md](experiment_roadmap.md) Stage D.)

### 7.3 AWR 핵심 아이디어 (한 문장)

> **"좋은 시연 구간은 더 세게, 나쁜 시연 구간은 더 약하게 모방한다."**

수식으로는 BC loss에 리워드 기반 샘플별 가중치를 곱한 것:

```
표준 BC:   L = E_(s,a)~D [ ‖ε_θ(s, a_noised) − ε‖² ]              (모든 샘플 동일 가중)
AWR:       L = E_(s,a)~D [ w(s,a) · ‖ε_θ(s, a_noised) − ε‖² ]
           w(s,a) = exp( A(s,a) / β )                             (advantage에 지수 가중)
```

이는 advantage-weighted regression / reward-weighted BC로 알려진 표준 오프라인 RL 계열
(AWR, AWAC, CRR 등)의 가장 단순한 형태다. **정책 구조는 그대로**, loss 가중치만 바뀐다 —
그래서 diffusion policy에 그대로 얹을 수 있다.

### 7.4 이 문제에서의 Advantage 정의 — "얼마나 빠르게 dock에 접근했나"

우리 데이터는 ICP 라벨(§8) 덕분에 **모든 프레임에 dock까지의 거리가 이미 있다**.
그래서 리워드를 따로 설계할 필요 없이 거리 감소율을 그대로 쓴다:

```
프레임 t에서 시작하는 horizon(H=60, 2초)에 대해:

  progress(t) = ( dock_dist(t) − dock_dist(t+H) ) / (H · dt)      [m/s]
                └ 이 2초 동안 dock에 초당 몇 m 가까워졌나

  A(t) = ( progress(t) − mean ) / std                             [z-score 정규화]

  w(t) = exp( clip(A(t), −adv_clip, +adv_clip) / adv_beta )
  w(t) ← w(t) / mean(w)                                           [배치 평균 1로 정규화]
```

- **빠르게 접근한 구간** → progress↑ → A↑ → **w > 1** → 더 세게 학습
- **꾸물대거나 정지한 구간** → progress≈0 → A<0 → **w < 1** → 약하게 학습
- **dock에서 멀어진 구간** → progress<0 → **w ≪ 1** → 거의 무시

구현 위치:
- `utils/docking_dataset.py`: `__init__`에서 전체 프레임의 progress를 미리 계산하고 z-score 통계
  (`_adv_mean`, `_adv_std`)를 저장 → `__getitem__`이 샘플별 `adv` 반환.
  라벨이 없는 프레임(t 또는 t+H가 unreliable)은 `adv=0` → `w=1`(중립).
- `scripts/train.py`: `w = exp(clip(adv/β)) / mean(w)` 를 계산해 `sample_weight`로 전달.
- `cleandiffuser/diffusion/*.py`: `loss(..., sample_weight=w)` — per-element loss에
  배치 차원으로 broadcast 후 평균. **기본값 `None`이면 기존 동작과 완전히 동일**(하위호환).

**하이퍼파라미터**:
- `adv_beta`(온도, 기본 1.0): 작을수록 advantage에 공격적으로 반응. 0에 가까우면 "가장 빠른
  구간만 학습"(= greedy filtering), 크면 균일 BC로 수렴.
- `adv_clip`(기본 2.0): z-score를 ±2로 클립. 한 배치가 극단값 소수 샘플에 지배되는 것을 방지.

### 7.5 aux(정밀도) loss는 건드리지 않는다 — 역할 분리

`sample_weight`는 **denoising loss(궤적 생성)에만** 적용되고, aux pose loss(정밀도)는 그대로다:

```
total_loss = denoise_loss(sample_weight=w)   ← AWR 가중 (속도/효율 담당)
           + aux_weight · aux_loss           ← 거리 가중 그대로 (정밀도 담당)
```

이 분리 덕분에 학습 곡선에서 **검증이 가능**하다: AWR을 켜도 `dock` 오차 곡선은
scratch20과 **동일하게** 움직여야 하고(aux 경로 불변), denoising loss만 달라져야 한다.
실제로 07-14 학습 중 관측: step 6800에서 dock **26.4mm(adv) vs 26.4mm(scratch20)** — 일치,
denoising loss 0.0775 vs 0.0758 — AWR 재가중으로 소폭 상승. **의도대로 동작 확인.**

### 7.6 AWR의 정직한 한계 (반드시 인지할 것)

> **AWR은 시연 분포 안에서 재가중할 뿐, 시연에 없는 행동을 창조하지 못한다.**

- 시연 중 가장 빠른 구간이 5cm/s라면, AWR 정책의 상한도 ~5cm/s다. 그 이상은 **온라인 RL만** 가능.
- 즉 AWR은 "느린 시연을 덜 배우기"이지 "시연보다 빠르기"가 아니다. 기대 효과는
  **시연 분포의 빠른 꼬리로 이동**하는 정도 — 개선은 있겠지만 극적이진 않을 것이다.
- 평가 시 주의: 표준 `vel_rmse`는 "시연보다 빠른 것"도 오차로 penalize하므로 AWR에 불리하다.
  그래서 `test/eval_openloop_metrics.py`에 **비대칭 지표**를 추가했다
  (`vel_progress_rmse`: 같은 방향으로 더 빠른 것은 0으로 처리, `speedup_frac`: 시연보다 빠른
  프레임 비율). AWR이 의도대로 작동하면 `speedup_frac`이 baseline 대비 올라가야 한다.

---

## 8. 배경 — ICP 라벨과 aux head는 무엇인가

> §2의 모든 실험이 "정밀도(near_mm)"를 말하는데, 그 숫자가 어디서 오는지의 정본 설명.
> 코드: `scripts/label_subgoals.py`(라벨 생성), `endgame/icp_matcher.py`(ICP),
> `cleandiffuser/nn_condition/sensor_fusion_condition.py`(aux head).

### 8.1 문제: 정답(GT)이 없다

정밀 도킹을 학습시키려면 "지금 dock이 정확히 어디 있는가"의 정답이 필요한데,
이 데이터셋에는 **marker_pose가 죽어 있어 GT가 없다.** 그래서 정답을 **우리가 만들어야** 했다.

### 8.2 ICP — 고전 기하 알고리즘으로 정답을 만든다 (오프라인 교사)

**ICP(Iterative Closest Point)**: 학습이 전혀 없는 고전 알고리즘. 재료는 두 가지 —
① dock의 알려진 모양(템플릿 점군), ② 방금 들어온 LiDAR 스캔.

```
1. 현재 추정 pose로 템플릿을 스캔 위에 얹는다
2. 템플릿의 각 점 → 가장 가까운 스캔 점을 짝짓는다 (correspondence)
3. 그 짝들의 거리를 최소화하는 강체변환(이동+회전)을 최소제곱으로 푼다
4. pose 갱신 → 1~3 반복 (수렴까지 수십 회, 수 ms)
```

"퍼즐 조각을 그림 위에 놓고 조금씩 밀고 돌려가며 가장 잘 맞는 자리를 찾는 것"과 같다.
결과: **dock의 SE(2) pose (x, y, θ)** — 그것도 **로봇 자신의 현재 좌표계 기준**이다
(로봇이 원점, dock이 앞쪽 어딘가. dock을 (0,0)으로 잡는 게 아니다).

**템플릿 출처**: `endgame/assets/dock_template_real.npy` — 154개 에피소드의 도킹 완료 스캔을
ICP로 한 좌표계에 겹쳐 누적한 1,640점의 U-노치 형상 (`scripts/build_dock_template.py`).

**라벨 생성 절차** (`scripts/label_subgoals.py`): 에피소드의 **마지막(도킹 완료) 프레임에서
시간을 거꾸로** 추적한다. 도킹 완료 시점은 dock이 코앞이라 ICP가 확실히 잠기고, 12.5Hz
연속 스캔은 프레임 간 이동이 작아 직전 pose를 시드로 쓰면 안정적으로 역추적된다.

**신뢰도 판정**: `inlier_ratio ≥ 0.5` AND `RMS 잔차 ≤ 2.5cm`를 만족하면 `reliable=1`.
이 두 필드(`dock_pose`, `reliable`)가 h5에 구워져 있다.

**교사의 노이즈 바닥 (핵심 상수)**: 도킹 정지 상태에서 ICP를 반복하면 pose가 **±5.7mm**
흔들린다(`test/icp_noise_floor.py`로 측정). **이게 라벨 자체의 정확도 한계**이고, 따라서
**학생(aux head)이 이보다 정확해질 수 없다** — §5-2의 "정밀도 포화"가 여기서 나온다.
또한 dock 거리 1.1m를 넘으면 라벨 자체가 cm급으로 틀리기 시작해서, aux loss에서 마스킹한다
(`aux_dist_max: 1.1`).

### 8.3 aux head — ICP를 신경망에 증류(distill)한다

**핵심 구도는 사제(師弟) 관계다:**

| | 선생 (teacher) | 학생 (student) |
|---|---|---|
| 무엇 | 고전 ICP 알고리즘 | aux head (신경망) |
| 언제 | **오프라인** (라벨 생성 시 1회) | **런타임** (매 프레임 추론) |
| 역할 | 정답 dock pose를 만든다 | 그 정답을 흉내 내도록 학습 |

> ⚠️ 흔한 오해: "모델 안에 ICP가 들어있다"가 **아니다.** 모델 안에 있는 건
> **ICP의 출력을 흉내 내도록 학습된 신경망**이다. 런타임에 ICP는 돌지 않는다.

**구조** (`sensor_fusion_condition.py`):
- 조건 네트워크가 카메라(DINO)+속도+LiDAR+goal을 융합하는데, 그 중 **LiDAR 포인트 토큰**에
  **cross-attention**하는 작은 head(`CrossAttnPoseHead`)를 붙였다.
- 출력: `[x_norm, y_norm, sin θ, cos θ]` (4차원) — 정규화된 dock pose.
- LiDAR가 없으면(nolidar 런) 융합 벡터에서 MLP로 뽑는 fallback head를 쓴다
  → 그래서 nolidar의 정밀도가 확 나빠진다(6.95mm, §2.7).

**손실** (`scripts/train.py`):
```
aux_loss = Σ w(t) · ‖aux_pred(t) − ICP_label(t)‖²  /  Σ w(t)
           w(t) = reliable(t) · (aux_dist_ref / dock_dist)^aux_dist_power · [dock_dist ≤ 1.1m]
           └ 신뢰 프레임만, 가까울수록 큰 가중 (정밀도가 중요한 건 근거리니까)

total_loss = denoise_loss + aux_weight · aux_loss
```

**두 가지 목적**:
1. **표현 강제**: 조건 네트워크가 "dock이 정확히 어디 있는지"를 내부적으로 표현하게 만들어,
   궤적 생성(denoising)도 정밀해지도록 유도한다.
2. **런타임 pose 산출**: ICP 없이도 dock pose를 즉답 → 도착 판정, MPC 랭킹(`test/mpc_rank.py`)의
   심판 등에 쓸 수 있다.

### 8.4 "정밀도 near_mm"이 정확히 무슨 숫자인가

§2/§4 표의 정밀도 = **aux head 예측과 ICP 라벨의 XY 오차(mm)의 median**,
단 **dock 거리 <0.6m 구간(도킹 직전)의 신뢰 프레임만** (`test/eval_run.py`).

- 왜 <0.6m만? 정밀 도킹에서 중요한 건 마지막 구간이다. 접근 전 구간을 평균 내면
  멀리 있는 프레임의 큰 오차가 섞여 지표가 흐려진다(07-10에 "29mm"로 보이던 게 실은 이 착시).
- 수천 프레임 기준이라 **FDE(에피소드 10개)보다 통계적으로 훨씬 신뢰도가 높다**(§4.1).

---

## 9. Graft6 — 구 baseline(100k) 접붙이기 ablation (2026-07-17)

> `test/queue_graft6.sh` 기준. 전부 `outputs/checkpoint_step_100000.pt`(§2.12 구 baseline, 2-camera
> DDPM, 실기 유일 성공작)에서 warm-start, 10 epoch, batch_size=128, live 2-camera DINO
> (`use_dino_cache=false`). 목적: 구 baseline에 신규 기능(goal 조건화 / LiDAR+aux / AWR)을 하나씩
> 접붙였을 때 held-out 지표가 어떻게 움직이는지 확인 — §2.12가 실기 성공의 원인 후보로 지목한
> "카메라 대수" 외에, "신규 기능 자체가 실기 실패의 원인이었는지"를 구 baseline 위에서 직접 검증.
> 학습 로그는 `outputs/train_graft_*.log`, 평가 원본은 `test/out/weekend/graft_*.json`.

> 07-18: `speedup_frac`(시연보다 같은 방향으로 빠르거나 같은 프레임 비율, §7.6) 열 추가.

| 실험 | 추가된 것 | train (mm/ADE/FDE/velRMSE/speedup) | held-out (mm/ADE/FDE/velRMSE/speedup) |
|---|---|---|---|
| `graft_g5_full` | 풀스택: goal-image + lidar + aux + goal-lidar + AWR(precision) | 5.9 / 2.6 / 2.9 / 15.9 / 41% | 5.6 / 4.9 / 8.9 / 21.6 / 44% |
| `graft_goalimg_lidar` | + goal-image + goal-lidar 함께 추가 (aux 없음, AWR 없음) | - / 0.7 / 2.0 / 7.7 / 33% | - / 2.1 / 3.6 / 9.9 / 30% |
| `graft_goalimg` | + goal-image 조건화만 추가 | - / 0.9 / 1.7 / 11.0 / 26% | - / 1.9 / 4.0 / 10.9 / 28% |
| `graft_g0_awr` | + AWR(precision)만 추가 — AWR 자체 효과만 분리 | - / 1.2 / 2.4 / 16.4 / 42% | - / 3.8 / 6.2 / 22.9 / 41% |
| `graft_g0_control` | control: 새 브랜치 없음, AWR 없음 (구 baseline + 배치/에폭만 변경) | - / 1.5 / 4.3 / 8.4 / 34% | - / 2.4 / 4.5 / 10.4 / 26% |
| `graft_goallidar` | + LiDAR 브랜치 + goal-lidar 조건화 추가 (goal-image 없음) | - / 2.4 / 6.8 / 7.4 / 51% | - / 2.1 / 3.9 / 9.1 / 33% |

<!-- GRAFT6:graft_goallidar -->

<!-- GRAFT6:graft_g0_control -->

<!-- GRAFT6:graft_g0_awr -->

<!-- GRAFT6:graft_goalimg -->

<!-- GRAFT6:graft_goalimg_lidar -->

<!-- GRAFT6:graft_g5_full -->

### 9.1 해석 (07-18, 6개 전부 완료 후)

**held-out FDE로 정렬**(goal-reaching, cm, 낮을수록 좋음):
`goalimg_lidar`(3.6) ≈ `goallidar`(3.9) ≈ `goalimg`(4.0) < `g0_control`(4.5) < `g0_awr`(6.2) < `g5_full`(8.9).

- **goal-image / lidar+goal-lidar 단독 그래프트는 control보다 나쁘지 않다** — 오히려 근소하게
  낫다(3.6~4.0 vs 4.5). 단, §4.1 기준 이 정도(1cm 미만) 차이는 통계력 밖(잡음)이라 "확실히
  개선"이라 단정하긴 이르다. 적어도 **이 두 기능이 단독으로는 실기 실패의 원인이 아니라는** 방향의
  증거.
- **AWR이 가장 뚜렷한 부정적 신호다.** `g0_awr`(AWR만 추가)은 control 대비 held-out FDE가
  4.5→6.2cm로 악화되고, 비대칭 속도 지표 `progRMSE`(demo보다 빠른 건 페널티 안 주는 지표)도
  10.1→22.5로 **2배 이상** 나빠진다 — `speedup_frac`(41% vs 26%)이 오른 건 AWR이 의도대로
  "빠른 시연 쪽으로" 재가중은 하고 있다는 뜻이지만, 그 대가로 궤적 자체가 더 노이즈해진다는 뜻.
  이 차이는 FDE 1~2cm대 잡음 수준을 넘어서는(§4.1 기준 ~4cm급 근접) 편이라 신뢰도가 더 높다.
- **`g5_full`(전체 스택)이 6개 중 최악**(held FDE 8.9, progRMSE 20.6)인데, 구성 요소별 결과를
  보면 이 악화의 주범은 goal/lidar 조건화가 아니라 **AWR로 보인다** — `g0_awr` 단독으로도 이미
  같은 방향(속도 노이즈↑, FDE↑)의 악화가 나타나고, `goalimg`/`goallidar`/`goalimg_lidar`는
  AWR 없이도 문제가 없었기 때문.
- **정밀도(mm)는 `g5_full`만 측정 가능**(aux head가 있는 유일한 셀) — held-out 5.6mm로, 다른
  스터디에서 반복 관측된 ICP 라벨 노이즈 바닥(~5.7mm, §8.2)에 붙어 있다. 포화, 특이사항 없음.
- **아직 열려 있는 질문**: 이 표는 전부 open-loop 오프라인 지표다. §2.12에서 이미 이 지표 체계가
  실기 성능을 예측하지 못한 전례가 있으므로("held-out 상위권 신규 모델 전부 실기 실패, 구
  baseline만 성공"), 여기서 AWR이 나쁘게 나온 것도 **실기와 반드시 일치한다는 보장은 없다.**
  다만 07-15 실기에서 `flow_goal_adv`(AWR 포함)가 "정면 도킹조차 못 함"으로 최악이었던 것과
  방향이 일치하긴 한다 — 두 개의 독립적인 관측(이전 실기 실패 + 이번 오프라인 열화)이 같은 곳을
  가리키므로, **AWR을 다음 실기 검증의 최우선 용의자로 삼는 것을 권장**. `g0_control` vs
  `g0_awr` 두 체크포인트만 골라 closed-loop로 재검증하면 저비용으로 확증 가능.
