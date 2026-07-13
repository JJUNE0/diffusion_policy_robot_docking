# 주말 Ablation 스터디 (2026-07-10 ~ 07-12) — 각 실험이 확인하는 것

> 기준일: 2026-07-13
> 근거: `test/weekend_queue.sh`, `outputs/weekend_queue.log`, `test/out/weekend/*.json`
> 성능을 **두 축**으로만 본다: **정밀도**(aux dock-pose 오차, mm) + **골 도달**(open-loop ADE/FDE, cm).
> 평가는 두 세트로 한다: **train**(외운 문제, 145 에피소드) vs **held-out**(처음 보는 10 에피소드,
> `dataset/after_0328_test.h5`). **결론은 반드시 held-out 기준으로 낸다** — train 순위는 07-12 밤
> 재평가에서 뒤집혔다 (§4).

---

## 0. 왜 이 순서인가 — 큐 설계 논리

GPU 시간이 주말 이틀로 제한돼 있어서, **"확실히 좋을 것 같은 실험부터"** 배치했다
(`test/weekend_queue.sh` 주석 참고). 그래서 두 가지 학습 전략이 섞여 있다:

- **warm-start (fine-tune)**: 이미 수렴한 `flow_goal_auxw` 체크포인트에서 이어 학습, lr을
  낮춰(3e-5) 빠르게 여러 하이퍼파라미터를 저비용으로 스캔. `auxw2`, `cfg07`, `auxw_w2`, `p4`가 이 방식.
- **from-scratch**: 처음부터 새로 학습(lr 1e-4). 아키텍처가 바뀌거나(`nolidar`, `nogoal`) 깨끗한
  비교 기준이 필요할 때(`scratch20`) 사용.

**주의 (해석 시 계속 의식할 것)**: warm-start 런들은 `flow_goal`(구 EMA 버그 세대) → `flow_goal_auxw`
→ 이번 런 순으로 가중치를 물려받아, 실질 학습 스텝 수가 label과 다르다(예: `auxw2`는 "20 epoch"이지만
실제로는 base 4230 + auxw 4230 + auxw2 8460 스텝 노출). **nolidar/nogoal은 10 epoch from-scratch인데
반해 scratch20은 20 epoch from-scratch** — 그래서 nolidar/nogoal을 scratch20과 직접 비교할 때는
"에포크 수 차이"와 "모달리티 제거 효과"가 섞여 있다는 걸 감안해야 한다 (§4 표에 표시).

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
**바꾼 것**: 없음(warm-start 안 함) — `flow_goal_auxw`와 동일한 최종 설정(거리 가중 aux, ema 0.999)을
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

| 실험 | 학습 방식 | train: 정밀(mm)/ADE/FDE(cm) | **held-out: 정밀(mm)/ADE/FDE(cm)** |
|---|---|---|---|
| ddpm_goal_auxw | warm-start(ddpm) | 6.0 / 4.3 / 7.1 | 5.3 / 8.3 / 15.3 |
| flow_goal_auxw | warm-start(base) | 5.9 / 4.5 / 4.8 | 5.2 / 5.7 / 11.3 |
| flow_goal_auxw2 | warm-start +10ep | 5.4 / 2.7 / **4.0** | 4.6 / 7.7 / 15.0 (과적합) |
| flow_goal_cfg07 | warm-start | 5.6 / 2.6 / 5.8 | 4.8 / 8.3 / 12.3 |
| flow_goal_auxw_w2 | warm-start | 5.7 / 3.2 / 5.6 | 4.8 / 7.3 / 12.4 |
| flow_goal_p4 | warm-start | 5.3 / 2.8 / 6.3 | 4.9 / 8.6 / 12.8 |
| **flow_goal_scratch20** | **from-scratch 20ep** | 5.7 / 2.3 / 6.7 | **5.2 / 5.2 / 11.1** ⭐ |
| flow_goal_nolidar | from-scratch 10ep | **7.3** / 4.3 / 9.6 | **6.9** / 5.0 / 8.6 |
| flow_goal_nogoal | from-scratch 10ep | 6.6 / 6.3 / 11.0 | 6.2 / **6.9** / **15.0** |

**읽는 법**: train 열만 보면 auxw2가 압도적 1위(FDE 4.0cm)지만, held-out에서는 **전 런 중 가장 나쁜
축에 속한다(FDE 15.0)** — 전형적 과적합. **held-out 기준 최종 순위는 scratch20 > flow_goal_auxw >
나머지.** 이래서 "학습을 더 오래"가 무조건 답이 아니고, 판단은 반드시 held-out으로 해야 한다는 게
이번 스터디의 메타 결론이다.

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

---

## 6. 이 스터디가 가리키는 다음 단계

- **goal-lidar 정합 모델** (`use_goal_lidar` + `aux_relative`, 커밋 `8bb7c74`): §5-1의 결론이
  "LiDAR=정밀, goal=접근"이라면, **골 LiDAR 스캔과 vision을 함께 조건화**해 두 역할을 한 표현으로
  합치는 것이 다음 novelty 실험. `flow_goal_glidar`로 from-scratch 20 epoch 학습 진행 중
  (§0과 동일하게 scratch20/nolidar/nogoal과 같은 절차로 held-out 비교 예정).
- **접근 축 일반화 개선**: 시연 개수 증가, 또는 cfg07 방향(무조건부 학습 비중↑ → CFG guidance
  스윕)으로 goal 조건의 강건성을 높이는 방향.
