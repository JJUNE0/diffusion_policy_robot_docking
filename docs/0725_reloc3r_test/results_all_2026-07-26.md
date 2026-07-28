# 전체 평가 결과 집계 (2026-07-26)

지금까지 held-out 평가를 마친 **모든 모델의 숫자**를 한 곳에 모았다. 논문 표 작성 시 이 문서를
단일 출처로 쓴다.

---

## 0. 읽기 전에 — 프로토콜과 교란 요인

**표를 가로로 비교하기 전에 이 절을 반드시 읽을 것.** 모델군마다 학습 조건이 다르고, 그 차이가
지표 차이보다 크다.

| 모델군 | 카메라 | 초기화 | 예산 | 아키텍처 |
|---|---|---|---|---|
| `r_*_b16s*` (시드 스윕) | **1** (dino_bottom) | scratch | batch 16 × 100k step ≈ 7.4 ep | token-seq + cross-attn |
| `r_*` (batch256) | **1** | scratch | batch 256 × 16,940 step ≈ 20 ep | token-seq + cross-attn |
| `r2cam_*` (시연용, §1a) | **2** (bottom+top) | scratch | batch 16 × 100k step ≈ 7.4 ep | token-seq + cross-attn |
| `graft_*` | **2** (room1+room2) | `checkpoint_step_100000.pt` | batch 128 × 10 ep **위에** 100k 사전학습 | legacy pooled-AdaLN |
| `old_baseline_100k` | **2** | scratch | batch 16 × 100k step ≈ 7.4 ep | legacy pooled-AdaLN |
| `s20_*` | 미확인 | 미확인 | 20 ep 표기 | legacy pooled-AdaLN |
| `r_geo_5f` (§5a, **다른 데이터셋**) | **1** (dino_bottom) | scratch | batch 256 × 7,760 step = 20 ep | token-seq + cross-attn |

> ⚠️ **graft가 다른 모든 군보다 ADE/FDE에서 3배 우월한 것은 조건화 때문이 아닐 가능성이 높다** —
> 카메라 2대 + 100k 사전학습이라는 두 요인이 같은 방향으로 작용한다. graft 군의 내부 비교
> (§6)만 조건화 효과로 읽을 수 있고, graft ↔ r_* 횡비교는 조건화 결론에 쓸 수 없다.

**평가 프로토콜**: 아래 전부 `dataset/after_0328_test.h5`, **10 에피소드(0–9) × 500 스텝**.
유일한 예외는 `test/out/weekend/old_baseline_100k_heldout.json`(구버전, **3 에피소드 × 250 스텝**)
이며 이 문서에서는 참고용으로만 표기한다.

> ⚠️ **§5a(`r_geo_5f`)는 학습·평가 데이터셋 자체가 다르다** (`front_dock_5f_*`, 5층 정면 도크).
> 본 문서의 다른 모든 행과 **어떤 지표로도 횡비교할 수 없다** — 도크 위치·조명·시연자·에피소드
> 구간 정의가 전부 다르다. §6 순위표와 §7 상관분석에도 **포함하지 않았다.**

### 지표 정의

| 지표 | 뜻 | 출처 |
|---|---|---|
| ADE / FDE (cm) | open-loop 500스텝 롤아웃의 평균/최종 변위 오차. **접근 구간이 지배** | `eval_run_rgeo.py` |
| velRMSE | 명령 (v, ω)의 RMSE | 동일 |
| **align (°)** | counterfactual 종단 정렬 잔차 median. θ_dock(t+H)=θ_dock(t)−∫ω dt | `eval_align_rgeo.py` |
| **xpos (mm)** | counterfactual 종단 전진 위치 오차 median | 동일 |
| p90 / xp90 | 위 두 지표의 90 백분위 — **꼬리** | 동일 |
| 우회전% / vs demo | 회전 프레임 중 우회전 비율, 시연(53.6%) 대비 부호 있는 편차. **부호를 반드시 볼 것** | `terminal_metric.py` |
| bias (%p) | 위 편차의 **절댓값**. `abs()`가 좌/우 방향을 지운다 — §1a 참고, 단독으로 쓰지 말 것 | `terminal_metric.py` |
| park (%) | 종단 100–200mm 밴드에서 사실상 정지한 프레임 비율 (시연 13.9%) | 동일 |
| ratio | 같은 밴드에서 시연 대비 전진 속도비 | 동일 |
| near (mm) | aux head의 dock-pose 인지 오차. **aux head 있는 모델만** | `eval_run.py` |

align/xpos의 기준 pose는 **ICP dock_pose**다. 학습은 ICP-free이며, 동일 계측기가 모든 모델과
시연을 함께 채점하므로 *비교*는 공정하고 *절대값*만 ICP 오차를 물려받는다.

---

## 1. 3×3 시드 스윕 — 핵심 결과 (batch 16 × 100k, 1카메라, scratch)

| run | align | p90 | xpos | xp90 | ADE | FDE | velRMSE | bias | park | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| r_nogoal s0 | 1.728 | 3.205 | 10.67 | 51.1 | 6.24 | 11.48 | 0.0232 | +8.3 | 3.1 | 0.87 |
| r_nogoal s1 | 1.495 | 3.296 | 11.37 | 49.5 | 7.24 | 13.40 | 0.0238 | **+13.1** | 2.4 | 0.86 |
| r_nogoal s2 | 1.846 | 3.718 | 12.44 | 55.1 | 8.56 | 14.04 | 0.0264 | +3.3 | 5.8 | 1.10 |
| r_goal s0 | 1.561 | 3.111 | 11.04 | 46.6 | **3.96** | **7.40** | **0.0163** | +1.2 | 2.4 | 0.92 |
| r_goal s1 | 1.904 | 3.284 | 11.39 | 48.6 | 7.23 | 13.43 | 0.0279 | +10.3 | 1.4 | 0.89 |
| r_goal s2 | 1.668 | 3.154 | 10.00 | 46.7 | 6.35 | 10.97 | 0.0234 | +5.1 | 2.7 | 0.80 |
| r_geo s0 | 1.538 | 3.048 | 10.68 | 45.3 | 7.21 | 12.57 | 0.0248 | +8.3 | 1.0 | 0.78 |
| r_geo s1 | 1.605 | 3.085 | 10.15 | 46.0 | 5.59 | 9.84 | 0.0229 | +8.4 | 2.4 | 0.99 |
| r_geo s2 | 1.579 | **2.947** | 10.17 | 50.2 | 6.45 | 11.36 | 0.0236 | +7.3 | 2.0 | 0.85 |

### arm별 평균 ± 표준편차

| arm | align | xpos | ADE | FDE | bias | p90 |
|---|---|---|---|---|---|---|
| r_nogoal | 1.690 ± 0.179 | 11.50 ± 0.89 | 7.35 ± 1.17 | 12.97 ± 1.33 | +8.2 ± 4.9 | 3.41 ± 0.27 |
| r_goal | 1.711 ± 0.175 | 10.81 ± 0.72 | 5.85 ± 1.69 | 10.60 ± 3.03 | +5.5 ± 4.6 | 3.18 ± 0.09 |
| r_geo | **1.574 ± 0.034** | **10.33 ± 0.30** | 6.42 ± 0.81 | 11.26 ± 1.37 | +8.0 ± 0.6 | **3.03 ± 0.07** |

### 검정 — 평균 효과는 전부 null

| 지표 | one-way ANOVA (arm 간 평균) | Levene (arm 간 분산) |
|---|---|---|
| align | F=0.77, **p=0.506** | p=0.444 |
| xpos | F=2.19, **p=0.193** | p=0.566 |
| bias | F=0.44, **p=0.662** | p=0.300 |
| ADE | F=1.06, **p=0.403** | p=0.737 |
| FDE | F=1.05, **p=0.407** | p=0.501 |

**결론: 3시드 기준 어떤 조건화도 어떤 종단 지표에서도 유의한 평균 차이를 만들지 않는다.**

### 관측된 패턴 (가설, 미확립)

r_geo의 시드 간 **표준편차**가 일관되게 가장 작다:

| 지표 | nogoal sd | goal sd | geo sd | 배율 |
|---|---|---|---|---|
| align | 0.179 | 0.175 | **0.034** | 5.2× |
| bias | 4.895 | 4.575 | **0.622** | 7.4× |
| xpos | 0.891 | 0.722 | **0.296** | 2.4× |

geo를 나머지 6런과 합쳐 비교하면 align F=22.3 (p=0.044), bias F=52.0 (p=0.019)로 유의하지만,
**적절한 3군 분산검정(Levene)은 전부 유의하지 않다(p=0.30–0.57).** n=3에서 sd 추정은 불확실성이
극단적으로 크다. **발견이 아니라 가설**로만 취급하고, 검증에는 arm당 6–8시드가 필요하다.

---

## 1a. 2카메라 현장 시연용 실험 (2026-07-27, batch 16 × 100k)

**배경**: 실기에서 유일하게 성공한 모델(`old_baseline_100k`)만 카메라 2대(room1+room2)이고
§1의 시드 스윕 9런은 전부 카메라 1대다. 07-27 시연을 앞두고 `dino_top`(room1) 캐시를
새로 만들어 `no_goal_2cam` / `goal_appearance_geometry_2cam` 두 config를 batch16×100k
(old_baseline과 동일 예산)로 학습하고, **60k 중간 체크포인트와 100k 최종 체크포인트를 모두**
평가했다. 두 체크포인트를 별도 디렉터리로 격리해 각각 그 시점의 값을 정확히 쟀다
(`outputs/eval60k_r2cam_*`, 학습 계속 진행 중인 원본 디렉터리와 분리).

| model | ckpt | 우회전% | vs demo | align | p90 | xpos | ADE | FDE | vel | park% | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **old_baseline_100k** | 100k | **42.9** | **−10.7** | — | — | — | 7.67 | 12.13 | 0.0261 | 2.4 | 1.12 |
| r2cam_geo | 60k | 43.8 | −9.8 | — | — | — | 6.9(med) | 14.1 | — | 1.0 | 1.12 |
| r2cam_geo | **100k** | 56.8 | +3.2 | 1.476 | 2.875 | 11.64 | 7.33 | 12.60 | 0.0224 | 1.4 | 0.97 |
| r2cam_nogoal | 60k | 50.3 | −3.3 | — | — | — | 8.3(med) | 14.2 | — | 3.0 | 1.03 |
| r2cam_nogoal | **100k** | 62.6 | +9.0 | 1.539 | 2.933 | 11.03 | 5.70 | 11.33 | 0.0209 | 0.3 | 0.87 |

> 60k 행의 align/p90/xpos/vel은 별도 align eval을 돌리지 않아 공란. ADE/FDE(med)는 rollout eval
> 로그의 median 값(요약 json 미보존)이므로 다른 행의 mean과 직접 비교하지 말 것.

### 핵심 발견 — 조향 편향은 같은 학습 런 안에서도 크게 드리프트한다

두 모델 모두 60k→100k 사이에 **거의 같은 크기로(+12.3%p, +13.0%p) 우측으로 이동**했다:

| model | 60k | 100k | 이동 |
|---|---|---|---|
| r2cam_geo | −9.8 (좌) | +3.2 (우) | **+13.0 %p** |
| r2cam_nogoal | −3.3 (좌) | +9.0 (우) | **+12.3 %p** |

이 이동 폭(12~13%p)은 §1에서 관측한 **arm 간 효과(1~13%p)나 시드 간 분산(±4.9~4.9%p)과 같은
크기**다. 즉 조향 편향은 arm·시드·**체크포인트** 세 축 모두에서 조건화 효과와 구분 안 되는
크기로 흔들린다.

> ⚠️ **이 발견은 다음 사실을 기각한다** (직전 세션에서 60k 데이터만으로 임시 결론 냈던 것):
> "카메라 2대 모델 3개는 전부 좌편향, 1대 12개는 전부 우편향으로 15/15 완벽 분리" — 100k
> 최종값을 §1의 100k arm들과 나란히 놓으면 r2cam_geo(56.8%)와 r2cam_nogoal(62.6%)은 1카메라
> 분포(54.7~66.6%) 한가운데로 섞여 들어간다. 카메라 수가 편향 방향을 가른다는 가설은 **성립하지
> 않는다.** 유일하게 예외적으로 남는 것은 old_baseline(42.9%)뿐인데, 이 모델은 카메라 수뿐 아니라
> 아키텍처(pooled-AdaLN, lidar/goal 부재)도 다르므로 카메라 수만으로 설명할 수 없다.

**실무적 함의**: 조향 편향을 모델 선택 기준으로 쓰려면 **배포할 그 체크포인트를 직접 측정**해야
한다 — "이 arm은 안전하다"는 식의 arm 단위 결론은 근거가 없다. 시연에 가져간
`eval60k_r2cam_geo/checkpoint_step_60000.pt`의 43.8%는 그 파일 자체를 측정한 값이라 유효하지만,
같은 런의 100k는 이미 56.8%로 넘어가 있다.

**검증 진행 중 (2026-07-27 오전 시작)**: 1카메라 arm(`r_nogoal_b16s0`, `r_goal_b16s0`,
`r_geo_b16s0`)의 60k 체크포인트와 `r_geo_b16s0`의 20k/40k/60k/80k 궤적을 동일 방식으로 평가해,
이 드리프트가 2카메라 특유의 현상인지 학습 일반의 현상인지 확인 중. 완료되면 본 절을 갱신한다.

---

## 2. batch 256 × 16,940 단일 시드 (원래 3 arm) + 시연 기준선

| model | align | p90 | xpos | xp90 | ADE | FDE | velRMSE | bias | park | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| **시연(demo)** | 1.028 (p25 0.604) | 2.336 | 8.83 | — | — | — | — | (우 53.6%) | 13.9 | 1.00 |
| r_nogoal | 1.444 | 2.652 | 8.96 | 41.4 | 8.00 | 12.59 | 0.0354 | +3.6 | 13.6 | 0.72 |
| r_goal | 1.498 | 2.598 | 9.98 | 42.3 | 8.26 | 15.17 | 0.0339 | +8.8 | 16.7 | 0.61 |
| r_geo | **1.281** | 2.834 | **8.19** | **39.1** | 7.92 | 14.35 | 0.0351 | +5.6 | 7.8 | 0.69 |

### 이 군의 paired Wilcoxon (동일 480 프레임)

| 비교 | align | xpos |
|---|---|---|
| geo vs nogoal | −0.153, p=1.4e−9 | −0.855, p=2.3e−3 |
| geo vs goal | −0.089, p=3.5e−5 | −0.874, p=3.8e−2 |
| goal vs nogoal | −0.068, p=8.6e−3 | −0.037, **p=0.47 (ns)** |

> ⚠️ **이 p값들은 §1에서 재현되지 않았다.** 프레임 짝짓기는 프레임 표본오차만 통제하고 학습 시드
> 분산을 통제하지 못한다. 시드 스윕의 arm 내 sd(align 0.18°)가 여기서 관측된 arm 간 효과(0.153°)
> 보다 크다. **이 표의 유의성은 논문 주장의 근거로 쓸 수 없다.**

**예산 교체의 부수 효과**: batch256→batch16/100k로 바꾸면 align이 악화(1.28–1.50 → 1.57–1.71)
되는 반면 ADE는 개선(7.9–8.3 → 3.96–8.56)된다. 두 축이 서로 반대로 움직인 또 하나의 사례.

---

## 3. old_baseline_100k (2카메라, ddpm, goal/lidar/aux 없음)

| 프로토콜 | ADE med | ADE mean | FDE med | FDE mean | velRMSE | align | xpos |
|---|---|---|---|---|---|---|---|
| **10ep × 500step (교정판)** | 5.47 | 7.67 | 10.07 | 12.13 | 0.0261 | — | — |
| 3ep × 250step (구버전, 참고) | 4.88 | — | 9.64 | — | 0.0264 | **2.105** | **12.19** |

아키텍처는 체크포인트의 `model_state_dict`로 직접 확인했다(`room1_resampler`+`room2_resampler`
존재, goal/lidar/aux 관련 가중치 부재). backbone은 DDPM이며 권위 있는 설정은
`ai-control/ai_models/postech_config.yaml` (git `dca73b2`): horizon 60, batch 16,
checkpoint_step 100000, ema_rate 0.9999, solver `ode_dpmsolver++_2M`.

**이 모델은 2026-07-15 현장 시연에서 유일하게 도킹에 성공한 구성이다** (`ablation_study_2026-07.md`
§2.12). 동시에 align 2.105°/xpos 12.19mm로 **정렬 지표에서는 측정된 전 모델 중 최악**이다.

---

## 4. graft 군 (2카메라 + 100k 사전학습 위 finetune)

| model | 조건화 | ADE | FDE | velRMSE | align | xpos | near |
|---|---|---|---|---|---|---|---|
| graft_goalimg | goal-img | **1.91** | 4.01 | 0.0109 | 1.419 | 9.79 | — |
| graft_lidar_goalimg | goal-img + lidar | **1.91** | 3.84 | 0.0104 | 1.479 | 9.40 | — |
| graft_gil_aux | goal-img+lidar+aux | 1.92 | 5.01 | 0.0122 | 1.425 | 9.27 | 5.09 |
| graft_goalimg_lidar | goal-img+lidar+goal-lidar | 2.07 | **3.60** | 0.0099 | 1.373 | 11.74 | — |
| graft_goallidar | lidar + goal-lidar | 2.11 | 3.90 | **0.0091** | 1.413 | 10.85 | — |
| graft_auxfb_lidar | aux-feedback + lidar | 2.25 | 5.79 | 0.0137 | 1.593 | 10.90 | 5.50 |
| graft_auxfb_full | aux-feedback full | 2.34 | 4.58 | 0.0148 | 1.464 | 9.66 | 5.58 |
| graft_g0_control | **대조군(조건화 없음)** | 2.42 | 4.51 | 0.0104 | 1.417 | 10.88 | — |
| graft_goallidar_aux | lidar+goal-lidar+aux | 2.42 | 5.05 | 0.0124 | 1.481 | 9.81 | 13.24 |
| graft_lidar | lidar만 | 2.49 | 3.85 | 0.0100 | 1.458 | 9.86 | — |
| graft_gil_awr_p2 | +AWR v2 | 3.60 | **3.71** | 0.0220 | 1.335 | 10.91 | — |
| graft_g0_awr | +AWR | 3.83 | 6.16 | 0.0229 | **1.226** | 9.47 | — |
| graft_gil_awr_p1 | +AWR v1 | 4.12 | 5.26 | 0.0207 | 1.345 | 10.35 | — |
| graft_g5_full | — | 4.86 | 8.87 | 0.0216 | 1.355 | 9.64 | 5.59 |

### graft 내부 paired 검정 — 대조군 대비 조건화 효과 (n=10 동일 에피소드)

| 조건화 | ADE | FDE |
|---|---|---|
| **goal-img** | −0.41cm, **10/10승, p=0.002 \*** | −0.65cm, 8/10승, **p=0.037 \*** |
| goal-img + lidar + goal-lidar | −0.56cm, 7/10, p=0.160 ns | −0.89cm, 8/10, p=0.105 ns |
| lidar + goal-lidar | −0.53cm, 6/10, p=0.375 ns | −0.23cm, 6/10, p=0.922 ns |
| lidar만 | +0.22cm, 4/10, p=0.432 ns | +0.22cm, 5/10, p=0.695 ns |

**2카메라 사전학습 트렁크 위에서는 goal-image가 ADE를 유의하게 개선한다.** 이는 §1(단일카메라
scratch에서 효과 없음)과 상충하지 않는다 — 조건화 이득이 트렁크 성숙도와 시각 입력 규모에
의존한다는 해석이 가능하다.

> ⚠️ 다만 이 검정도 **단일 시드 쌍**의 에피소드 짝짓기다. §2의 p=1.4e−9가 시드에서 무너진 것과
> 구조가 같으므로, 이 결과를 논문 주장으로 쓰려면 graft 쪽도 시드 복제가 선행돼야 한다.

---

## 5. s20 / weekend 군

| model | ADE | FDE | velRMSE | align | xpos | near |
|---|---|---|---|---|---|---|
| s20_batch256 | 8.18 | 14.89 | 0.0319 | 1.280 | 8.62 | 4.85 |
| s20_nogoal | 6.68 | 13.57 | 0.0267 | 1.392 | 9.66 | 5.18 |
| s20_nolidar | 6.60 | 11.31 | 0.0289 | 1.303 | — | 6.45 |

`ablation_study_2026-07.md` §4.1이 **통계력 충분**으로 인정한 유일한 조건화 결론이 여기 있다:
**LiDAR 제거 시 인지 정밀도 악화 (near 6.45 vs ~5.0mm, 수천 프레임 규모).**

---

## 5a. front_dock_5f — 신규 데이터셋 R-Geo scratch 학습 (2026-07-27)

`dataset/front_dock_5th_floor/*.zip`(15개 아카이브, 86 레코드)로 **5층 정면 도크** 전용 R-Geo를
scratch 학습했다. **평가는 아직 실시하지 않았다** — 아래는 데이터셋 구축·학습 수렴까지의 기록이다.

> ⚠️ **본 문서 다른 절과 횡비교 불가.** 학습·held-out 모두 `front_dock_5f_*`이며
> `after_0328_test.h5`가 아니다. §6 순위·§7 상관에 포함하지 않았다.

### 데이터셋 구축 — 두 개의 필수 필터

`dataset/front_dock_5th_floor/readme.md`가 지정한 두 규칙을 `scripts/build_front_dock_5f.py`가
강제한다.

1. **`metadata.json`의 `labels.success == true`인 레코드만.**
2. **segment 마커 이후 구간만.** `segment_01`은 도킹 시작 자세로 접근하는 주행이며 모방 대상이
   아니다. 센서 CSV/JSONL은 exporter가 이미 segment별로 쪼개 두므로 `segment_02..N` 연결로
   충분하고(3-segment 레코드 2건 = `episode_476`, `episode_406`은 02+03 사용), **카메라 프레임은
   분할돼 있지 않아 타임스탬프로 필터**한다.

| 항목 | 수 |
|---|---|
| 원본 레코드 | 86 |
| `success=false`로 제외 | 4 (`509`, `530`, `532`, `536`) |
| 마커 없음으로 제외 | 1 (`515`, success=true이나 segment 전무) |
| **최종 에피소드** | **81** |

마커 필터의 효과는 크다 — 예: `record_00474`는 전체 92.2초 중 **도킹 구간이 42.7초**(46%)로,
나머지 절반은 접근 주행이다.

### 스플릿 · 규모

| 스플릿 | 에피소드 | 프레임 | 에피소드당 프레임 min/med/max | 총 시간 |
|---|---|---|---|---|
| train | 71 | 103,654 | 775 / 1,278 / 3,530 | 57.6 분 |
| held-out | 10 | 17,896 | 1,043 / 1,936 / 2,759 | 9.9 분 |

레코드 ID 정렬 후 앞 71 / 뒤 10으로 나눴다. 그 결과 **held-out 10개가 전부 `0713` 세션**에
몰린다 — i.i.d.가 아니라 **세션 단위 일반화 테스트**다(더 엄격한 쪽). 필요하면 교차 분할로
재빌드해야 한다.

### 전처리에서 드러난 잠복 버그 — ns/s 단위 불일치

이 exporter 포맷은 타임스탬프가 **나노초**인데 `utils/preprocessing.py`는 **초**를 가정한다
(`target_interval = 1/target_hz`, `max_time_diff=0.05`). `np.arange(min_t, max_t, 0.0333)`가
4.3e10 span에 대해 ~1.3e12 원소를 할당하려다 **SIGKILL(exit 137, 트레이스백 없음)** 로 죽는다.
로그에 아무것도 남지 않아 원인 파악이 어렵다.

→ sync 코드가 아니라 **컨버터에서 ns→s로 정규화**했다(CSV 컬럼, JSONL 필드, `_<ts>.jpg` 파일명
접미사 전부). 기존 `after_0328`이 이미 쓰던 규약에 맞추는 방향이다. **이 exporter 포맷을 쓰는
다른 데이터셋도 동일하게 걸린다.**

부수 변경: 이 데이터셋은 도크가 보이는 카메라가 `camera_orbbec-0`(→ room2) 하나뿐이라
`utils/preprocessing.py`에 **`--no_room1`** 을 추가했다(가짜 room1을 복제해 넣는 대신 `image_top`
자체를 쓰지 않음 → h5 절반). `utils/docking_dataset.py`의 `z_img1`도 `use_room1` 가드를 걸었다.

### 캐시

| 파일 | train | held-out |
|---|---|---|
| `front_dock_5f_*.h5` | 13.7 GB | 2.3 GB |
| `*_dino_bottom.h5` (DINOv3 patch feat) | 31.2 GB | 5.4 GB |
| `*_reloc3r_bottom.h5` (+`geometry_bottom`) | 41.6 GB | 7.2 GB |

전 캐시 행 정렬 1:1 확인, NaN 0건. 빌드는 `bash scripts/prepare_front_dock_5f.sh` 한 방.

> 참고: `lidar_npoints` 평균이 **42.9** 로 `after_0328`(≈80, `docs/plan/02_preprocessing.md` §5)의
> 절반이다. crop 반경은 동일한 0.8m이므로 5층 도크 주변 스캔이 더 성기다는 뜻. point 브랜치는
> npoints 마스킹을 하므로 학습은 진행되지만, LiDAR 정밀도 관련 결론을 낼 때 고려해야 한다.

### 캘리브레이션 전이 검증 (측정된 결과)

R-Geo의 geometry 토큰은 `reloc3r/body_frame_calibration_odometry.json`(camera→body 회전)에
의존하는데, 이 값은 **`after_0328`에서 적합된 것**이다. 전이 가정을 검증 없이 쓰지 않기 위해
동일 스크립트를 5F 오도메트리로 재적합해 비교했다(ICP-free 경로 그대로).

| | axis→z 오차 median | fwd-translation dir 오차 median |
|---|---|---|
| `after_0328` 적합 (재사용 중) | 6.98° | 10.04° |
| **5F 재적합** | 4.85° | 7.46° |

**두 회전의 geodesic 각도 = 1.65°** — 적합 자체의 잔차(5–7°)보다 훨씬 작다. 같은 로봇·같은
카메라 마운트이므로 예상대로 전이되며, **재사용한 캘리브레이션으로 만든 geometry 토큰은 유효**
하다(재빌드 불필요).

토큰 자체의 정합성도 확인: `dx,dy` 단위노름 1.0, `sin²+cos²=1`, NaN 0, **goal 프레임에서
yaw ≈ −0.03°**(자기 자신 대비 → 0이어야 함), 에피소드 진행에 따라 |yaw| **1.42° → 0.03°** 감소.

재적합 산출물은 별도 경로에 저장했고 `reloc3r/body_frame_calibration_odometry.json`은
**덮어쓰지 않았다**(§1 이후 모든 런과의 일관성 유지).

### 학습 설정 · 수렴

`configs/robot/smr_rgeo_5f.yaml` + `sensors_variant/goal_appearance_geometry_5f.yaml`.
네트워크·옵티마이저·batch·`action_norm: minmax` 전부 `smr_rgeo.yaml`과 동일하고 **데이터셋과
캐시 경로만 다르다** — §2의 `r_geo`(batch256 × 20 ep)와 예산이 일치한다.

조건 토큰 5종 전부 결선 확인: `wheel(60,2)` / `rgb_history(5,196,768)` / `lidar(256,2)` /
`goal(196,768)` / `geometry(4)`.

batch 256 × 7,760 step = **20 epoch**, 5.5시간(H100 1장).

| step 구간 | 평균 loss |
|---|---|
| 0–999 | 0.0774 |
| 1000–1999 | 0.0489 |
| 2000–2999 | 0.0465 |
| 3000–3999 | 0.0454 |
| 4000–4999 | 0.0416 |
| 5000–5999 | 0.0431 |
| 6000–6999 | 0.0418 |
| 7000–7760 | **0.0400** |

산출물: `outputs/train/r_geo_5f/2026-07-27_18-48-31/` — `checkpoint_step_{1000..7000,7760}.pt`
(8개), `metrics.jsonl`(776행), `train_convergence.png`.

### 미실시 — 다음 단계

**held-out 평가를 아직 돌리지 않았다.** 따라서 이 모델에 대해서는 align/xpos/ADE/FDE/bias 어느
것도 본 문서에 숫자가 없다. 평가 시 §7·§8의 확립된 결론에 따라 **ADE/FDE가 아니라
align_deg/xpos_mm를 1차 지표로** 읽어야 하며, §1a에 따라 **배포 후보 체크포인트의 조향 편향은
그 파일에서 직접** 재야 한다(arm 단위 일반화 근거 없음). 단일 시드이므로 §8-3에 의해
**조건화 결론에는 쓸 수 없다** — 5F에서 arm 비교를 하려면 `no_goal`/`goal_appearance` 5F 변형을
같은 예산으로 학습해 시드 스윕까지 가야 한다.

---

## 6. 전체 모델 순위

### align (°) — 낮을수록 좋음

`graft_g0_awr` 1.226 < **r_geo(b256) 1.281** ≈ `s20_batch256` 1.280 < `s20_nolidar` 1.303 <
`graft_gil_awr_p2` 1.335 < `graft_gil_awr_p1` 1.345 < `graft_g5_full` 1.355 <
`graft_goalimg_lidar` 1.373 < `s20_nogoal` 1.392 < `graft_goallidar` 1.413 <
`graft_g0_control` 1.417 < `graft_goalimg` 1.419 < `graft_gil_aux` 1.425 <
**r_nogoal(b256) 1.444** < `graft_lidar` 1.458 < `graft_auxfb_full` 1.464 <
`graft_lidar_goalimg` 1.479 < `graft_goallidar_aux` 1.481 < **r_goal(b256) 1.498** <
`r_geo(b16) 1.574` < `graft_auxfb_lidar` 1.593 < `r_nogoal(b16) 1.690` < `r_goal(b16) 1.711` <
**`old_baseline_100k` 2.105 (최악)**

### xpos (mm) — 낮을수록 좋음

**r_geo(b256) 8.19** < `s20_batch256` 8.62 < **시연 8.83** < r_nogoal(b256) 8.96 <
`graft_gil_aux` 9.27 < `graft_lidar_goalimg` 9.40 < `graft_g0_awr` 9.47 < `graft_g5_full` 9.64 <
`graft_auxfb_full`/`s20_nogoal` 9.66 < `graft_goalimg` 9.79 < `graft_goallidar_aux` 9.81 <
`graft_lidar` 9.86 < r_goal(b256) 9.98 < `r_geo(b16) 10.33` < `graft_gil_awr_p1` 10.35 <
`graft_goallidar` 10.85 < `graft_g0_control` 10.88 < `graft_auxfb_lidar` 10.90 <
`graft_gil_awr_p2` 10.91 < `graft_goalimg_lidar` 11.74 < **`old_baseline_100k` 12.19 (최악)**

### ADE (cm) — 낮을수록 좋음

`graft_goalimg`/`graft_lidar_goalimg` 1.91 < `graft_gil_aux` 1.92 < `graft_goalimg_lidar` 2.07 <
`graft_goallidar` 2.11 < … < `r_goal_b16s0` 3.96 < … < `old_baseline_100k` 7.67 <
r_geo(b256) 7.92 < r_nogoal(b256) 8.00 < `s20_batch256` 8.18 < r_goal(b256) 8.26

---

## 7. 두 지표군의 탈상관 — 확립된 결과

측정된 **21개 모델** 전체에 대한 순위 상관:

```
Spearman  ADE vs align_deg :  rho = -0.286,  p = 0.209   (유의하지 않음)
Spearman  ADE vs xpos_mm   :  rho = -0.230,  p = 0.329   (유의하지 않음)
```

**"역전"이 아니라 "무상관"이다.** 궤적 지표는 종단 정렬에 대해 거의 정보를 주지 않는다.

극단적 해리 사례:

| model | ADE 순위 | align 순위 |
|---|---|---|
| `old_baseline_100k` | 중위권 (7.67) | **최악 (2.105)** — 그런데 실기 유일 성공 |
| `s20_batch256` | **최하위권 (8.18)** | 상위권 (1.280) |
| `graft_g0_awr` | 중하위 (3.83) | **최상위 (1.226)** |

같은 체크포인트에서 추론 파라미터만 바꾼 비교에서도 동일하게 나타난다: 실행 구간 K를 2→16으로
늘리면 ADE가 2–3배·velRMSE가 3–5배 악화되지만(3모델 × 2설정 = 6/6 단조 일관) align은
1.32–1.56° 구간에서 사실상 불변이다. **이 비교는 학습을 공유하므로 시드 교란이 원천적으로 없다.**

---

## 8. 무엇이 확립되었고 무엇이 안 되었나

### 확립됨

1. **궤적 지표와 종단 정렬 지표는 탈상관** (ρ=−0.25~−0.29, ns, n=21). K 실험이 시드 교란 없는
   독립 증거.
2. **LiDAR 제거는 인지 정밀도를 악화** (near 6.45 vs ~5.0mm, 수천 프레임).
3. **단일 시드 ablation은 이 체제에서 해석 불가**: 시드 sd(align 0.179°, bias 4.9%p)가 관측된
   조건화 효과(0.153°, 5.2%p)보다 크다.
4. **1cm급 종단 전진 정렬**: xpos 중앙값 8.2mm(단일시드) / 10.3mm(3시드 평균), 시연 자체 8.83mm.
   ICP-free 학습으로 달성.
5. **old_baseline은 정렬 지표 최악인데 실기 유일 성공** — offline 지표 체계의 예측력 문제.
6. **조향 편향은 학습 진행에 따라 arm/시드 효과와 같은 크기(12~13%p)로 드리프트한다** (§1a,
   r2cam_geo·r2cam_nogoal 둘 다 60k→100k에서 우측으로 동일 크기 이동). 조향 편향을 모델 선택
   기준으로 쓰려면 배포할 그 체크포인트를 직접 측정해야 하며, arm 단위 일반화는 근거가 없다.

### 확립되지 않음

1. geometry 토큰의 종단 정렬 개선 (ANOVA p=0.506) — **기각**
2. goal appearance의 해악 / dose–response (bias ANOVA p=0.662) — **기각**
3. geometry 토큰의 시드 분산 감소 — **가설** (Levene ns, 6–8시드 필요)
4. graft에서의 goal-image 이득 (p=0.002) — **단일 시드, 시드 복제 미실시**
5. 실로봇 도킹 성공률 — **실험 0회**
6. lateral(y) 정렬 — **측정 안 함** (ICP 노이즈 지배). 실제 도킹 실패의 주 원인이 lateral이라는
   현장 보고와 어긋나는 중요한 공백.
7. **카메라 수가 조향 편향의 방향을 결정한다 — 기각** (§1a). 60k 중간 체크포인트만 놓고 "2카메라
   3개 전부 좌편향, 1카메라 12개 전부 우편향"이라 결론 냈던 것이 100k 최종값에서 무너짐. 유일하게
   남는 이례적 사례는 old_baseline(42.9%)뿐이며, 이는 카메라 수 외에도 아키텍처가 다르므로
   카메라 수만의 효과로 귀속할 수 없다.
8. **`r_geo_5f`의 성능 — 평가 0회** (§5a). 학습 수렴(loss 0.077→0.040)만 확인했다. 지표가 하나도
   없으므로 5층 도크 일반화에 대해 어떤 주장도 할 수 없다. 단일 시드라 평가를 마쳐도 §8-3에
   의해 조건화 결론에는 쓸 수 없다.

### §5a에서 부수적으로 확립된 것

- **Reloc3r camera→body 캘리브레이션은 데이터셋 간 전이된다** — `after_0328` 적합과 5F 재적합의
  geodesic 차이 1.65°로 적합 잔차(5–7°)보다 작다. 같은 로봇/마운트라면 재적합 불필요.
  (단, 이는 캘리브레이션의 전이일 뿐 **정책의 전이가 아니다**.)

---

## 9. 재현

```bash
# 시드 스윕 (arm당 3시드, 9런 병렬)
bash test/queue_rgeo_seeds.sh <gpu> <arm> <sensors_variant> <seed>

# 궤적 지표
EVAL_H5=dataset/after_0328_test.h5 EVAL_STATS_H5=dataset/after_0328_train.h5 \
EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout \
  python test/eval_run_rgeo.py <run_dir>

# 종단 정렬 지표 (paired, 480 프레임, per-frame npz 동시 저장)
EVAL_H5=dataset/after_0328_test.h5 EVAL_STATS_H5=dataset/after_0328_train.h5 EVAL_TAG=heldout \
  python test/eval_align_rgeo.py <run_dir>

# old_baseline 교정 평가
python test/eval_run_old_baseline_100k.py
```

### §5a — front_dock_5f 빌드 · 학습

```bash
# 1) zip -> 에피소드 레이아웃 (success=true + segment_02.. 만, ns->s 정규화)
python scripts/build_front_dock_5f.py \
  --zip_dir dataset/front_dock_5th_floor --out dataset/front_dock_5f

# 2) h5 + DINO + Reloc3r rot/dir + geometry 토큰 (train 71 / held-out 10)
bash scripts/prepare_front_dock_5f.sh

# 3) 학습 (batch 256 x 20 ep = 7,760 step)
python scripts/train.py --config-name smr_rgeo_5f

# (선택) 캘리브레이션 전이 재확인 — 기존 json을 덮어쓰지 않도록 --out 지정
python scripts/calibrate_reloc3r_odometry.py \
  --h5 dataset/front_dock_5f_train.h5 \
  --cache dataset/front_dock_5f_train_reloc3r_bottom.h5 \
  --out /tmp/calib_5f.json

# 평가(미실시) — EVAL_H5/EVAL_STATS_H5를 5f로 바꿔야 한다
EVAL_H5=dataset/front_dock_5f_test.h5 EVAL_STATS_H5=dataset/front_dock_5f_train.h5 \
EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout5f \
  python test/eval_align_rgeo.py outputs/train/r_geo_5f/2026-07-27_18-48-31
```

산출물: `test/out/rgeo/*.json`, `test/out/rgeo/*_align.json`,
`test/out/rgeo/*_align_perframe.npz` (paired 검정용).

---

## 관련 문서

- `docs/ablation_study_2026-07.md` — 선행 ablation. §2.12(구 baseline 실기 성공), §4.1(통계력), 결론 5·7
- `docs/paper_draft_2026-07-26.md` — 논문 초안 (§5.2·§5.3은 본 문서 §1 결과로 무효)
- `docs/paper_draft_guide_2026-07-26.md` — 작성 가이드 (시드 판정 규칙은 본 문서 §1로 확정됨)
- `docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md` — R-NoGoal/R-Goal/R-Geo 스펙
