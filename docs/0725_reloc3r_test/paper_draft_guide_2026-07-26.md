# 논문 초안 작성 가이드 (2026-07-26)

**목적**: 지금까지 나온 결과로 *오늘 밤~내일* 쓸 수 있는 주장과 쓰면 안 되는 주장을 갈라주고,
숫자를 붙여넣기 가능한 형태로 정리한다.

**핵심 한 줄**: 지금 데이터로는 **"성능을 X% 올렸다" 논문은 못 쓴다. 하지만 "무엇을 재야 하는지를
바로잡고, goal 조건화의 숨은 실패 모드를 규명한" 분석 논문은 쓸 수 있다.** 후자가 더 방어하기 쉽고,
지금 증거로 실제로 지지된다.

---

## 0. 상태 요약 — 무엇이 준비됐나

| 항목 | 상태 |
|---|---|
| 3-arm ablation (R-NoGoal / R-Goal / R-Geo), 각 1시드 | 완료, 평가 완료 |
| 종단 정렬 지표(align_deg / xpos_mm) 측정 | 완료 (`test/eval_align_rgeo.py`, 신규) |
| ICP-free hand-eye 캘리브레이션 | 완료, 독립 검증됨 |
| **3시드 × 3arm 재현 스윕 (batch16 × 100k)** | **학습 중, 2026-07-26 ~20:37 완료 예정** |
| 실로봇 실험 | **없음** |

---

## 1. 확정된 사실 (논문에 써도 되는 것)

### 1.1 종단 정렬 축 — held-out, 동일 480 프레임, paired

| model | align_deg (median) | align p90 | xpos_mm (median) |
|---|---|---|---|
| demonstration (참조) | 1.028 (p25 0.60) | — | 8.83 |
| **R-Geo** | **1.281** | 2.834 | **8.19** |
| R-NoGoal | 1.444 | 2.652 | 8.96 |
| R-Goal | 1.498 | 2.598 | 9.98 |
| old baseline (100k) | 2.10 | — | 12.19 |

paired Wilcoxon (n=480, 세 arm이 동일 프레임 채점):

| 비교 | align_deg | xpos_mm |
|---|---|---|
| Geo vs NoGoal | **−0.153, p=1.4e−9** | **−0.855, p=2.3e−3** |
| Geo vs Goal | −0.089, p=3.5e−5 | −0.874, p=3.8e−2 |
| Goal vs NoGoal | −0.068, p=8.6e−3 | −0.037, **p=0.47 (ns)** |

> ⚠️ 이 p값은 **프레임** 짝짓기다. **학습 시드 분산은 통제하지 못한다.** §3 참조.

### 1.2 종단 거동 축 — steering bias / 정지

| arm | 우회전 비율 (demo 53.6%) | parked @100–200mm (demo 13.9%) | term vx ratio |
|---|---|---|---|
| R-NoGoal | 57.2% (**+3.6**%p) | 13.6% | 0.72 |
| R-Goal | 62.4% (**+8.8**%p) | **16.7%** | **0.61** |
| R-Geo | 59.2% (+5.6%p) | **7.8%** | 0.69 |

n=3,713–3,755 (SE 0.008), 종단 밴드 n=294.

**dose–response가 단조롭다**: goal appearance만 넣으면 세 지표 전부 악화, geometry를 얹으면 회복.
이는 `configs/robot/sensors_variant/goal_appearance_geometry.yaml`에 사전 기록된 가설
("goal-estimate 토큰이 항상 켜져 있으면 trunk가 거기 기대고, OOD 추정 오차가 조향 편향으로 샌다")과
방향이 일치한다 — **사후 해석이 아니라 사전 등록된 예측의 확인**이라는 점이 논문에서 강하다.

### 1.3 지표 축 자체의 발견 (가장 방어하기 쉬운 기여)

**궤적 지표(ADE/FDE/velRMSE)와 종단 정렬 지표가 모델을 정반대로 순위매긴다.**

| model | ADE mean | velRMSE | align_deg | xpos_mm |
|---|---|---|---|---|
| old baseline | **7.67 cm** | **0.0261** | 2.10 | 12.19 |
| s20_batch256 | 8.18 cm | 0.0319 | **1.28** | **8.62** |
| R-Geo | 7.92 cm | 0.0351 | **1.28** | **8.19** |

이유: ADE/FDE는 500스텝 open-loop 궤적을 적분하므로 **접근 구간이 지배**하고, 결국 *전체 궤적의
모방 충실도*를 잰다. 도킹 성패는 마지막 수 cm에서 갈린다. 게다가 ADE/FDE는 에피소드 n=10의
median이라 통계력이 없다(`docs/ablation_study_2026-07.md` §4.1: 짝지은 차이의 표준편차 4~5cm인데
모델 간 차이는 1~2cm). 반면 align은 n=480 프레임 paired라 검정이 가능하다.

→ **논문 주장**: last-centimeter docking 연구는 trajectory-imitation 지표로 모델을 고르면 안 된다.

### 1.4 방법론 기여 — ICP-free 파이프라인

- 학습 전 구간에서 ICP/dock_pose를 GT로 쓰지 않는다.
- camera→body 외부 파라미터를 **wheel odometry만으로** hand-eye(Kabsch/SVD) 추정
  (`scripts/calibrate_reloc3r_odometry.py`, 출력 `reloc3r/body_frame_calibration_odometry.json`,
  `uses_icp_or_dock_pose: false` 어서션 포함).
- 독립적으로 구한 ICP 기반 추정과 **5.66° 이내 일치** → 두 경로가 서로를 교차검증.
- SLAM / 사전 지도 / 고전 pose controller / 시뮬레이터 없음. 실 시연만으로 학습.

이건 abstract의 "시뮬레이션 학습, SLAM, 사전 구축 지도 또는 고전적 pose controller 없이"를
**그대로 유지해도 되는** 부분이다. 오히려 강화할 수 있다.

### 1.5 선행 연구에서 이미 확정된 것 (인용해서 쓸 것)

`docs/ablation_study_2026-07.md` §4.1이 "통계력 충분"으로 인정한 것만:
- LiDAR 제거 → 인지 정밀도 악화 (near_mm 6.9 vs ~5.0, 수천 프레임) — **LiDAR의 기여는 실재**
- `aux_relative` 타깃 → 정밀도 2배 악화 (9.7 ↔ 5.0mm)

---

## 2. 쓰면 안 되는 주장 (그리고 왜)

| 초안의 문장 | 문제 |
|---|---|
| "실제 로봇 실험에서 성공률 [X%p] 향상, first-attempt [Y%p]" | **실로봇 실험이 0회.** 빈칸이 아니라 미실시. |
| "LiDAR와 goal-relative geometry의 결합이 기여함을 확인" | goal 조건화는 오히려 **해롭다**(§1.2). geometry는 그 해악을 되돌리는 역할. 문장 방향이 반대. |
| "충전이나 물리적 결합이 가능한 위치와 방향으로 정확하게 정렬" | 시연이 **설계상 도킹 입구에서 멈춘다**(과전류 위험). 최종 결합 구간이 학습 데이터에 없음. "도킹 입구 정렬"로 완화 필요. |
| "Reloc3r의 상대 회전 **및 translation direction**" | 유지 가능. geometry 토큰이 `[dx_body, dy_body, sin ψ, cos ψ]`로 direction을 실제로 포함함. 단 **translation은 scale-ambiguous(단위 방향뿐)** 임을 본문에 명시할 것. |

### 2.1 반드시 disclose 해야 할 것

- **Reloc3r 라이선스: CC-BY-NC-SA 4.0 (비상업)**. 논문은 문제없지만 명시할 것.
- **평가에 ICP를 쓴다**: align_deg / xpos_mm의 기준 pose는 ICP dock_pose다. 학습은 ICP-free.
  → "**ICP-free training, ICP-instrumented evaluation**"으로 정직하게 쓰면 오히려 셋업이 깔끔하다.
  ICP 오차는 policy와 demo, 그리고 모든 arm에 공통이므로 *비교*는 공정하고, *절대 각도*만 ICP
  오차를 물려받는다.
- **단일 카메라**: 세 arm 모두 `dino_bottom` 하나. §4 참조 — 이게 잠재적 치명 결함.

---

## 3. 가장 큰 위험 두 가지 (리뷰어가 반드시 찌른다)

### 3.1 시드 1개 — 오늘 밤 해소됨

§1.1의 p=1.4e−9는 **프레임** 짝짓기다. "이 두 체크포인트가 이 프레임들에서 다르다"만 보였고,
"geometry 토큰이 원인이다"는 못 보였다. 학습 시드 분산이 통제되지 않았다.

**노이즈 바닥의 실측 근거**: `docs/ablation_study_2026-07.md`에서
`graft_goalimg_lidar` (align 1.373) vs `graft_lidar_goalimg` (1.479) 는 **0.106°** 차이가 나는데,
두 런은 그 문서 스스로 "무효과 / 기각"으로 결론낸 goal-lidar 조건화 하나만 다르다.
→ **런 수준 노이즈 바닥 ≈ 0.1°**. 우리 효과 0.153°는 그 **1.5배에 불과**하다.

**진행 중**: 3 arm × 3 seed = 9런, batch16 × 100k step (= old baseline과 동일 예산 ≈ 7.4 epoch),
2026-07-26 ~20:37 완료 예정. `test/queue_rgeo_seeds.sh`.

### 3.2 held-out 지표가 실기를 예측하지 못한 전례 — 미해소

`docs/ablation_study_2026-07.md` 결론 7 / §2.12:

> scratch20·glidar_abs·flow_goal_adv는 held-out 순위표 상위권이었지만 실제 시연에서 **전부 실패**했고,
> 반대로 이 체계에 한 번도 들어온 적 없는 구 baseline이 **유일하게 성공**했다.

그리고 §2.12 아키텍처 표에서 성공한 구 baseline만 **카메라 2대(room1+room2)**, 실패한 신규 모델은
전부 단일 카메라다. **우리 세 arm도 전부 단일 카메라다.**
→ 실기 성공을 가른 유일한 관측 변수를 세 arm이 공유해서 결여하고 있다. §2.12의 미확정 3가설
(카메라 수 / 샘플링 스텝 / 배치 크기)은 아직 분리되지 않았다.

**논문에서의 처리**: Limitations에 명시할 것. 숨기면 리뷰어가 찾아낸다. 오히려
"offline 지표와 closed-loop 성능의 괴리"를 본 논문의 **문제의식**으로 끌어올리면 §1.3의 기여와
자연스럽게 이어진다.

---

## 4. 권장 논문 프레이밍

### 제목 방향
> "무엇이 실제로 last-centimeter docking에 전이되는가: diffusion policy의 goal·geometry
> 조건화에 대한 체계적 분석"

### Story arc
1. **문제**: 정밀 도킹은 goal 근처 도달이 아니라 종단 정렬 문제다. NoMaD류는 종단 정렬을 명시적으로
   다루지 않고, AnyImageNav도 최종 오차 0.21–0.27m 수준.
2. **관찰 1 (방법론)**: 이 분야가 쓰는 궤적 지표(ADE/FDE)는 종단 정렬과 **정반대로** 모델을
   순위매긴다 (§1.3). 우리는 counterfactual 종단 정렬 지표를 도입한다.
3. **방법**: ICP-free, 시뮬레이터/SLAM/지도/pose controller 없는 end-to-end diffusion policy.
   RGB history + LiDAR + wheel odometry + Reloc3r 상대 기하를 **token 수준**에서 융합
   (cross-attention DiT), DPM-Solver++로 (v, ω) 궤적 생성.
   ICP-free hand-eye 캘리브레이션 (§1.4).
4. **관찰 2 (핵심 발견)**: goal appearance 조건화는 단독으로는 **해롭다** — 조향 편향 +8.8%p,
   종단 정지 16.7%. explicit relative geometry를 추가하면 회복(+5.6%p, 7.8%)되고 종단 정렬
   중앙값이 유의하게 개선된다.
5. **한계**: 시드 N개, 단일 카메라, 실로봇 미검증, offline↔closed-loop 괴리 전례.

### Contribution 목록 (이 순서로)
1. 종단 정렬 counterfactual 지표 + 궤적 지표와의 순위 역전 실증
2. goal 조건화의 실패 모드(조향 편향·종단 정지) 규명과 geometry에 의한 완화 — dose–response
3. ICP-free 학습 및 캘리브레이션 파이프라인
4. (시드 결과가 받쳐주면) explicit relative geometry의 종단 정렬 개선

### Abstract 재작성안 (한국어)

> Autonomous Mobile Robot의 정밀 도킹은 목표 부근 도달을 넘어 충전 결합이 가능한 자세로 정렬돼야
> 하는 last-centimeter navigation 문제이다. NoMaD류 diffusion navigation policy는 goal-directed
> navigation에 초점을 두어 종단 정렬을 명시적으로 다루지 않으며, AnyImageNav 역시 최종 위치 오차가
> 0.21–0.27 m 수준이다. **본 연구는 먼저, 이 분야가 관행적으로 사용하는 궤적 지표(ADE/FDE)가
> 종단 정렬 품질과 모델을 정반대로 순위매긴다는 것을 보인다** — 궤적 지표 최상위 모델이 정렬
> 지표에서는 최하위였다(2.10° / 12.2 mm vs 1.28° / 8.2 mm). 이에 counterfactual 종단 정렬 지표를
> 도입하고, 목표 영상과 관측 영상으로부터 Reloc3r의 상대 회전 및 (스케일 미정의) translation
> direction을 추출해 RGB history, LiDAR, wheel-velocity history와 token 수준에서 융합하는
> diffusion policy를 제안한다. 시뮬레이션, SLAM, 사전 지도, 고전적 pose controller를 사용하지 않고
> 실 시연만으로 학습하며, 외부 파라미터 역시 wheel odometry만으로 추정한다(ICP 기반 독립 추정과
> 5.66° 일치). 체계적 ablation 결과, **목표 appearance만을 조건으로 주면 조향 편향이 +8.8 %p로
> 증폭되고 종단 정지 비율이 13.6%→16.7%로 악화되는 반면, explicit relative geometry를 추가하면
> 이 열화가 회복되고(+5.6 %p, 7.8%) 종단 정렬 오차 중앙값이 유의하게 개선된다**
> (−0.153°, −0.86 mm; paired Wilcoxon, n=480). 이는 목표 조건화가 정밀 도킹에서 양날임을 보이며,
> 명시적 상대 기하가 그 부작용을 억제하는 정규화 역할을 함을 시사한다.

### Abstract (English)

> Precision docking for autonomous mobile robots is a last-centimeter navigation problem: the
> robot must not merely reach the vicinity of a goal but align to a pose that permits physical
> mating. Diffusion navigation policies such as NoMaD target goal-directed navigation and do not
> explicitly address terminal alignment, while AnyImageNav still leaves 0.21–0.27 m of final
> position error. **We first show that the trajectory metrics conventionally used in this
> literature (ADE/FDE) rank models in the *opposite* order to terminal-alignment quality** — the
> top model by trajectory error was the worst by alignment (2.10°/12.2 mm vs 1.28°/8.2 mm). We
> therefore introduce a counterfactual terminal-alignment metric, and propose a diffusion policy
> that extracts Reloc3r relative rotation and (scale-ambiguous) translation direction from a goal
> image and the current observation, fusing them at the token level with RGB history, LiDAR, and
> wheel-velocity history. The policy is trained end-to-end from real demonstrations only, without
> simulation, SLAM, prior maps, or a classical pose controller; the camera-to-body extrinsic is
> likewise recovered from wheel odometry alone (agreeing with an independent ICP-based estimate
> to 5.66°). In a systematic ablation, **conditioning on goal appearance alone is harmful —
> steering bias grows by +8.8 %p and terminal stalling rises from 13.6% to 16.7% — whereas adding
> explicit relative geometry recovers both (+5.6 %p, 7.8%) and significantly improves median
> terminal alignment** (−0.153°, −0.86 mm; paired Wilcoxon, n=480). Goal conditioning is thus
> double-edged for precision docking, and explicit relative geometry acts as a regularizer that
> suppresses its failure modes.

---

## 5. 내일 아침 분기 규칙 (시드 결과에 따라)

오늘 밤 9런이 끝나면 arm별 3시드의 align_deg 평균과 **arm 내 스프레드**를 본다.

| 결과 | 판정 | 논문 조치 |
|---|---|---|
| arm 내 스프레드 < arm 간 격차, 순서(Geo<NoGoal) 3시드 모두 재현 | **주장 성립** | §4 abstract 그대로. contribution 4 살림. |
| 순서는 유지되나 스프레드가 격차와 비슷 | **약한 지지** | "trend, not significant"로 낮추고 contribution 4를 제거. 1~3만 주장. |
| 순서가 시드에 따라 뒤집힘 | **기각** | geometry 정렬 주장 삭제. §1.2(조향 편향·종단 정지 dose–response)만 유지 — 이건 효과 크기가 SE 대비 훨씬 커서 살아남을 가능성 높음. 논문은 §1.3(지표 축) 중심으로. |

집계 명령 (내일 아침):
```bash
python3 - <<'EOF'
import json, glob, numpy as np
for arm in ['r_nogoal','r_goal','r_geo']:
    v=[json.load(open(f))['align']['policy_median_deg']
       for f in sorted(glob.glob(f'test/out/rgeo/{arm}_b16s*_heldout_align.json'))]
    x=[json.load(open(f))['align']['x_policy_median_mm']
       for f in sorted(glob.glob(f'test/out/rgeo/{arm}_b16s*_heldout_align.json'))]
    if v: print(f'{arm}: align {np.mean(v):.3f} +-{np.std(v,ddof=1) if len(v)>1 else 0:.3f} {v} | xpos {np.mean(x):.2f} {x}')
EOF
```

---

## 6. 재현 정보 (Methods/Appendix용)

| 항목 | 값 |
|---|---|
| 데이터 | `dataset/after_0328_train.h5` (145 ep, 225,465 프레임) / `after_0328_test.h5` (10 ep, held-out) |
| 학습 윈도 | 216,910 |
| backbone | DDPM (`ContinuousDiffusionSDE`), solver `ode_dpmsolver++_2M` |
| 조건화 | `TokenSequenceFusionCondition` (unpooled token sequence) + `DiTCrossAttn1d` (cross-attention; AdaLN은 diffusion timestep만) |
| DiT | d_model 384, heads 6, depth 12 |
| horizon / obs_horizon | 60 / 60 (dt=0.0333, ≈2.0 s) |
| 시드 스윕 예산 | batch 16 × 100,000 step ≈ 7.4 epoch (old baseline과 동일 레시피, `ai-control/ai_models/postech_config.yaml` @ git `dca73b2`) |
| Reloc3r | ViT-L encoder, 224 입력, 14×14=196 patch, **frozen, 오프라인 캐시** |
| geometry token | `[dx_body, dy_body, sin ψ, cos ψ]`, 1 token, dropout 0.2 |
| 평가 | 종단 정렬 `test/eval_align_rgeo.py` (n=480 paired) / 궤적 `test/eval_run_rgeo.py` (10 ep × 500 step) |

**주의**: 이번 시드 스윕(batch16/100k)은 기존 seed-0 런(batch256/16,940)과 **예산이 다르다.**
스윕 내부끼리만 비교하고 기존 런과 섞지 말 것.

---

## 7. 오늘 밤 할 수 있는 일 (학습 대기 중)

논문에서 시드 결과와 **무관한** 부분은 지금 다 쓸 수 있다:

- [ ] Introduction / Related Work (NoMaD, AnyImageNav, Reloc3r, diffusion policy)
- [ ] Method §: 아키텍처, token-level fusion, geometry token 정의, ICP-free 캘리브레이션 (§1.4)
- [ ] Metric §: counterfactual 정렬 지표 유도 (`θ_dock(t+H) = θ_dock(t) − ∫ω dt`, 실증 상관 0.991)
- [ ] Results §1: **지표 순위 역전** 표 (§1.3) — 시드와 무관, 지금 확정
- [ ] Limitations: §2.1 disclose 항목 + §3.2 offline↔closed-loop 괴리
- [ ] Results §2 (정렬 수치)와 Abstract 수치만 내일 아침 시드 결과로 채우기

---

## 관련 문서

- `docs/ablation_study_2026-07.md` — 선행 ablation, §2.12(구 baseline 실기 성공), §4.1(통계력), 결론 5·7
- `docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md` — R-NoGoal/R-Goal/R-Geo 스펙
- `test/eval_align_rgeo.py` — 종단 정렬 지표 (신규, 2026-07-26)
- `test/queue_rgeo_seeds.sh` — 3×3 시드 스윕
