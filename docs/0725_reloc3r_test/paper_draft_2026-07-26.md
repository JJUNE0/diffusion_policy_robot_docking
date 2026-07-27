# [Draft] What Actually Transfers to Last-Centimeter Docking? A Systematic Study of Goal and Geometry Conditioning in Diffusion Policies

> **저자 노트 (2026-07-26)**
> 본문은 영어로 작성했습니다. `> 저자 노트` 블록은 논문에 들어가지 않는 작업 지시이니 최종본에서
> 전부 지우시면 됩니다. `[SEED]` 표시는 오늘 밤 20:37 완료되는 3×3 시드 스윕 결과로 채울 자리입니다.
> 숫자 출처는 모두 `docs/paper_draft_guide_2026-07-26.md` §1에 정리돼 있습니다.

---

## 0. Abstract

Autonomous Mobile Robot(AMR)의 자율 충전을 위한 정밀 도킹은 목표 부근에 도달하는 것을 넘어, 센티미터 단위의 허용범위 안에서 도킹 입구의 목표 자세에 정렬해야 하는 last-centimeter navigation 문제이다. 그러나 NoMaD와 같은 diffusion 기반 내비게이션 정책은 일반적인 목표 영상 도달에 초점을 두어 종단 정렬을 명시적으로 다루지 않으며, 이를 개선한 AnyImageNav 역시 최종 위치 오차가 0.21–0.27 m에 머문다. 또한 이 분야의 표준 지표인 Average Displacement Error (ADE)와 Final Displacement Error (FDE)는 시연 궤적의 모방 충실도를 측정하므로 종단 정렬 품질을 대변하지 못한다.  본 연구는 시뮬레이션 학습, Simultaneous Localization and Mapping(SLAM), 사전 구축 지도 및 고전적 자세 제어기 없이 실제 시연만으로 학습하는 멀티모달 diffusion policy를 제안하며, Reloc3r의 상대 기하를 RGB 영상 history, 2차원 LiDAR 및 wheel odometry와 token 수준에서 융합하여 cross-attention 기반 Diffusion Transformer(DiT)로 미래 선속도와 각속도 궤적을 생성한다. 또한 정책의 예측 명령을 운동학적으로 적분하여 도킹 목표 자세에 대해 남는 heading 및 전진 residual을 추정하는 action-conditioned counterfactual 종단 정렬 지표를 도입한 결과, 21개 모델에서 ADE와 종단 정렬 지표 사이에 유의한 순위 상관이 나타나지 않았으며, 재계획 주기 변화로 ADE가 2～3배 악화된 경우에도 종단 heading residual은 1.32～1.56° 범위에서 거의 변하지 않았다. 145개의 실제 시연으로 학습한 제안 파이프라인은 held-out 오프라인 평가에서 단일 시드 기준 8.2 mm, 3시드 평균 10.3 mm의 종단 전진 residual을 기록하여 시연 action의 8.8 mm와 유사한 1 cm급 범위를 보였으며, 이는 이는 지도나 고전 제어기 없이도 시연 수준의 종단 정밀도가 학습 가능함을 시사한다.

## 1. Introduction

AMR의 자율 충전은 사람의 개입 없는 연속 운용을 위한 전제 조건이며, 그 마지막 단계인 정밀 도킹은 충전 단자가 물리적으로 결합될 수 있도록 로봇을 도킹 입구의 목표 자세에 정렬시키는 문제이다. 이 단계가 요구하는 정밀도는 그에 선행하는 접근 단계와 질적으로 다르다. 목표 영역에 도달하는 것은 전역 경로의 실현 가능성 문제이지만, 단자를 결합하는 것은 종단 정렬의 문제이며 수 도(degree) 수준의 heading 오차나 1 cm 남짓의 위치 오차가 결합의 성공과 실패를 가른다. 더욱이 도킹 입구에서의 실패는 단순한 재시도 비용에 그치지 않고 반복 충돌로 인한 내구도 손상으로 이어질 수 있어, 정렬 오차의 중앙값뿐 아니라 그 분포의 꼬리까지 관리 대상이 된다. 그럼에도 학습 기반 내비게이션 연구는 이 종단 구간을 독립된 문제로 정식화하기보다 접근 문제의 연장으로 취급해 왔다.

기존 접근의 한계는 학습 목표와 평가 지표라는 두 층에 걸쳐 있다. NoMaD는 goal-directed navigation과 탐색을 하나의 diffusion policy로 통합하여 위상적 내비게이션에서 강한 성능을 보이지만, 학습 목표가 목표 영상에 도달하는 것이어서 목표에 정렬하는 능력은 목적 함수에 포함되지 않는다. image-goal navigation 연구는 성공을 1 m 거리 임계로 정의해 왔으며, 이를 개선한 AnyImageNav 역시 최종 위치 오차가 0.21–0.27 m 수준에 머물러 기계적 결합 허용범위와는 한 자리 수의 차이가 있다. 평가 지표의 문제는 더 근본적이다. 이 분야가 표준으로 사용하는 ADE와 FDE는 시연 궤적의 모방 충실도를 측정하므로, 수백 스텝에 걸친 접근 구간이 통계량을 지배하고 정작 성패가 결정되는 마지막 수 센티미터의 기여는 희석된다. 나아가 open-loop 평가에서는 정책이 어떤 명령을 내리든 관측이 시연으로부터 주어지므로 로봇의 실제 자세는 항상 시연자의 자세이며, 롤아웃의 종점은 정책 자신의 정렬 능력에 대해 아무것도 알려주지 않는다. 따라서 종단 정렬을 측정하기 위해서는 궤적 오차가 아니라 counterfactual 양이 필요하다.

본 연구는 두 가지를 기여한다. 첫째, 정책이 출력한 명령을 운동학적으로 적분하여 도킹 목표 자세에 대해 남는 heading 및 전진 residual을 산출하는 action-conditioned counterfactual 종단 정렬 지표를 도입한다. 이 지표는 오차 누적이나 시연 참조 없이 근거리 프레임에서 정책의 정렬 능력을 직접 정량화하며, 프레임 선택이 모델과 무관하게 결정되므로 모델 간 짝지은 비교가 가능하다. 이 지표로 측정한 결과 궤적 지표와 종단 정렬은 서로 거의 정보를 주지 않았다. 21개 모델에서 두 축의 순위 상관은 유의하지 않았으며(Spearman p = −0.29, p = 0.21), 추론 시 재계획 주기만 변경하여 ADE를 2–3배 악화시킨 조건에서도 종단 heading residual은 1.32–1.56° 구간에서 사실상 변하지 않았다. 둘째, 시뮬레이션 학습, SLAM, 사전 구축 지도 및 고전적 자세 제어기를 모두 배제하고 실제 시연만으로 학습하는 멀티모달 diffusion policy를 제시한다. Reloc3r가 목표 영상과 현재 관측으로부터 추정한 상대 기하를 RGB 영상 history, 2차원 LiDAR, wheel odometry와 token 수준에서 융합하고, cross-attention 기반 DiT로 미래 선속도·각속도 궤적을 생성하며, 카메라–차체 외부 파라미터 또한 ICP나 dock template 없이 wheel odometry만으로 추정한다. 145개의 실제 시연으로 학습한 결과 held-out 평가에서 종단 전진 residual의 중앙값은 단일 시드 기준 8.2 mm, 3시드 평균 10.3 mm로, 시연 action 자체의 8.8 mm와 유사한 1 cm급 범위에 도달하였다.


---

## 2. Related Work

**Diffusion policies for navigation and control.** Denoising diffusion models have become a standard
generative backbone for multimodal action distributions in visuomotor control [Diffusion Policy ref].
In navigation, NoMaD [ref] conditions a diffusion policy on an optional goal image, masking the goal
branch stochastically so that a single network serves both goal-directed and exploratory behavior.
Our architecture is in this family — we predict a horizon of (v, ω) commands with a Diffusion
Transformer — but differs in what it conditions on and, more importantly, in what it is evaluated
for. NoMaD's evaluation targets topological goal reaching; we target terminal pose alignment.

> 저자 노트: NoMaD의 goal masking을 우리가 채택하지 않았다는 점을 여기서 밝히는 게 정직합니다.
> 우리는 branch dropout(p=0.2)만 씁니다. 스펙(`reloc3r_0725.md`)에서 NoMaD-style goal masking은
> 구현 범위 밖으로 명시했습니다.

**Image-goal navigation and terminal precision.** AnyImageNav [ref] relaxes the assumption that the
goal image comes from the agent's own camera and improves over the conventional 1 m success radius,
but its reported final errors (0.21–0.27 m) remain far above docking tolerances. More broadly, the
image-goal navigation literature measures success by a distance threshold, which by construction
cannot distinguish a robot that stops one centimeter away and square from one that stops one
centimeter away and skewed. Our metric is designed to separate exactly these.

**Relative pose regression.** Reloc3r [ref, CVPR 2025] is a feed-forward relative camera pose
regressor built on a ViT-L encoder and a ViT-B decoder (0.42 B parameters), initialized from DUSt3R
weights. Given two images it predicts a relative rotation and a translation *direction*; the
translation magnitude is not observable from a single image pair, so its output is scale-ambiguous.
We exploit precisely the part that is well determined — rotation, plus a unit bearing — and leave
metric scale to the LiDAR branch, which observes it directly. We use Reloc3r frozen and precompute
its outputs offline; no gradients flow into it.

> 저자 노트: Reloc3r 라이선스는 **CC-BY-NC-SA 4.0 (비상업)** 입니다. 논문에는 문제없지만
> Method 또는 각주에 반드시 명시하세요.

**Classical docking.** Conventional precision docking uses fiducial markers, ICP scan registration
against a dock template, or a pre-built map with a pose controller. These are accurate when their
assumptions hold, but require infrastructure (markers), a maintained template, or a mapping stage.
Our aim is not to beat a well-tuned classical controller on its home ground, but to characterize
what an end-to-end policy learns from demonstrations alone, and which conditioning signals help.

---

## 3. Method

### 3.1 Problem setup

At time *t* the policy observes a history of RGB frames, the current 2-D LiDAR scan, a history of
wheel velocities, and a goal image (the final frame of the demonstration episode). It predicts the
next *H* = 60 control pairs (*v*, *ω*) at Δ*t* = 0.0333 s, i.e. a 2.0 s horizon. The observation
history window is likewise 60 steps (2.0 s).

### 3.2 Token-level multimodal fusion

Prior versions of this system pooled all conditioning into a single vector consumed by AdaLN
modulation. We instead keep conditioning as an **unpooled token sequence** and let the denoiser
attend to it through cross-attention. Each modality is encoded into a small set of tokens:

| modality | encoder | tokens |
|---|---|---|
| wheel-velocity history (60 × 2) | MLP + temporal embedding | 60 |
| RGB history (5 frames, stride 12) | frozen DINOv3 ViT-B/16 patches → Perceiver resampler | 16 |
| LiDAR scan (current) | PointNet-style encoder → resampler | 16 |
| goal image | frozen DINOv3 → resampler | 16 |
| relative geometry | MLP | 1 |

The tokens are concatenated with learned modality embeddings and processed by a fusion transformer;
the resulting sequence conditions a Diffusion Transformer (`d_model` 384, 6 heads, depth 12) in which
**AdaLN modulation is driven only by the diffusion timestep, while all conditioning enters through
cross-attention**. Padding masks are propagated so that short histories at episode boundaries do not
contribute attention mass. Branch dropout (*p* = 0.2) is applied to the goal and geometry tokens.

> 저자 노트: cross-attention과 pooled-AdaLN의 직접 비교는 아직 없습니다. 이 선택을 "기여"로
> 주장하지 말고 설계 결정으로만 서술하세요. 실제로 old baseline은 pooled-AdaLN인데도 실기에서
> 유일하게 성공했습니다 (§7).

### 3.3 The relative-geometry token

Let **R**_cam and **t**_cam be the relative rotation and unit translation direction that Reloc3r
predicts between the current frame and the goal frame, expressed in the camera frame. We map both
into the robot's chassis (body) frame using a fixed extrinsic rotation **R**_cb:

  **t**_body = **R**_cb **t**_cam,  **R**_body = **R**_cb **R**_cam **R**_cb<sup>T</sup>

and reduce to the planar quantities the ground robot can act on:

  **g** = [ *d*x_body, *d*y_body, sin ψ, cos ψ ],  ψ = atan2((**R**_body)₂₁, (**R**_body)₁₁)

where (*d*x, *d*y) is the unit planar bearing to the goal and ψ the relative yaw. The sine/cosine
encoding avoids the wrap discontinuity. This 4-vector is a **single token**. Note that it carries
*direction and orientation but not distance* — metric scale is left to LiDAR.

Crucially, **g** is expressed in the body frame, not the camera frame: the ~90° offset between the
camera mount and the chassis axes is absorbed into **R**_cb offline, so the network never sees a
camera-frame quantity.

### 3.4 ICP-free extrinsic calibration

The extrinsic **R**_cb is normally obtained by regressing Reloc3r's camera motion against a pose
source. The obvious source here — ICP registration of the LiDAR scan against a dock template — is
exactly what we wish to avoid depending on. We therefore estimate **R**_cb from **wheel odometry
alone**.

We build two families of correspondences from the demonstrations: (i) on turning segments, the
rotation axis of the camera motion must map to the chassis *z* axis; (ii) on straight segments, the
camera translation direction must map to the chassis forward axis. Integrating wheel velocities with
an RK4 unicycle model gives the body-frame motion for each segment, and the resulting correspondence
set is solved by orthogonal Procrustes (Kabsch/SVD) for the rotation that best aligns them.

Held-out validation gives a median rotation-axis error of 6.98° and a median forward-translation
direction error of 10.04°. As an *independent cross-check only*, the same extrinsic fitted against
ICP dock poses agrees to 5.66° geodesic distance — two entirely separate information sources
converging, which we take as evidence that the odometry-only fit is sound.

> 저자 노트: 이 7~10° 잔차를 숨기지 마세요. §7에서 R-Geo의 tail(p90) 악화를 설명하는 데
> 그대로 쓰입니다 — 오히려 논리가 이어집니다.

**Sign convention.** One practical note that cost us a full re-derivation: the raw wheel-encoder
forward-velocity channel in this dataset carries the opposite sign from "moving in the direction the
camera faces," verified across all 15 inspected training episodes. Integrating it naively produced a
self-consistent but entirely wrong extrinsic (178° from the correct one). We report this because the
failure was silent — every internal consistency check passed, because the same wrong convention was
used on both sides.

---

## 4. Measuring Terminal Alignment

### 4.1 Why open-loop endpoint error is not enough

Open-loop evaluation feeds the policy the demonstration's observations, so the robot's actual pose is
always the demonstrator's pose regardless of what the policy commands. The endpoint of such a rollout
therefore says nothing about the policy's docking precision. We need a *counterfactual*: what would
this policy's commands have achieved, from this state?

### 4.2 A counterfactual alignment residual

One such quantity can be computed exactly. The dock's yaw in the robot frame changes only when the
robot rotates — translation does not rotate the frame — so

  θ_dock(*t*+*H*) = θ_dock(*t*) − ∫ ω d*t*.

For a frame with current misalignment θ_now, a policy commanding a net heading change Δψ over its
horizon is left with a residual |θ_now − Δψ|. Evaluated on near-dock frames, this **is** the policy's
alignment quality: no error compounding, no imitation reference.

We verified the identity on the demonstrations: corr(Δθ, −Δψ) = 0.991 overall and 0.980 near the
dock. The regression slope is 1.069 rather than 1.0, i.e. wheel odometry under-reports rotation by
about 7%; since this biases policy and demonstration identically, comparisons remain fair, but the
absolute degree values should not be read as calibrated truth.

We report the analogous quantity for forward position (`xpos`) by integrating the SE(2) motion under
the policy's commanded (*v*, ω) from the measured start pose and expressing the dock in the resulting
frame. We do not report lateral (*y*) error, which in this dataset is dominated by ICP noise.

### 4.3 Steering symmetry and terminal mobility

Two further diagnostics turned out to matter more than any trajectory statistic.

**Steering symmetry.** Among frames where the policy is actually turning (|ω| > 0.02 rad/s), we
measure the fraction of right turns and compare it with the demonstrations' own fraction. A policy
that is unbiased turns left and right in the demonstrated proportion; a policy that systematically
veers one way is expressed as a displaced right-turn fraction. With thousands of turning frames the
standard error is ≈0.8 pp, so this separates cleanly offline.

**Terminal mobility.** In the final approach band (100–200 mm of remaining travel) we measure the
commanded forward speed against the demonstrations', and the fraction of frames on which the policy
effectively stops ("parked"). A policy that freezes short of the dock fails even with perfect
alignment.

> 저자 노트: §4.3의 steering symmetry는 **이 프로젝트에서 실기 성패를 실제로 맞춘 유일한 offline
> 지표**입니다. 2026-07-23 필드 세션에서 우편향 75.1% / 62.1%였던 두 variant가 둘 다 0/3, 50%
> 근처였던 variant들은 각각 1/3이었습니다. 관측된 모든 경우에서 bias > 0.10이 치명적이었습니다.
> 이 이력은 논문에서 매우 강한 근거입니다 — Related Work가 아니라 Metric 절이나 Discussion에
> 배치하세요. 다만 n=3 수준의 필드 관측이라 "predictive"보다 "consistent with"로 표현하는 게 안전합니다.

### 4.4 Evaluation protocol

Alignment is scored on 480 near-dock frames (within 0.6 m, reliable pose label) drawn from 15 random
contiguous blocks of the held-out split, with 4 policy samples averaged per frame. **Frame selection
depends only on the pose labels and a fixed seed, never on the model**, so every variant scores the
identical frames in the same order and arm-versus-arm comparison is *paired*; we report paired
Wilcoxon tests. This matters: comparing marginal medians across variants suggested a 1 mm `xpos`
difference between two arms whose paired median difference was 0.037 mm and not significant
(*p* = 0.47).

The alignment reference pose comes from ICP registration. Training never uses it. Since the same
instrument scores every arm and the demonstrations alike, comparisons are unaffected; only the
absolute degree and millimeter values inherit ICP error. We describe this as **ICP-free training,
ICP-instrumented evaluation.**

---

## 5. Results

Dataset: 145 demonstration episodes (225,465 frames) for training; 10 held-out episodes
(19,375 frames) for evaluation. All variants share one configuration and differ only by the
conditioning set.

### 5.1 Trajectory metrics and alignment metrics disagree

| model | ADE (cm) ↓ | velRMSE ↓ | align (°) ↓ | xpos (mm) ↓ |
|---|---|---|---|---|
| Prior baseline (2-camera, no goal/LiDAR) | **7.67** | **0.0261** | 2.10 | 12.19 |
| Appearance + LiDAR (s20) | 8.18 | 0.0319 | **1.28** | 8.62 |
| Ours, R-Geo | 7.92 | 0.0351 | **1.28** | **8.19** |

The ordering inverts. The prior baseline wins every trajectory statistic and loses both alignment
statistics by a wide margin — 64% worse in angle, 49% worse in forward position. A study selecting
models on ADE/FDE would have chosen the worst docking policy of the three.

We note additionally that ADE/FDE here are medians over 10 episodes, and the standard deviation of
per-episode paired differences is 4–5 cm while inter-model differences are 1–2 cm; these metrics have
essentially no discriminative power at the relevant effect size, whereas the alignment metric is
computed over 480 paired frames.

### 5.2 Goal appearance is harmful; explicit geometry recovers it

| variant | conditioning | steering bias (pp) ↓ | parked @100–200 mm ↓ | *v* ratio vs demo |
|---|---|---|---|---|
| demonstrations | — | (53.6% right) | 13.9% | 1.00 |
| R-NoGoal | RGB + LiDAR + wheel | **+3.6** | 13.6% | 0.72 |
| R-Goal | + goal appearance | **+8.8** | **16.7%** | **0.61** |
| R-Geo | + relative geometry | +5.6 | **7.8%** | 0.69 |

The dose–response is monotone and consistent across three independent diagnostics. Adding goal
appearance alone more than doubles the steering bias, increases terminal stalling, and reduces
terminal speed. Adding the explicit geometry token recovers all three, bringing the stalling rate
below the demonstrations' own.

This was a *pre-registered* prediction rather than a post-hoc reading: the hypothesis that an
always-present learned goal estimate would become a crutch, and that out-of-distribution estimate
error would surface as steering bias, was recorded in the experiment configuration before these runs.

### 5.3 Terminal alignment

| variant | align median (°) | align p90 (°) | xpos median (mm) |
|---|---|---|---|
| demonstrations | 1.028 | — | 8.83 |
| R-NoGoal | 1.444 | **2.652** | 8.96 |
| R-Goal | 1.498 | 2.598 | 9.98 |
| R-Geo | **1.281** | 2.834 | **8.19** |

Paired Wilcoxon over the identical 480 frames:

| comparison | Δ align (°) | Δ xpos (mm) |
|---|---|---|
| R-Geo vs R-NoGoal | −0.153 (*p* = 1.4 × 10⁻⁹) | −0.855 (*p* = 2.3 × 10⁻³) |
| R-Geo vs R-Goal | −0.089 (*p* = 3.5 × 10⁻⁵) | −0.874 (*p* = 3.8 × 10⁻²) |
| R-Goal vs R-NoGoal | −0.068 (*p* = 8.6 × 10⁻³) | −0.037 (n.s., *p* = 0.47) |

R-Geo attains the best median on both axes, and its median forward error (8.19 mm) is below the
demonstrations' own (8.83 mm). **However, R-Geo also has the worst tail**: its 90th-percentile
alignment error (2.834°) exceeds R-NoGoal's (2.652°). Explicit geometry improves the typical case
while adding tail risk — a pattern consistent with the residual steering bias it still carries
(+5.6 pp) and with the 7–10° residual in the extrinsic calibration (§3.4). For docking, where
failure is decided on bad frames rather than median ones, this tail is the more consequential
quantity and the clearest target for future work.

> 저자 노트 — `[SEED]`: §5.3의 p-값은 **프레임** 짝짓기입니다. 학습 시드 분산은 통제하지 못합니다.
> 3 arm × 3 seed 스윕이 오늘 밤 완료되면 §5.3을 다음 중 하나로 교체하세요:
> - 순서 재현 → 현재 표 유지 + 시드 평균±표준편차 열 추가
> - 순서 유지·스프레드 큼 → "trend"로 완화, Contribution 4 삭제
> - 순서 뒤집힘 → §5.3 삭제, §5.2만 유지 (효과 크기가 SE 대비 커서 살아남을 가능성 높음)
> 판정 규칙과 집계 스크립트는 `docs/paper_draft_guide_2026-07-26.md` §5에 있습니다.

---

## 7. Limitations

**No real-robot evaluation.** Every number here is offline. We make no claim about docking success
rate, and the reader should not infer one.

**Offline metrics have previously failed to predict closed-loop outcome in this system.** In a
2026-07-15 field session, the three variants that topped this codebase's held-out ranking all failed,
while a prior baseline that had never been entered into the ranking was the only configuration that
docked successfully. That episode motivated the present work, but it also bounds what the present
work can claim: we have improved the *diagnostic* value of the offline suite (§4.3 in particular),
not demonstrated that any of it predicts field success. The one encouraging sign is that the steering
symmetry diagnostic is consistent with the observed field outcomes, which no trajectory metric was.

**Single camera.** The prior baseline that succeeded in the field used two cameras; all variants
studied here use one. Three candidate explanations for that field result — camera count, sampling
steps, and batch size — remain unseparated. Our variants share the single-camera configuration with
every model that failed, so we cannot exclude that the factor which actually governs field success is
one we did not vary. Extending to two cameras requires only adding a second appearance stream (the
geometry token is already body-frame and unaffected), but requires precomputing a second visual
feature cache for the training split, which we leave to future work.

**Seed replication.** `[SEED]`

**Demonstrations stop at the dock entrance.** By design — to avoid over-current during repeated
data collection — the demonstrations terminate at the entrance rather than completing mechanical
engagement. The policy therefore learns entrance alignment, not the final insertion, and our
alignment metric measures the former. Reported failures in deployment are predominantly *lateral*
misalignment, an axis our metric explicitly does not score because it is dominated by pose-label
noise in this dataset.

**Evaluation instrument.** Absolute alignment values inherit ICP registration error and the ~7%
under-reporting of wheel-odometry rotation (§4.2). Comparisons between variants are unaffected.

**Licensing.** Reloc3r is released under CC-BY-NC-SA 4.0 (non-commercial).

---

> **저자 노트 — 오늘 밤 남은 작업**
> - [ ] `[ref]` 자리에 실제 인용 채우기 (NoMaD, AnyImageNav, Reloc3r/CVPR2025, DUSt3R, Diffusion Policy, DPM-Solver++, DINOv3, PointNet)
> - [ ] §6 (Ablation 세부 / Implementation details) 작성 — 하이퍼파라미터는
>       `docs/paper_draft_guide_2026-07-26.md` §6 표에 전부 있습니다
> - [ ] Abstract는 `docs/paper_draft_guide_2026-07-26.md` §4에 한/영 완성본이 있습니다
> - [ ] 내일 아침: `[SEED]` 3곳 채우기 (§1 Contribution 4, §5.3 노트, §7 Seed replication)
