# RA-L 논문 개정 초안 — 일반화 중심

> 개정일: 2026-08-06
> 대표 모델: `r_relfeat_only`
> 정량 결과와 현재까지 확인된 정성적 실로봇 결과를 구분한다. 반복 횟수가 기록되지 않은 OOD 시험에는 성공률을 임의로 부여하지 않는다.

## 작업 제목

**약 2시간의 실제 시연으로 미관측 도크와 시각 목표에 일반화하는 Last-Centimeter Robot Docking**

**Generalizable Last-Centimeter Robot Docking for Unseen Docks and Visual Goals from Two Hours of Real-World Demonstrations**

## 초록

자율이동로봇의 정밀 도킹은 목표 부근에 도달하는 것을 넘어, 충전 단자가 결합될 수 있도록 센티미터 단위의 위치 오차와 수 도 이내의 자세 오차로 로봇을 정렬해야 하는 last-centimeter navigation 문제이다. 기존의 학습 기반 접근은 충분한 양의 다환경 데이터나 명시적 위치 추정에 의존하거나, 학습에 사용된 도킹 스테이션의 외형과 배경을 암기할 위험이 있다. 본 연구의 핵심 기여는 특정 visual-localization backbone 자체가 아니라, 범용 관계 표현을 소량의 실제 데이터로 정밀 폐루프 제어에 전이하는 **Relational Token Docking Policy (RTDP)**이다. 내부 실험명 **r_relfeat_only**인 RTDP는 단일 RGB 카메라, wheel-velocity history 및 한 장의 goal image로부터 미래 속도 궤적을 생성한다.

도킹 상태가 단일 영상의 절대 appearance보다 current–goal 사이의 상대 관점으로 정의된다는 점에 착안하여, 두 영상을 함께 처리하고 양방향 관계 token을 제공하는 ReLoc3R를 frozen feature backbone으로 선택하였다. 그러나 ReLoc3R의 localization용 pose head와 final pose 출력은 사용하지 않는다. pose head는 dense relation을 하나의 전역 rotation과 translation으로 축약하여 localization에는 적합하지만, 정밀 제어에 유용한 국소 correspondence, 가려짐 및 ambiguity를 제거할 수 있기 때문이다. RTDP는 pose head 이전의 dense token을 우리가 설계한 Perceiver resampler, temporal–multimodal self-attention fusion 및 cross-attention Diffusion Transformer에 연결한다. 따라서 ReLoc3R는 교체 가능한 관계 특징 공급자이고, 어떤 정보를 보존하고 압축하여 wheel history와 결합하고 행동 궤적으로 변환할지를 학습하는 RTDP가 제안 방법이다.

정책 학습에는 하나의 수집 환경에서 얻은 145개의 실제 성공 시연만 사용하였다. 데이터는 30 Hz 기준 225,465 frame, 약 2시간에 해당하며, 정책은 20 epoch, 16,940 gradient update로 학습되었다. 실행 시 LiDAR, depth, SLAM, 사전 지도, 명시적 goal-pose label 및 별도의 고전 제어기를 사용하지 않는다. 학습 분포와 동일한 도크에 대한 기존 실로봇 시험에서는 20/20 도킹 성공을 기록하였다. 더 나아가 학습에 포함되지 않은 도킹 스테이션, 처음 보는 배경과 새로운 종류의 도킹 스테이션의 조합, 그리고 데이터셋에 전혀 존재하지 않는 일반 물체를 포함한 goal image에 대해서도 폐루프 도킹 성공이 관찰되었다. 현장 시험에서 최종 위치 오차는 센티미터급, heading 오차는 5° 이내로 유지되었다. 새 OOD 조건의 반복 횟수와 오차 분포는 아직 정량화 중이지만, 이 결과는 RTDP가 특정 도크의 외형을 단순히 암기하기보다 goal image와 현재 관측 사이의 상대적 시각 관계를 제어 신호로 변환하고 있음을 보여준다.

**RA-L Index Terms:** Vision-Based Navigation, Robot Docking, Imitation Learning, Diffusion Policy, Visual Generalization, Sensor-Based Control

## 1. Introduction

자율이동로봇(Autonomous Mobile Robot, AMR)이 사람의 개입 없이 장시간 운용되기 위해서는 충전 스테이션으로 이동하는 것뿐 아니라 충전 단자를 실제로 결합하는 마지막 정렬 단계까지 자율적으로 수행해야 한다. 이 단계는 목표 영역에 도달하는 일반적인 navigation과 요구 정밀도가 다르다. 수 센티미터의 위치 오차나 수 도의 heading 오차만으로도 단자 결합이 실패할 수 있으며, 접근 중 발생한 오차를 폐루프에서 복구할 수 있어야 한다.

전통적인 정밀 도킹 시스템은 LiDAR 또는 depth sensor, SLAM, 사전 지도, fiducial marker, 명시적 relative-pose estimator 및 도크별 controller를 결합할 수 있다. 이러한 modular pipeline은 구조화된 환경에서 높은 정확도를 제공하지만, 새로운 도크나 배경으로 이동할 때 센서 교정, 지도 갱신 또는 도크별 제어 규칙의 재설계가 필요할 수 있다. 반대로 학습 기반 정책은 관측에서 행동까지의 변환을 데이터로 학습할 수 있지만, 소규모 데이터에서는 도크의 색상과 모양 또는 배경을 목표 그 자체로 암기하는 shortcut을 사용할 가능성이 있다.

본 연구가 다루는 핵심 질문은 다음과 같다.

> 하나의 환경에서 수집한 약 2시간의 실제 시연만으로 학습한 정책이, 학습하지 않은 도킹 스테이션과 배경에서도 센티미터급 정밀 도킹을 수행할 수 있는가?

이 질문에 답하기 위해 먼저 결정해야 할 것은 backbone의 이름이 아니라 도킹에 적합한 표현의 형태이다. 단일 영상 appearance embedding은 무엇이 보이는지를 잘 표현하지만, 도킹 제어에 필요한 것은 현재 관측이 목표 관측에 대해 얼마나 이동·회전되어 있는가이다. 따라서 RTDP는 current와 goal을 독립적으로 부호화한 뒤 단순 비교하는 구조보다, 두 영상을 입력 단계부터 함께 처리하는 pairwise relational representation을 선택한다.

본 구현에서는 이러한 설계 조건을 만족하는 frozen backbone으로 ReLoc3R [4]를 사용한다. 선택 이유는 ReLoc3R의 이름이나 최종 pose 정확도 자체가 아니라, current가 goal을 참조한 token과 goal이 current를 참조한 token을 공간 해상도를 유지한 채 동시에 제공하기 때문이다. ReLoc3R는 RTDP 안의 사전학습된 feature extractor일 뿐이며, 본 연구는 ReLoc3R를 개선하거나 그 localization 성능을 기여로 주장하지 않는다.

더 중요한 설계는 ReLoc3R를 **어디까지 사용하는가**이다. RTDP는 원래 pose head 직전에서 backbone을 절단하고 final rotation·translation 출력은 사용하지 않는다. pose head는 dense decoder grid를 하나의 전역 pose로 축약하도록 학습되어 localization에는 적합하지만, 정밀 행동 생성에는 다음과 같은 병목이 될 수 있다. 첫째, global pooling은 도크 모서리, 접촉부, 가려짐과 같은 국소 관계 정보를 제거한다. 둘째, 하나의 pose point estimate는 correspondence의 모호성과 신뢰도를 숨긴다. 셋째, camera-pose regression과 wheel-velocity trajectory generation은 서로 다른 목적 함수이다. 넷째, pose 좌표계·scale·camera-to-body calibration 오차가 controller 입력으로 직접 전파될 수 있다. RTDP는 이 병목 앞의 양방향 token을 보존하고, task-learned resampler와 fusion network가 도킹 행동에 필요한 정보를 직접 선택하도록 한다.

그 이후가 본 연구가 개발한 제어 정책이다. RTDP는 양방향 dense token을 고정 길이 latent로 학습 압축하고, wheel 및 시간 history와 token 수준에서 융합하며, Diffusion Transformer로 2초 길이의 속도 궤적을 생성하고 최신 관측으로 반복 재계획한다. 별도의 DINO appearance branch, LiDAR, depth 또는 final-pose vector는 사용하지 않는다.

실로봇 시험에서 나타난 가장 중요한 결과는 학습 도크에 대한 성공 자체가 아니라 학습 분포 밖으로의 전이다. RTDP는 수집에 사용한 도크뿐 아니라 (i) 학습하지 않은 도킹 스테이션, (ii) 처음 보는 배경에 놓인 새로운 종류의 도킹 스테이션, (iii) 데이터셋에 존재하지 않는 일반 물체를 포함한 goal image에 대해서도 도킹 행동을 완수하였다. 이는 RTDP가 이 물체가 학습 도크인가를 분류하는 대신 현재 영상이 주어진 goal 영상과 같은 상대 관측이 되려면 어떻게 움직여야 하는가를 학습했다는 해석과 일치한다.

본 연구의 기여는 다음과 같다.

1. 단일 RGB 카메라와 wheel history만을 사용하는 map- and range-sensor-free last-centimeter docking을 goal-conditioned offline imitation learning 문제로 정식화한다.
2. localization용 final pose를 제어 입력으로 사용하는 대신, pre-head 양방향 dense token을 보존하여 학습 압축하고 wheel·시간 정보와 융합한 뒤 행동 sequence로 변환하는 **RTDP 제어 아키텍처**를 제안한다.
3. 범용 pairwise backbone을 frozen feature supplier로 제한하고, pose-head의 정보 병목과 localization–control objective mismatch를 피하는 task-specific pre-head interface를 설계한다.
4. 145개, 225,465 frame, 약 2.09시간의 task-specific 실제 시연만으로 RTDP를 학습하여 동일 도크에서 20/20의 실로봇 성공과 센티미터급·5° 이내 종단 정밀도를 보인다.
5. 미관측 도크, 미관측 배경과 도크의 동시 변화, 그리고 비도크 일반 물체를 포함한 goal image까지 성공적으로 추종하는 정성적 cross-instance·cross-environment·cross-category 전이를 보인다.

## 2. Related Work

### 2.1 Goal-Conditioned Visual Navigation

Image-goal navigation은 목표 위치에서 촬영한 영상을 입력으로 제공함으로써 정책을 고정된 함수 \(\pi(A\mid O)\)에서 재지정 가능한 함수 \(\pi(A\mid O,G)\)로 확장한다. NoMaD [1]는 goal-conditioned navigation과 goal-agnostic exploration을 diffusion policy 하나로 통합하여 미지 환경에서의 장거리 이동 가능성을 보였다. 그러나 일반적인 ImageNav는 통상 미터 단위의 성공 반경을 사용하며, 충전 단자 결합에 필요한 last-centimeter alignment를 직접 평가하지 않는다.

AnyImageNav [2]는 goal image를 geometric query로 취급하고 dense correspondence를 통해 6-DoF goal pose를 복원한다. Gibson과 HM3D에서 각각 0.27 m/3.41°와 0.21 m/1.23°의 pose-recovery error를 보고하여 last-meter ImageNav의 정밀도를 크게 향상시켰다. 본 연구는 전역 탐색이나 명시적인 6-DoF pose recovery가 아니라, 이미 도크의 접근 구간에 들어온 로봇이 goal-relative visual relation을 직접 velocity sequence로 변환하여 실제 단자 결합 수준의 정렬을 수행하는 문제를 다룬다.

### 2.2 Diffusion Policies for Closed-Loop Control

Diffusion Policy [3]는 행동 정책을 conditional denoising process로 표현하여 multimodal action distribution과 시간적으로 일관된 action sequence를 모델링한다. Receding-horizon execution과 결합하면 정책이 최신 관측으로 반복 재계획할 수 있다. 본 연구에서 diffusion은 여러 가능한 접근 및 복구 궤적을 평균 행동으로 붕괴시키지 않기 위한 action generator이며, 일반화의 직접적인 원천은 아니다. 새로운 도크에 대한 전이를 결정하는 핵심은 diffusion model에 어떤 goal-relative representation을 조건으로 제공하는가이다.

### 2.3 Pairwise Visual Representation as a Policy Component

ReLoc3R [4]는 약 8백만 개의 posed image pair로 사전학습된 relative camera-pose regression model이다. 본 연구는 ReLoc3R의 새 변형을 제안하지 않으며, 그 final-pose 성능을 연구 기여로 사용하지도 않는다. RTDP에서 ReLoc3R는 current–goal pair를 함께 처리하고 공간적으로 조밀한 양방향 token을 제공하는 frozen backbone의 한 구현이다. 같은 인터페이스를 제공하는 다른 pairwise backbone으로 교체할 수 있다는 점에서, 제안 방법과 backbone은 구분된다.

기존 localization pipeline은 decoder token을 pose head에 통과시켜 하나의 rotation과 translation으로 완결한다. 반면 RTDP는 localization 결과를 소비하는 controller가 아니라, pre-head representation을 task-specific하게 다시 읽는 policy이다. 학습 가능한 Perceiver query가 국소 관계를 선택하고, fusion Transformer가 시간·wheel·양방향 visual token의 상호작용을 형성하며, action DiT가 이를 폐루프 velocity trajectory로 변환한다. 즉, 기여는 pretrained token의 존재가 아니라 **pose로 축약되기 전의 표현을 정밀 제어용 정보 인터페이스로 재설계한 것**에 있다.

따라서 본 연구의 약 2시간 데이터는 backbone을 처음부터 학습한 총 데이터량이 아니라 **RTDP의 도킹 행동을 학습하는 데 사용한 task-specific robot demonstration의 양**을 의미한다. 사전학습 관계 표현을 고정한 채 소량의 실제 로봇 데이터로 압축·융합·행동 생성을 학습한다는 점이 데이터 효율의 핵심이다.

## 3. Method

### 3.1 Problem Formulation

시점 \(t\)에서 정책은 현재까지의 관측 history \(O_t\)와 goal image \(G\)를 받아, 향후 60 step의 선속도와 각속도 sequence를 출력한다.

\[
A_t=[a_t,\ldots,a_{t+59}]\in\mathbb{R}^{60\times2},
\qquad a_t=(v_t,\omega_t).
\]

관측은 30 Hz로 수집된 60-frame wheel-velocity history와 동일한 약 2초 구간에서 stride 12로 선택한 다섯 장의 RGB frame으로 구성된다. 학습 중에는 각 성공 episode의 마지막 도킹 frame을 goal image로 사용한다. 정책은 조건부 행동 분포 \(p_\theta(A_t\mid O_t,G)\)를 학습하고, 실행 중에는 새로운 goal image로 목표를 교체할 수 있다.

### 3.2 Backbone Selection and Pre-Head Control Interface

#### 3.2.1 Why a Pairwise Relational Backbone?

도킹 제어의 목표 상태는 특정 물체 class가 아니라 goal image가 정의하는 상대 관측이다. 따라서 RTDP가 필요한 primitive는 single-image recognition feature보다 current–goal 사이의 correspondence와 viewpoint change이다. ReLoc3R는 두 영상을 jointly encode하고 양방향 cross-attention decoder stream을 제공하므로 이 표현 요건에 부합한다. 또한 대규모 image-pair 사전학습 결과를 frozen 상태로 사용할 수 있어, 145개 도킹 episode로 visual relation 자체를 처음부터 학습해야 하는 부담을 줄인다.

여기서 ReLoc3R를 선택한 논리는 **task–representation alignment**와 **data-efficient transfer**이며, ReLoc3R pose output을 정답으로 신뢰해서가 아니다. RTDP는 backbone의 pose loss로 재학습되지 않으며, downstream 행동 loss도 backbone으로 전달하지 않는다. 따라서 backbone은 관계 token을 공급하고, 실제 도킹 지식은 그 token을 읽어 행동으로 바꾸는 RTDP의 학습 모듈에 저장된다.

#### 3.2.2 Why Stop Before the Original Pose Head?

원래 ReLoc3R pose head는 양방향 decoder grid를 projection, convolution 및 global pooling으로 축약하여 camera rotation과 translation을 회귀한다. 이는 localization에는 타당하지만 RTDP가 원하는 control representation과는 다르다.

- **Spatial bottleneck:** 196개 위치에 분산된 접촉부·모서리·바닥 경계와 같은 국소 단서를 하나의 pose로 압축한다.
- **Uncertainty loss:** 여러 대응 후보나 부분 가림에서 나타나는 ambiguity가 하나의 point estimate 뒤에 숨는다.
- **Objective mismatch:** camera-pose regression에 최적인 요약이 wheel-velocity sequence 생성에도 최적이라는 보장은 없다.
- **Error coupling:** pose 좌표계, metric scale 및 camera-to-body calibration 오차가 명시적 controller 입력에 직접 결합될 수 있다.
- **Control sufficiency:** 정책은 완전한 6-DoF pose를 복원할 필요 없이, 현재 관측을 goal 관측으로 이동시키는 데 필요한 행동 단서만 선택하면 된다.

따라서 RTDP는 pose head를 성능이 낮다고 간주하여 제거한 것이 아니다. head가 해결하는 문제와 정책이 해결하는 문제가 다르기 때문에, 정보가 아직 조밀하게 남은 경계에서 interface를 형성한다. 이 선택을 통해 downstream policy가 localization용 hand-designed bottleneck을 거치지 않고 docking loss로 필요한 관계 정보를 직접 선택할 수 있다.

각 history frame \(H_i\)와 goal image \(G\)는 weight-shared frozen encoder를 통과한다. 이어지는 양방향 decoder는 다음 두 stream을 생성한다.

- **dec1:** goal feature를 참조하여 갱신된 current-frame token
- **dec2:** current-frame feature를 참조하여 갱신된 goal token

각 stream의 출력은 프레임당 \(196\times768\) token이다. 두 stream은 같은 image pair를 반대 방향에서 표현한다. RTDP는 두 출력을 모두 사용하여 current-side correction cue와 goal-side reference cue를 함께 보존한다. 한 stream만 사용하는 것은 backbone의 요구사항이 아니라 후속 directional ablation으로 검증할 RTDP의 설계 변수이다.

### 3.3 Perceiver Resampling and Condition Fusion

pre-head token 이후의 projection, resampling, multimodal fusion 및 action generation은 모두 RTDP가 새로 구성한 policy head이다. 이 모듈들은 ReLoc3R의 원래 pose head를 재사용한 것이 아니며, localization objective가 아니라 docking action loss로 학습된다.

각 decoder stream은 linear projection으로 768차원에서 384차원으로 변환된다. 이후 학습 가능한 16개의 latent query가 196개의 patch token을 cross-attend하는 Perceiver resampler를 적용한다.

\[
[B,5,196,768]\rightarrow[B,5,16,384].
\]

따라서 한 stream은 980개 patch token에서 80개 latent token으로, 두 stream은 총 1,960개에서 160개로 압축된다. 여기에 60개의 wheel token을 결합하여 총 220개의 condition token을 구성한다. 이 압축은 단순 평균 pooling과 달리 여러 latent가 서로 다른 공간적·관계적 정보를 선택하게 하면서, 후속 self-attention의 이차 계산량을 크게 줄인다.

4-layer, 6-head self-attention fusion Transformer는 wheel, `dec1`, `dec2` token 전체를 대칭적으로 갱신한다. 특정 modality를 query로 고정하는 cross-attention 대신 self-attention을 사용하는 이유는 이 단계의 목적이 압축이 아니라 세 token 집합 사이의 시간적·관계적 정합성을 공동으로 형성하는 것이기 때문이다. 최종 condition sequence \(C_t\in\mathbb{R}^{220\times384}\)는 pooling되지 않고 그대로 action denoiser에 제공된다.

### 3.4 Goal-Conditioned Diffusion Transformer

학습 시 clean action sequence \(A_t^0\)에 diffusion time \(\tau\)에 따른 Gaussian noise를 추가한다.

\[
A_t^\tau=\alpha_\tau A_t^0+\sigma_\tau\epsilon.
\]

12-layer cross-attention Diffusion Transformer는 noisy action token이 condition sequence \(C_t\)를 참조하도록 하며, 다음 denoising objective를 최소화한다.

\[
\mathcal{L}_{\mathrm{diff}}
=\mathbb{E}\left[\|\epsilon-\epsilon_\theta(A_t^\tau,\tau,C_t)\|_2^2\right].
\]

실로봇 실행에서는 EMA weight와 DPM-Solver++(2M), 30 sampling step을 사용한다. 정책은 최신 관측을 이용해 반복적으로 재계획하므로, 한 번 계산한 pose에 의존하지 않고 접근 과정에서 발생하는 오차를 폐루프에서 수정할 수 있다.

## 4. Dataset and Training Efficiency

학습 데이터는 하나의 수집 환경에서 사람이 수행한 145개의 성공 도킹 episode로 구성된다. 모든 frame을 합치면 225,465개이며, 30 Hz 기준 총 시연 시간은 다음과 같다.

\[
225{,}465/30=7{,}515.5\ \mathrm{s}
=125.26\ \mathrm{min}
=2.0876\ \mathrm{h}.
\]

| 항목 | 값 |
|---|---:|
| 성공 시연 episode | 145 |
| 전체 frame | 225,465 |
| 수집 주기 | 30 Hz |
| 전체 궤적 시간 | 7,515.5 s = 약 2시간 5분 |
| episode 평균 길이 | 51.83 s |
| episode 중앙 길이 | 48.50 s |
| 학습 epoch | 20 |
| gradient update | 16,940 |
| task-specific pose supervision | 없음 |
| 실행 시 LiDAR/depth/SLAM/map | 없음 |

225,465개의 frame은 서로 독립적인 시연 225,465개를 의미하지 않는다. 연속 trajectory에서 생성된 상관된 관측이므로, 데이터 규모는 frame 수와 함께 145 episode 및 약 2.09시간으로 보고한다. 또한 ReLoc3R backbone은 대규모 image-pair 사전학습을 사용하므로, 본 결과는 “2시간 만에 시각 관계를 처음부터 학습했다”가 아니라 “2시간의 도킹 시연으로 pretrained relational representation을 정밀 제어에 효율적으로 전이했다”는 의미이다.

## 5. Real-Robot Evaluation

### 5.1 Evaluation Questions

실로봇 평가는 다음 질문을 분리한다.

1. **In-distribution precision:** 학습 데이터에 등장한 도크와 배경에서 반복 도킹이 가능한가?
2. **Cross-instance generalization:** 학습하지 않은 도킹 스테이션에도 도킹할 수 있는가?
3. **Cross-environment and cross-type generalization:** 배경과 도크 종류가 동시에 바뀌어도 성공하는가?
4. **Non-dock visual-goal transfer:** 데이터셋에 없는 일반 물체를 포함한 goal image도 목표 관측으로 추종할 수 있는가?
5. **Terminal precision:** 성공한 trial이 센티미터급 위치 오차와 5° 이내 heading 오차를 만족하는가?

### 5.2 Observed Generalization

| 평가 조건 | 학습 데이터 포함 여부 | 현재 실로봇 결과 | 증거 수준 |
|---|---|---|---|
| 수집에 사용한 도킹 스테이션과 배경 | 포함 | 20/20 성공 | 정량 반복 시험 |
| 처음 보는 도킹 스테이션 | 미포함 | 도킹 성공 관찰 | 정성 시험; 반복 수 기록 필요 |
| 처음 보는 배경 + 새로운 종류의 도킹 스테이션 | 모두 미포함 | 도킹 성공 관찰 | 정성 시험; 반복 수 기록 필요 |
| 데이터셋에 없는 일반 물체를 포함한 goal image | 미포함 | 목표 영상 기준 도킹 성공 관찰 | 정성 stress test; 반복 수 기록 필요 |

동일 도크에 대한 20/20 결과는 정책이 학습 task를 안정적으로 수행함을 보여준다. 그러나 더 중요한 관찰은 외형과 환경의 변화가 동시에 발생해도 성공이 유지되었다는 점이다. 특히 일반 물체를 포함한 goal image에서의 성공은 정책이 고정된 “도킹 스테이션” category detector에만 의존하지 않음을 시사한다. 정책은 goal image에 지정된 종단 관측과 current observation 사이의 관계를 줄이는 방향으로 행동을 생성하는 visual servoing behavior를 형성한 것으로 해석할 수 있다.

다만 이 결과만으로 임의의 모든 object category나 scene에 일반화한다고 주장할 수는 없다. OOD 조건별 trial 수, 초기 위치·각도 분포 및 실패 사례가 기록되기 전까지 OOD 결과는 **강한 정성적 증거**로 보고하며, 20/20과 같은 정량 성공률과 혼합하지 않는다.

### 5.3 Terminal Precision

현재 실로봇 관찰에서 `r_relfeat_only`는 다음 종단 정밀도 범위를 보였다.

| 지표 | 관찰 결과 |
|---|---:|
| 최종 위치 오차 | 센티미터급 |
| 최종 heading 오차 | 5° 이내 |
| 학습 도크 성공률 | 20/20 (100%) |

이 결과는 일반적인 ImageNav의 meter-scale success radius보다 훨씬 엄격한 실제 단자 결합 수준의 목표를 다룬다는 점을 강조한다. 다만 “센티미터급”을 논문의 최종 정량 결과로 사용하려면 측정 장비와 기준 좌표계, 평균·중앙값·표준편차·최댓값 및 trial 수를 함께 보고해야 한다. 현재 문구는 관찰된 정밀도 범위를 나타내며, 아직 분포 통계가 확보되었다는 의미는 아니다.

## 6. Why Does RTDP Generalize?

관찰된 일반화의 주체는 frozen backbone 단독이 아니다. backbone token만으로는 선속도·각속도를 출력할 수 없고, 도킹의 action horizon, 종단 감속, 각도 복구 또는 폐루프 correction도 알지 못한다. RTDP가 적은 시연으로 이러한 관계 표현을 **어떻게 선택·압축·융합하고 행동으로 변환했는가**가 결과의 핵심이다.

### 6.1 A Control-Oriented Pre-Head Interface

RTDP는 pretrained model의 최종 답인 pose를 사용하는 대신, 답을 만들기 전의 양방향 dense representation을 제어 인터페이스로 사용한다. 이 때문에 국소 대응, 부분 가림 및 ambiguity를 downstream에 남겨 두고, docking loss가 실제 행동에 유효한 정보를 결정할 수 있다. 이는 off-the-shelf pose estimator와 controller를 직렬 연결한 구조가 아니라, pretrained representation과 task policy 사이의 새로운 연결 방식이다.

### 6.2 Goal-Relative Task Formulation

고정 도크만 입력받는 정책은 도크 색상, QR pattern 또는 배경을 action과 직접 연결할 수 있다. 반면 RTDP는 매 trial마다 current–goal image pair를 함께 처리한다. 따라서 절대적인 물체 identity보다 두 영상의 상대적 관점과 correspondence가 행동을 결정한다. 새로운 물체를 goal로 제공했을 때도 행동이 형성된 것은 이 해석과 일치한다.

### 6.3 Learned Compression and Temporal–Multimodal Fusion

RTDP의 Perceiver query는 프레임별 196개 patch를 단순 평균하지 않고 action loss에 유용한 16개 latent로 선택 압축한다. 이어지는 4-layer self-attention은 dec1, dec2, 다섯 시점의 변화 및 60-step wheel history를 공동으로 융합한다. 즉, RTDP는 pretrained feature를 그대로 사용하는 것이 아니라, 도킹에 맞는 고정 길이의 동적 condition representation을 학습한다.

### 6.4 Diffusion Trajectory Generation and Closed-Loop Correction

12-layer action DiT는 fused token에서 단일 steering command가 아니라 2초 길이의 일관된 velocity trajectory를 생성한다. 실행 중에는 최신 RGB와 wheel history로 반복 재계획하므로 작은 translation 또는 heading 오차를 이후 예측에서 보정할 수 있다. 센티미터급 정밀도는 관계 특징의 존재만이 아니라 RTDP가 학습한 trajectory generation과 반복 error correction의 결과이다.

### 6.5 Pretraining as a Data-Efficient Starting Point

frozen pairwise backbone은 다양한 scene과 viewpoint에 대한 유용한 초기 relation prior를 제공한다. 이 prior 덕분에 RTDP는 약 2.09시간의 도킹 데이터로 visual relation을 처음부터 학습할 필요가 없다. 그러나 어떤 token을 남기고, wheel과 어떻게 결합하며, 언제 감속·회전·정지할지는 RTDP가 task-specific demonstration에서 학습한다. 따라서 사전학습은 데이터 효율을 가능하게 하는 기반이고, 관찰된 정밀 도킹과 일반화를 구현한 시스템은 RTDP 전체이다.

이 설명은 현재 결과와 일치하는 가설이며 각 요소의 인과적 기여는 controlled ablation으로 검증해야 한다. 특히 appearance-only, final-pose interface, dec1 only, dec2 only, no-goal 및 다른 pairwise backbone을 같은 RTDP control stack과 데이터 예산에서 비교하면 backbone 자체의 효과와 우리의 interface·policy 설계 효과를 분리할 수 있다.

## 7. Required Quantitative Generalization Protocol

현재의 놀라운 성공 사례를 재현 가능한 논문 결과로 전환하기 위해 다음 protocol을 사용한다.

| 축 | 권장 구성 |
|---|---|
| 도크 | seen 1종 + unseen 최소 3종 |
| 환경 | seen 1곳 + unseen 최소 3곳 |
| 일반 물체 goal | 형상·색상·재질이 다른 최소 5종 |
| 초기 조건 | 거리와 heading offset을 사전 정의한 bin으로 균형화 |
| 반복 | 각 조건당 최소 10회, 가능하면 20회 |
| 성공 기준 | 제한 시간 내 무충돌 정지 및 사전에 정의한 위치·각도 tolerance 충족 |
| 정밀도 | position/heading의 median, mean±std, p90, worst case |
| 효율 | completion time, inference latency, recovery 횟수 |

반드시 다음 control을 포함한다.

- **Goal shuffle:** 현재 scene과 무관한 goal을 주었을 때 행동 목표가 실제로 바뀌는지 확인한다.
- **No-goal control:** goal을 제거한 정책이 unseen dock에서 같은 행동을 하는지 비교한다.
- **Appearance baseline:** 독립적인 RGB embedding만 사용하는 조건과 비교한다.
- **Final-pose baseline:** ReLoc3R final pose를 사용하는 modular controller 또는 동일 DiT와 비교한다.
- **Directional ablation:** `dec1` only, `dec2` only, `dec1+dec2`를 비교한다.
- **Data-scaling:** 10, 25, 50, 100, 145 episode로 학습하여 데이터 효율 곡선을 보고한다.

이 protocol은 성공 사례를 단순한 domain coincidence와 구분하고, “무엇이 일반화를 만들었는가”를 검증하는 데 필요하다.

## 8. Limitations

첫째, 학습 도크에서의 20/20 시험과 달리 새로운 도크·배경·일반 물체 goal 시험은 현재 반복 횟수와 실패 사례가 완전하게 기록되지 않았다. 따라서 본 초안은 이를 정성적 일반화 결과로 보고한다.

둘째, 센티미터급 위치 오차와 5° 이내 heading 오차를 최종 논문에 정량 주장으로 제시하려면 외부 motion capture, 정밀 fiducial 측정 또는 반복 가능한 기계적 기준을 이용해 오차 분포를 측정해야 한다. 충전 성공 여부와 pose error는 관련되지만 동일한 지표가 아니다.

셋째, task-specific demonstration은 약 2.09시간이지만 frozen ReLoc3R는 대규모 외부 데이터로 사전학습되었다. 그러므로 본 방법의 데이터 효율은 foundation representation의 transfer를 포함하며, 2시간의 데이터만으로 전체 시각 능력을 처음부터 학습한 결과가 아니다.

넷째, 현재 결과는 한 종류의 robot platform과 한 개의 bottom RGB camera에서 얻었다. 카메라 높이·intrinsic, 로봇 운동학 또는 조명 조건이 크게 변할 때의 성능은 별도로 검증해야 한다.

다섯째, `r_relfeat_only`의 성공만으로 diffusion, bidirectional token 및 4-layer fusion 각각이 필수라고 결론 내릴 수 없다. 동일 조건의 controlled ablation과 여러 random seed가 필요하다.

## 9. Conclusion

본 연구는 정밀 도킹 정책이 반드시 대규모 task-specific robot dataset이나 명시적 pose-control pipeline을 필요로 하는 것은 아니라는 가능성을 보여준다. 제안한 RTDP의 구현체 **r_relfeat_only**는 145개, 약 2.09시간의 실제 도킹 시연으로 학습되었으며, 단일 RGB 카메라와 wheel history만을 사용해 학습 도크에서 20/20의 성공을 기록했다. 더 나아가 미관측 도킹 스테이션, 미관측 배경과 새로운 도크 종류의 조합, 그리고 데이터셋에 존재하지 않는 일반 물체를 포함한 goal image에서도 폐루프 도킹 성공이 관찰되었다. 성공한 실행은 센티미터급 위치 오차와 5° 이내 heading 오차를 유지하였다.

이 결과에서 pretrained backbone은 current–goal 관계의 초기 표현을 제공했을 뿐이다. 정밀 도킹을 가능하게 한 제안 방법은 localization head 이전의 정보를 보존하고, 이를 행동 중심으로 압축하며, 양방향 시각 관계와 wheel·시간 정보를 융합하고, diffusion trajectory와 폐루프 재계획으로 변환하는 RTDP 전체이다. 후속 정량 시험이 현재의 정성 결과를 확인한다면, RTDP는 새로운 충전 스테이션마다 데이터를 다시 수집하거나 전용 pose controller를 설계하지 않고도 goal image 한 장으로 정밀 도킹 목표를 재지정할 수 있는 실용적 접근이 될 수 있다.

## 제출 전 반드시 갱신할 항목

- OOD 세 조건별 성공/전체 trial 수와 95% confidence interval
- 각 조건의 초기 거리·heading 분포
- 최종 position/heading error의 mean±std, median, p90 및 worst case
- 위치·각도 측정 장비와 success 판정 기준
- 일반 물체 goal의 정확한 의미와 goal image 예시 figure
- 실패 사례 및 failure taxonomy
- `dec1`/`dec2`, final pose, appearance 및 no-goal controlled ablation
- 10/25/50/100/145 episode data-scaling curve

## 참고문헌 초안

[1] A. Sridhar, D. Shah, C. Glossop, and S. Levine, “NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration,” in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2024.

[2] Y. Deng, S. Yuan, and Y. Fang, “AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation,” arXiv:2604.05351, 2026.

[3] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” in *Proc. Robot Sci. Syst. (RSS)*, 2023.

[4] S. Dong, S. Wang, S. Liu, L. Cai, Q. Fan, J. Kannala, and Y. Yang, “Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, pp. 16739–16752, 2025.
