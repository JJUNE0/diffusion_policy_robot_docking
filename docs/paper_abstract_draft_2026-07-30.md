# RA-L 논문 개정 초안 — 일반화 중심

> 개정일: 2026-08-06
> 대표 모델: `r_relfeat_only`
> 정량 결과와 현재까지 확인된 정성적 실로봇 결과를 구분한다. 반복 횟수가 기록되지 않은 OOD 시험에는 성공률을 임의로 부여하지 않는다.

## 작업 제목

**약 2시간의 실제 시연으로 미관측 도크와 시각 목표에 일반화하는 Last-Centimeter Robot Docking**

**Generalizable Last-Centimeter Robot Docking for Unseen Docks and Visual Goals from Two Hours of Real-World Demonstrations**

## 초록

자율이동로봇의 정밀 도킹은 목표 부근에 도달하는 것을 넘어, 충전 단자가 결합될 수 있도록 센티미터 단위의 위치 오차와 수 도 이내의 자세 오차로 로봇을 정렬해야 하는 last-centimeter navigation 문제이다. 기존의 학습 기반 접근은 충분한 양의 다환경 데이터나 명시적 위치 추정에 의존하거나, 학습에 사용된 도킹 스테이션의 외형과 배경을 암기할 위험이 있다. 본 연구는 단일 RGB 카메라, wheel-velocity history 및 한 장의 goal image만으로 미래 속도 궤적을 생성하는 goal-conditioned diffusion policy를 제안한다. 제안 모델 `r_relfeat_only`는 일반 RGB appearance branch나 ReLoc3R의 최종 pose 출력을 사용하지 않고, frozen ReLoc3R의 양방향 decoder가 생성한 goal-relative relational token을 유일한 시각 조건으로 사용한다. 두 방향의 관계 토큰은 Perceiver resampler와 self-attention fusion을 거쳐 Diffusion Transformer의 조건으로 제공되며, 정책은 매 제어 주기마다 2초 길이의 선속도·각속도 궤적을 다시 계획한다.

정책 학습에는 하나의 수집 환경에서 얻은 145개의 실제 성공 시연만 사용하였다. 데이터는 30 Hz 기준 225,465 frame, 총 7,515.5초, 즉 약 2.09시간에 해당하며, 정책은 20 epoch, 16,940 gradient update로 학습되었다. 실행 시 LiDAR, depth, SLAM, 사전 지도, 명시적 goal-pose label 및 별도의 고전 제어기를 사용하지 않는다. 학습 분포와 동일한 도크에 대한 기존 실로봇 시험에서는 20/20 도킹 성공을 기록하였다. 더 나아가 학습에 포함되지 않은 도킹 스테이션, 처음 보는 배경과 새로운 종류의 도킹 스테이션의 조합, 그리고 데이터셋에 전혀 존재하지 않는 일반 물체를 포함한 goal image에 대해서도 폐루프 도킹 성공이 관찰되었다. 현장 시험에서 최종 위치 오차는 센티미터급, heading 오차는 5° 이내로 유지되었다. 새 OOD 조건의 반복 횟수와 오차 분포는 아직 정량화 중이지만, 이 결과는 정책이 특정 도크의 외형을 단순히 암기하기보다 goal image와 현재 관측 사이의 상대적 시각 관계를 제어 신호로 변환하고 있음을 보여준다. 특히 범용 시각 관계 표현을 이용하면 약 2시간의 task-specific 실제 시연만으로도 도크 인스턴스와 배경을 넘어서는 정밀 도킹 정책을 학습할 수 있다는 가능성을 제시한다.

**RA-L Index Terms:** Vision-Based Navigation, Robot Docking, Imitation Learning, Diffusion Policy, Visual Generalization, Sensor-Based Control

## 1. Introduction

자율이동로봇(Autonomous Mobile Robot, AMR)이 사람의 개입 없이 장시간 운용되기 위해서는 충전 스테이션으로 이동하는 것뿐 아니라 충전 단자를 실제로 결합하는 마지막 정렬 단계까지 자율적으로 수행해야 한다. 이 단계는 목표 영역에 도달하는 일반적인 navigation과 요구 정밀도가 다르다. 수 센티미터의 위치 오차나 수 도의 heading 오차만으로도 단자 결합이 실패할 수 있으며, 접근 중 발생한 오차를 폐루프에서 복구할 수 있어야 한다.

전통적인 정밀 도킹 시스템은 LiDAR 또는 depth sensor, SLAM, 사전 지도, fiducial marker, 명시적 relative-pose estimator 및 도크별 controller를 결합할 수 있다. 이러한 modular pipeline은 구조화된 환경에서 높은 정확도를 제공하지만, 새로운 도크나 배경으로 이동할 때 센서 교정, 지도 갱신 또는 도크별 제어 규칙의 재설계가 필요할 수 있다. 반대로 학습 기반 정책은 관측에서 행동까지의 변환을 데이터로 학습할 수 있지만, 소규모 데이터에서는 도크의 색상과 모양 또는 배경을 목표 그 자체로 암기하는 shortcut을 사용할 가능성이 있다.

본 연구가 다루는 핵심 질문은 다음과 같다.

> 하나의 환경에서 수집한 약 2시간의 실제 시연만으로 학습한 정책이, 학습하지 않은 도킹 스테이션과 배경에서도 센티미터급 정밀 도킹을 수행할 수 있는가?

이를 위해 목표 상태에서 촬영한 한 장의 영상을 task specification으로 사용한다. 그러나 goal image를 독립적인 appearance embedding으로 처리하는 것만으로는 작은 횡방향 오차와 heading 오차를 안정적으로 표현하기 어렵다. 본 연구는 ReLoc3R [4]가 두 영상의 상대 pose를 추정하기 직전에 형성하는 양방향 decoder token에 주목한다. 이 token은 current image가 goal image를 참조한 표현과 goal image가 current image를 참조한 표현을 모두 포함하므로, 특정 도크의 의미적 identity보다 두 관측 사이의 상대적인 correspondence와 관점 변화를 표현할 가능성이 있다.

제안 모델 `r_relfeat_only`는 이 관계 표현을 정책의 유일한 시각 입력으로 사용한다. 즉, 별도의 DINO appearance feature, LiDAR, depth 또는 ReLoc3R final-pose vector를 사용하지 않는다. frozen ReLoc3R가 생성한 dense relational token과 wheel history를 학습 가능한 condition fusion network로 결합하고, cross-attention Diffusion Transformer가 미래 선속도와 각속도 sequence를 생성한다. 이 구성은 범용 relative-geometry prior를 task-specific한 도킹 행동으로 변환하면서, 적은 실제 시연만으로 학습할 수 있도록 설계되었다.

실로봇 시험에서 나타난 가장 중요한 결과는 학습 도크에 대한 성공 자체가 아니라 학습 분포 밖으로의 전이다. `r_relfeat_only`는 수집에 사용한 도크뿐 아니라 (i) 학습하지 않은 도킹 스테이션, (ii) 처음 보는 배경에 놓인 새로운 종류의 도킹 스테이션, (iii) 데이터셋에 존재하지 않는 일반 물체를 포함한 goal image에 대해서도 도킹 행동을 완수하였다. 이는 정책이 “이 물체가 학습 도크인가?”를 분류하는 대신 “현재 영상이 주어진 goal 영상과 같은 상대 관측이 되려면 어떻게 움직여야 하는가?”를 학습했다는 해석과 일치한다.

본 연구의 기여는 다음과 같다.

1. 단일 RGB 카메라와 wheel history만을 사용하는 map- and range-sensor-free last-centimeter docking을 goal-conditioned offline imitation learning 문제로 정식화한다.
2. ReLoc3R의 final pose가 아니라 pose head 이전의 양방향 relational token을 직접 사용하는 `r_relfeat_only` 조건 구조를 제안한다.
3. 145개, 225,465 frame, 약 2.09시간의 task-specific 실제 시연만으로 정책을 학습하고, 동일 도크에서 20/20의 실로봇 성공을 보인다.
4. 미관측 도크, 미관측 배경과 도크의 동시 변화, 그리고 비도크 일반 물체를 포함한 goal image까지 성공적으로 추종하는 정성적 cross-instance·cross-environment·cross-category 전이를 보인다.
5. 이러한 전이를 유지하면서 센티미터급 최종 위치 오차와 5° 이내 heading 오차의 정밀 도킹 가능성을 실로봇에서 확인한다.

## 2. Related Work

### 2.1 Goal-Conditioned Visual Navigation

Image-goal navigation은 목표 위치에서 촬영한 영상을 입력으로 제공함으로써 정책을 고정된 함수 \(\pi(A\mid O)\)에서 재지정 가능한 함수 \(\pi(A\mid O,G)\)로 확장한다. NoMaD [1]는 goal-conditioned navigation과 goal-agnostic exploration을 diffusion policy 하나로 통합하여 미지 환경에서의 장거리 이동 가능성을 보였다. 그러나 일반적인 ImageNav는 통상 미터 단위의 성공 반경을 사용하며, 충전 단자 결합에 필요한 last-centimeter alignment를 직접 평가하지 않는다.

AnyImageNav [2]는 goal image를 geometric query로 취급하고 dense correspondence를 통해 6-DoF goal pose를 복원한다. Gibson과 HM3D에서 각각 0.27 m/3.41°와 0.21 m/1.23°의 pose-recovery error를 보고하여 last-meter ImageNav의 정밀도를 크게 향상시켰다. 본 연구는 전역 탐색이나 명시적인 6-DoF pose recovery가 아니라, 이미 도크의 접근 구간에 들어온 로봇이 goal-relative visual relation을 직접 velocity sequence로 변환하여 실제 단자 결합 수준의 정렬을 수행하는 문제를 다룬다.

### 2.2 Diffusion Policies for Closed-Loop Control

Diffusion Policy [3]는 행동 정책을 conditional denoising process로 표현하여 multimodal action distribution과 시간적으로 일관된 action sequence를 모델링한다. Receding-horizon execution과 결합하면 정책이 최신 관측으로 반복 재계획할 수 있다. 본 연구에서 diffusion은 여러 가능한 접근 및 복구 궤적을 평균 행동으로 붕괴시키지 않기 위한 action generator이며, 일반화의 직접적인 원천은 아니다. 새로운 도크에 대한 전이를 결정하는 핵심은 diffusion model에 어떤 goal-relative representation을 조건으로 제공하는가이다.

### 2.3 Relative Visual Geometry and Transfer

ReLoc3R [4]는 약 8백만 개의 posed image pair로 사전학습되어 새로운 scene으로 일반화되는 relative camera-pose regression을 목표로 한다. ReLoc3R의 final pose head는 decoder token을 하나의 rotation과 translation으로 축약하지만, 이 과정에서 정책에 유용할 수 있는 국소 correspondence, 공간 구조 및 ambiguity가 제거될 수 있다. 본 연구는 final pose를 외부 localization 결과로 사용하지 않고, 양방향 decoder의 pre-head token을 그대로 downstream policy에 전달한다.

따라서 본 연구의 “약 2시간 데이터”는 ReLoc3R를 처음부터 학습한 총 데이터량이 아니라 **도킹 행동을 학습하는 데 사용한 task-specific robot demonstration의 양**을 의미한다. 광범위한 사전학습으로 획득한 시각 관계 prior를 고정하고, 소량의 실제 로봇 데이터로 그 표현을 정밀 제어 행동에 연결한다는 점이 데이터 효율의 핵심이다.

## 3. Method

### 3.1 Problem Formulation

시점 \(t\)에서 정책은 현재까지의 관측 history \(O_t\)와 goal image \(G\)를 받아, 향후 60 step의 선속도와 각속도 sequence를 출력한다.

\[
A_t=[a_t,\ldots,a_{t+59}]\in\mathbb{R}^{60\times2},
\qquad a_t=(v_t,\omega_t).
\]

관측은 30 Hz로 수집된 60-frame wheel-velocity history와 동일한 약 2초 구간에서 stride 12로 선택한 다섯 장의 RGB frame으로 구성된다. 학습 중에는 각 성공 episode의 마지막 도킹 frame을 goal image로 사용한다. 정책은 조건부 행동 분포 \(p_\theta(A_t\mid O_t,G)\)를 학습하고, 실행 중에는 새로운 goal image로 목표를 교체할 수 있다.

### 3.2 Bidirectional ReLoc3R Relational Tokens

각 history frame \(H_i\)와 goal image \(G\)는 weight-shared frozen ReLoc3R ViT encoder를 통과한다. 이어지는 양방향 decoder는 다음 두 stream을 생성한다.

- `dec1`: goal feature를 참조하여 갱신된 current-frame token
- `dec2`: current-frame feature를 참조하여 갱신된 goal token

각 stream의 출력은 프레임당 \(196\times768\) token이다. 두 stream은 같은 image pair를 반대 방향에서 표현한다. `dec1`은 현재 관측의 어느 부분이 goal과 대응하는지에 유리하고, `dec2`는 goal의 어느 부분이 현재 관측에서 관찰되는지 또는 가려지는지를 표현할 수 있다. 본 연구는 두 출력을 모두 사용하되, 한 stream만 사용하는 선택도 가능한 설계 변수로 남긴다.

### 3.3 Perceiver Resampling and Condition Fusion

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

## 6. Why Does `r_relfeat_only` Generalize?

관찰된 일반화는 다음 네 요소가 결합된 결과로 해석할 수 있다.

### 6.1 Appearance Classification 대신 Goal-Relative Matching

고정 도크만 입력받는 정책은 도크 색상, QR pattern 또는 배경을 action과 직접 연결할 수 있다. 반면 `r_relfeat_only`는 매 trial마다 current–goal image pair를 함께 처리한다. 따라서 절대적인 물체 identity보다 두 영상의 상대적 관점과 correspondence가 행동을 결정한다. 새로운 물체를 goal로 제공했을 때도 행동이 형성된 것은 이 해석과 일치한다.

### 6.2 Large-Scale Pretrained Relational Prior

ReLoc3R는 약 8백만 image pair에서 학습된 상대 시각 표현을 제공한다. 도킹 데이터가 약 2.09시간으로 작더라도, 정책은 다양한 scene과 viewpoint에 대한 관계 표현을 처음부터 학습할 필요가 없다. 도킹 시연은 이 범용 표현에서 정렬 행동에 필요한 부분을 선택하고 velocity command로 연결하는 역할을 한다.

### 6.3 Dense Pre-Head Token의 정보 보존

final pose vector는 localization에 필요한 정보를 하나의 point estimate로 축약한다. 반면 pre-head `dec1+dec2` token은 국소 대응, 가려짐, 구조 및 추정 ambiguity를 더 풍부하게 유지한다. downstream policy가 이 정보 중 실제 도킹에 유용한 부분을 직접 선택할 수 있기 때문에, 새로운 도크에서도 final-pose head의 calibration에 덜 종속될 수 있다.

### 6.4 Closed-Loop Replanning

정책은 한 번 예측한 궤적을 끝까지 open-loop로 실행하지 않는다. 최신 RGB와 wheel history로 계속 다시 계획하기 때문에 작은 translation 또는 heading 오차가 발생해도 이후 예측에서 보정할 수 있다. 센티미터급 정밀도는 relational representation뿐 아니라 이러한 반복적인 error-correction 과정의 결과이다.

이 네 설명은 현재 결과와 일치하는 가설이며, 각각의 인과적 기여가 완전히 증명된 것은 아니다. DINO appearance, ReLoc3R final pose, `dec1` only, `dec2` only, `dec1+dec2` 및 goal-shuffle 조건을 동일한 데이터와 제어 설정에서 비교해야 한다.

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

본 연구는 정밀 도킹 정책이 반드시 대규모 task-specific robot dataset이나 명시적 pose-control pipeline을 필요로 하는 것은 아니라는 가능성을 보여준다. `r_relfeat_only`는 145개, 약 2.09시간의 실제 도킹 시연으로 학습되었으며, 단일 RGB 카메라와 wheel history만을 사용해 학습 도크에서 20/20의 성공을 기록했다. 더 나아가 미관측 도킹 스테이션, 미관측 배경과 새로운 도크 종류의 조합, 그리고 데이터셋에 존재하지 않는 일반 물체를 포함한 goal image에서도 폐루프 도킹 성공이 관찰되었다. 성공한 실행은 센티미터급 위치 오차와 5° 이내 heading 오차를 유지하였다.

이 결과는 도크의 절대 appearance를 암기하는 대신 current–goal 관계를 표현하는 pretrained relational token을 행동 생성의 조건으로 사용하는 것이 데이터 효율과 실제 환경 일반화를 동시에 얻는 유효한 방향임을 시사한다. 후속 정량 시험이 현재의 정성 결과를 확인한다면, 본 방법은 새로운 충전 스테이션마다 데이터를 다시 수집하거나 전용 pose controller를 설계하지 않고도 goal image 한 장으로 정밀 도킹 목표를 재지정할 수 있는 실용적 접근이 될 수 있다.

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
