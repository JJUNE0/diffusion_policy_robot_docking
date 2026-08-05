# RA-L 논문 초안

## 작업 제목

**지도와 거리 센서 없이 수행하는 Last-Centimeter Robot Docking: ReLoc3R 관계 토큰 기반 Goal-Conditioned Diffusion Policy**

**Map- and Range-Sensor-Free Last-Centimeter Robot Docking with a Goal-Conditioned Diffusion Policy Using ReLoc3R Relational Tokens**

## 초록

자율이동로봇의 정밀 도킹은 목표 부근 도달을 넘어, 충전 단자와 결합할 수 있는 위치와 자세로 정렬해야 하는 last-centimeter navigation 문제이다. 본 연구는 도킹 상태의 목표 영상, 단일 RGB 카메라의 관측 이력 및 wheel-velocity 이력으로 미래 속도 궤적을 생성하는 goal-conditioned diffusion policy를 제안한다. 제안 방법은 ReLoc3R의 최종 pose 출력 대신 pose head 이전의 양방향 cross-attention decoder token을 조건으로 사용하고, 이를 wheel token과 융합하여 Diffusion Transformer로 선속도와 각속도를 생성한다. 정책은 명시적 도킹 pose ground truth나 시뮬레이션 없이 145개의 실제 시연으로 학습되며, 실행 시 LiDAR, depth, SLAM, 사전 지도 또는 별도의 고전적 복구 제어기를 사용하지 않는다. 단일 복도 환경의 현재 실로봇 평가에서 20회 시도 모두 도킹에 성공했으며, 이는 조밀한 goal-relative visual representation을 이용한 최소 센서 정책의 1 cm급 정밀 도킹 가능성을 보여준다.

**RA-L Index Terms:** Vision-Based Navigation, Imitation Learning, Machine Learning for Robot Control, Sensor-based Control, Wheeled Robots

## 1. Introduction

자율이동로봇(Autonomous Mobile Robot, AMR)의 지속적인 무인 운용을 위해서는 배터리 충전 과정까지 사람의 개입 없이 수행할 수 있어야 한다. 이때 정밀 도킹은 단순히 충전 스테이션 근처에 도달하는 문제가 아니라, 제한된 기계적 허용오차 안에서 로봇의 위치와 heading을 동시에 맞추고 접근 중 발생한 각도 오차를 복구해야 하는 last-centimeter navigation 문제이다. 기존의 규칙 기반 도킹 시스템은 정확한 상태 추정과 제어를 위해 LiDAR 또는 depth 센서, 사전 지도, SLAM, 인공 표식이나 도크별 pose controller를 조합할 수 있으며, 이러한 구성은 잘 정의된 환경에서 높은 정확도를 제공하지만 센서 교정, 지도 유지 및 예외 상황별 로직 설계에 추가적인 시스템 통합 비용을 요구한다. 본 연구의 목적은 학습 기반 방법이 본질적으로 계산 비용이 더 낮다고 주장하는 데 있지 않다. 대신 단일 RGB 카메라와 wheel encoder만으로 구성된 정책이 복잡한 인지·제어 모듈 없이 실제 도킹에 필요한 종단 정렬과 폐루프 복구 동작을 학습할 수 있는지를 규명한다.

정밀 도킹 정책은 로봇이 현재 무엇을 보고 있는지뿐 아니라 어떤 종단 상태에 정렬해야 하는지를 알아야 한다. 하나의 고정된 도크만 학습하는 no-goal policy도 특정 환경을 암기할 수 있지만, 목표가 모델 입력으로 명시되지 않으면 원하는 도킹 자세가 배경과 학습 궤적에 암묵적으로 결합되어 다른 도크나 목표 자세로 재지정하기 어렵다. Image-goal navigation은 목표 장소에서 촬영한 영상을 별도의 task specification으로 제공하여 이 문제를 완화한다. NoMaD는 goal-conditioned navigation과 goal-agnostic exploration을 하나의 diffusion policy로 통합해 미지 환경에서의 장거리 목표 도달 능력을 보였지만 [1], 목표 부근에 도달한 이후의 센티미터급 위치 및 자세 정렬을 직접 다루지는 않는다. AnyImageNav는 목표 영상을 geometric query로 사용하여 6-DoF goal pose를 복원하고 기존 ImageNav의 1 m 성공 반경을 last-meter 수준으로 좁혔으나, Gibson과 HM3D에서 보고된 최종 위치 오차는 각각 0.27 m와 0.21 m이다 [2]. 따라서 정밀 도킹에는 goal appearance의 유사성뿐 아니라 현재 관측과 목표 관측 사이의 미세한 상대 관계를 보존하고, 이를 폐루프 속도 명령으로 변환하는 방법이 필요하다.

본 연구는 별도의 reward 설계나 시뮬레이터 없이 실제 도킹 시연을 활용하기 위해 문제를 offline imitation learning으로 정식화한다. 그중 diffusion policy를 선택한 이유는 조건부 행동 분포의 여러 가능한 복구 궤적을 단일 평균 행동으로 축약하지 않고, 시간적으로 일관된 action sequence로 모델링할 수 있기 때문이다 [3]. 그러나 diffusion은 행동 분포를 생성하는 방법일 뿐 정밀한 goal-relative 상태를 스스로 보장하지 않는다. 이를 위해 도킹 완료 영상을 goal reference로 사용하고, ReLoc3R [4]의 최종 pose parameter 대신 pose head 이전의 양방향 cross-attention decoder stream을 조건으로 제공한다. 본 연구의 기여는 (i) 최소 센서 기반 image-goal 정밀 도킹을 goal-conditioned offline imitation learning 문제로 정식화하고, (ii) 조밀한 ReLoc3R relational token과 wheel history로 조건화된 trajectory diffusion policy를 설계하며, (iii) 145개의 실제 시연으로 학습한 모델이 단일 복도 환경의 20회 실로봇 시험에서 모두 성공함을 보인 것이다. 다만 diffusion, goal conditioning 및 pre-head token 각각의 인과적 기여는 동일 조건의 ablation으로 검증되어야 한다.

## 2. Related Work

### 2.1 Imitation Learning and Diffusion Policies

Offline imitation learning은 전문가가 수행한 관측–행동 궤적을 직접 학습하므로 정밀 도킹을 위한 reward shaping과 시뮬레이터 구축을 피할 수 있다. 그러나 제곱오차로 학습하는 결정론적 행동 회귀는 유사한 관측에서 서로 다른 유효한 조향 또는 복구 동작이 존재할 때 그 평균을 출력할 수 있다. Diffusion Policy는 정책을 조건부 denoising process로 표현하여 multimodal action distribution과 고차원 action sequence를 모델링하고, receding-horizon control과 결합할 수 있음을 보였다 [3]. NoMaD는 이 아이디어를 visual navigation에 적용했지만 [1], 본 연구는 장거리 goal reaching이 아니라 마지막 수 센티미터의 위치·각도 정렬을 목표로 한다. 따라서 diffusion backbone 자체보다 어떤 goal-relative 정보로 행동 생성을 조건화하는지가 핵심 설계 문제가 된다.

### 2.2 Goal Conditioning and Relative Visual Geometry

Goal image는 원하는 종단 관측을 명시하므로 정책을 고정된 행동 함수 \(\pi(A\mid O)\)에서 재지정 가능한 함수 \(\pi(A\mid O,G)\)로 확장한다. 하지만 독립적으로 추출한 current/goal appearance embedding의 유사성은 장소 인식에는 유용해도, 도킹에 필요한 작은 횡방향 및 heading 오차를 직접 나타내지 않는다. ReLoc3R는 두 영상 사이의 상대 camera pose를 회귀하기 위해 양방향 cross-attention decoder를 학습한다 [4]. 본 연구는 ReLoc3R의 최종 pose를 외부 localization 결과로 사용하지 않고, current와 goal이 서로를 참조한 decoder token을 diffusion policy의 내부 조건으로 사용한다. 이 설계는 명시적 pose supervision 없이도 정책이 task-specific한 상대 시각 정보를 선택하도록 한다는 점에서, pose estimator와 controller를 직렬 연결하는 전통적 modular pipeline과 구별된다.

## 3. Method

### 3.1 Problem Formulation and Goal-Conditioned Demonstrations

각 시점 \(t\)에서 관측 \(O_t\)는 30 Hz로 수집한 60-frame wheel-velocity history와 같은 구간에서 stride 12로 선택한 다섯 RGB frame으로 구성된다. 각 episode의 마지막 도킹 완료 frame을 정적 goal image \(G\)로 사용하며, 학습 target \(A_t=[a_t,\ldots,a_{t+59}]\in\mathbb{R}^{60\times2}\)는 약 2초 동안의 선속도와 각속도이다. 데이터셋은 145개의 성공 시연에서 \((O_t,G,A_t)\) tuple을 구성하고, 정책은 조건부 분포 \(p_\theta(A_t\mid O_t,G)\)를 학습한다. 여기서 goal conditioning의 역할은 현재 위치를 추정하는 데 그치지 않고 “어떤 관측 상태가 도킹 완료인가”를 명시하는 것이다. 따라서 서로 다른 goal image로 재지정하는 능력은 학습 도크를 암기한 no-goal policy와 구분되어야 하며, 별도의 cross-goal 평가로 검증한다.

### 3.2 Conditional Diffusion Imitation Policy

행동은 학습 데이터의 범위에 따라 \([-1,1]\)로 정규화된다. 학습 시 clean action sequence \(A_t^0\)에 diffusion time \(\tau\)에 따른 Gaussian noise \(\epsilon\)을 더해 \(A_t^\tau=\alpha_\tau A_t^0+\sigma_\tau\epsilon\)을 만들고, 조건 \(C_t=f_\phi(O_t,G)\)가 주어졌을 때 noise를 복원하도록 \(\mathcal{L}_{\mathrm{diff}}=\mathbb{E}[\|\epsilon-\epsilon_\theta(A_t^\tau,\tau,C_t)\|_2^2]\)를 최소화한다.

Denoiser는 12-layer cross-attention Diffusion Transformer이며, 60개의 noisy action token이 전체 condition sequence에 attention한다. 실기에서는 EMA weight와 DPM-Solver++(2M) 30 denoising step을 사용하여 60-step 속도 궤적을 생성한다. 정책은 매 inference cycle마다 최신 관측으로 다시 계획하므로, 한 번의 open-loop pose 추정에 의존하지 않고 진행 중 발생한 오차를 후속 action sequence에서 수정할 수 있다. Diffusion을 사용한 주된 목적은 이러한 action-sequence distribution을 모델링하는 것이며, ReLoc3R는 그 분포를 선택하는 조건 정보를 제공한다.

### 3.3 Pre-Head ReLoc3R Relational Tokens

샘플된 각 current frame \(H_i\)와 goal image \(G\)는 frozen ReLoc3R ViT-L encoder를 공유한다. 이어지는 양방향 decoder는 `dec1_i`, 즉 goal을 cross-attend한 current stream과 `dec2_i`, 즉 current를 cross-attend한 goal stream을 각각 \(\mathbb{R}^{196\times768}\) token으로 생성한다. 원래 ReLoc3R pose head는 이 token grid를 projection과 residual convolution에 통과시킨 뒤 adaptive global average pooling하고, 하나의 translation vector와 rotation matrix로 축약한다. 이 출력은 localization에는 적합하지만, spatial correspondence, appearance cue, 추정의 모호성과 같이 도킹 정책에 유용할 수 있는 정보를 단일 pose estimate 안에서 제거할 수 있다.

본 방법은 global pooling 직전의 두 decoder stream을 가져와 각각 학습 가능한 projection과 Perceiver resampler로 196개 patch token을 16개 latent token으로 압축한다. 즉 pose head의 task-agnostic point estimate를 사용하는 대신, 관계 정보가 남아 있는 token을 정책 목적에 맞게 압축한다. 다섯 시점에서 `dec1`과 `dec2`가 각각 80개 token을 만들고, 60개의 wheel token과 결합하여 총 220개의 384-dimensional condition token을 구성한다. 4-layer token fusion network가 이 sequence를 처리한 후 action DiT에 제공한다. ReLoc3R backbone은 frozen 상태이며, resampler, condition network와 diffusion policy만 도킹 시연으로 학습한다.

### 3.4 What Must Be Demonstrated

Pre-head token이 더 높은 실기 성능을 보인다는 설명은 현재 구조만으로 증명되지 않는다. 제안한 가설은 (i) final pose head의 global pooling이 정밀 제어에 필요한 국소 관계 정보를 제거하고, (ii) ReLoc3R의 translation이 metric scale을 직접 제공하지 않으며, (iii) pretrained pose head의 목적과 도킹 action generation의 목적이 다르기 때문에 downstream policy가 dense token에서 필요한 정보를 직접 선택하는 편이 유리하다는 것이다. 이를 검증하려면 동일한 diffusion backbone과 학습 예산에서 ReLoc3R encoder token, 최종 pose output 및 pre-head `dec1/dec2` token을 비교해야 한다.

## 4. Experimental Questions and Ablation Plan

| Research question | Controlled comparison | Required evidence |
|---|---|---|
| 왜 diffusion policy인가? | deterministic Transformer/MLP behavior cloning vs. DDPM, 동일 condition·데이터·action horizon | 실로봇 success, 종단 위치·각도 오차, 복구 성공률 |
| 왜 goal-conditioned인가? | no-goal vs. goal appearance vs. goal-relative condition | 새로운 goal reference와 초기 자세에서의 성능 |
| 왜 ReLoc3R인가? | 일반 vision encoder vs. ReLoc3R encoder feature vs. ReLoc3R final pose | 동일 파라미터·학습 예산의 paired comparison |
| 왜 pre-head token인가? | final pose vs. `dec1` only vs. `dec2` only vs. `dec1+dec2` | success와 final error, seed별 평균 및 분산 |
| recovery가 가능한가? | 초기 거리와 heading offset을 구간별로 변화 | offset bin별 성공률, correction time, collision 및 timeout |

Main evaluation은 도킹 성공 여부뿐 아니라 최종 위치 오차(mm), heading 오차(deg), 초기 오차 구간별 recovery rate, collision, completion time 및 inference latency를 함께 보고한다. 모든 비교는 같은 demonstration split, action horizon, diffusion sampling 설정과 실로봇 초기 조건을 사용해야 한다. 특히 이전 diffusion variant의 실기 실패와 현재 `r_relfeat_only`의 성공은 여러 설정 차이가 섞여 있으므로, 그것만으로 ReLoc3R의 인과적 효과를 주장하지 않는다.

## 5. Current Real-Robot Result

현재 대표 모델 `r_relfeat_only`는 145개 실제 시연, 20 epoch(16,940 gradient step)로 학습되었다. 이 모델은 한 대의 RGB 카메라, 60-frame wheel history와 ReLoc3R `dec1+dec2` relational token만 사용하며, LiDAR, depth, ICP pose label 또는 별도의 goal-pose controller를 사용하지 않는다. 단일 4층 복도 task에서 현재까지 수행한 20회 실로봇 시험은 모두 성공했다. 이 결과는 제안 조합의 feasibility를 보여주지만, goal conditioning이나 pre-head token 중 어느 요소가 성공을 만들었는지는 위의 controlled ablation이 완료된 뒤에만 결론 내릴 수 있다.

## 저자 메모 — 제출 전 갱신할 항목

- 현재 확인된 실기 결과는 단일 환경에서의 `20/20` 성공이다. “10 mm 위치 오차를 달성했다”는 계측 결과와 동일하지 않으므로, 별도의 위치·각도 오차 측정 전에는 `1 cm급 정밀 도킹 가능성`으로 표현한다.
- `20/20`의 성공 판정 기준을 명시해야 한다. 예: 충전 단자 결합, 최종 위치 오차 10 mm 이하, 무충돌 정지, 제한 시간 내 완료.
- 최종 논문에는 환경 수, 초기 위치·각도 분포, 전체 trial 수, 평균 오차, 최악 오차, 성공률의 신뢰구간을 추가한다.
- ReLoc3R pose head 출력과 pose-head 이전 decoder token의 비교 실험을 추가하여 relational token 사용의 효과를 검증한다.
- ReLoc3R 이외의 vision encoder 및 NoMaD 기반 모델과 동일 데이터·동일 평가 조건으로 비교한다.
- 본 방법은 LiDAR·depth 등의 센서 및 인프라 의존성을 줄이지만, ReLoc3R와 diffusion inference를 사용하므로 기존 규칙 기반 방법보다 계산 비용이 낮다고 주장하지 않는다. 센서 비용, 연산 지연 및 전력 소비는 별도 항목으로 측정한다.
- ReLoc3R는 frozen encoder/decoder로 사용되므로, 최종 표현에서는 “전체 네트워크를 end-to-end로 공동 학습했다”보다 “관측에서 속도 명령까지 직접 생성하는 end-to-end control pipeline”이라고 기술하는 편이 정확하다.

## 참고문헌 초안

[1] A. Sridhar, D. Shah, C. Glossop, and S. Levine, “NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration,” in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2024.

[2] Y. Deng, S. Yuan, and Y. Fang, “AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation,” arXiv:2604.05351, 2026.

[3] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. C. M. Burchfiel, and S. Song, “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” in *Proc. Robot. Sci. Syst. (RSS)*, 2023.

[4] S. Dong, S. Wang, S. Liu, L. Cai, Q. Fan, J. Kannala, and Y. Yang, “Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2025.
