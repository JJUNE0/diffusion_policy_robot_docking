<div align="center">

# Diffusion Policy for Autonomous Robot Docking

**Single-Camera ReLoc3R Relational Features + DDPM DiT for Real-Robot Docking**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

This README documents **`r_relfeat_only`**, the only experiment in this
repository that has succeeded on the real robot so far. The policy uses one
`orbbec0` camera, wheel-velocity history, and ReLoc3R's bidirectional
cross-attention decoder tokens to generate a 60-step velocity trajectory with
a conditional DDPM DiT. Other model families remain in the repository for
ablation history, but they are not the reproduction target of this README.

## Overview

| | |
|---|---|
| **Task** | Autonomous charging-station docking for a differential-drive mobile robot |
| **Model** | `r_relfeat_only` |
| **Input** | `orbbec0` RGB history + docked-position goal RGB + wheel velocity history |
| **Output** | 60-step `(linear velocity, angular velocity)` trajectory |
| **Vision** | Frozen ReLoc3R-224 encoder + bidirectional decoder features (`dec1`, `dec2`) |
| **Policy** | Token-sequence fusion + cross-attention DiT + DDPM |
| **Training** | Offline imitation learning, 20 epochs / 16940 steps |
| **Real robot** | `step 12000` and `step 16940`: all trials successful to date (user report) |

## 실기 검증 성공 모델: `r_relfeat_only`

> **실기 결과(사용자 보고, 2026-07-30):** 현재까지 이 run이 유일하게
> 실기 도킹에 성공한 실험이며, 테스트한 `step 12000`과 `step 16940`은
> 각각 수행한 모든 실기 시도에 성공했다(현재까지 success rate 100%).
> 시험 횟수가 기록되어 있지 않으므로 성공 횟수나 통계적 신뢰구간까지
> 의미하는 결과로 확대 해석하지 않는다.

성공 run과 검증된 체크포인트는 다음과 같다.

| 항목 | 값 |
|---|---|
| Run | `outputs/train/r_relfeat_only/2026-07-28_22-47-34` |
| 실기 성공 checkpoint 1 | `r_relfeat_only_step_12000.pt` (내부 step도 `12000`) |
| 실기 성공 checkpoint 2 | `r_relfeat_only_step_16940.pt` (20 epoch 최종 checkpoint, 내부 step도 `16940`) |
| 저장 config | `r_relfeat_only_step_12000.config.yaml` |
| 현재 배포 config | `ai-control-260729/config.yml`은 `step 12000`, DDPM, EMA weight, 30 denoising steps, 1 sample을 사용 |

`step 16940`도 동일한 학습 run과 모델 구조를 사용한다. 다만 현재
`ai-control-260729/config.yml`의 기본 선택이 `step 12000`일 뿐이며, 이것이
`step 16940`의 실기 성공 여부와는 무관하다.

### 이 모델이 실제로 사용하는 입력과 출력

`r_relfeat_only`의 입력·출력 계약은 다음과 같다.

- **입력 카메라:** `image_bottom` 한 대(배포 이름 `orbbec0`)만 사용한다.
- **Goal:** 각 학습 episode의 마지막 docked frame을 고정 goal image로 사용한다.
- **관측 시간:** wheel velocity `(vx, wz)` 60 frame과, 같은 60-frame
  history에서 stride 12로 뽑은 RGB 5 frame을 사용한다.
- **ReLoc3R 특징:** 각 history frame과 episode goal frame을 한 쌍으로
  ReLoc3R decoder에 넣고, pose head 직전의 두 cross-attention stream인
  `dec1`과 `dec2`를 사용한다.
- **사용하지 않는 것:** DINO, 두 번째 카메라(`image_top`/`usb0`), LiDAR,
  ICP auxiliary pose head, 별도 4-D geometry token을 사용하지 않는다.
- **출력:** 30 Hz 기준 미래 60 step의 `(linear velocity, angular velocity)`,
  즉 약 2초 길이의 속도 trajectory를 한 번에 생성한다.

조건 토큰은 다음과 같이 만들어진다.

```text
wheel 60 x 2
  -> MLP + temporal embedding
  -> 60 tokens

5 history frames H_i + one static episode goal G
  -> shared ReLoc3R ViT-L encoder
  -> ReLoc3R cross-attention decoder
     dec1_i = history stream H_i after attending to G
     dec2_i = goal stream G after attending to H_i
  -> each [196, 768] stream is compressed by a Perceiver to 16 tokens
  -> dec1: 5 x 16 = 80 tokens, dec2: 5 x 16 = 80 tokens

total condition sequence = 60 + 80 + 80 = 220 tokens, each 384-D
  -> 4-layer TokenSequenceFusionCondition (unpooled)
  -> 12-layer, 6-head DiTCrossAttn1d
  -> DDPM / ContinuousDiffusionSDE
  -> future action [60, 2]
```

Action DiT는 미래 60개 action token을 non-causal self-attention으로 함께
denoise하고, 매 block에서 위 220개 condition token에 cross-attention한다.
학습 action은 `minmax`로 `[-1, 1]`에 정규화된다. 추론은
`ode_dpmsolver++_2M`, 30 sampling steps가 실기 검증 설정이다.

엄밀히 말하면 현재 stride 구현은 60-frame window의 index
`[0, 12, 24, 36, 48]`을 선택한다. 즉 가장 최근 RGB는 window 마지막보다
11 frame 앞이고, wheel만 마지막 frame까지 포함한다. 재현할 때 RGB index를
`[11, 23, 35, 47, 59]`처럼 바꾸면 성공 모델과 다른 입력이 된다.

### 실기 추론 설정

실기 성공 당시와 현재 `ai-control-260729/config.yml`에 기록된 주요 값은
다음과 같다. `step 12000`과 `step 16940` 모두 이 모델 구조/입력 계약을
유지해야 한다.

```text
demo_backbone=ddpm
demo_steps=30
demo_nsamples=1
demo_use_ema=true
demo_agg=medoid       # sample이 1개라 결과에는 영향 없음
demo_ema=1.0          # trajectory frame EMA 끔
demo_continuity_blend=0.2
window_size=60
inference_size=32
action_horizon=32
require_encoder=true
require_lidar=false
```

배포 시 episode 마지막 training image를 읽는 것이 아니라, 같은 카메라의
실제 docked-position goal image를 지정한다. 이 goal은 시작할 때 한 번
ReLoc3R encoder로 처리되고 매 추론의 5개 history frame과 pair를 이룬다.

### 정확한 학습 재현

아래 명령은 저장소 root
`/home/work/.postech/diffusion_policy_robot_docking`에서 실행한다.

#### 1. 필요한 데이터와 cache 확인

성공 run은 아래 두 파일이 row 단위로 정확히 정렬되어 있다는 전제에서
학습되었다.

```text
dataset/after_0328_train.h5
  episode_ends: 145 episodes / 225465 rows
  encoder:      [225465, 2]
  image_bottom: [225465, 3, 240, 320]

dataset/after_0328_train_reloc3r_bottom.h5
  reloc3r_bottom:      [225465, 196, 1024] float16
  reloc3r_dec1_bottom: [225465, 196,  768] float16
  reloc3r_dec2_bottom: [225465, 196,  768] float16
```

현재 cache가 그대로 있다면 이 단계는 다시 실행하지 않는다. Cache를
원본 HDF5에서 재생성해야 할 때만 다음을 실행한다. 두 번째 명령은 첫 번째
명령이 만든 HDF5에 `dec1/dec2` dataset을 in-place로 추가한다.

```bash
cd /home/work/.postech/diffusion_policy_robot_docking

python scripts/precompute_reloc3r_cache.py \
  --h5 dataset/after_0328_train.h5 \
  --camera image_bottom \
  --out dataset/after_0328_train_reloc3r_bottom.h5

python scripts/precompute_reloc3r_dec_features.py \
  --cache dataset/after_0328_train_reloc3r_bottom.h5 \
  --camera image_bottom
```

두 cache script 모두 episode마다 마지막 frame을 goal로 정하고, 전체
episode를 끝까지 처리한다. `--limit_episodes`는 smoke test용이므로 실제
재현 명령에는 넣지 않는다.

#### 2. 성공 run과 같은 학습 실행

```bash
cd /home/work/.postech/diffusion_policy_robot_docking

python scripts/train.py \
  --config-name smr_rgeo \
  sensors_variant=reloc3r_relfeat_only \
  experiment_name=r_relfeat_only
```

이 명령은 현재 config 기준으로 저장된 성공 run과 동일하게 다음 설정을
합성한다.

```text
seed=0, device=cuda:0
batch_size=256, num_epochs=20
learning_rate=1e-4, weight_decay=1e-5
action_norm=minmax
diffusion_backbone=ddpm
d_model=384, n_heads=6, depth=12
condition_num_layers=4, dropout=0.1
ema_rate=0.999
horizon=60, obs_horizon=60
```

현재 dataset에서는 학습 sample이 `216910`개이고 `drop_last=true`이므로
epoch당 `floor(216910 / 256) = 847` gradient steps, 총
`20 x 847 = 16940` steps가 된다. 따라서 성공 run의 최종 checkpoint가
`step 16940`인 것도 재현 조건의 일부다. 데이터 행 수, episode 수,
batch size 중 하나라도 바뀌면 최종 step은 달라진다.

Hydra는 새 결과를 다음 형식의 별도 timestamp 디렉터리에 저장한다.

```text
outputs/train/r_relfeat_only/YYYY-MM-DD_HH-MM-SS/
```

#### 3. 기존 성공 artifact 무결성 확인

```bash
cd /home/work/.postech/diffusion_policy_robot_docking

sha256sum \
  outputs/train/r_relfeat_only/2026-07-28_22-47-34/r_relfeat_only_step_12000.pt \
  outputs/train/r_relfeat_only/2026-07-28_22-47-34/r_relfeat_only_step_16940.pt \
  outputs/train/r_relfeat_only/2026-07-28_22-47-34/r_relfeat_only_step_12000.config.yaml
```

기대값:

```text
803ab3cc494d17e241a2b806da93ab2bfb24baa874f95326ea40cd0be0a51152  r_relfeat_only_step_12000.pt
a298f6a71f0ab579eb7b161a2b5be993630e7bc65b00987d69f7db3ec93cf6a5  r_relfeat_only_step_16940.pt
d0cfc35afb523e87a5c750a32d453284e72bed67db29794af00e616895f2db08  r_relfeat_only_step_12000.config.yaml
```

### `dec1`과 `dec2`가 같은 file을 보는 이유

두 sensor가 같은 `file`을 가리키는 것은 중복이나 오류가 아니다. 하나의
HDF5 sidecar 안에 **서로 다른 dataset key**를 함께 저장한 것이다.

```yaml
reloc3r_dec1:
  file:   dataset/after_0328_train_reloc3r_bottom.h5
  source: reloc3r_dec1_bottom

reloc3r_dec2:
  file:   dataset/after_0328_train_reloc3r_bottom.h5
  source: reloc3r_dec2_bottom
```

Episode의 고정 goal을 `G = last_frame(episode)`라 하고 sampled history
frame을 `H_i`라 하면, cache는 매 row마다 다음 두 값을 저장한다.

```text
(dec1_i, dec2_i) = ReLoc3RDecoder(Encoder(H_i), Encoder(G))
```

- `dec1_i`는 `G`를 본 history-side stream이다.
- `dec2_i`는 `H_i`를 본 goal-side stream이다.

따라서 **goal image 자체는 episode 마지막 frame 하나만 주면 되고, 현재
코드도 이미 그렇게 동작한다.** Offline cache는 마지막 frame의 encoder
feature를 episode 내에서 재사용하고, live 배포 코드도 goal image를 시작할
때 한 번만 encode한다.

반면 `dec2`의 **출력**은 goal만의 정적 feature가 아니다. 같은 goal
`G`라도 상대 history frame `H_i`가 달라질 때 cross-attention 결과
`dec2_i`도 달라진다. 마지막 row의 `dec2(G, G)` 하나를 5개 시점에
broadcast하면 `H_i`에 대한 관계 정보가 사라지므로 현재 모델과 동등하지
않고, 기존 checkpoint에도 사용할 수 없다.

`dec2`가 실제로 불필요한지는 가능한 ablation 질문이지만, 그것은 다음 중
하나로 **별도 모델을 재학습해서** 비교해야 한다.

1. `dec1`만 유지하는 모델
2. `dec2`만 유지하는 모델
3. 현재의 `dec1 + dec2` 성공 모델

현재 성공률 100%를 재현하는 목적에서는 `dec1 + dec2`를 그대로 유지한다.

## 현재 모델 코드 맵

```text
configs/robot/
├── smr_rgeo.yaml
│   └── DDPM, DiT, optimizer, batch/epoch 및 공통 학습 설정
└── sensors_variant/reloc3r_relfeat_only.yaml
    └── wheel + reloc3r_dec1 + reloc3r_dec2 입력 계약

scripts/
├── precompute_reloc3r_cache.py
│   └── image_bottom -> ReLoc3R ViT-L encoder cache
├── precompute_reloc3r_dec_features.py
│   └── history/goal pair -> dec1/dec2 cache
└── train.py
    └── Hydra 학습 entrypoint

utils/
├── modular_dataset.py
│   └── row-aligned HDF5 history/action sampling
└── setups.py
    └── TokenSequenceFusionCondition + DiTCrossAttn1d + DDPM 생성

cleandiffuser/
├── nn_condition/modality_encoders.py
│   └── wheel 및 reloc3r_relation encoder
├── nn_condition/token_sequence_condition.py
│   └── 220개 condition token의 self-attention fusion
├── nn_diffusion/dit.py
│   └── DiTCrossAttn1d action denoiser
└── diffusion/diffusionsde.py
    └── ContinuousDiffusionSDE(DDPM) 학습/샘플링

reloc3r/
└── vendored ReLoc3R-224 implementation

ai-control-260729/
├── config.yml
│   └── 실기 checkpoint와 sampler/runtime 설정
└── ai_models/
    ├── reloc3r_geometry.py
    │   └── goal을 한 번 encode하고 live dec1/dec2 생성
    └── plugins/run_postech_docking_demo.py
        └── live sensor context, DDPM sampling, trajectory command 생성

outputs/train/r_relfeat_only/2026-07-28_22-47-34/
└── 실기 성공 checkpoint, 저장 config, metrics
```

## 환경 설정

성공 run은 `device=cuda:0`에서 실행되었다. 정확한 GPU/CUDA/PyTorch 버전이
checkpoint에 저장되지는 않으므로, 아래는 저장소의 지원 범위이고 새로운
학습 결과가 기존 checkpoint와 bit-for-bit 동일함을 보장하지는 않는다.

- Python 3.11+
- CUDA 사용 가능한 PyTorch 2.0+
- 대용량 HDF5 cache를 저장할 충분한 디스크 공간
- ReLoc3R-224 checkpoint `siyan824/reloc3r-224`에 접근 가능한 Hugging Face
  cache 또는 최초 다운로드 환경

```bash
conda create -n diffusion_policy python=3.11 -y
conda activate diffusion_policy

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

설치 후에는 앞의 **정확한 학습 재현** 절에 있는 두 cache 명령과 명시적인
Hydra 명령을 사용한다. 인자 없이 `python scripts/train.py`를 실행하면 다른
기본 config가 선택되므로 `r_relfeat_only` 재현 명령이 아니다.

## 데이터 계약

현재 모델 학습에 직접 필요한 데이터는 다음뿐이다.

- Main HDF5: `episode_ends`, `encoder`
- ReLoc3R sidecar: `episode_ends`, `reloc3r_dec1_bottom`,
  `reloc3r_dec2_bottom`
- Cache 재생성 시에만 필요한 main HDF5 입력: `image_bottom`
- Action target: main HDF5의 미래 `encoder` 60 frame

`image_top`, DINO cache, LiDAR, `dock_pose`, `reliable`, ICP label은
`r_relfeat_only`의 조건 입력이나 loss에 사용되지 않는다. Main HDF5와
ReLoc3R sidecar는 `episode_ends`와 전체 row 수가 반드시 일치해야 한다.
현재 성공 데이터의 정확한 shape는 앞의 재현 절에 기록되어 있다.

## 실기 배포

실기 경로의 기준 파일은 다음과 같다.

- `ai-control-260729/config.yml`: checkpoint와 runtime knob
- `ai-control-260729/ai_models/plugins/run_postech_docking_demo.py`: 추론 plugin
- `ai-control-260729/ai_models/reloc3r_geometry.py`:
  `Reloc3rRelationFeatures`
- `ai-control-260729/ai_models/cleandiffuser/`: 배포용 모델 코드

Token-sequence checkpoint는 weight만으로 temporal sensor 설정을 복구할 수
없으므로 checkpoint와 그 학습 config를 함께 배포해야 한다. Plugin은 다음
순서로 config를 찾는다.

```text
<checkpoint_stem>.config.yaml
<checkpoint_stem>_config.yaml
config.yaml
```

`step 16940`을 배포할 때도 `step 12000`과 동일한 `r_relfeat_only` 학습
config를 해당 checkpoint 옆에 두어야 한다. 실기 전에는 명령 전송을 끈
상태에서 카메라 stream, docked-position goal image, 60-frame window와 생성
trajectory를 먼저 확인한다.

## 다른 실험 브랜치

저장소에는 DINO, 두 카메라, LiDAR, ICP auxiliary head, explicit geometry,
Rectified Flow를 사용하는 과거 실험 코드와 config도 남아 있다. 이들은
ablation 및 기록 보존용이며 현재까지의 실기 성공 모델 설명이나 위 재현
명령에 포함되지 않는다. 관련 역사와 비교 결과는 `docs/` 아래 문서를
참조한다.

## Acknowledgments

This project builds on the following excellent work:

- **[ReLoc3R](https://github.com/ffrivera0/reloc3r)** -- Relative camera pose regression and the bidirectional decoder used for relational visual tokens
- **[CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)** -- Diffusion-model components for decision-making
- **[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)** -- Visuomotor policy learning with diffusion models
