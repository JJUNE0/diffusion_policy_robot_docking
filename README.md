<div align="center">

# Diffusion Policy for Autonomous Robot Docking

**Single-Camera ReLoc3R Relational Features + DDPM DiT for Real-Robot Docking**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

이 저장소는 **`r_relfeat_only`** 하나를 위한 코드다. 카메라 한 대(`orbbec0`),
wheel velocity history, 그리고 ReLoc3R의 양방향 cross-attention decoder token을
조건으로 60-step 속도 trajectory를 DDPM DiT로 생성한다. 현재까지 실기 도킹에
성공한 유일한 모델이며, 나머지 sensors_variant는 이 모델의 통제 ablation이다.

| | |
|---|---|
| **Task** | 차동구동 이동로봇의 충전 스테이션 자동 도킹 |
| **Input** | `orbbec0` RGB history + docked goal RGB + wheel velocity history |
| **Output** | 60-step `(linear velocity, angular velocity)` trajectory (~2s @ 30Hz) |
| **Vision** | Frozen ReLoc3R-224 encoder + bidirectional decoder feature (`dec1`, `dec2`) |
| **Policy** | Token-sequence fusion → cross-attention DiT → DDPM |
| **Training** | Offline imitation learning, 20 epoch / 16940 step |
| **Run** | `outputs/train/r_relfeat_only/2026-07-28_22-47-34` |

> **실기 결과(사용자 보고, 2026-07-30):** 검증한 `step 12000`, `step 16940`
> 두 checkpoint 모두 수행한 모든 실기 시도에 성공했다. 시험 횟수가 기록되어
> 있지 않으므로 통계적 신뢰구간을 의미하는 결과로 확대 해석하지 않는다.

## 입출력 계약

- **카메라:** `image_bottom` 한 대만 사용. `image_top`/`usb0`는 쓰지 않는다.
- **Goal:** 각 episode의 마지막 docked frame 하나가 고정 goal image다.
- **관측:** wheel velocity `(vx, wz)` 60 frame + 같은 window에서 stride 12로
  뽑은 RGB 5 frame.
- **ReLoc3R:** history frame과 goal frame을 pair로 decoder에 넣고, pose head
  **직전**의 두 cross-attention stream `dec1`/`dec2`를 사용한다.
- **출력:** 미래 60 step의 `(v, w)`를 한 번에 생성.

```text
wheel 60 x 2  -> MLP + temporal embedding                        -> 60 tokens

5 history frames H_i + static goal G
  -> shared ReLoc3R ViT-L encoder -> ReLoc3R cross-attention decoder
       dec1_i = H_i stream after attending to G   ("goal-aware history")
       dec2_i = G   stream after attending to H_i ("current-aware goal")
  -> 각 [196, 768] stream을 Perceiver로 16 token 압축
  -> dec1: 5 x 16 = 80 tokens,  dec2: 5 x 16 = 80 tokens

condition sequence = 60 + 80 + 80 = 220 tokens x 384-D
  -> 4-layer TokenSequenceFusionCondition (unpooled)
  -> 12-layer 6-head DiTCrossAttn1d  -> DDPM(ContinuousDiffusionSDE)
  -> action [60, 2]
```

Action DiT는 60개 action token을 non-causal self-attention으로 함께 denoise하고
매 block에서 위 220개 condition token에 cross-attention한다. Action은 `minmax`로
`[-1, 1]` 정규화, 추론은 `ode_dpmsolver++_2M` / 30 step이 실기 검증 설정이다.

`dec1`과 `dec2`는 같은 sidecar 파일의 서로 다른 dataset key다(중복 아님).
`dec2`는 goal의 정적 feature가 **아니라** 상대 history frame마다 달라지는
cross-attention 출력이므로, 한 row를 broadcast해 대체할 수 없다.

> **재현 주의:** 현재 stride 구현은 60-frame window의 index
> `[0, 12, 24, 36, 48]`을 고른다. 가장 최근 RGB가 window 끝보다 11 frame
> 앞이고 wheel만 마지막 frame을 포함한다. `[11, 23, 35, 47, 59]`로 바꾸면
> 성공 모델과 다른 입력이 된다.

## 재현

### 1. 데이터와 cache

```text
dataset/after_0328_train.h5
  episode_ends: 145 episodes / 225465 rows
  encoder:      [225465, 2]          image_bottom: [225465, 3, 240, 320]

dataset/after_0328_train_reloc3r_bottom.h5
  reloc3r_bottom:      [225465, 196, 1024] f16
  reloc3r_dec1_bottom: [225465, 196,  768] f16
  reloc3r_dec2_bottom: [225465, 196,  768] f16
```

두 파일은 row 단위로 정확히 정렬되어 있어야 한다. cache가 이미 있으면 이
단계는 건너뛴다. 재생성이 필요할 때만(두 번째 명령이 `dec1/dec2`를 첫 번째
파일에 in-place 추가한다):

```bash
python scripts/precompute_reloc3r_cache.py \
  --h5 dataset/after_0328_train.h5 --camera image_bottom \
  --out dataset/after_0328_train_reloc3r_bottom.h5

python scripts/precompute_reloc3r_dec_features.py \
  --cache dataset/after_0328_train_reloc3r_bottom.h5 --camera image_bottom
```

`--limit_episodes`는 smoke test용이므로 재현 명령에는 넣지 않는다.

### 2. 학습

```bash
python scripts/train.py --config-name smr_rgeo \
  sensors_variant=reloc3r_relfeat_only experiment_name=r_relfeat_only
```

```text
seed=0, device=cuda:0, batch_size=256, num_epochs=20
learning_rate=1e-4, weight_decay=1e-5, ema_rate=0.999, action_norm=minmax
diffusion_backbone=ddpm, d_model=384, n_heads=6, depth=12
condition_num_layers=4, dropout=0.1, horizon=60, obs_horizon=60
```

학습 sample `216910`개에 `drop_last=true`이므로 epoch당
`floor(216910/256) = 847` step, 총 `20 x 847 = 16940` step이다. 즉 최종
checkpoint가 `step 16940`인 것도 재현 조건의 일부이며, 행 수·episode 수·batch
size 중 하나만 바뀌어도 달라진다. 결과는
`outputs/train/r_relfeat_only/YYYY-MM-DD_HH-MM-SS/`에 저장된다.

### 3. 기존 artifact 무결성

```bash
cd outputs/train/r_relfeat_only/2026-07-28_22-47-34
sha256sum r_relfeat_only_step_{12000,16940}.pt r_relfeat_only_step_12000.config.yaml
```

```text
803ab3cc494d17e241a2b806da93ab2bfb24baa874f95326ea40cd0be0a51152  ..._step_12000.pt
a298f6a71f0ab579eb7b161a2b5be993630e7bc65b00987d69f7db3ec93cf6a5  ..._step_16940.pt
d0cfc35afb523e87a5c750a32d453284e72bed67db29794af00e616895f2db08  ..._step_12000.config.yaml
```

## 데이터 계약

| 필요한 것 | 어디서 |
|---|---|
| `episode_ends`, `encoder` | main HDF5 |
| `reloc3r_dec1_bottom`, `reloc3r_dec2_bottom`, `episode_ends` | ReLoc3R sidecar |
| `image_bottom` | main HDF5 (cache 재생성 시에만) |
| action target | main HDF5의 미래 `encoder` 60 frame |

main HDF5와 sidecar는 `episode_ends`와 전체 row 수가 반드시 일치해야 한다.
`image_top`, `dock_pose`, `reliable`은 조건 입력이나 loss에 쓰이지 않는다
(`dock_pose`는 `test/eval_align_rgeo.py`의 계측 용도로만 읽힌다).

## 코드 맵

```text
configs/robot/smr_rgeo.yaml            공통 학습 설정 (기본 arm = reloc3r_relfeat_only)
configs/robot/sensors_variant/         arm별 입력 계약. relfeat_only가 실기 검증 모델,
  reloc3r_*.yaml                       pose_only/posthead_only/dec1·dec2/goal_pool은 ablation

scripts/precompute_reloc3r_cache.py    image_bottom -> ViT-L encoder cache
scripts/precompute_reloc3r_dec_features.py   history/goal pair -> dec1/dec2
scripts/precompute_reloc3r_head_features.py  pose head 이후 tap (ablation용)
scripts/train.py                       Hydra 학습 entrypoint

utils/modular_dataset.py               row-aligned HDF5 history/action sampling
utils/setups.py                        condition + DiT + diffusion backbone 생성
utils/inference.py                     저장된 config로 학습 시점 network 재구성

cleandiffuser/nn_condition/modality_encoders.py       motion / reloc3r_relation / reloc3r_goal_pair
cleandiffuser/nn_condition/token_sequence_condition.py 220 condition token fusion
cleandiffuser/nn_diffusion/dit.py                      DiTCrossAttn1d
cleandiffuser/diffusion/diffusionsde.py                DDPM 학습/샘플링

test/test_reloc3r_*.py                 shape / no-future-leak / padding mask / 좌표계
test/eval_run_rgeo.py, eval_align_rgeo.py, terminal_metric.py
                                       held-out open-loop, terminal-band, alignment 평가

reloc3r/                               vendored ReLoc3R-224
```

## 환경

- Python 3.11+, CUDA 지원 PyTorch 2.0+
- ReLoc3R-224 checkpoint `siyan824/reloc3r-224` (HF cache 또는 최초 다운로드)
- 대용량 HDF5 cache를 담을 디스크

```bash
conda create -n diffusion_policy python=3.11 -y && conda activate diffusion_policy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

인자 없이 `python scripts/train.py`를 실행하면 config 기본값으로 돌아가므로,
재현할 때는 위의 명시적 Hydra 명령을 쓴다.

## 실기 배포

배포 코드는 별도 저장소(`ai-control-*`)로 분리되었다. 실기 검증 설정은
`ddpm` / 30 sampling step / 1 sample / EMA weight / `window_size=60` /
`action_horizon=32`이고, goal은 episode 마지막 학습 이미지가 아니라 같은
카메라로 찍은 **실제 docked-position 이미지**를 지정한다. 이 goal은 시작 시
한 번 ReLoc3R encoder를 통과한 뒤 매 추론의 history 5 frame과 pair를 이룬다.

Token-sequence checkpoint는 weight만으로 temporal sensor 설정을 복구할 수
없으므로 checkpoint와 그 학습 config(`<checkpoint_stem>.config.yaml`)를 반드시
함께 배포한다. 실기 전에는 명령 전송을 끈 상태에서 카메라 stream, goal image,
60-frame window, 생성 trajectory를 먼저 확인한다.

## 제거된 과거 실험

DINO appearance branch, 2-camera 전용 arm, LiDAR point branch, ICP auxiliary
pose head(`endgame/`), 4-D geometry token, pooled `ModularSensorFusionCondition`
및 legacy `SensorFusionConditionNetwork` 경로와 그에 딸린 config·eval·queue
script는 2026-08-11에 삭제되었다. Rectified Flow는 `diffusion_backbone` flag로
남아 있다. 삭제 직전 상태는 git tag `pre-cleanup-2026-08-11`로 복구할 수 있고,
실험 기록과 비교 결과는 `docs/` 아래 문서에 남아 있다.

## Acknowledgments

- **[ReLoc3R](https://github.com/ffrivera0/reloc3r)** — relative pose regression과 relational visual token을 제공하는 bidirectional decoder
- **[CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)** — 의사결정용 diffusion 모델 구성요소
- **[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)** — diffusion 기반 visuomotor policy learning
