<div align="center">

# Diffusion Policy for Autonomous Robot Docking

**Single-Camera ReLoc3R Relational Features + DDPM DiT for Real-Robot Docking**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

| | |
|---|---|
| **Task** | AMR의 충전 스테이션 자동 도킹 |
| **Input** | `orbbec0` RGB history + docked goal RGB + wheel velocity history |
| **Output** | 60-step `(linear velocity, angular velocity)` trajectory (~2s @ 30Hz) |
| **Vision** | Frozen ReLoc3R-224 encoder + bidirectional decoder feature (`dec1`, `dec2`) |
| **Policy** | Token-sequence fusion → cross-attention DiT → DDPM |
| **Training** | Offline imitation learning, 20 epoch / 16940 step |
| **Run** | `outputs/train/r_relfeat_only/2026-07-28_22-47-34` |


## 입출력

- **카메라:** `image_bottom` 한 대만 사용. `image_top`/`usb0`는 쓰지 않는다.
- **Goal:** 각 episode의 마지막 docked frame 하나가 고정 goal image다.
- **관측:** wheel velocity `(vx, wz)` 60 frame + 같은 window에서 stride 12로
  뽑은 RGB 5 frame.
- **ReLoc3R:** history frame과 goal frame을 pair로 decoder에 넣고, pose head
  **직전**의 두 cross-attention stream `dec1`/`dec2`를 사용한다.
- **출력:** 미래 60 step의 `(v, w)`를 한 번에 생성.



Action DiT는 60개 action token을 non-causal self-attention으로 함께 denoise하고
매 block에서 위 220개 condition token에 cross-attention한다. Action은 `minmax`로
`[-1, 1]` 정규화, 추론은 `ode_dpmsolver++_2M` / 30 step이 실기 검증 설정이다.

`dec1`과 `dec2`는 같은 sidecar 파일의 서로 다른 dataset key다(중복 아님).
`dec2`는 goal의 정적 feature가 **아니라** 상대 history frame마다 달라지는
cross-attention 출력이므로, 한 row를 broadcast해 대체할 수 없다.

> **재현 주의:** 현재 stride 구현은 60-frame window의 index
> `[0, 12, 24, 36, 48]`을 고른다.

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


## 학습 파이프라인


### 1. 전처리 — 학습/평가 데이터셋 분리

```bash
python scripts/build_f4hall150_vds.py
```
원본 h5의 앞 150 episode를 zero-copy virtual dataset으로
`dataset/f4hall150_train.h5`에 발행한다. dock camera(`image_bottom` =
orbbec-0, swap 불필요)가 맞는지 pinned sha256 fingerprint + 전 episode
gradient-energy 비교로 검증하고, 어긋나면 조용히 넘어가지 않고 빌드가 실패한다.

```bash
python scripts/build_f4hall150_val_vds.py
```
나머지 held-out episode 151-154를 같은 방식으로 `dataset/f4hall150_val.h5`에
발행한다 (평가 전용, 학습에는 쓰이지 않음).

### 2. 캐싱(precompute) — ReLoc3R feature

```bash
python scripts/precompute_reloc3r_dec_features.py \
  --cache dataset/f4hall150_train_reloc3r_bottom.h5 --camera image_bottom
```
학습 150 episode의 `dec1`/`dec2`(cross-attention decoder token, 각
`[N,196,768]` fp16)를 계산한다. ViT-L encoder token
(`reloc3r_bottom`)은 같은 소스로 이미 계산되어 있던 캐시를 `build_f4hall150_vds.py`가
VDS로 재매핑해 두므로 인코더 pass를 새로 돌릴 필요가 없다.

```bash
python scripts/precompute_reloc3r_cache.py \
  --h5 dataset/f4hall150_val.h5 --camera image_bottom \
  --out dataset/f4hall150_val_reloc3r_bottom.h5

python scripts/precompute_reloc3r_dec_features.py \
  --cache dataset/f4hall150_val_reloc3r_bottom.h5 --camera image_bottom
```

### 3. 학습


```bash
python scripts/train.py --config-name smr_rgeo \
  sensors_variant=reloc3r_relfeat_only_now_f4hall150 \
  experiment_name=r_relfeat_only_now_f4hall150 \
  train_data_path=dataset/f4hall150_train.h5 \
  num_workers=8 prefetch_factor=1 +loader_microbatch_size=16 \
  sensors.reloc3r_dec1.cache_mmap=<memfd 캐시 경로> \
  sensors.reloc3r_dec2.cache_mmap=<memfd 캐시 경로>
```
`sensors_variant`만 바꾸면 다른 dataset 변형이 되도록
`configs/robot/sensors_variant/reloc3r_relfeat_only_now_f4hall150.yaml`이
`train_data_path`와 sidecar 경로 세 줄만 원본(`reloc3r_relfeat_only`)과 다르고
나머지는 `smr_rgeo.yaml` 기본값 그대로다. `cache_mmap`은
`scripts/cache_h5_dataset_memfd.py`가 실행 중에 할당하는
`/proc/<pid>/fd/N` 경로라 고정 명령으로 못 박을 수 없어
`run_f4hall150_pipeline.sh`가 그 경로를 직접 채워 실행한다
(수동으로 하려면 `test/queue_f4hall150_relfeat.sh` 참고). 결과 checkpoint:
`checkpoint_step_18900.pt` (20 epoch, 945 step/epoch, final loss 0.0272).

### 4. 평가

```bash
EVAL_H5=dataset/f4hall150_val.h5 \
EVAL_STATS_H5=dataset/f4hall150_train.h5 \
EVAL_EPISODES=0,1,2,3 EVAL_TAG=heldout \
python test/eval_run_rgeo.py \
  outputs/train/r_relfeat_only_now_f4hall150/2026-08-27_15-30-39
```
held-out 4개 episode(151-154)로 open-loop rollout을 돌려 ADE/FDE/velRMSE를
계산한다. `EVAL_STATS_H5`는 action 정규화 통계(`action_min`/`action_scale`)의
출처이며 반드시 학습에 쓴 h5와 같아야 한다. 결과:
`test/out/rgeo/r_relfeat_only_now_f4hall150_heldout.json`
(ADE 중앙값 11.9cm / FDE 14.3cm / velRMSE 0.0366).

```bash
EVAL_H5=dataset/f4hall150_val.h5 \
EVAL_STATS_H5=dataset/f4hall150_train.h5 \
EVAL_EPISODES=0,1,2,3 EVAL_TAG=heldout \
python test/eval_traj_video_rgeo.py \
  outputs/train/r_relfeat_only_now_f4hall150/2026-08-27_15-30-39 \
  --check-json test/out/rgeo/r_relfeat_only_now_f4hall150_heldout.json
```
같은 rollout 프로토콜(`test/eval_run_rgeo.py`를 그대로 import)로 policy 경로와
demonstration 경로를 겹쳐 그린 PNG와, 카메라 영상 위에 두 경로가 함께 자라나는
MP4를 episode마다 만든다. `--check-json`은 이 스크립트가 계산한
ADE/FDE/velRMSE가 위 평가기의 결과와 정확히 일치하는지 대조해, 두 스크립트의
rollout 로직이 어긋나지 않았음을 보장한다. 결과:
`test/out/rgeo_traj/r_relfeat_only_now_f4hall150_episode_*_dock.{png,mp4}`.

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




## Acknowledgments

- **[ReLoc3R](https://github.com/ffrivera0/reloc3r)** — relative pose regression과 relational visual token을 제공하는 bidirectional decoder
- **[CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)** — 의사결정용 diffusion 모델 구성요소
- **[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)** — diffusion 기반 visuomotor policy learning
