<div align="center">

# Diffusion Policy for Autonomous Robot Docking

**Learning Multimodal Visuomotor Policies via Rectified Flow, with LiDAR-ICP-Distilled Precision Docking**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

A mobile robot learns to autonomously dock at a charging station by observing two room cameras, its wheel encoder velocities, and a raw 2D LiDAR scan. A **Diffusion Transformer (DiT)**, trained as a **Rectified Flow** velocity-field model and integrated with a simple **Euler ODE**, generates future velocity trajectories conditioned on fused vision--motion--LiDAR representations. A single model handles the whole approach-to-dock path: the main head outputs the velocity trajectory, while an **ICP-distilled auxiliary head** regresses the dock pose (`x, y, θ`) from the raw LiDAR points for mm-level precision/arrival judgment — so no online ICP runs at deployment.

## Overview

| | |
|---|---|
| **Task** | Autonomous charging-station docking for a differential-drive mobile robot |
| **Input** | RGB (room1, room2) + encoder velocity + **raw 2D LiDAR points** + goal feature |
| **Output** | 60-step velocity trajectory **+ ICP-distilled dock pose** (precision/arrival, aux head) |
| **Method** | Conditional **Rectified Flow** policy + LiDAR-ICP-distilled precision head |
| **Training** | Offline imitation learning from expert demonstrations |

## Method

### Pipeline

단일 모델(Option A): 네 모달리티를 토큰으로 융합 → (주) 속도 궤적(**Rectified Flow**, Euler ODE) + (aux) **ICP-distill 도크 포즈**.
ICP는 **오프라인 교사**(라벨)일 뿐 런타임엔 비전+LiDAR 정책만 동작.

```
   Room1 RGB     Room2 RGB     Encoder vel.   raw LiDAR pts     Goal frame(docked)
  [5,3,240,320] [5,3,240,320]    [30,2]      [256,2]+npoints     [3,240,320]
       │             │             │              │                  │
   Frozen        Frozen          MLP        PointNet-style       Frozen DINO
   DINO-v3       DINO-v3      + Pos.Enc.     point encoder       (goal feature)
       │             │             │              │                  │
   Perceiver     Perceiver         │        masked Perceiver     Perceiver
   Resampler     Resampler         │         Resampler          (goal) + NoMaD
       └─────────────┴──────┬──────┴──────────────┴──────────────────┘  mask
                            │   (image / velocity / lidar / goal modality tokens)
                ┌───────────▼────────────┐
                │ Sensor Fusion Transformer│ → readout = Condition Vector [d=384]
                └───────────┬────────────┘
                  ┌─────────┴───────────────────┐
        ┌─────────▼──────────┐        ┌─────────▼──────────────┐
        │   DiT1d Velocity   │        │   ICP aux head         │
        │ + Euler ODE (RF)   │        │  (dock pose x,y,θ)     │
        └─────────┬──────────┘        └─────────┬──────────────┘
                  ▼                             ▼
        Velocity trajectory [60,2]     Dock pose  →  정밀/도착 판정(≤1cm)
        (v_linear, v_angular)          (ICP-distilled, 런타임 ICP 없음)
```

> 나중(`docs/plan/00_overview.md` §5): goal을 **멀티 sub-goal DINO 정합**으로 확장 + anomaly 헤드.

### Key Design Choices

- **Rectified Flow generative backbone** -- The DiT is trained to regress the straight-line velocity field between noise and data (`x0 - x1`) rather than a score. The learned flow is nearly straight, so a plain **Euler ODE** samples high-quality trajectories in very few steps (~5-20) instead of DPM-Solver++ diffusion sampling. Because the precision (aux ICP head) and sensor-fusion branches live entirely in the condition network, this backbone is a clean drop-in swap over the diffusion version via CleanDiffuser's `ContinuousRectifiedFlow` -- the training loop is unchanged.

- **Role-split precision (LiDAR ICP distillation)** -- The learned policy handles robust wide-area approach; a **cross-attention aux head** attends a pose query to the raw LiDAR point tokens and regresses the dock pose, distilling an offline point-to-line ICP teacher. This yields mm-level arrival judgment (≤1 cm) without any runtime ICP, and lets the policy learn from few coarse demos (precision is not imitated).

- **Frozen DINOv2 backbone** -- No fine-tuning needed. The pretrained ViT-B/16 provides rich spatial features (196 patches x 768-dim per frame) that generalize well to the indoor docking environment.

- **Sparse temporal vision sampling** -- From 30 observation steps, only 5 frames (stride=6) are fed to the vision branch. This reduces computation 6x while preserving sufficient visual context for docking alignment.

- **Perceiver Resampler** -- Compresses 196 DINO patches per frame into 16 latent tokens via cross-attention, reducing the vision sequence from 980 tokens to 80 tokens per camera.

- **Offline DINO feature caching (training speedup)** -- The DINO backbone is frozen and the input frames are fixed, so the `[196, 768]` patch features are identical every step/epoch. Recomputing them on the GPU each step was the dominant training bottleneck (~1.5k ViT-B forwards/step). `scripts/precompute_dino_cache.py` runs the backbone **once offline** and stores the features (row-aligned, fp16) in a cache HDF5; training then reads features from disk and skips the backbone entirely. Features are **bit-identical** to the live path (float32 diff = 0), so results are unchanged — only faster. Enabled via `use_dino_cache` / `dino_cache_path`.

- **Single-camera option** -- `use_room1=false` uses only room2 (`image_bottom`), dropping the room1 (`image_top`) vision branch. Halves DINO compute and cache size (~66 GB → for one camera at 225k frames). This is a model change (retrain required); default `true` keeps the two-camera setup.

- **Velocity-conditioned fusion** -- The encoder velocity history provides proprioceptive grounding, enabling the policy to reason about the robot's current motion state alongside visual observations.

- **Classifier-Free Guidance (CFG)** -- Supports conditional dropout (p=0.1) during training for optional guidance-weighted sampling at inference time.

## Project Structure

> 단일 모델(Option A): 하나의 flow-matching(Rectified Flow) 정책이 **DINO(room1/2) + encoder velocity + raw LiDAR 점 + goal feature**
> 를 받아 (주) 미래 속도 궤적과 (aux) **ICP-distill 도크 포즈**를 출력. ICP는 **오프라인 라벨 생성에만** 쓰이고
> 런타임엔 비전만 동작. 자세한 설계·진행은 `docs/plan/00..05_*.md`, `docs/CLAUDE.md`.

```
.
├── cleandiffuser/                  # Diffusion 프레임워크 (modified CleanDiffuser)
│   ├── diffusion/                  #   생성 백본 (ContinuousRectifiedFlow: loss/update/sample, Euler ODE) + DDPM/SDE/DPM-Solver
│   ├── nn_diffusion/               #   DiT1d 디노이저 (adaLN-Zero)
│   ├── nn_condition/
│   │   └── sensor_fusion_condition.py  # ★멀티모달 조건망: DINO room1/2 + velocity
│   │                               #     + raw-LiDAR point 브랜치 + goal 토큰 + ICP aux head
│   ├── dataset/                    #   base dataset 클래스
│   └── utils/                      #   Transformer 블록, normalizer, tensor ops
│
├── endgame/                        # ★오프라인 ICP (라벨 생성 전용, 런타임 미사용)
│   ├── icp_matcher.py              #   raw-point known-shape ICP (point-to-line, aliasing 가드)
│   ├── target_model.py             #   도크 형상 템플릿 (make_template, real_dock 로드)
│   ├── se2.py / config.py          #   SE(2) 유틸 / ICPConfig
│   └── assets/dock_template_real.* #   155개 에피소드로 만든 공식 도크 템플릿
│
├── dino/
│   ├── dino_detector.py            # frozen DINOv3 (facebook/dinov3-vitb16) feature 추출
│   └── master_vector.pt            # 유사도용 기준 벡터
│
├── utils/                          # 데이터 로딩 + 셋업 (preprocessing/loader 여기로 통합)
│   ├── preprocessing.py            # ★원본 에피소드 → 학습 h5 (이미지/encoder/raw-lidar점/ICP 라벨)
│   ├── docking_dataset.py          # ★h5 로더 (sparse-uint8 vision, lidar_points, dock_pose 반환)
│   ├── setups.py                   # 모델/로거 초기화 (model_setups)
│   ├── utils.py                    # Logger, RK4 궤적 복원, plotting
│   └── check_dataset.py / viz_*    # 데이터 점검 / 시각화
│
├── scripts/
│   ├── train.py                    # ★단일 모델 학습 (denoising + ICP aux 합산 손실)
│   ├── eval_heldout.py             # held-out 평가 (dock mm, denoising)
│   ├── plot_training.py            # 학습 수렴 곡선 plot
│   ├── label_subgoals.py           # ★오프라인 ICP 라벨러 → dataset/after_0328/icp_labels/
│   ├── build_dock_template.py      # 공식 도크 템플릿 생성 (endgame/assets/)
│   ├── icp_real_data.py            # 실데이터 ICP 정밀도 검증
│   └── inference_ema.py / _rtc.py  # 배포 추론 (EMA / 랭킹 선택, ZMQ 스트리밍)
│
├── configs/robot/smr.yaml          # Hydra config (architecture/training + lidar/aux/goal/sparse_vision 플래그)
│
├── dataset/
│   ├── after_0328/
│   │   ├── dock/episode_*_dock/    # 원본: room1/2 jpg, encoder.csv, lidar.jsonl, marker_pose.csv
│   │   ├── icp_labels/             # ★ICP 산출 라벨: <ep>.npz(프레임별 도크 포즈=최종골, reliable)
│   │   │                           #   + <ep>.json(sub-goal 마일스톤, handoff onset, success)
│   │   └── after_0328_{train,test}.h5   # 빌드된 학습/테스트 데이터
│   └── new/record_46/              # 신규 포맷(orbbec 깊이+usb) — 미래 깊이 브랜치용
│
├── docs/
│   ├── CLAUDE.md                   # 설계 기준 문서
│   ├── plan/00..05_*.md            # 단일모델 구현 계획·진행(정리→데이터→전처리→학습→결과→plot)
│   ├── design_of_endgame.md        # 전체 진행 로그
│   ├── inference.md / icp_background.md / ai_server_robot_pipeline.md
│
├── rc_server/  plugins/            # 외부 서버 스택 + AI 플러그인 (배포 인프라; 이 repo 핵심 아님)
├── outputs/  results/              # 학습 산출(체크포인트/로그/plot)
└── requirements*.txt / README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA 12.1+ with a GPU (16 GB+ VRAM recommended)

### Installation

```bash
# Create and activate conda environment
conda create -n diffusion_policy python=3.11 -y
conda activate diffusion_policy

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

## Data Preparation

### Expected Raw Data Layout

```
data_root/
└── validation/
    ├── episode_001/
    │   ├── image/
    │   │   ├── room1/            # Top-view camera (*.jpg, timestamped filenames)
    │   │   └── room2/            # Side-view camera (*.jpg, timestamped filenames)
    │   └── encoder.csv           # Columns: ts, vx, wz
    ├── episode_002/
    └── ...
```

### Convert to HDF5

```bash
# Raw episodes -> training h5 with raw LiDAR points + offline ICP dock-pose labels
python utils/preprocessing.py \
    --data_root /path/to/data_root \
    --save_path dataset/after_0328_train.h5 \
    --use_lidar --lidar_format points \
    --with_labels --labels_dir dataset/after_0328/icp_labels
```

> The ICP labels are generated offline first (`scripts/build_dock_template.py` to build the dock template, then `scripts/label_subgoals.py` to label every frame); `--with_labels` bakes them into the h5.

The resulting HDF5 contains:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `encoder` | `[N, 2]` | Wheel encoder velocities (vx, wz) |
| `image_top` | `[N, 3, 240, 320]` | Room 1 camera frames |
| `image_bottom` | `[N, 3, 240, 320]` | Room 2 camera frames |
| `episode_ends` | `[E]` | Episode boundary indices |
| `lidar_points` | `[N, M, 2]` | Raw robot-frame LiDAR points, zero-padded (M=256) |
| `lidar_npoints` | `[N]` | Number of valid points per frame |
| `dock_pose` | `[N, 3]` | ICP dock-pose label `[x, y, θ]` (aux head teacher) |
| `reliable` | `[N]` | 1 = ICP-reliable frame (used to mask the aux loss) |

## Training

```bash
# Run from the repository root -- config paths resolve against the launch directory
python scripts/train.py
```

The single model trains with a **combined loss**: the Rectified Flow denoising loss (main) plus a masked ICP dock-pose loss on the reliable frames (aux). Each step logs both the flow loss and the aux pose error in mm. All hyperparameters are managed via [Hydra](https://hydra.cc/) and can be overridden from the command line:

```bash
# Custom training run
python scripts/train.py batch_size=48 learning_rate=5e-5 diffusion_gradient_steps=140000

# Toggle the precision / sensor-fusion branches
python scripts/train.py use_lidar_points=true use_aux_pose=true use_goal=true aux_weight=1.0

# Resume from checkpoint
python scripts/train.py resume_path=/path/to/checkpoint_step_100000.pt
```

### DINO Feature Caching (optional, big speedup)

Precompute the frozen-DINO features once and train from the cache instead of running the backbone every step.

```bash
# 1) Precompute room2 (image_bottom) DINO features -> cache HDF5 (run once).
#    Reproduces the exact backbone path of DinoBatchDetector.get_heatmap
#    (interpolate->224 bicubic, ImageNet normalize, last_hidden_state[:, 5:, :]);
#    the discarded sim-map / OpenCV path is skipped. Resumable.
python scripts/precompute_dino_cache.py \
    --h5 dataset/after_0328_train.h5 \
    --camera image_bottom \
    --out dataset/after_0328_train_dino_bottom.h5

# 2) Train reading from the cache (skips the DINO backbone entirely).
python scripts/train.py use_dino_cache=true use_room1=false \
    dino_cache_path=dataset/after_0328_train_dino_bottom.h5
```

**Notes**
- The cache is **row-aligned 1:1** with the source image rows and stored fp16, so it must be regenerated if the source h5 changes. Size ≈ `N_frames × 196 × 768 × 2 B` per camera (~66 GB for 225k frames, one camera).
- Both files are used during training: the source h5 still provides encoder velocity / action targets, LiDAR points, and ICP labels; the cache provides only the DINO vision features (`dino_feat_room2`, `goal_feat_room2`). Raw image pixels are no longer read.
- For `use_room1=true` with a cache, precompute the top camera too (`--camera image_top`) and provide a `dino_top` dataset; otherwise set `use_room1=false`.
- Features are bit-identical to the live path, so training dynamics are unchanged — only faster.

### Configuration Reference

<details>
<summary><b>Full hyperparameter table</b> (click to expand)</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Data** | | |
| `horizon` | 60 | Future trajectory length (steps) |
| `obs_horizon` | 30 | Observation history window |
| `vision_stride` | 6 | Temporal subsampling for vision (30/6 = 5 frames) |
| `batch_size` | 48 | Training batch size |
| **Architecture** | | |
| `d_model` | 384 | Transformer hidden dimension |
| `n_heads` | 6 | Number of attention heads |
| `depth` | 12 | DiT block depth |
| `dropout` | 0.1 | CFG condition masking probability |
| **Precision / Sensor Fusion** | | |
| `use_lidar_points` | true | Enable the raw-LiDAR point-set branch |
| `num_lidar_latents` | 16 | Perceiver latents for the LiDAR branch |
| `use_aux_pose` | true | Enable the ICP-distilled dock-pose aux head |
| `aux_weight` | 1.0 | Weight of the masked aux pose loss vs the flow loss |
| `use_goal` | true | Enable goal-feature (docked frame) conditioning |
| `goal_mask_prob` | 0.5 | P(goal active) for NoMaD-style masking |
| `sparse_vision` | true | Dataset returns sparse uint8 frames (RAM saver) |
| `use_room1` | true | Use the room1 (`image_top`) branch; `false` = single camera (room2 only) |
| `use_dino_cache` | false | Read precomputed DINO features from `dino_cache_path` and skip the backbone |
| `dino_cache_path` | null | Path to the cache HDF5 (from `scripts/precompute_dino_cache.py`) |
| **Training** | | |
| `diffusion_gradient_steps` | 140,000 | Total gradient steps |
| `learning_rate` | 1e-4 | Initial learning rate (cosine decay) |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `ema_rate` | 0.9999 | EMA model decay rate |
| `save_interval` | 10,000 | Checkpoint save frequency |
| **Inference** | | |
| `solver` | `euler` | Euler ODE integrator for Rectified Flow sampling |
| `inference_sampling_steps` | 20 | Number of Euler ODE steps (RF needs only a few) |
| `w_cfg` | 1.0 | Classifier-free guidance weight |
| `num_samples` | 8 | Trajectory samples per step |

</details>

## Inference

Two inference strategies are provided for different deployment scenarios:

### EMA Trajectory Smoothing

Applies exponential moving average across consecutive predictions for temporally smooth control:

```bash
cd scripts
python inference_ema.py checkpoint_step=160000
```

### Ranking-based Trajectory Continuity (RTC)

Generates multiple trajectory samples, ranks them by continuity with the previous prediction, and applies exponential blending:

```bash
cd scripts
python inference_rtc.py checkpoint_step=160000
```

Both scripts output trajectory comparison plots and stream predictions via ZMQ for real-time visualization.

## Acknowledgments

This project builds on the following excellent work:

- **[CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)** -- A clean and modular diffusion model library for decision-making
- **[DINOv3](https://github.com/facebookresearch/dinov3)** -- Self-supervised vision transformer for feature extraction
- **[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)** -- Visuomotor policy learning with diffusion models
