<div align="center">

# Diffusion Policy for Autonomous Robot Docking

**Learning Multimodal Visuomotor Policies via Score-Based Diffusion for Charging Station Docking**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

A mobile robot learns to autonomously dock at a charging station by observing two room cameras and its own wheel encoder velocities. A **Diffusion Transformer (DiT)** generates future velocity trajectories conditioned on fused vision--motion representations, producing smooth and reliable docking behaviors from offline demonstration data.

## Overview

| | |
|---|---|
| **Task** | Autonomous charging-station docking for a differential-drive mobile robot |
| **Input** | RGB (room1, room2) + encoder velocity + **raw 2D LiDAR points** + goal feature |
| **Output** | 60-step velocity trajectory **+ ICP-distilled dock pose** (precision/arrival, aux head) |
| **Method** | Conditional diffusion policy with multimodal sensor fusion |
| **Training** | Offline imitation learning from expert demonstrations |

## Method

### Pipeline

단일 모델(Option A): 네 모달리티를 토큰으로 융합 → (주) 속도 궤적 + (aux) **ICP-distill 도크 포즈**.
ICP는 **오프라인 교사**(라벨)일 뿐 런타임엔 비전만 동작.

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
        │   DiT1d Denoiser   │        │   ICP aux head         │
        │ + DPM-Solver++(ODE)│        │  (dock pose x,y,θ)     │
        └─────────┬──────────┘        └─────────┬──────────────┘
                  ▼                             ▼
        Velocity trajectory [60,2]     Dock pose  →  정밀/도착 판정(≤1cm)
        (v_linear, v_angular)          (ICP-distilled, 런타임 ICP 없음)
```

> 나중(`docs/plan/00_overview.md` §5): goal을 **멀티 sub-goal DINO 정합**으로 확장 + anomaly 헤드.

### Key Design Choices

- **Frozen DINOv2 backbone** -- No fine-tuning needed. The pretrained ViT-B/16 provides rich spatial features (196 patches x 768-dim per frame) that generalize well to the indoor docking environment.

- **Sparse temporal vision sampling** -- From 30 observation steps, only 5 frames (stride=6) are fed to the vision branch. This reduces computation 6x while preserving sufficient visual context for docking alignment.

- **Perceiver Resampler** -- Compresses 196 DINO patches per frame into 16 latent tokens via cross-attention, reducing the vision sequence from 980 tokens to 80 tokens per camera.

- **Velocity-conditioned fusion** -- The encoder velocity history provides proprioceptive grounding, enabling the policy to reason about the robot's current motion state alongside visual observations.

- **Classifier-Free Guidance (CFG)** -- Supports conditional dropout (p=0.1) during training for optional guidance-weighted sampling at inference time.

## Project Structure

> 단일 모델(Option A): 하나의 diffusion 정책이 **DINO(room1/2) + encoder velocity + raw LiDAR 점 + goal feature**
> 를 받아 (주) 미래 속도 궤적과 (aux) **ICP-distill 도크 포즈**를 출력. ICP는 **오프라인 라벨 생성에만** 쓰이고
> 런타임엔 비전만 동작. 자세한 설계·진행은 `docs/plan/00..05_*.md`, `docs/CLAUDE.md`.

```
.
├── cleandiffuser/                  # Diffusion 프레임워크 (modified CleanDiffuser)
│   ├── diffusion/                  #   DDPM/SDE/DPM-Solver 등 (ContinuousDiffusionSDE: loss/update/sample)
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
python dataset/preprocessing.py \
    --data_root /path/to/data_root \
    --save_path dataset/validation_dataset1.h5
```

The resulting HDF5 contains:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `encoder` | `[N, 2]` | Wheel encoder velocities (vx, wz) |
| `image_top` | `[N, 3, 240, 320]` | Room 1 camera frames |
| `image_bottom` | `[N, 3, 240, 320]` | Room 2 camera frames |
| `episode_ends` | `[E]` | Episode boundary indices |

## Training

```bash
cd scripts
python train.py
```

All hyperparameters are managed via [Hydra](https://hydra.cc/) and can be overridden from the command line:

```bash
# Custom training run
python train.py batch_size=32 learning_rate=5e-5 diffusion_gradient_steps=200000

# Resume from checkpoint
python train.py resume_path=/path/to/checkpoint_step_100000.pt
```

### Configuration Reference

<details>
<summary><b>Full hyperparameter table</b> (click to expand)</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Data** | | |
| `horizon` | 60 | Future trajectory length (steps) |
| `obs_horizon` | 30 | Observation history window |
| `vision_stride` | 6 | Temporal subsampling for vision (30/6 = 5 frames) |
| `batch_size` | 16 | Training batch size |
| **Architecture** | | |
| `d_model` | 384 | Transformer hidden dimension |
| `n_heads` | 6 | Number of attention heads |
| `depth` | 12 | DiT block depth |
| `dropout` | 0.1 | CFG condition masking probability |
| **Training** | | |
| `diffusion_gradient_steps` | 400,001 | Total gradient steps |
| `learning_rate` | 1e-4 | Initial learning rate (cosine decay) |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `ema_rate` | 0.9999 | EMA model decay rate |
| `save_interval` | 10,000 | Checkpoint save frequency |
| **Inference** | | |
| `solver` | `ode_dpmsolver++_2M` | ODE solver for sampling |
| `inference_sampling_steps` | 100 | Number of denoising steps |
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
