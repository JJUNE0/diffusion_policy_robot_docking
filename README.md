<div align="center">

# Diffusion Policy for Autonomous Robot Docking

**Learning Multimodal Visuomotor Policies via Rectified Flow for Charging Station Docking**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

A mobile robot learns to autonomously dock at a charging station by observing two room cameras and its own wheel encoder velocities. A **Diffusion Transformer (DiT)**, trained as a **Rectified Flow** velocity-field model, generates future velocity trajectories conditioned on fused vision--motion representations, producing smooth and reliable docking behaviors from offline demonstration data. The straight-line flow is integrated with a simple **Euler ODE**, enabling high-quality generation in very few sampling steps.

## Overview

| | |
|---|---|
| **Task** | Autonomous charging-station docking for a differential-drive mobile robot |
| **Input** | Two RGB camera streams (room1, room2) + wheel encoder velocity history |
| **Output** | 60-step future velocity trajectory (linear vel, angular vel) |
| **Method** | Conditional Rectified Flow policy with multimodal sensor fusion |
| **Training** | Offline imitation learning from expert demonstrations |

## Method

### Pipeline

```
                              ┌─────────────────────────────────────────────┐
                              │          Observation History (T=30)         │
                              └──────┬──────────────┬──────────────┬───────┘
                                     │              │              │
                              Room 1 Images   Room 2 Images   Encoder Vel.
                              [5, 3, 240, 320] [5, 3, 240, 320]  [30, 2]
                                     │              │              │
                              ┌──────▼──────┐┌──────▼──────┐┌─────▼──────┐
                              │  Frozen     ││  Frozen     ││   MLP +    │
                              │  DINO-v3    ││  DINO-v3    ││  Pos.Enc.  │
                              │  ViT-B/16   ││  ViT-B/16   ││            │
                              └──────┬──────┘└──────┬──────┘└─────┬──────┘
                              [5, 196, 768]  [5, 196, 768]    [30, 384]
                                     │              │              │
                              ┌──────▼──────┐┌──────▼──────┐       │
                              │  Perceiver  ││  Perceiver  │       │
                              │  Resampler  ││  Resampler  │       │
                              │ (16 latents)│| (16 latents)│       │
                              └──────┬──────┘└──────┬──────┘       │
                                [5, 16, 384]  [5, 16, 384]         │
                                     │              │              │
                                     └──────┬───────┘              │
                                            │                      │
                              ┌─────────────▼──────────────────────▼──────┐
                              │       Sensor Fusion Transformer           │
                              │     (Cross-Attention, 2 layers)           │
                              └─────────────────────┬─────────────────────┘
                                                    │
                                            Condition Vector
                                                    │
                              ┌─────────────────────▼─────────────────────┐
                              │        DiT1d Velocity Field               │
                              │    (12 blocks, d=384, 6 heads)            │
                              │                                           │
                              │   x_1 ──► Euler ODE ──► ... ──► x_0       │
                              │   (noise)              (trajectory)       │
                              └─────────────────────┬─────────────────────┘
                                                    │
                                     Rectified Flow (Euler ODE)
                                         20 sampling steps
                                                    │
                                                    ▼
                                    Predicted Velocity Trajectory
                                            [60, 2]
                                       (v_linear, v_angular)
```

### Key Design Choices

- **Rectified Flow generative model** -- Instead of a score-based diffusion process, the DiT is trained to regress the straight-line velocity field between noise and data (`x0 - x1`). The learned flow is nearly straight, so a plain **Euler ODE** produces high-quality trajectories in far fewer steps (~5-20) than DPM-Solver++ diffusion sampling. Backbone, conditioning, and training pipeline are otherwise unchanged, making this a clean drop-in swap via CleanDiffuser's `ContinuousRectifiedFlow`.

- **Frozen DINOv2 backbone** -- No fine-tuning needed. The pretrained ViT-B/16 provides rich spatial features (196 patches x 768-dim per frame) that generalize well to the indoor docking environment.

- **Sparse temporal vision sampling** -- From 30 observation steps, only 5 frames (stride=6) are fed to the vision branch. This reduces computation 6x while preserving sufficient visual context for docking alignment.

- **Perceiver Resampler** -- Compresses 196 DINO patches per frame into 16 latent tokens via cross-attention, reducing the vision sequence from 980 tokens to 80 tokens per camera.

- **Velocity-conditioned fusion** -- The encoder velocity history provides proprioceptive grounding, enabling the policy to reason about the robot's current motion state alongside visual observations.

- **Classifier-Free Guidance (CFG)** -- Supports conditional dropout (p=0.1) during training for optional guidance-weighted sampling at inference time.

## Project Structure

```
.
├── cleandiffuser/               # Diffusion model framework (modified CleanDiffuser)
│   ├── diffusion/               #   DDPM, SDE, EDM, DPM-Solver, Rectified Flow, ...
│   ├── nn_diffusion/            #   DiT1d, UNet, Transformer denoiser architectures
│   ├── nn_condition/            #   SensorFusionConditionNetwork, image/MLP conditioners
│   ├── dataset/                 #   Base dataset classes and utilities
│   └── utils/                   #   Transformer blocks, normalizers, tensor ops
│
├── configs/
│   └── robot/smr.yaml           # Hydra config: architecture, training, inference
│
├── dataset/
│   ├── docking_dataset.py       # HDF5 multimodal dataset loader
│   └── preprocessing.py         # Raw episode data --> HDF5 converter
│
├── dino/
│   ├── dino_detector.py         # DINOv2 feature extraction + heatmap visualization
│   └── master_vector.pt         # Pre-computed reference vector for similarity
│
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── inference_ema.py         # Inference with EMA trajectory smoothing
│   ├── inference_rtc.py         # Inference with ranking-based trajectory selection
│   └── chunk_transfer.py        # Split/rejoin large datasets for cloud upload (+sha256 verify)
│
├── utils/
│   ├── setups.py                # Model & logger initialization
│   └── utils.py                 # Logger, RK4 trajectory reconstruction, plotting
│
├── rc_server/                   # Real-time control server (ZMQ-based robot interface)
├── requirements.txt
└── README.md
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
# Run from the repository root -- config paths resolve against the launch directory
python scripts/train.py
```

Training runs for `num_epochs` epochs; the total gradient steps are derived automatically from the dataloader length (`num_epochs x floor(len(dataset) / batch_size)`). A final checkpoint is always saved when training ends.

All hyperparameters are managed via [Hydra](https://hydra.cc/) and can be overridden from the command line:

```bash
# Custom training run
python scripts/train.py batch_size=64 num_epochs=10 learning_rate=5e-5

# Resume from checkpoint
python scripts/train.py resume_path=/path/to/checkpoint_step_100000.pt
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
| `batch_size` | 64 | Training batch size |
| **Architecture** | | |
| `d_model` | 384 | Transformer hidden dimension |
| `n_heads` | 6 | Number of attention heads |
| `depth` | 12 | DiT block depth |
| `dropout` | 0.1 | CFG condition masking probability |
| **Training** | | |
| `num_epochs` | 10 | Number of epochs (auto-converted to gradient steps) |
| `diffusion_gradient_steps` | 400,001 | Fallback total steps, used only when `num_epochs` is unset |
| `learning_rate` | 1e-4 | Initial learning rate (cosine decay) |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `ema_rate` | 0.9999 | EMA model decay rate |
| `save_interval` | 10,000 | Checkpoint save frequency |
| **Inference** | | |
| `solver` | `euler` | Euler ODE integrator for Rectified Flow sampling |
| `inference_sampling_steps` | 20 | Number of Euler ODE steps (Rectified Flow needs only a few) |
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
