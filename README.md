# Diffusion Policy for Mobile Robot Charging Station Docking

A diffusion-based policy learning framework for autonomous mobile robot docking at charging stations. The system fuses multi-camera vision (DINOv2 features) with encoder velocity history through a Sensor Fusion Conditioning Network, then generates future velocity trajectories via a Diffusion Transformer (DiT).

## Architecture

```
Camera Room1 ──► DINOv2 ──► Perceiver Resampler ──┐
Camera Room2 ──► DINOv2 ──► Perceiver Resampler ──┤
Encoder Velocity ──► MLP + Positional Encoding ───┘
                            │
                   Sensor Fusion (Cross-Attention)
                            │
                      DiT1d Denoiser
                            │
                    DPM-Solver++ (ODE)
                            │
                  Predicted Velocity Trajectory [H, 2]
```

**Key Components:**
- **Vision Backbone**: Frozen DINOv2 ViT-B/16 extracts 196 patch features (768-dim) per frame
- **Sparse Temporal Sampling**: 30-step observation history subsampled at stride 6 (5 vision frames)
- **Conditioning**: Perceiver Resampler compresses DINO features + Transformer-based velocity-vision fusion
- **Denoiser**: DiT1d (12 blocks, 384-dim, 6 heads) predicts 60-step velocity trajectories
- **Diffusion Process**: Continuous SDE with DPM-Solver++ 2M for fast inference (100 steps)

## Project Structure

```
.
├── cleandiffuser/              # Diffusion model framework (modified CleanDiffuser)
│   ├── diffusion/              # Diffusion process implementations (DDPM, SDE, EDM, ...)
│   ├── nn_diffusion/           # Denoiser architectures (DiT, UNet, MLP, ...)
│   ├── nn_condition/           # Conditioning networks (sensor fusion, image, MLP, ...)
│   ├── dataset/                # Dataset base classes and utilities
│   └── utils/                  # Building blocks, normalizers, tensor utilities
├── configs/
│   └── robot/smr.yaml          # Hydra training/inference configuration
├── dino/
│   ├── dino_detector.py        # DINOv2 feature extraction with heatmap visualization
│   └── master_vector.pt        # Pre-computed master vector for similarity matching
├── scripts/
│   ├── train.py                # Training entry point
│   ├── inference_ema.py        # Inference with EMA trajectory smoothing
│   ├── inference_rtc.py        # Inference with ranking-based sample selection
│   └── dataset/
│       ├── docking_dataset.py  # HDF5 multimodal dataset loader
│       └── preprocessing.py    # Raw episode data to HDF5 converter
├── utils/
│   ├── setups.py               # Model and logger initialization
│   └── utils.py                # Logger, trajectory plotting, RK4 reconstruction
├── rc_server/                  # Real-time control server (ZMQ-based robot interface)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (GPU required for training)

### Installation

```bash
# Create conda environment
conda create -n cleandiffuser python=3.11
conda activate cleandiffuser

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Raw Data Format

Organize raw episode data in the following structure:

```
data_root/
└── validation/
    ├── episode_001/
    │   ├── image/
    │   │   ├── room1/          # Top camera images (*.jpg, timestamped)
    │   │   └── room2/          # Bottom camera images (*.jpg, timestamped)
    │   └── encoder.csv         # Columns: ts, vx, wz
    ├── episode_002/
    └── ...
```

### Convert to HDF5

```bash
cd scripts
python -m dataset.preprocessing \
    --data_root /path/to/data_root \
    --save_path ./dataset/validation_dataset1.h5
```

The resulting HDF5 file contains:
- `encoder`: Normalized velocity history `[N, 2]` (vx, wz)
- `image_top`: Room 1 images `[N, 3, 240, 320]`
- `image_bottom`: Room 2 images `[N, 3, 240, 320]`
- `episode_ends`: Episode boundary indices

## Training

```bash
cd scripts
python train.py
```

### Key Configuration (`configs/robot/smr.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 60 | Future trajectory length (steps) |
| `obs_horizon` | 30 | Observation history length |
| `vision_stride` | 6 | Vision temporal subsampling stride |
| `d_model` | 384 | Transformer hidden dimension |
| `depth` | 12 | Number of DiT blocks |
| `diffusion_gradient_steps` | 400001 | Total training steps |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `ema_rate` | 0.9999 | EMA decay rate |

### Resume Training

```bash
cd scripts
python train.py resume_path=/path/to/checkpoint_step_100000.pt
```

## Inference

### EMA-based Inference

Uses exponential moving average over predicted trajectories for smooth control:

```bash
cd scripts
python inference_ema.py checkpoint_step=160000
```

### RTC (Ranking-based Trajectory Continuity) Inference

Generates multiple trajectory samples and selects the most continuous one:

```bash
cd scripts
python inference_rtc.py checkpoint_step=160000
```

## Acknowledgments

- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) - Diffusion model framework
- [DINOv2](https://github.com/facebookresearch/dinov2) - Vision feature backbone
