import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
from dataset.docking_dataset import DockingDataset

from .utils import Logger


def logger_setups(args):
    if args.resume_path:
        save_path = args.resume_path
        timestamp = Path(save_path.rstrip("/")).name
        config_load_path = os.path.join(save_path, "config.yaml")
        if os.path.exists(config_load_path):
            saved_config = OmegaConf.load(config_load_path)
            saved_config.resume_path = args.resume_path
            saved_config.mode = getattr(args, "mode", None)
            args = saved_config
        else:
            print(f"Warning: Configuration file not found at {config_load_path}. Using current settings.")
    else:
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"results/{args.experiment_name}/{timestamp}/"
        os.makedirs(save_path, exist_ok=True)
        config_save_path = os.path.join(save_path, "config.yaml")
        OmegaConf.save(config=args, f=config_save_path)

    plot_save_path = os.path.join(save_path, "plots")
    os.makedirs(plot_save_path, exist_ok=True)

    logger_cfg = OmegaConf.create(
        {
            "project": "polaris3d_diff_flow",
            "group": f"{args.experiment_name}",
            "exp_name": f"{args.experiment_name}-{timestamp}",
            "wandb_mode": "online",
        }
    )
    logger = Logger(Path(save_path), logger_cfg)
    return logger, save_path


def model_setups(args):
    obs_horizon = args.get("obs_horizon", 30)
    vision_stride = args.get("vision_stride", 6)
    vision_horizon = len(range(0, obs_horizon, vision_stride))

    dataset = DockingDataset(
        npz_path=args.train_data_path,
        train_npz_path=args.train_data_path,
        horizon=args.horizon,
        obs_horizon=obs_horizon,
        dt=args.get("dt", 0.0333),
    )

    num_workers = args.get("num_workers", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        drop_last=True,
    )

    nn_condition = SensorFusionConditionNetwork(
        state_dim=args.state_dim,
        obs_horizon=obs_horizon,
        vision_horizon=vision_horizon,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.get("condition_num_layers", 2),
        dropout=args.dropout,
        num_image_latents=args.get("num_image_latents", 16),
        velocity_dim=args.get("velocity_dim", 2),
        velocity_dropout_prob=args.get("velocity_dropout_prob", 0.0),
        vision_backend=args.get("vision_backend", "raw_cnn"),
    ).to(args.device)

    nn_diffusion_model = DiT1d(
        in_dim=2,
        emb_dim=args.d_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=0.0,
    ).to(args.device)

    nn_diffusion = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion_model,
        nn_condition=nn_condition,
        ema_rate=args.ema_rate,
        device=args.device,
    )

    return dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion
