import os
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
from dataset.docking_dataset import DockingDataset

from .utils import Logger


def logger_setups(args):
    """
    Resolve the run directory used for checkpoints / metrics / plots.

    Layout (controlled by configs/robot/smr.yaml hydra.run.dir):
        outputs/<run_kind>/<experiment_name>/<timestamp>/

    On resume:
        save_path = args.resume_path (the existing run folder)
    """
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
        save_path = HydraConfig.get().runtime.output_dir
        timestamp = Path(save_path.rstrip("/")).name
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

    use_goal = bool(args.get("use_goal", True))

    dataset = DockingDataset(
        npz_path=args.train_data_path,
        train_npz_path=args.train_data_path,
        horizon=args.horizon,
        obs_horizon=obs_horizon,
        vision_stride=vision_stride,
        dt=args.get("dt", 0.0333),
        with_goal=use_goal,
        goal_offset_min=args.get("goal_offset_min", None),
        goal_offset_max=args.get("goal_offset_max", None),
    )

    # NOTE: keep DataLoader memory pressure low.
    #   * num_workers small (1) -> at most 1 prefetch_factor batches buffered
    #   * pin_memory=False      -> avoid extra page-locked CPU copies
    #   * persistent_workers=False -> release worker memory between epochs
    #   * prefetch_factor=1     -> at most 1 batch waiting per worker
    # Combined with the uint8 + sparse history changes in DockingDataset, this
    # cuts CPU resident memory per buffered batch by ~24x (4x dtype, 6x stride).
    num_workers = int(args.get("num_workers", 1))
    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1
    dataloader = DataLoader(dataset, **dl_kwargs)

    use_lidar = bool(args.get("use_lidar", False))
    if use_lidar and dataset.z_lidar is None:
        raise ValueError(
            "use_lidar=True 인데 HDF5에 'lidar_map'이 없습니다. "
            "preprocessing.py를 --use_lidar 옵션으로 다시 실행하세요."
        )

    lidar_in_ch = (
        int(dataset.lidar_meta["channels"])
        if (use_lidar and dataset.lidar_meta is not None)
        else int(args.get("lidar_channels", 2))
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
        use_lidar=use_lidar,
        lidar_in_ch=lidar_in_ch,
        num_lidar_latents=args.get("num_lidar_latents", 16),
        lidar_dropout_prob=args.get("lidar_dropout_prob", 0.0),
        use_goal=use_goal,
        num_goal_latents=args.get("num_goal_latents", 16),
        goal_mask_prob=args.get("goal_mask_prob", 0.5),
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
