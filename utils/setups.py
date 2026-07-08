import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cleandiffuser.diffusion import ContinuousRectifiedFlow, ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
from cleandiffuser.nn_condition.modular_fusion_condition import ModularSensorFusionCondition
from utils.modular_dataset import ModularDockingDataset
from utils.docking_dataset import DockingDataset

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
        save_path = f"train/{args.experiment_name}/{timestamp}/"
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


def _select_backbone(args):
    """flow vs ddpm ablation = one config flag, orthogonal to the sensors."""
    name = str(args.get("diffusion_backbone", "rectified_flow")).lower()
    if name in ("rectified_flow", "flow", "rf"):
        return ContinuousRectifiedFlow
    if name in ("ddpm", "diffusion", "sde"):
        return ContinuousDiffusionSDE
    raise ValueError(f"unknown diffusion_backbone '{name}'")


def _modular_setups(args):
    from omegaconf import OmegaConf, open_dict
    obs_horizon = args.get("obs_horizon", 30)
    sensors = OmegaConf.to_container(args.sensors, resolve=True)

    # The `head: aux_pose` field on any sensor is the SINGLE knob for the ICP
    # precision head: it turns the head on in the condition net AND the targets
    # on in the dataset. Mirror it into args.use_aux_pose so the existing
    # train.py aux-loss path activates without a second, separate flag.
    has_aux = any(s.get("head") == "aux_pose" for s in sensors.values())
    with open_dict(args):
        args.use_aux_pose = has_aux

    dataset = ModularDockingDataset(
        h5_path=args.train_data_path,
        sensors=sensors,
        horizon=args.horizon,
        obs_horizon=obs_horizon,
        action_key=args.get("action_key", "encoder"),
        train_h5_path=args.train_data_path,
    )
    num_workers = args.get("num_workers", 4)
    loader_kwargs = dict(
        batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=args.get("pin_memory", True), drop_last=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.get("prefetch_factor", 2)
    dataloader = DataLoader(dataset, **loader_kwargs)

    nn_condition = ModularSensorFusionCondition(
        sensors=sensors,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.get("condition_num_layers", 4),
        dropout=args.dropout,
    ).to(args.device)

    nn_diffusion_model = DiT1d(
        in_dim=2, emb_dim=args.d_model, d_model=args.d_model,
        n_heads=args.n_heads, depth=args.depth, dropout=0.0).to(args.device)

    Backbone = _select_backbone(args)
    nn_diffusion = Backbone(
        nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
        ema_rate=args.ema_rate, device=args.device)

    return dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion


def model_setups(args):
    # --- Modular sensor-fusion path (opt-in via config). Spec-driven: the
    # `sensors` dict in the YAML defines the dataset reads AND the fusion net
    # branches, so ablations are config-only. See configs/robot/modular.yaml.
    if args.get("use_modular_fusion", False):
        return _modular_setups(args)

    obs_horizon = args.get("obs_horizon", 30)
    vision_stride = args.get("vision_stride", 6)
    vision_horizon = len(range(0, obs_horizon, vision_stride))

    use_goal = args.get("use_goal", False)
    use_lidar = args.get("use_lidar_points", False)
    use_aux = args.get("use_aux_pose", False)
    use_room1 = args.get("use_room1", True)
    dino_cache_path = args.get("dino_cache_path", None) if args.get("use_dino_cache", False) else None
    dataset = DockingDataset(
        npz_path=args.train_data_path,
        train_npz_path=args.train_data_path,
        horizon=args.horizon,
        obs_horizon=obs_horizon,
        dt=args.get("dt", 0.0333),
        with_goal=use_goal,
        goal_mask_prob=args.get("goal_mask_prob", 0.5),
        with_lidar=use_lidar,
        with_aux=use_aux,
        vision_stride=vision_stride,
        sparse_vision_uint8=args.get("sparse_vision", False),
        dino_cache_path=dino_cache_path,
        use_room1=use_room1,
    )

    # RAM note: in-flight host memory ≈ num_workers × prefetch_factor × batch_size
    # samples. On the 15 GB machine, large batch × many workers OOMs -> keep these
    # configurable (prefetch_factor=1 and pin_memory=False save the most RAM).
    num_workers = args.get("num_workers", 4)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.get("pin_memory", True),
        drop_last=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.get("prefetch_factor", 2)
    dataloader = DataLoader(dataset, **loader_kwargs)

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
        use_goal=use_goal,
        num_goal_latents=args.get("num_goal_latents", 16),
        use_lidar_points=use_lidar,
        num_lidar_latents=args.get("num_lidar_latents", 16),
        use_aux_pose=use_aux,
        use_room1=use_room1,
    ).to(args.device)

    nn_diffusion_model = DiT1d(
        in_dim=2,
        emb_dim=args.d_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=0.0,
    ).to(args.device)

    # Rectified Flow backbone: learns the straight-line velocity field (x0 - x1)
    # and integrates it with a simple Euler ODE. The aux ICP head + lidar/goal
    # branches live in nn_condition, so this backbone swap is orthogonal to the
    # end-game precision-docking design (train loop uses .loss(), which RF also
    # exposes -> no training-loop change needed).
    nn_diffusion = ContinuousRectifiedFlow(
        nn_diffusion=nn_diffusion_model,
        nn_condition=nn_condition,
        ema_rate=args.ema_rate,
        device=args.device,
    )

    return dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion
