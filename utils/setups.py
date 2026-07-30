import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cleandiffuser.diffusion import ContinuousRectifiedFlow, ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d, DiTCrossAttn1d
from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
from cleandiffuser.nn_condition.modular_fusion_condition import ModularSensorFusionCondition
from cleandiffuser.nn_condition.token_sequence_condition import TokenSequenceFusionCondition
from utils.modular_dataset import ModularDockingDataset
from utils.docking_dataset import DockingDataset

from .utils import Logger


def _tree_batch_size(tree):
    if isinstance(tree, torch.Tensor):
        return int(tree.shape[0])
    if isinstance(tree, dict):
        return _tree_batch_size(next(iter(tree.values())))
    raise TypeError(f"unsupported microbatch leaf: {type(tree)!r}")


def _allocate_batch_like(tree, batch_size):
    if isinstance(tree, torch.Tensor):
        return torch.empty((batch_size, *tree.shape[1:]), dtype=tree.dtype)
    if isinstance(tree, dict):
        return {key: _allocate_batch_like(value, batch_size) for key, value in tree.items()}
    raise TypeError(f"unsupported microbatch leaf: {type(tree)!r}")


def _copy_batch_slice(dst, src, start, stop):
    if isinstance(src, torch.Tensor):
        dst[start:stop].copy_(src)
        return
    for key in src:
        _copy_batch_slice(dst[key], src[key], start, stop)


class _RebatchedDataLoader:
    """Assemble worker microbatches into one large batch outside /dev/shm.

    A full random-goal batch is multiple GiB and cannot pass through this
    container's 2-GiB shared-memory mount. Workers instead return compact
    float16 microbatches; each is copied immediately into ordinary host RAM,
    releasing its shared-memory segment before the next one arrives.
    """

    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = int(batch_size)
        self.epoch = 0

    def __len__(self):
        return len(self.source.sampler) // self.batch_size

    def __iter__(self):
        if isinstance(self.source.sampler, DistributedSampler):
            self.source.sampler.set_epoch(self.epoch)
            self.epoch += 1
        dst, filled = None, 0
        for microbatch in self.source:
            count = _tree_batch_size(microbatch)
            if dst is None:
                dst = _allocate_batch_like(microbatch, self.batch_size)
            if filled + count > self.batch_size:
                raise RuntimeError(
                    f"microbatch boundary {filled}+{count} exceeds {self.batch_size}"
                )
            _copy_batch_slice(dst, microbatch, filled, filled + count)
            filled += count
            if filled == self.batch_size:
                yield dst
                dst, filled = None, 0


def _build_dataloader(args, dataset):
    num_workers = int(args.get("num_workers", 4))
    batch_size = int(args.batch_size)
    microbatch_size = args.get("loader_microbatch_size", None)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=True)
        if distributed else None
    )
    if microbatch_size is not None:
        microbatch_size = int(microbatch_size)
        if num_workers < 1:
            raise ValueError("loader_microbatch_size requires num_workers >= 1")
        if batch_size % microbatch_size:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by "
                f"loader_microbatch_size={microbatch_size}"
            )
        source = DataLoader(
            dataset,
            batch_size=microbatch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=int(args.get("prefetch_factor", 1)),
        )
        print(
            f"Rebatched loader: {num_workers} workers x microbatch "
            f"{microbatch_size} -> batch {batch_size}"
        )
        return _RebatchedDataLoader(source, batch_size)

    loader_kwargs = dict(
        batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=num_workers,
        pin_memory=args.get("pin_memory", True), drop_last=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.get("prefetch_factor", 2)
    return DataLoader(dataset, **loader_kwargs)


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
        action_norm=args.get("action_norm", "minmax"),
    )
    dataloader = _build_dataloader(args, dataset)

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


def _token_sequence_setups(args):
    """R-NoGoal / R-Goal / R-Geo path (docs/0725_reloc3r_test/reloc3r/
    reloc3r_0725.md): TokenSequenceFusionCondition (unpooled [B,N,D]) +
    DiTCrossAttn1d (action tokens cross-attend to it, no causal mask, no
    AdaLN-vector conditioning). Same `sensors:`-dict-is-the-only-ablation-knob
    pattern as _modular_setups; the three named variants differ ONLY in which
    sensors are listed (no_goal drops `goal`/`geometry`, goal_appearance drops
    `geometry`, goal_appearance_geometry has all three RGB/goal/geometry
    sensors) -- enforced identical elsewhere by
    test/test_reloc3r_acceptance_criteria.py's param/config-equality check.
    No aux head, no NoMaD goal masking (not implemented, per spec)."""
    from omegaconf import OmegaConf, open_dict
    obs_horizon = args.get("obs_horizon", 30)
    sensors = OmegaConf.to_container(args.sensors, resolve=True)
    with open_dict(args):
        args.use_aux_pose = False  # not implemented for this path (per spec)

    dataset = ModularDockingDataset(
        h5_path=args.train_data_path,
        sensors=sensors,
        horizon=args.horizon,
        obs_horizon=obs_horizon,
        action_key=args.get("action_key", "encoder"),
        train_h5_path=args.train_data_path,
        action_norm=args.get("action_norm", "minmax"),
    )
    dataloader = _build_dataloader(args, dataset)

    nn_condition = TokenSequenceFusionCondition(
        sensors=sensors,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.get("condition_num_layers", 4),
        dropout=args.dropout,
    ).to(args.device)

    nn_diffusion_model = DiTCrossAttn1d(
        in_dim=2, emb_dim=args.d_model, d_model=args.d_model,
        n_heads=args.n_heads, depth=args.depth, dropout=0.0).to(args.device)

    Backbone = _select_backbone(args)
    nn_diffusion = Backbone(
        nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
        ema_rate=args.ema_rate, device=args.device)

    return dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion


def model_setups(args):
    # --- R-NoGoal/R-Goal/R-Geo cross-attention path (opt-in). Checked before
    # use_modular_fusion so both flags can coexist in a saved config without
    # ambiguity (this one wins).
    if args.get("use_token_sequence_fusion", False):
        return _token_sequence_setups(args)

    # --- Modular sensor-fusion path (opt-in via config). Spec-driven: the
    # `sensors` dict in the YAML defines the dataset reads AND the fusion net
    # branches, so ablations are config-only. See configs/robot/smr_new.yaml
    # (new dataset) and the `sensors:` block in configs/robot/smr.yaml (legacy
    # after_0328 dataset).
    if args.get("use_modular_fusion", False):
        return _modular_setups(args)

    obs_horizon = args.get("obs_horizon", 30)
    vision_stride = args.get("vision_stride", 6)
    vision_horizon = len(range(0, obs_horizon, vision_stride))

    use_goal = args.get("use_goal", False)
    use_lidar = args.get("use_lidar_points", False)
    use_aux = args.get("use_aux_pose", False)
    use_room1 = args.get("use_room1", True)
    use_goal_lidar = args.get("use_goal_lidar", False)
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
        with_goal_lidar=use_goal_lidar,
        aux_relative=args.get("aux_relative", False),
        adv_weight=args.get("adv_weight", False),
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
        use_goal_lidar=use_goal_lidar,
        use_aux_feedback=args.get("use_aux_feedback", False),
    ).to(args.device)

    nn_diffusion_model = DiT1d(
        in_dim=2,
        emb_dim=args.d_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=0.0,
    ).to(args.device)

    # Backbone is config-selected (`diffusion_backbone`: rectified_flow | ddpm),
    # same knob as the modular path. The aux ICP head + lidar/goal branches live
    # in nn_condition, so this backbone swap is orthogonal to the end-game
    # precision-docking design (train loop uses .loss(), which both expose).
    Backbone = _select_backbone(args)
    nn_diffusion = Backbone(
        nn_diffusion=nn_diffusion_model,
        nn_condition=nn_condition,
        ema_rate=args.ema_rate,
        device=args.device,
    )

    return dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion
