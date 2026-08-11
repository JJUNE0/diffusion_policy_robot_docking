#!/usr/bin/env python3
"""Train DP-Image-Goal with the unmodified official Diffusion Policy modules."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = REPO_ROOT / "third_party" / "diffusion_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from utils.dp_image_goal_dataset import DPImageGoalDataset

OFFICIAL_COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"


class LossModule(torch.nn.Module):
    """Expose official ``compute_loss`` through DDP's forward path."""

    def __init__(self, policy: DiffusionUnetImagePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robot/dp_image_goal.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--image-cache-mmap")
    parser.add_argument("--batch-size", type=int, help="per-GPU override")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-steps", type=int, help="finite smoke/debug run")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true", help="smoke test only")
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world_size, rank, local_rank, device


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(tree, device):
    if isinstance(tree, torch.Tensor):
        return tree.to(device, non_blocking=True)
    return {key: to_device(value, device) for key, value in tree.items()}


def build_dataset(cfg, cache_override=None):
    d = cfg.data
    cache_path = cache_override if cache_override is not None else d.image_cache_mmap
    return DPImageGoalDataset(
        h5_path=str(REPO_ROOT / d.h5_path),
        goal_pool_db=str(REPO_ROOT / d.goal_pool_db),
        image_key=d.image_key,
        goal_camera=d.goal_camera,
        horizon=d.horizon,
        obs_horizon=d.obs_horizon,
        vision_horizon=d.vision_horizon,
        vision_stride=d.vision_stride,
        action_key=d.action_key,
        goal_splits=d.goal_splits,
        goal_variant_weights=OmegaConf.to_container(d.goal_variant_weights, resolve=True),
        cross_goal_prob=d.cross_goal_prob,
        cross_goal_map=OmegaConf.to_container(d.cross_goal_map, resolve=True),
        image_cache_mmap=cache_path,
    )


def build_policy(cfg, dataset):
    p = cfg.policy
    shape_meta = dataset.shape_meta()
    rgb_model = get_resnet(name=p.resnet, weights=None)
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=tuple(p.image_resize) if p.image_resize else None,
        crop_shape=tuple(p.image_crop) if p.image_crop else None,
        random_crop=True,
        use_group_norm=bool(p.use_group_norm),
        share_rgb_model=bool(p.share_rgb_model),
        imagenet_norm=bool(p.imagenet_norm),
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(p.num_train_timesteps),
        beta_start=float(p.beta_start),
        beta_end=float(p.beta_end),
        beta_schedule=str(p.beta_schedule),
        variance_type=str(p.variance_type),
        clip_sample=True,
        prediction_type=str(p.prediction_type),
    )
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        obs_encoder=obs_encoder,
        horizon=int(cfg.data.horizon),
        n_action_steps=int(p.n_action_steps),
        n_obs_steps=int(p.n_obs_steps),
        num_inference_steps=int(p.num_inference_steps),
        obs_as_global_cond=True,
        diffusion_step_embed_dim=int(p.diffusion_step_embed_dim),
        down_dims=tuple(int(x) for x in p.down_dims),
        kernel_size=int(p.kernel_size),
        n_groups=int(p.n_groups),
        cond_predict_scale=bool(p.cond_predict_scale),
    )
    policy.set_normalizer(dataset.get_normalizer())
    return policy


def atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def checkpoint_payload(policy, ema_policy, optimizer, scheduler, ema, epoch, step, cfg):
    return {
        "official_diffusion_policy_commit": OFFICIAL_COMMIT,
        "epoch": int(epoch),
        "global_step": int(step),
        "model_state_dict": policy.state_dict(),
        "ema_state_dict": ema_policy.state_dict() if ema_policy is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_optimization_step": int(ema.optimization_step) if ema is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def main():
    cli = parse_args()
    world_size, rank, local_rank, device = init_distributed()
    is_main = rank == 0

    cfg = OmegaConf.load(REPO_ROOT / cli.config)
    if cli.batch_size is not None:
        cfg.training.per_gpu_batch_size = cli.batch_size
    if cli.num_workers is not None:
        cfg.training.num_workers = cli.num_workers
    seed = int(cfg.seed)
    seed_everything(seed + rank)

    if torch.cuda.is_available():
        # PyTorch >=2.9 API. Keep the compatibility branch for older runtimes.
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except (AttributeError, RuntimeError):
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if cli.output_dir:
        output_dir = Path(cli.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = REPO_ROOT / "outputs" / "train" / "dp_image_goal" / stamp
    if world_size > 1:
        payload = [str(output_dir) if is_main else None]
        dist.broadcast_object_list(payload, src=0)
        output_dir = Path(payload[0])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")

    dataset = build_dataset(cfg, cli.image_cache_mmap)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
        seed=seed, drop_last=True
    ) if world_size > 1 else None
    workers = int(cfg.training.num_workers)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=int(cfg.training.per_gpu_batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(cfg.training.pin_memory),
        drop_last=True,
    )
    if workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(cfg.training.persistent_workers),
            prefetch_factor=int(cfg.training.prefetch_factor),
        )
    loader = DataLoader(**loader_kwargs)

    policy = build_policy(cfg, dataset)
    ema_policy = copy.deepcopy(policy) if bool(cfg.training.use_ema) else None
    policy.to(device)
    if ema_policy is not None:
        ema_policy.to(device)
    train_module = LossModule(policy).to(device)
    if world_size > 1:
        train_module = DDP(
            train_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    live_policy = train_module.module.policy if world_size > 1 else train_module.policy

    optimizer_kwargs = dict(
        lr=float(cfg.training.learning_rate),
        betas=tuple(float(x) for x in cfg.training.betas),
        eps=float(cfg.training.eps),
        weight_decay=float(cfg.training.weight_decay),
    )
    if bool(cfg.training.fused_adamw) and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(train_module.parameters(), **optimizer_kwargs)
    epochs = int(cfg.training.epochs)
    total_steps = len(loader) * epochs
    scheduler = get_scheduler(
        str(cfg.training.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=total_steps,
    )
    ema = None
    if ema_policy is not None:
        ema = EMAModel(
            model=ema_policy,
            update_after_step=int(cfg.training.ema_update_after_step),
            inv_gamma=float(cfg.training.ema_inv_gamma),
            power=float(cfg.training.ema_power),
            min_value=float(cfg.training.ema_min_value),
            max_value=float(cfg.training.ema_max_value),
        )

    start_epoch = 0
    global_step = 0
    last_ckpt = output_dir / "last.ckpt"
    resume = bool(cfg.training.resume) and not cli.no_resume
    if resume and last_ckpt.is_file():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        live_policy.load_state_dict(state["model_state_dict"])
        if ema_policy is not None and state.get("ema_state_dict") is not None:
            ema_policy.load_state_dict(state["ema_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if ema is not None and state.get("ema_optimization_step") is not None:
            ema.optimization_step = int(state["ema_optimization_step"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])

    amp_dtype = str(cfg.training.amp_dtype).lower()
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    log_path = output_dir / "train.jsonl"
    if is_main:
        params = sum(x.numel() for x in live_policy.parameters())
        contract = {
            "event": "start",
            "official_commit": OFFICIAL_COMMIT,
            "seed": seed,
            "world_size": world_size,
            "per_gpu_batch_size": int(cfg.training.per_gpu_batch_size),
            "global_batch_size": int(cfg.training.per_gpu_batch_size) * world_size,
            "episodes": len(dataset.episode_ends),
            "samples": len(dataset),
            "steps_per_epoch": len(loader),
            "epochs": epochs,
            "total_steps": total_steps,
            "parameters": params,
            "output": "raw [B,60,2] = (v,w)",
            "rgb_keys": list(dataset.rgb_keys),
            "wheel_shape": [1, dataset.obs_horizon * 2],
            "goal_variant_weights": dict(dataset.variant_weights),
            "cross_goal_prob": dataset.cross_goal_prob,
            "image_cache_mmap": dataset.image_cache_mmap,
        }
        print(json.dumps(contract, ensure_ascii=False), flush=True)
        with log_path.open("a") as stream:
            stream.write(json.dumps(contract, ensure_ascii=False) + "\n")

    optimizer.zero_grad(set_to_none=True)
    run_started = time.monotonic()
    run_start_step = global_step
    stop_requested = False
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        live_policy.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                loss = train_module(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}: {loss}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.step(live_policy)

            reduced = loss.detach()
            if world_size > 1:
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced /= world_size
            value = float(reduced.item())
            epoch_loss += value
            epoch_steps += 1
            global_step += 1

            if is_main and global_step % int(cfg.training.log_every_steps) == 0:
                elapsed = time.monotonic() - run_started
                completed = max(1, global_step - run_start_step)
                step_s = elapsed / completed
                remaining = max(0, total_steps - global_step)
                record = {
                    "event": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": value,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "step_seconds": step_s,
                    "global_samples_per_second": (
                        int(cfg.training.per_gpu_batch_size) * world_size / step_s
                    ),
                    "eta_seconds": remaining * step_s,
                }
                print(json.dumps(record), flush=True)
                with log_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            if cli.max_steps is not None and global_step >= cli.max_steps:
                stop_requested = True
                break

        if is_main:
            record = {
                "event": "epoch",
                "epoch": epoch,
                "step": global_step,
                "mean_loss": epoch_loss / max(1, epoch_steps),
                "peak_gpu_gib": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda" else 0.0
                ),
            }
            print(json.dumps(record), flush=True)
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            state = checkpoint_payload(
                live_policy, ema_policy, optimizer, scheduler, ema,
                epoch, global_step, cfg
            )
            if not cli.skip_checkpoint:
                if (
                    stop_requested
                    or (epoch + 1) % int(cfg.training.checkpoint_every_epochs) == 0
                ):
                    atomic_torch_save(state, last_ckpt)
                if (epoch + 1) % int(cfg.training.milestone_every_epochs) == 0:
                    atomic_torch_save(
                        state, output_dir / f"checkpoint_epoch_{epoch + 1:03d}.ckpt"
                    )
        if world_size > 1:
            dist.barrier()
        if stop_requested:
            break

    if is_main:
        print(
            json.dumps({"event": "complete", "step": global_step, "output_dir": str(output_dir)}),
            flush=True,
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
