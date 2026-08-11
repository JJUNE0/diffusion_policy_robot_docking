
import glob
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import hydra
import torch
import torch.distributed as dist
from omegaconf import open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

# Avoid "/dev/shm ... Resource temporarily unavailable" from DataLoader workers:
# use file_system sharing instead of the default file_descriptor strategy.
torch.multiprocessing.set_sharing_strategy("file_system")

# Avoid "/dev/shm ... Resource temporarily unavailable" from DataLoader workers:
# use file_system sharing instead of the default file_descriptor strategy.
torch.multiprocessing.set_sharing_strategy("file_system")

from utils.setups import logger_setups, model_setups
from cleandiffuser.utils import loop_dataloader, threaded_prefetch, set_seed
from scripts.plot_training import plot_from_jsonl


def _resolve_resume_checkpoint(resume_path: str | None) -> str | None:
    """
    Resolve a checkpoint file to resume from.

    Supported cases:
      1) resume_path is None -> return None
      2) resume_path is a .pt file -> return that file
      3) resume_path is a directory -> return the latest checkpoint_step_*.pt inside it
    """
    if not resume_path:
        return None

    if os.path.isfile(resume_path):
        if resume_path.endswith(".pt"):
            return resume_path
        raise FileNotFoundError(f"resume_path is a file but not a .pt checkpoint: {resume_path}")

    if not os.path.isdir(resume_path):
        raise FileNotFoundError(f"resume_path does not exist: {resume_path}")

    pattern = os.path.join(resume_path, "checkpoint_step_*.pt")
    ckpt_files = glob.glob(pattern)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(
            f"No checkpoint_step_*.pt found under resume_path: {resume_path}"
        )

    def _extract_step(path: str) -> int:
        m = re.search(r"checkpoint_step_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    ckpt_files = sorted(ckpt_files, key=_extract_step)
    return ckpt_files[-1]


def _load_resume_state(
    ckpt_path: str,
    nn_diffusion,
    lr_scheduler,
    device: torch.device,
) -> int:
    """
    Load model / EMA / optimizer / scheduler state and return next gradient step.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    nn_diffusion.model.load_state_dict(ckpt["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ckpt["ema_state_dict"])
    nn_diffusion.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    loaded_step = int(ckpt.get("step", -1))
    next_step = loaded_step + 1

    print("======================================================")
    print(f"Resumed from checkpoint: {ckpt_path}")
    print(f"Loaded step: {loaded_step}")
    print(f"Next training step: {next_step}")
    print("======================================================")

    return next_step


<<<<<<< HEAD
def _resolve_total_gradient_steps(args, dataset) -> int:
    """
    Decide how many gradient steps to train for.

    If `num_epochs` is set (> 0) in the config, derive the step count from the
    dataset size and batch size:

        steps_per_epoch = len(dataset) // batch_size   # drop_last=True
        total_steps     = num_epochs * steps_per_epoch

    Otherwise fall back to the explicit `diffusion_gradient_steps`.
    """
    num_epochs = args.get("num_epochs", None)
    if not num_epochs or num_epochs <= 0:
        return int(args.diffusion_gradient_steps)

    steps_per_epoch = len(dataset) // args.batch_size  # DataLoader uses drop_last=True
    if steps_per_epoch <= 0:
        raise ValueError(
            f"dataset size ({len(dataset)}) is smaller than batch_size "
            f"({args.batch_size}); cannot compute steps for num_epochs={num_epochs}"
        )

    total_steps = int(num_epochs) * steps_per_epoch
    print(
        f"[num_epochs] {num_epochs} epochs x {steps_per_epoch} steps/epoch "
        f"(dataset={len(dataset)}, batch_size={args.batch_size}) "
        f"-> {total_steps} gradient steps"
    )
    return total_steps


@hydra.main(config_path="../configs/robot", config_name="smr", version_base=None)
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
=======
def _checkpoint_model_state(model):
    """Return a DDP-prefix-free state dict compatible with existing runs."""
    state = model.state_dict()
    clean = {}
    for key, value in state.items():
        key = key.replace("diffusion.module.", "diffusion.", 1)
        key = key.replace("condition.module.", "condition.", 1)
        clean[key] = value
    return clean


def _init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, torch.device(args.device if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    with open_dict(args):
        args.device = f"cuda:{local_rank}"
    return True, rank, local_rank, torch.device(args.device)


@hydra.main(config_path="../configs/robot", config_name="smr_rgeo", version_base=None)
def main(args):
    distributed, rank, local_rank, device = _init_distributed(args)
    is_main = rank == 0
    set_seed(int(args.seed) + rank)

    if is_main:
        logger, save_path = logger_setups(args)
    else:
        logger, save_path = None, None
    if distributed:
        shared = [save_path]
        dist.broadcast_object_list(shared, src=0)
        save_path = shared[0]
>>>>>>> endgame

    dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion = model_setups(args)

<<<<<<< HEAD
    total_gradient_steps = _resolve_total_gradient_steps(args, dataset)
=======
    print(f"Total Samples {len(dataset)}\n")

    # Resolve total gradient steps. If `num_epochs` is set, derive it from the
    # dataloader length (drop_last=True -> floor(len(dataset) / batch_size)) so
    # that "N epochs" is honored regardless of dataset size. Otherwise fall back
    # to the explicit `diffusion_gradient_steps`.
    num_epochs = args.get("num_epochs", None)
    steps_per_epoch = len(dataloader)
    if num_epochs is not None:
        total_gradient_steps = int(num_epochs) * steps_per_epoch
        print(
            f"num_epochs={num_epochs} x steps_per_epoch={steps_per_epoch} "
            f"-> total_gradient_steps={total_gradient_steps}"
        )
    else:
        total_gradient_steps = int(args.diffusion_gradient_steps)
>>>>>>> endgame

    print("Start Training...")
    lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
        nn_diffusion.optimizer, T_max=total_gradient_steps
    )

    # ----------------------------------------------------------
    # Resume support
    # ----------------------------------------------------------
    resume_path = args.get("resume_path")
    resume_ckpt_path = _resolve_resume_checkpoint(resume_path)
    if resume_ckpt_path is not None:
        n_gradient_step = _load_resume_state(
            ckpt_path=resume_ckpt_path,
            nn_diffusion=nn_diffusion,
            lr_scheduler=lr_schedulers,
            device=device,
        )
        # A resumed run may intentionally change batch_size. That changes
        # steps_per_epoch and therefore the live cosine horizon. The checkpoint
        # scheduler restores its old T_max, so adapt only that horizon while
        # preserving optimizer state, current LR, and last_epoch.
        saved_t_max = int(lr_schedulers.T_max)
        if saved_t_max != total_gradient_steps:
            lr_schedulers.T_max = total_gradient_steps
            print(
                f"Adjusted resumed scheduler T_max: {saved_t_max} -> "
                f"{total_gradient_steps} (live batch/epoch plan)"
            )
    else:
        n_gradient_step = 0
        # Warm-start (init_from): load WEIGHTS ONLY from a previous checkpoint
        # into a fresh run (new run dir/config/optimizer/scheduler/step count).
        # Unlike resume_path, the live config applies -> use this to continue
        # training under a changed config.
        # The EMA is re-seeded from the loaded weights: the source runs' EMA
        # still carries most of its random init (ema_rate 0.9999 vs 4230 steps
        # -> ~65% init, see memory ema-undertrained-4230-steps), so copying the
        # trained weights is the only sane EMA start.
        init_from = args.get("init_from", None)
        if init_from:
            ck = torch.load(init_from, map_location=device, weights_only=False)
            src = ck["model_state_dict"]
            # Same-name/different-shape params (e.g. TokenSequenceFusionCondition's
            # `modality_emb`, sized by sensor count -- grows when new sensor
            # branches are added) can't be loaded at all: strict=False only
            # tolerates missing/unexpected KEYS, not shape mismatches on a key
            # present in both, so load_state_dict would still raise. Drop them
            # up front so they fall back to the live model's fresh init, same
            # "leave new branches fresh" intent as the missing/unexpected
            # handling below, just for same-name params instead of new-name ones.
            tgt_sd = nn_diffusion.model.state_dict()
            reshaped = [k for k in src if k in tgt_sd and src[k].shape != tgt_sd[k].shape]
            for k in reshaped:
                del src[k]
            # Partial (graft) loading: when the live architecture ADDS sensor
            # branches the source checkpoint never had, load every matching key
            # and leave the new branches at their fresh init. strict load stays
            # the default path so exact warm-starts still catch real mismatches
            # loudly.
            missing, unexpected = nn_diffusion.model.load_state_dict(src, strict=False)
            nn_diffusion.model_ema.load_state_dict(src, strict=False)
            print("======================================================")
            print(f"Warm-started weights from: {init_from}")
            print(f"  loaded {len(src) - len(unexpected)}/{len(src)} source params"
                  f" | new (randomly initialized) params: {len(missing)}")
            if reshaped:
                print(f"  shape-changed params reinitialized: {len(reshaped)} e.g. {reshaped[:3]}")
            if unexpected:
                print(f"  source-only params ignored: {len(unexpected)} e.g. {unexpected[:3]}")
            if missing:
                print(f"  fresh branches e.g. {missing[:3]}")
            if len(missing) == 0 and len(unexpected) == 0 and len(reshaped) == 0:
                print("  (exact match — plain warm-start)")
            print("(fresh optimizer/scheduler/EMA-seed/step counter)")
            print("======================================================")
        else:
            print("======================================================")
            print("Starting training from scratch...")
            print("======================================================")

    # Load plain checkpoints first, then wrap the two trainable branches. DDP
    # broadcasts rank 0 parameters and averages gradients across both H100s.
    if distributed:
        ddp_kwargs = dict(
            device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, gradient_as_bucket_view=True,
            static_graph=True,
        )
        nn_diffusion.model["diffusion"] = DDP(
            nn_diffusion.model["diffusion"], **ddp_kwargs)
        nn_diffusion.model["condition"] = DDP(
            nn_diffusion.model["condition"], **ddp_kwargs)
        if is_main:
            print(f"DDP enabled: {dist.get_world_size()} x H100 | per-rank batch={args.batch_size}")

    nn_diffusion.train()

    # The dataset already returns the condition dict keyed exactly as
    # TokenSequenceFusionCondition expects (sensor names from the YAML
    # `sensors:` block), so each batch is forwarded verbatim. ReLoc3R features
    # are precomputed (in-h5 or sidecar `file:`) -- no vision backbone runs
    # here, except the frozen ReLoc3R decoder inside the goal-pair encoder.

    amp_name = str(args.get("amp_dtype", "none")).lower()
    amp_enabled = device.type == "cuda" and amp_name in ("bf16", "bfloat16")
    if amp_name not in ("none", "fp32", "bf16", "bfloat16"):
        raise ValueError(f"unsupported amp_dtype={amp_name!r}; use none or bf16")
    if is_main:
        amp_label = "BF16" if amp_enabled else "FP32"
        print(f"AMP: {amp_label}")

<<<<<<< HEAD
    for batch in loop_dataloader(dataloader):
        if n_gradient_step >= total_gradient_steps:
            print("End Training")
=======
    batch_stream = threaded_prefetch(loop_dataloader(dataloader))
    for batch in batch_stream:
        if n_gradient_step >= total_gradient_steps:
            if is_main:
                print("End Training")
                plot_from_jsonl(os.path.join(save_path, "metrics.jsonl"))
>>>>>>> endgame
            break

        action = batch["act"].to(device, non_blocking=True)  # [B, horizon, 2]
        context = {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()}

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            total_loss = nn_diffusion.loss(x0=action, condition=context)

        nn_diffusion.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = (torch.nn.utils.clip_grad_norm_(nn_diffusion.model.parameters(),
                     nn_diffusion.grad_clip_norm) if nn_diffusion.grad_clip_norm else None)
        nn_diffusion.optimizer.step()
        nn_diffusion.ema_update()
        lr_schedulers.step()
        loss_for_log = total_loss.detach().float()
        if distributed and n_gradient_step % args.get("log_interval", 100) == 0:
            dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM)
            loss_for_log /= dist.get_world_size()

        if n_gradient_step == 0:
            print(f"Action target shape: {action.shape}")
            print(f"Token-sequence fusion | backbone="
                  f"{args.get('diffusion_backbone', 'rectified_flow')}")
            for k, v in context.items():
                print(f"  obs[{k}]: {tuple(v.shape)}")

        if is_main and n_gradient_step % args.get("log_interval", 100) == 0:
            logger.log(
                {
                    "step": n_gradient_step,
                    "loss": loss_for_log.item(),
                    "grad_norm": grad_norm,
                },
                category="train",
            )
            print(f"Step {n_gradient_step} | Loss: {loss_for_log.item():.4f}")

        if is_main and n_gradient_step > 0 and n_gradient_step % args.save_interval == 0:
            save_ckpt_path = os.path.join(save_path, f"checkpoint_step_{n_gradient_step}.pt")
            torch.save(
                {
                    "step": n_gradient_step,
                    "model_state_dict": _checkpoint_model_state(nn_diffusion.model),
                    "ema_state_dict": nn_diffusion.model_ema.state_dict(),
                    "optimizer_state_dict": nn_diffusion.optimizer.state_dict(),
                    "scheduler_state_dict": lr_schedulers.state_dict(),
                    "action_min": dataset.action_min,
                    "action_scale": dataset.action_scale,
                },
                save_ckpt_path,
            )
            print(f"Checkpoint saved at step {n_gradient_step}")

        n_gradient_step += 1

    # The infinite loader was stopped by the step limit; finish its in-flight
    # batch and worker queues before distributed/CUDA teardown.
    batch_stream.close()

    # Always persist a final checkpoint. For short epoch-based runs the step
    # counter may never land on a save_interval boundary, so save explicitly.
    if distributed:
        dist.barrier()
    if is_main:
        final_ckpt_path = os.path.join(save_path, f"checkpoint_step_{n_gradient_step}.pt")
        torch.save(
            {
                "step": n_gradient_step,
                "model_state_dict": _checkpoint_model_state(nn_diffusion.model),
                "ema_state_dict": nn_diffusion.model_ema.state_dict(),
                "optimizer_state_dict": nn_diffusion.optimizer.state_dict(),
                "scheduler_state_dict": lr_schedulers.state_dict(),
                "action_min": dataset.action_min,
                "action_scale": dataset.action_scale,
            },
            final_ckpt_path,
        )
        print(f"Final checkpoint saved at step {n_gradient_step}: {final_ckpt_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()