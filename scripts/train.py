
import glob
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import hydra
import torch

# Avoid "/dev/shm ... Resource temporarily unavailable" from DataLoader workers:
# use file_system sharing instead of the default file_descriptor strategy.
torch.multiprocessing.set_sharing_strategy("file_system")

from utils.setups import logger_setups, model_setups
from dino.dino_detector import DinoBatchDetector
from cleandiffuser.utils import loop_dataloader, set_seed
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


@hydra.main(config_path="../configs/robot", config_name="smr", version_base=None)
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger, save_path = logger_setups(args)
    dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion = model_setups(args)

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
    else:
        n_gradient_step = 0
        print("======================================================")
        print("Starting training from scratch...")
        print("======================================================")

    nn_diffusion.train()

    # Modular spec-driven path: the dataset already returns the condition dict
    # keyed exactly as ModularSensorFusionCondition expects (sensor names from
    # the YAML `sensors:` block), so the batch is forwarded as-is. DINO features
    # are precomputed (in-h5 or sidecar `file:`) -> no live backbone here.
    use_modular = args.get("use_modular_fusion", False)

    # Vision uses sparse temporal sampling from 30-step history.
    vision_stride = args.get("vision_stride", 6)
    use_goal = args.get("use_goal", False)
    use_lidar = args.get("use_lidar_points", False)
    use_aux = args.get("use_aux_pose", False)
    aux_weight = args.get("aux_weight", 1.0)
    sparse_vision = args.get("sparse_vision", False)
    use_room1 = args.get("use_room1", True)
    # When DINO features are precomputed (scripts/precompute_dino_cache.py), the
    # dataset returns cached [Tv,196,768] features directly and we skip the DINO
    # backbone entirely (see plan: single-camera + offline caching).
    use_dino_cache = args.get("use_dino_cache", False)

    # Frozen DINO feature extractor used only to build condition inputs. Skipped
    # (not even loaded) when reading precomputed features from the cache.
    dino_detector = None if (use_dino_cache or use_modular) else DinoBatchDetector(device=device)

    for batch in loop_dataloader(dataloader):
        if n_gradient_step >= total_gradient_steps:
            print("End Training")
            plot_from_jsonl(os.path.join(save_path, "metrics.jsonl"))
            break

        action = batch["act"].to(device, non_blocking=True)  # [B, horizon, 2]
        obs_dict = batch["obs"]

        # ------------------------------------------------------------------
        # Multimodal condition inputs. Three paths:
        #   (M) use_modular_fusion: obs dict is already the condition dict
        #       (spec-driven names) -> forward verbatim.
        #   (A) use_dino_cache: dataset returns precomputed [B,Tv,196,768] DINO
        #       features directly -> skip the backbone (the fast training path).
        #   (B) live: DINO-encode sparse image history on the fly.
        # room1 (image_top) is only used when use_room1 is True (else room2 only).
        # ------------------------------------------------------------------
        if use_modular:
            context = {k: v.to(device, non_blocking=True) for k, v in obs_dict.items()}
        elif use_dino_cache:
            velocity = obs_dict["velocity"].to(device, non_blocking=True)
            dino_feat2 = obs_dict["dino_feat_room2"].to(device, non_blocking=True).float()
            B, T_vis = dino_feat2.shape[0], dino_feat2.shape[1]
            context = {"dino_feat2": dino_feat2.view(B, T_vis, 196, 768), "velocity": velocity}
            if use_room1:
                context["dino_feat1"] = (obs_dict["dino_feat_room1"]
                                         .to(device, non_blocking=True).float().view(B, T_vis, 196, 768))
            if use_goal:
                context["goal_feat2"] = (obs_dict["goal_feat_room2"]
                                         .to(device, non_blocking=True).float().view(B, 1, 196, 768))
                if use_room1:
                    context["goal_feat1"] = (obs_dict["goal_feat_room1"]
                                             .to(device, non_blocking=True).float().view(B, 1, 196, 768))
                context["goal_mask"] = obs_dict["goal_mask"].to(device, non_blocking=True)
        else:
            velocity = obs_dict["velocity"].to(device, non_blocking=True)

            def _encode(img_key):
                if sparse_vision:
                    img = obs_dict[img_key].to(device, non_blocking=True).float() / 255.0
                else:
                    img = obs_dict[img_key][:, ::vision_stride].to(device, non_blocking=True)
                b, tv, c, h, w = img.shape
                with torch.no_grad():
                    feat, _, _ = dino_detector.get_heatmap(img.reshape(b * tv, c, h, w))
                return feat.view(b, tv, 196, 768), b, tv

            dino_feat2, B, T_vis = _encode("image_room2")
            context = {"dino_feat2": dino_feat2, "velocity": velocity}
            if use_room1:
                context["dino_feat1"] = _encode("image_room1")[0]

            # Goal-feature conditioning (CLAUDE.md §2.3 Loss A): DINO-encode the
            # goal (docked) frame and pass it + its NoMaD mask as condition inputs.
            if use_goal:
                def _encode_goal(img_key):
                    goal = obs_dict[img_key].to(device, non_blocking=True)
                    if sparse_vision:
                        goal = goal.float() / 255.0
                    with torch.no_grad():
                        feat, _, _ = dino_detector.get_heatmap(goal)
                    return feat.view(B, 1, 196, 768)

                context["goal_feat2"] = _encode_goal("goal_image_room2")
                if use_room1:
                    context["goal_feat1"] = _encode_goal("goal_image_room1")
                context["goal_mask"] = obs_dict["goal_mask"].to(device, non_blocking=True)

        # Raw LiDAR points (Option A): point-set branch input (legacy path only;
        # the modular obs dict already carries its lidar keys by sensor name).
        if use_lidar and not use_modular:
            context["lidar_points"] = obs_dict["lidar_points"].to(device, non_blocking=True)
            context["lidar_npoints"] = obs_dict["lidar_npoints"].to(device, non_blocking=True)

        # Combined loss = denoising (main) + ICP-distilled aux pose (precision).
        # nn_diffusion.loss() runs the condition net once and caches its aux pred.
        denoise_loss = nn_diffusion.loss(x0=action, condition=context)
        aux_val, mm_val = None, None
        if use_aux:
            aux_pred = nn_condition._aux_pred
            dock_target = batch["dock_target"].to(device, non_blocking=True)
            m = batch["reliable"].to(device, non_blocking=True).float()
            se = ((aux_pred - dock_target) ** 2).sum(dim=1)
            aux_loss = (se * m).sum() / m.sum().clamp(min=1.0)
            total_loss = denoise_loss + aux_weight * aux_loss
            aux_val = aux_loss.item()
            with torch.no_grad():
                std = torch.as_tensor(dataset.dock_xy_std, device=device)
                d_xy = (aux_pred[:, :2] - dock_target[:, :2]) * std
                mm_val = ((torch.hypot(d_xy[:, 0], d_xy[:, 1]) * 1000.0 * m).sum()
                          / m.sum().clamp(min=1.0)).item()
        else:
            total_loss = denoise_loss

        nn_diffusion.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = (torch.nn.utils.clip_grad_norm_(nn_diffusion.model.parameters(),
                     nn_diffusion.grad_clip_norm) if nn_diffusion.grad_clip_norm else None)
        nn_diffusion.optimizer.step()
        nn_diffusion.ema_update()
        lr_schedulers.step()
        diff_log = {"loss": denoise_loss.item(), "grad_norm": grad_norm,
                    "aux": aux_val, "aux_mm": mm_val}

        if n_gradient_step == 0:
            print(f"Action target shape: {action.shape}")
            if use_modular:
                print(f"Modular fusion | backbone={args.get('diffusion_backbone', 'rectified_flow')}")
                for k, v in context.items():
                    print(f"  obs[{k}]: {tuple(v.shape)}")
            else:
                print(f"use_room1={use_room1} | use_dino_cache={use_dino_cache}")
                print(f"DINO room2 feature shape: {context['dino_feat2'].shape}")
                if use_room1:
                    print(f"DINO room1 feature shape: {context['dino_feat1'].shape}")
                print(f"Velocity history shape: {velocity.shape}")

        if n_gradient_step % args.get("log_interval", 100) == 0:
            logger.log(
                {
                    "step": n_gradient_step,
                    "loss": diff_log["loss"],
                    "grad_norm": diff_log["grad_norm"],
                    "aux_loss": diff_log["aux"],
                    "aux_pose_mm": diff_log["aux_mm"],
                },
                category="train",
            )
            extra = (f" | aux {diff_log['aux']:.4f} | dock {diff_log['aux_mm']:.1f}mm"
                     if diff_log["aux"] is not None else "")
            print(f"Step {n_gradient_step} | Loss: {diff_log['loss']:.4f}{extra}")

        if n_gradient_step > 0 and n_gradient_step % args.save_interval == 0:
            save_ckpt_path = os.path.join(save_path, f"checkpoint_step_{n_gradient_step}.pt")
            torch.save(
                {
                    "step": n_gradient_step,
                    "model_state_dict": nn_diffusion.model.state_dict(),
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

    # Always persist a final checkpoint. For short epoch-based runs the step
    # counter may never land on a save_interval boundary, so save explicitly.
    final_ckpt_path = os.path.join(save_path, f"checkpoint_step_{n_gradient_step}.pt")
    torch.save(
        {
            "step": n_gradient_step,
            "model_state_dict": nn_diffusion.model.state_dict(),
            "ema_state_dict": nn_diffusion.model_ema.state_dict(),
            "optimizer_state_dict": nn_diffusion.optimizer.state_dict(),
            "scheduler_state_dict": lr_schedulers.state_dict(),
            "action_min": dataset.action_min,
            "action_scale": dataset.action_scale,
        },
        final_ckpt_path,
    )
    print(f"Final checkpoint saved at step {n_gradient_step}: {final_ckpt_path}")


if __name__ == "__main__":
    main()