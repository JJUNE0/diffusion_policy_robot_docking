
import glob
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import hydra
import torch

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
    print("Start Training...")
    lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
        nn_diffusion.optimizer, T_max=args.diffusion_gradient_steps
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

    # Frozen DINO feature extractor used only to build condition inputs.
    dino_detector = DinoBatchDetector(device=device)

    # Vision uses sparse temporal sampling from 30-step history.
    vision_stride = args.get("vision_stride", 6)
    use_goal = args.get("use_goal", False)
    use_lidar = args.get("use_lidar_points", False)
    use_aux = args.get("use_aux_pose", False)
    aux_weight = args.get("aux_weight", 1.0)
    sparse_vision = args.get("sparse_vision", False)

    for batch in loop_dataloader(dataloader):
        if n_gradient_step >= args.diffusion_gradient_steps:
            print("End Training")
            plot_from_jsonl(os.path.join(save_path, "metrics.jsonl"))
            break

        action = batch["act"].to(device, non_blocking=True)  # [B, horizon, 2]
        obs_dict = batch["obs"]

        # ------------------------------------------------------------------
        # Multimodal condition inputs:
        #   - image_room1: full 30-step history -> sparse history via ::vision_stride
        #   - image_room2: full 30-step history -> sparse history via ::vision_stride
        #   - velocity:    full 30-step normalized encoder history
        # ------------------------------------------------------------------
        if sparse_vision:
            # dataset already returned the sparse uint8 frames; convert on GPU.
            image_room1 = obs_dict["image_room1"].to(device, non_blocking=True).float() / 255.0
            image_room2 = obs_dict["image_room2"].to(device, non_blocking=True).float() / 255.0
        else:
            image_room1 = obs_dict["image_room1"][:, ::vision_stride].to(device, non_blocking=True)
            image_room2 = obs_dict["image_room2"][:, ::vision_stride].to(device, non_blocking=True)
        velocity = obs_dict["velocity"].to(device, non_blocking=True)

        B, T_vis, C, H, W = image_room1.shape
        image_room1_flat = image_room1.reshape(B * T_vis, C, H, W)
        image_room2_flat = image_room2.reshape(B * T_vis, C, H, W)

        with torch.no_grad():
            dino_feat1, _, _ = dino_detector.get_heatmap(image_room1_flat)
            dino_feat2, _, _ = dino_detector.get_heatmap(image_room2_flat)

        context = {
            "dino_feat1": dino_feat1.view(B, T_vis, 196, 768),
            "dino_feat2": dino_feat2.view(B, T_vis, 196, 768),
            "velocity": velocity,
        }

        # Goal-feature conditioning (CLAUDE.md §2.3 Loss A): DINO-encode the goal
        # (docked) frame and pass it + its NoMaD mask as extra condition inputs.
        if use_goal:
            goal_room1 = obs_dict["goal_image_room1"].to(device, non_blocking=True)  # [B, 3, H, W]
            goal_room2 = obs_dict["goal_image_room2"].to(device, non_blocking=True)
            if sparse_vision:
                goal_room1 = goal_room1.float() / 255.0
                goal_room2 = goal_room2.float() / 255.0
            with torch.no_grad():
                goal_feat1, _, _ = dino_detector.get_heatmap(goal_room1)
                goal_feat2, _, _ = dino_detector.get_heatmap(goal_room2)
            context["goal_feat1"] = goal_feat1.view(B, 1, 196, 768)
            context["goal_feat2"] = goal_feat2.view(B, 1, 196, 768)
            context["goal_mask"] = obs_dict["goal_mask"].to(device, non_blocking=True)

        # Raw LiDAR points (Option A): point-set branch input.
        if use_lidar:
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
            print(f"Room1 image sparse history shape: {image_room1.shape}")
            print(f"Room2 image sparse history shape: {image_room2.shape}")
            print(f"DINO room1 feature shape: {dino_feat1.shape}")
            print(f"DINO room2 feature shape: {dino_feat2.shape}")
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


if __name__ == "__main__":
    main()