
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
from cleandiffuser.utils import loop_dataloader, set_seed

# Lazy singleton for DINO when vision_backend=dino
_dino_detector_cache = None


def _get_dino_detector(device: torch.device):
    global _dino_detector_cache
    if _dino_detector_cache is None:
        from dino.dino_detector import DinoBatchDetector

        _dino_detector_cache = DinoBatchDetector(device=device)
    return _dino_detector_cache


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

    # The DockingDataset already returns sparse uint8 image / lidar histories
    # (subsampled by vision_stride) so we only need to move them to GPU and
    # convert to float here. This dramatically reduces CPU resident memory
    # and DataLoader queue size.
    vision_stride = args.get("vision_stride", 6)
    vision_backend = args.get("vision_backend", "raw_cnn")
    use_lidar = bool(args.get("use_lidar", False))
    use_goal = bool(args.get("use_goal", True))

    for batch in loop_dataloader(dataloader):
        if n_gradient_step >= args.diffusion_gradient_steps:
            print("End Training")
            break

        action = batch["act"].to(device, non_blocking=True)  # [B, horizon, 2]
        obs_dict = batch["obs"]

        # ------------------------------------------------------------------
        # Multimodal condition inputs:
        #   - image_room1: sparse history (T_vis frames) uint8 -> float on GPU
        #   - image_room2: sparse history (T_vis frames) uint8 -> float on GPU
        #   - velocity:    full obs_horizon normalized encoder history
        # ------------------------------------------------------------------
        image_room1 = obs_dict["image_room1"].to(device, non_blocking=True).float().div_(255.0)
        image_room2 = obs_dict["image_room2"].to(device, non_blocking=True).float().div_(255.0)
        velocity = obs_dict["velocity"].to(device, non_blocking=True)

        B, T_vis, C, H, W = image_room1.shape

        if vision_backend == "dino":
            dino_detector = _get_dino_detector(device)
            image_room1_flat = image_room1.reshape(B * T_vis, C, H, W)
            image_room2_flat = image_room2.reshape(B * T_vis, C, H, W)
            with torch.no_grad():
                dino_feat1 = dino_detector.get_features(image_room1_flat)
                dino_feat2 = dino_detector.get_features(image_room2_flat)
            context = {
                "dino_feat1": dino_feat1.view(B, T_vis, 196, 768),
                "dino_feat2": dino_feat2.view(B, T_vis, 196, 768),
                "velocity": velocity,
            }
        else:
            # raw_cnn: pass RGB (0~1) into condition net; learned encoder lives in SensorFusionConditionNetwork
            context = {
                "raw_image1": image_room1,
                "raw_image2": image_room2,
                "velocity": velocity,
            }

        if use_lidar:
            if "lidar_map" not in obs_dict:
                raise KeyError(
                    "use_lidar=True 인데 batch에 'lidar_map'이 없습니다. "
                    "preprocessing.py를 --use_lidar 옵션으로 실행했는지 확인하세요."
                )
            lidar_map = obs_dict["lidar_map"].to(device, non_blocking=True).float().div_(255.0)
            context["lidar_map"] = lidar_map

        # NoMaD-style goal frames. The condition network samples the goal mask
        # internally during training (Bernoulli(goal_mask_prob)), so we only
        # supply the goal observation here.
        if use_goal:
            if "goal_image1" not in obs_dict:
                raise KeyError(
                    "use_goal=True 인데 batch에 'goal_image1'이 없습니다. "
                    "DockingDataset(with_goal=True)로 생성되었는지 확인하세요."
                )
            goal_image1 = obs_dict["goal_image1"].to(device, non_blocking=True).float().div_(255.0)
            goal_image2 = obs_dict["goal_image2"].to(device, non_blocking=True).float().div_(255.0)
            if vision_backend == "dino":
                dino_detector = _get_dino_detector(device)
                with torch.no_grad():
                    goal_feat1 = dino_detector.get_features(goal_image1)
                    goal_feat2 = dino_detector.get_features(goal_image2)
                context["goal_feat1"] = goal_feat1.view(B, 196, 768)
                context["goal_feat2"] = goal_feat2.view(B, 196, 768)
            else:
                context["goal_image1"] = goal_image1
                context["goal_image2"] = goal_image2

        diff_log = nn_diffusion.update(x0=action, condition=context)
        lr_schedulers.step()

        if n_gradient_step == 0:
            print(f"Action target shape: {action.shape}")
            print(f"Room1 image sparse history shape: {image_room1.shape}")
            print(f"Room2 image sparse history shape: {image_room2.shape}")
            print(f"vision_backend: {vision_backend}")
            if vision_backend == "dino":
                print(f"DINO room1 feature shape: {context['dino_feat1'].shape}")
                print(f"DINO room2 feature shape: {context['dino_feat2'].shape}")
            else:
                print("Using raw CNN patch encoder inside condition network (no DINO forward in train step).")
            print(f"Velocity history shape: {velocity.shape}")
            if use_lidar:
                print(f"Lidar map sparse history shape: {context['lidar_map'].shape}")
            if use_goal:
                goal_key = "goal_feat1" if vision_backend == "dino" else "goal_image1"
                print(f"Goal frame shape ({goal_key}): {context[goal_key].shape}")
                print(
                    f"Goal masking: goal_mask_prob={args.get('goal_mask_prob', 0.5)} "
                    f"(per-step expected ~{args.get('goal_mask_prob', 0.5):.2f} of samples blocked)"
                )

        if n_gradient_step % args.get("log_interval", 100) == 0:
            logger.log(
                {
                    "step": n_gradient_step,
                    "loss": diff_log["loss"],
                    "grad_norm": diff_log["grad_norm"],
                },
                category="train",
            )
            print(f"Step {n_gradient_step} | Loss: {diff_log['loss']:.4f}")

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