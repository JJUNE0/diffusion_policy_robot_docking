"""Held-out evaluation of the single model: aux dock-pose error (mm) + denoising loss
on episodes NOT seen in training.

Run (from repo root):
  WANDB_MODE=disabled HF_HUB_OFFLINE=1 python scripts/eval_heldout.py \
    train_data_path=$PWD/dataset/after_0328_heldout.h5 \
    eval_checkpoint=outputs/results/single_model_feasibility/<ts>/checkpoint_step_1500.pt \
    train_stats_path=$PWD/dataset/after_0328_train.h5 eval_batches=60
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import hydra
import numpy as np
import torch

from utils.setups import model_setups
from utils.docking_dataset import DockingDataset
from dino.dino_detector import DinoBatchDetector


@hydra.main(config_path="../configs/robot", config_name="smr", version_base=None)
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vision_stride = args.get("vision_stride", 6)
    sparse_vision = args.get("sparse_vision", False)
    use_goal = args.get("use_goal", False)
    use_lidar = args.get("use_lidar_points", False)

    # held-out dataset + model (args.train_data_path overridden to the held-out h5)
    dataset, dataloader, nn_condition, _, nn_diffusion = model_setups(args)

    # use the TRAINING dock-pose stats (the space the model learned) for correct mm
    train_ds = DockingDataset(args.train_stats_path, args.train_stats_path,
                              horizon=args.horizon, obs_horizon=args.get("obs_horizon", 30),
                              with_aux=True)
    train_mean = np.asarray(train_ds.dock_xy_mean)
    train_std = np.asarray(train_ds.dock_xy_std)
    dataset.dock_xy_mean = train_ds.dock_xy_mean      # so dock_target matches the model space
    dataset.dock_xy_std = train_ds.dock_xy_std

    ckpt = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
    nn_diffusion.model.load_state_dict(ckpt["model_state_dict"])
    nn_diffusion.eval()
    dino = DinoBatchDetector(device=device)
    print(f"loaded {args.eval_checkpoint} | held-out {len(dataset)} frames")

    std_t = torch.as_tensor(train_std, device=device, dtype=torch.float32)
    mm_all, dn_all = [], []
    n_batches = int(args.get("eval_batches", 60))
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= n_batches:
                break
            obs = batch["obs"]
            if sparse_vision:
                im1 = obs["image_room1"].to(device).float() / 255.0
                im2 = obs["image_room2"].to(device).float() / 255.0
            else:
                im1 = obs["image_room1"][:, ::vision_stride].to(device)
                im2 = obs["image_room2"][:, ::vision_stride].to(device)
            B, T, C, H, W = im1.shape
            f1, _, _ = dino.get_heatmap(im1.reshape(B * T, C, H, W))
            f2, _, _ = dino.get_heatmap(im2.reshape(B * T, C, H, W))
            ctx = {"dino_feat1": f1.view(B, T, 196, 768), "dino_feat2": f2.view(B, T, 196, 768),
                   "velocity": obs["velocity"].to(device)}
            if use_goal:
                g1 = obs["goal_image_room1"].to(device); g2 = obs["goal_image_room2"].to(device)
                if sparse_vision:
                    g1 = g1.float() / 255.0; g2 = g2.float() / 255.0
                gf1, _, _ = dino.get_heatmap(g1); gf2, _, _ = dino.get_heatmap(g2)
                ctx["goal_feat1"] = gf1.view(B, 1, 196, 768); ctx["goal_feat2"] = gf2.view(B, 1, 196, 768)
                ctx["goal_mask"] = torch.ones(B, device=device)            # conditioned at eval
            if use_lidar:
                ctx["lidar_points"] = obs["lidar_points"].to(device)
                ctx["lidar_npoints"] = obs["lidar_npoints"].to(device)

            denoise = nn_diffusion.loss(x0=batch["act"].to(device), condition=ctx)
            dn_all.append(denoise.item())

            pred = nn_condition._aux_pred
            tgt = batch["dock_target"].to(device)
            m = batch["reliable"].to(device).float()
            d_xy = (pred[:, :2] - tgt[:, :2]) * std_t
            mm = torch.hypot(d_xy[:, 0], d_xy[:, 1]) * 1000.0
            mm_all.append((mm[m > 0]).detach().cpu().numpy())

    mm_all = np.concatenate(mm_all) if mm_all else np.array([0.0])
    print("\n================= HELD-OUT =================")
    print(f"denoising loss (held-out): {np.mean(dn_all):.4f}")
    print(f"dock pose error: median {np.median(mm_all):.1f} mm | mean {mm_all.mean():.1f} mm | "
          f"p90 {np.percentile(mm_all,90):.1f} mm")
    print(f"within 1cm: {np.mean(mm_all < 10)*100:.0f}%  | within 3cm: {np.mean(mm_all < 30)*100:.0f}%  "
          f"(n={len(mm_all)} reliable frames)")
    print("===========================================")


if __name__ == "__main__":
    main()
