"""Held-out evaluation of the single model: aux dock-pose error (mm) + denoising loss
on episodes NOT seen in training.

Builds the condition context exactly like scripts/train.py (three paths):
  (M) use_modular_fusion : obs dict forwarded verbatim (spec-driven names)
  (A) use_dino_cache     : precomputed DINO features from the cache h5
  (B) live               : DINO-encode image history on the fly

Run (from repo root):
  WANDB_MODE=disabled HF_HUB_OFFLINE=1 python scripts/eval_heldout.py \
    train_data_path=$PWD/dataset/after_0328_heldout.h5 \
    eval_checkpoint=outputs/results/.../checkpoint_step_XXXX.pt \
    train_stats_path=$PWD/dataset/after_0328_train.h5 eval_batches=60

NOTE (use_dino_cache=true): dino_cache_path must be the cache precomputed from
the HELD-OUT h5, not the train cache (row alignment is validated). If you have
no held-out cache, override use_dino_cache=false to DINO-encode live.
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


@hydra.main(config_path="../configs/robot", config_name="smr", version_base=None)
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vision_stride = args.get("vision_stride", 6)
    sparse_vision = args.get("sparse_vision", False)
    use_modular = args.get("use_modular_fusion", False)
    use_goal = args.get("use_goal", False)
    use_lidar = args.get("use_lidar_points", False)
    use_room1 = args.get("use_room1", True)
    use_dino_cache = args.get("use_dino_cache", False)

    # held-out dataset + model (args.train_data_path overridden to the held-out h5)
    dataset, dataloader, nn_condition, _, nn_diffusion = model_setups(args)

    # use the TRAINING dock-pose stats (the space the model learned) for correct mm
    train_ds = DockingDataset(args.train_stats_path, args.train_stats_path,
                              horizon=args.horizon, obs_horizon=args.get("obs_horizon", 30),
                              with_aux=True)
    train_std = np.asarray(train_ds.dock_xy_std)
    dataset.dock_xy_mean = train_ds.dock_xy_mean      # so dock_target matches the model space
    dataset.dock_xy_std = train_ds.dock_xy_std

    ckpt = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
    nn_diffusion.model.load_state_dict(ckpt["model_state_dict"])
    nn_diffusion.eval()

    # Frozen DINO backbone only needed on the live path (B).
    dino = None
    if not (use_modular or use_dino_cache):
        from dino.dino_detector import DinoBatchDetector
        dino = DinoBatchDetector(device=device)
    print(f"loaded {args.eval_checkpoint} | held-out {len(dataset)} frames")

    def build_context(obs):
        """Mirror the train.py context construction for all three paths."""
        if use_modular:
            return {k: v.to(device) for k, v in obs.items()}

        ctx = {"velocity": obs["velocity"].to(device)}
        if use_dino_cache:
            f2 = obs["dino_feat_room2"].to(device).float()
            B, Tv = f2.shape[0], f2.shape[1]
            ctx["dino_feat2"] = f2.view(B, Tv, 196, 768)
            if use_room1:
                ctx["dino_feat1"] = obs["dino_feat_room1"].to(device).float().view(B, Tv, 196, 768)
            if use_goal:
                ctx["goal_feat2"] = obs["goal_feat_room2"].to(device).float().view(B, 1, 196, 768)
                if use_room1:
                    ctx["goal_feat1"] = obs["goal_feat_room1"].to(device).float().view(B, 1, 196, 768)
                ctx["goal_mask"] = torch.ones(B, device=device)     # conditioned at eval
        else:
            def encode(img):
                img = img.to(device)
                img = img.float() / 255.0 if sparse_vision else img[:, ::vision_stride]
                B, T, C, H, W = img.shape
                f, _, _ = dino.get_heatmap(img.reshape(B * T, C, H, W))
                return f.view(B, T, 196, 768)

            ctx["dino_feat2"] = encode(obs["image_room2"])
            B = ctx["dino_feat2"].shape[0]
            if use_room1:
                ctx["dino_feat1"] = encode(obs["image_room1"])
            if use_goal:
                def encode_goal(img):
                    img = img.to(device)
                    if sparse_vision:
                        img = img.float() / 255.0
                    f, _, _ = dino.get_heatmap(img)
                    return f.view(B, 1, 196, 768)

                ctx["goal_feat2"] = encode_goal(obs["goal_image_room2"])
                if use_room1:
                    ctx["goal_feat1"] = encode_goal(obs["goal_image_room1"])
                ctx["goal_mask"] = torch.ones(B, device=device)
        if use_lidar:
            ctx["lidar_points"] = obs["lidar_points"].to(device)
            ctx["lidar_npoints"] = obs["lidar_npoints"].to(device)
        return ctx

    use_aux = args.get("use_aux_pose", False)   # modular path mirrors it from the spec
    std_t = torch.as_tensor(train_std, device=device, dtype=torch.float32)
    mm_all, dn_all = [], []
    n_batches = int(args.get("eval_batches", 60))
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= n_batches:
                break
            ctx = build_context(batch["obs"])
            denoise = nn_diffusion.loss(x0=batch["act"].to(device), condition=ctx)
            dn_all.append(denoise.item())

            if use_aux:
                pred = nn_condition._aux_pred
                tgt = batch["dock_target"].to(device)
                m = batch["reliable"].to(device).float()
                d_xy = (pred[:, :2] - tgt[:, :2]) * std_t
                mm = torch.hypot(d_xy[:, 0], d_xy[:, 1]) * 1000.0
                mm_all.append((mm[m > 0]).detach().cpu().numpy())

    print("\n================= HELD-OUT =================")
    print(f"denoising loss (held-out): {np.mean(dn_all):.4f}")
    if mm_all:
        mm_all = np.concatenate(mm_all)
        print(f"dock pose error: median {np.median(mm_all):.1f} mm | mean {mm_all.mean():.1f} mm | "
              f"p90 {np.percentile(mm_all,90):.1f} mm")
        print(f"within 1cm: {np.mean(mm_all < 10)*100:.0f}%  | within 3cm: {np.mean(mm_all < 30)*100:.0f}%  "
              f"(n={len(mm_all)} reliable frames)")
    else:
        print("(no aux head -> dock-pose metrics skipped)")
    print("===========================================")


if __name__ == "__main__":
    main()
