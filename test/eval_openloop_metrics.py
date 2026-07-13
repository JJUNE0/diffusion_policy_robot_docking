"""Multi-episode open-loop inference evaluation with an explicit metric suite.

Metrics (all open-loop: GT observations in, predicted actions integrated out):
  * vel RMSE   -- per-step error of the predicted (vx, wz) vs the demo, in the
                  model's native output space. Measures raw action accuracy.
  * ADE [cm]   -- Average Displacement Error: mean position gap between the
                  dead-reckoned predicted path and the GT path over ALL steps.
                  Measures whole-trajectory fidelity.
  * FDE [cm]   -- Final Displacement Error (= endpoint error): position gap at
                  the last step. Open-loop errors compound, so this is a
                  stress-test upper bound, not the deployed (closed-loop) error.
  * path%      -- FDE / GT path length: normalizes FDE by how far the robot
                  actually traveled in the segment.

The true deployment metric (closed-loop docking success within 1 cm) needs the
robot or a simulator; these are its best offline proxies. Docking-precision
itself is tracked separately by the aux head eval (test/dist_binned_error.py).

Run (from repo root):  CUDA_VISIBLE_DEVICES=1 python test/eval_openloop_metrics.py
Outputs: test/out/eval_openloop_metrics.{png,json}
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scripts.inference_ema_v2 import build_model_from_cfg, reconstruct_pose_rk4  # noqa: E402
from utils.docking_dataset import denormalize  # noqa: E402

# Env-overridable (same convention as dist_binned_error.py) so the identical
# tooling scores held-out data: EVAL_H5 / EVAL_CACHE.
H5 = os.environ.get("EVAL_H5", "dataset/after_0328_train.h5")
DINO_CACHE = os.environ.get("EVAL_CACHE", "dataset/after_0328_train_dino_bottom.h5")
RUNS = {
    "flow": ("outputs/train/flow_goal/2026-07-09_12-02-10", False),        # old, raw weights
    "auxw": ("outputs/train/flow_goal_auxw/2026-07-10_15-36-56", True),    # new, healthy EMA
}
CKPT_STEP = 4230
EPISODES = [0, 20, 50, 80, 110, 140]
MAX_STEPS = 500
N_SAMPLES = 8
SAMPLE_STEPS = 20
TRAJ_EMA_ALPHA = 0.3
OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
DT = 0.0333
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"   # remap via CUDA_VISIBLE_DEVICES
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load_episode(ep_idx):
    f = h5py.File(H5, "r")
    ends = f["episode_ends"][:].astype(int)
    s = 0 if ep_idx == 0 else ends[ep_idx - 1]
    e = ends[ep_idx]
    enc = f["encoder"][s:e].astype(np.float32)
    lid = f["lidar_points"][s:e].astype(np.float32)
    nlid = f["lidar_npoints"][s:e].astype(np.int64)
    cf = h5py.File(DINO_CACHE, "r")
    dino = cf["dino_bottom"][s:e]
    f.close(), cf.close()
    return enc, lid, nlid, dino


def rollout(nn_diffusion, solver, ep, act_min, act_scale, use_ema):
    """scripts/inference_ema_v2.py open-loop protocol, cache-fed."""
    enc, lid, nlid, dino = ep
    ep_steps = min(len(enc) - HORIZON + 1, MAX_STEPS)
    a_min, a_scale = torch.as_tensor(act_min), torch.as_tensor(act_scale)
    goal = torch.from_numpy(np.ascontiguousarray(dino[-1])).float().to(DEVICE)
    goal = goal.view(1, 1, 196, 768).repeat(N_SAMPLES, 1, 1, 1)

    torch.manual_seed(0)
    prev_ema, selected = None, []
    for t in range(ep_steps):
        rows = np.clip(np.arange(t - OBS_HORIZON + 1, t + 1), 0, None)
        vel = (2.0 * (torch.as_tensor(enc[rows]) - a_min) / a_scale - 1.0).float()
        feats = torch.from_numpy(dino[rows[::VISION_STRIDE]].astype(np.float32)).to(DEVICE)
        context = {
            "velocity": vel.unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1),
            "dino_feat2": feats.unsqueeze(0).repeat(N_SAMPLES, 1, 1, 1, 1).view(N_SAMPLES, -1, 196, 768),
            "goal_feat2": goal,
            "goal_mask": torch.ones(N_SAMPLES, device=DEVICE),
            "lidar_points": torch.from_numpy(lid[t]).unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1),
            "lidar_npoints": torch.tensor([nlid[t]], device=DEVICE).repeat(N_SAMPLES),
        }
        with torch.no_grad():
            prior = torch.randn(N_SAMPLES, HORIZON, 2, device=DEVICE)
            out = nn_diffusion.sample(
                solver=solver, w_cfg=1, prior=prior, condition_cfg=context,
                n_samples=N_SAMPLES, sample_steps=SAMPLE_STEPS, use_ema=use_ema)
            out = out[0] if isinstance(out, tuple) else out
            res = denormalize(out.cpu().numpy(), act_scale, act_min)
        current = res.mean(axis=0)
        ema = current if prev_ema is None else TRAJ_EMA_ALPHA * current + (1 - TRAJ_EMA_ALPHA) * prev_ema
        prev_ema = ema.copy()
        selected.append(ema[0, :])

    ai = np.array(selected)
    gt = enc[:ep_steps]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
    disp = np.hypot(*(gt_path[:, :2] - ai_path[:, :2]).T)
    path_len = float(np.sum(np.hypot(*np.diff(gt_path[:, :2], axis=0).T)))
    return dict(
        vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())),
        ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100),
        path_len_cm=path_len * 100,
        fde_over_path=float(disp[-1] / max(path_len, 1e-9)),
        gt_path=gt_path.tolist(), ai_path=ai_path.tolist(), steps=int(ep_steps))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    models = {}
    for name, (ckpt_dir, use_ema) in RUNS.items():
        cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
        _, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
        ck = torch.load(os.path.join(ckpt_dir, f"checkpoint_step_{CKPT_STEP}.pt"),
                        map_location=DEVICE, weights_only=False)
        nn_diffusion.model.load_state_dict(ck["model_state_dict"])
        nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
        nn_diffusion.eval()
        solver = "euler" if cfg.get("diffusion_backbone") == "rectified_flow" else "ode_dpmsolver++_2M"
        models[name] = (nn_diffusion, solver, ck["action_min"], ck["action_scale"], use_ema)

    results = {name: {} for name in models}
    for ep_idx in EPISODES:
        ep = load_episode(ep_idx)
        for name, (nnd, solver, amin, ascale, use_ema) in models.items():
            r = rollout(nnd, solver, ep, amin, ascale, use_ema)
            results[name][ep_idx] = r
            print(f"ep{ep_idx:3d} {name}: velRMSE {r['vel_rmse']:.4f} | ADE {r['ade_cm']:5.1f} cm | "
                  f"FDE {r['fde_cm']:5.1f} cm | GT path {r['path_len_cm']:5.0f} cm "
                  f"({r['steps']} steps)", flush=True)

    summary = {}
    for name in models:
        rs = results[name].values()
        summary[name] = {k: float(np.median([r[k] for r in rs]))
                         for k in ("vel_rmse", "ade_cm", "fde_cm", "fde_over_path")}
        print(f"{name} MEDIAN: velRMSE {summary[name]['vel_rmse']:.4f} | "
              f"ADE {summary[name]['ade_cm']:.1f} cm | FDE {summary[name]['fde_cm']:.1f} cm | "
              f"FDE/path {summary[name]['fde_over_path']*100:.0f}%")

    json.dump({"episodes": {n: {str(e): {k: v for k, v in r.items() if not k.endswith("_path")}
                                for e, r in rr.items()} for n, rr in results.items()},
               "summary": summary},
              open(os.path.join(OUT_DIR, "eval_openloop_metrics.json"), "w"), indent=1)

    n = len(EPISODES)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 8.6))
    for ax, ep_idx in zip(axes.ravel(), EPISODES):
        gt = np.array(results["flow"][ep_idx]["gt_path"])
        ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.4, alpha=0.6, label="GT")
        for name, col in (("flow", "#9ecae1"), ("auxw", "#e6550d")):
            p = np.array(results[name][ep_idx]["ai_path"])
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.1, alpha=0.9,
                    label=f"{name} (FDE {results[name][ep_idx]['fde_cm']:.1f} cm)")
        ax.scatter(*gt[0, :2], c="limegreen", s=60, zorder=5)
        ax.scatter(*gt[-1, :2], c="black", s=50, marker="X", zorder=5)
        ax.set_title(f"episode {ep_idx}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=7)
    plt.suptitle("open-loop rollouts: flow (old, raw) vs auxw (new, EMA)")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "eval_openloop_metrics.png")
    plt.savefig(png, dpi=110)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
