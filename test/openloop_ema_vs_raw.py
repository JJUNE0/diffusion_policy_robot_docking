"""Open-loop rollout: EMA vs raw weights on the step-4230 goal checkpoints.

Follow-up to test/dist_binned_error.py, which showed the EMA weights of the
2026-07-09 runs still carry ~65% of the random init (ema_rate 0.9999 vs only
4230 steps). This re-runs the scripts/inference_ema_v2.py open-loop protocol
(same n_samples/trajectory-EMA/solver) on one episode with use_ema True vs
False to quantify how much past EMA-based evaluations were distorted.

Reads DINO features from the precomputed cache (one sequential slice per
episode) instead of live-encoding pixels, so no DINO backbone is needed and
the whole episode fits in RAM. See memory h5-io-quirks-this-machine.

Run (from repo root):  python test/openloop_ema_vs_raw.py
Outputs: test/out/openloop_ema_vs_raw.png, test/out/openloop_ema_vs_raw.json
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

H5 = "dataset/after_0328_train.h5"
DINO_CACHE = "dataset/after_0328_train_dino_bottom.h5"
RUNS = {
    "flow": "outputs/train/flow_goal/2026-07-09_12-02-10",
    "auxw": "outputs/train/flow_goal_auxw/2026-07-10_15-36-56",
}
CKPT_STEP = 4230
EPISODE = 0
N_SAMPLES = 8
SAMPLE_STEPS = 20
TRAJ_EMA_ALPHA = 0.3
OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
DT = 0.0333
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load_episode():
    """Preload everything episode EPISODE needs (sequential h5 reads)."""
    f = h5py.File(H5, "r")
    ends = f["episode_ends"][:].astype(int)
    s = 0 if EPISODE == 0 else ends[EPISODE - 1]
    e = ends[EPISODE]
    enc = f["encoder"][s:e].astype(np.float32)
    lid = f["lidar_points"][s:e].astype(np.float32)
    nlid = f["lidar_npoints"][s:e].astype(np.int64)
    cf = h5py.File(DINO_CACHE, "r")
    dino = cf["dino_bottom"][s:e]                      # [T,196,768] fp16, one seq read
    return enc, lid, nlid, dino


def rollout(nn_diffusion, cfg, ep, act_min, act_scale, use_ema: bool):
    """scripts/inference_ema_v2.py open-loop protocol, cache-fed."""
    enc, lid, nlid, dino = ep
    T = len(enc)
    ep_steps = T - HORIZON + 1
    a_min = torch.as_tensor(act_min)
    a_scale = torch.as_tensor(act_scale)

    def norm(a):
        return (2.0 * (torch.as_tensor(a) - a_min) / a_scale - 1.0).float()

    goal = torch.from_numpy(np.ascontiguousarray(dino[-1])).float().to(DEVICE)
    goal = goal.view(1, 1, 196, 768).repeat(N_SAMPLES, 1, 1, 1)
    if cfg.get("diffusion_backbone") == "rectified_flow":
        solver = "euler"
    else:
        # the run configs say "dpmsolver++", which this cleandiffuser version
        # spells "ode_dpmsolver++_2M" (see diffusionsde.SUPPORTED_SOLVERS)
        solver = "ode_dpmsolver++_2M"

    torch.manual_seed(0)                               # same priors for both variants
    prev_ema, selected = None, []
    for t in range(ep_steps):
        rows = np.clip(np.arange(t - OBS_HORIZON + 1, t + 1), 0, None)
        vel = norm(enc[rows]).unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1)
        feats = torch.from_numpy(dino[rows[::VISION_STRIDE]].astype(np.float32)).to(DEVICE)
        context = {
            "velocity": vel,
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
        if (t + 1) % 100 == 0:
            print(f"    step {t + 1}/{ep_steps}", flush=True)

    ai = np.array(selected)
    gt = enc[:ep_steps]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
    end_err = float(np.hypot(*(gt_path[-1, :2] - ai_path[-1, :2])))
    vel_rmse = float(np.sqrt(((ai - gt) ** 2).mean()))
    return dict(gt_path=gt_path, ai_path=ai_path, end_err_cm=end_err * 100, vel_rmse=vel_rmse)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ep = load_episode()
    print(f"episode {EPISODE}: {len(ep[0])} frames")

    results = {}
    for run, ckpt_dir in RUNS.items():
        cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
        nn_condition, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
        ck = torch.load(os.path.join(ckpt_dir, f"checkpoint_step_{CKPT_STEP}.pt"),
                        map_location=DEVICE, weights_only=False)
        nn_diffusion.model.load_state_dict(ck["model_state_dict"])
        nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
        nn_diffusion.eval()
        for use_ema in (False, True):
            name = f"{run}-{'ema' if use_ema else 'raw'}"
            print(f"[{name}] rolling out ...", flush=True)
            r = rollout(nn_diffusion, cfg, ep, ck["action_min"], ck["action_scale"], use_ema)
            results[name] = r
            print(f"[{name}] endpoint err {r['end_err_cm']:.1f} cm | vel RMSE {r['vel_rmse']:.4f}")

    json.dump({k: dict(end_err_cm=v["end_err_cm"], vel_rmse=v["vel_rmse"]) for k, v in results.items()},
              open(os.path.join(OUT_DIR, "openloop_ema_vs_raw.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, run in zip(axes, RUNS):
        gt = results[f"{run}-raw"]["gt_path"]
        ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.5, alpha=0.6, label="ground truth")
        for use_ema, col in ((False, "#e6550d"), (True, "#3182bd")):
            name = f"{run}-{'ema' if use_ema else 'raw'}"
            p = results[name]["ai_path"]
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.2, alpha=0.85,
                    label=f"{'EMA' if use_ema else 'raw'} (end err {results[name]['end_err_cm']:.1f} cm)")
            ax.scatter(*p[-1, :2], color=col, s=70, marker="X", zorder=5)
        ax.scatter(*gt[0, :2], c="limegreen", s=90, marker="o", zorder=5, label="start")
        ax.scatter(*gt[-1, :2], c="black", s=70, marker="X", zorder=5)
        ax.set_title(f"{run}_goal step {CKPT_STEP} | ep {EPISODE} open-loop")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "openloop_ema_vs_raw.png")
    plt.savefig(png, dpi=120)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
