"""Sample-ranking "MPC" at inference time (no training) on the best checkpoint.

At every step the policy samples N trajectories; the deployed code currently
averages them. Here we compare three aggregation strategies on the SAME samples
in one rollout pass:

  mean   -- current baseline (average all N, then trajectory-EMA)
  medoid -- pick the sample closest to the others (robust: outliers can't
            drag the average; still purely imitation-shaped)
  dock   -- task-aware ranking: integrate each sample, pick the one whose
            endpoint gets closest to the aux-head-predicted dock point
            (sensor->robot bearing rotation calibrated offline: -87.9 deg,
            matches the documented ~90 deg extrinsic). Falls back to mean
            when the predicted dock is outside the trusted range (>1.2 m).

Run (from repo root):  CUDA_VISIBLE_DEVICES=1 python test/mpc_rank.py
Outputs: test/out/mpc_rank.{png,json}
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "test"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scripts.inference_ema_v2 import build_model_from_cfg, reconstruct_pose_rk4  # noqa: E402
from utils.docking_dataset import denormalize  # noqa: E402
from eval_openloop_metrics import load_episode  # noqa: E402

RUN_DIR = "outputs/train/flow_goal_auxw2/2026-07-11_07-36-24"
CKPT_STEP = 8460
EPISODES = [0, 50, 110]
MAX_STEPS = 500
N_SAMPLES = 8
SAMPLE_STEPS = 20
TRAJ_EMA_ALPHA = 0.3
OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
DT = 0.0333
THETA_CAL = np.radians(-87.9)          # sensor->robot bearing rotation (calibrated)
DOCK_TRUST_M = 1.2                     # aux pred only trusted inside this range
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(REPO, "test", "out")
STRATS = ("mean", "medoid", "dock")


def dock_stats():
    with h5py.File("dataset/after_0328_train.h5", "r") as f:
        pose, rel = f["dock_pose"][:], f["reliable"][:].astype(bool)
    xy = pose[rel & ~np.isnan(pose).any(axis=1)][:, :2]
    return xy.mean(0), xy.std(0) + 1e-6


def integrate_batch(vw):
    """[N,H,2] (vx,wz) -> endpoint positions [N,2] in the current robot frame."""
    th = np.cumsum(np.concatenate([np.zeros((len(vw), 1)), vw[:, :-1, 1] * DT], axis=1), axis=1)
    x = np.sum(vw[:, :, 0] * np.cos(th) * DT, axis=1)
    y = np.sum(vw[:, :, 0] * np.sin(th) * DT, axis=1)
    return np.stack([x, y], axis=1)


def rollout_all(nn_diffusion, ep, act_min, act_scale, xy_mean, xy_std):
    enc, lid, nlid, dino = ep
    ep_steps = min(len(enc) - HORIZON + 1, MAX_STEPS)
    a_min, a_scale = torch.as_tensor(act_min), torch.as_tensor(act_scale)
    goal = torch.from_numpy(np.ascontiguousarray(dino[-1])).float().to(DEVICE)
    goal = goal.view(1, 1, 196, 768).repeat(N_SAMPLES, 1, 1, 1)
    ema_cond = nn_diffusion.model_ema["condition"]     # aux pred lives on the EMA copy
    c, s = np.cos(THETA_CAL), np.sin(THETA_CAL)
    R = np.array([[c, -s], [s, c]])

    torch.manual_seed(0)
    prev = {k: None for k in STRATS}
    sel = {k: [] for k in STRATS}
    dock_used = 0
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
                solver="euler", w_cfg=1, prior=prior, condition_cfg=context,
                n_samples=N_SAMPLES, sample_steps=SAMPLE_STEPS, use_ema=True)
            out = out[0] if isinstance(out, tuple) else out
            res = denormalize(out.cpu().numpy(), act_scale, act_min)   # [N,H,2]
            aux = ema_cond._aux_pred[0].cpu().numpy()                  # same for all N

        # --- three aggregations of the same N samples -----------------------
        cur = {"mean": res.mean(axis=0)}
        d2 = ((res[:, None] - res[None]) ** 2).sum(axis=(2, 3))        # pairwise, action space
        cur["medoid"] = res[d2.sum(axis=1).argmin()]
        dock_xy = R @ (aux[:2] * xy_std + xy_mean)                     # robot frame
        if np.hypot(*dock_xy) <= DOCK_TRUST_M:
            ends = integrate_batch(res)
            cur["dock"] = res[np.hypot(*(ends - dock_xy).T).argmin()]
            dock_used += 1
        else:
            cur["dock"] = cur["mean"]

        for k in STRATS:
            e = cur[k] if prev[k] is None else TRAJ_EMA_ALPHA * cur[k] + (1 - TRAJ_EMA_ALPHA) * prev[k]
            prev[k] = e.copy()
            sel[k].append(e[0, :])

    gt = enc[:ep_steps]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    out = {"dock_ranking_used_frac": dock_used / ep_steps}
    for k in STRATS:
        ai = np.array(sel[k])
        p = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
        disp = np.hypot(*(gt_path[:, :2] - p[:, :2]).T)
        out[k] = dict(ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100),
                      vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())), path=p.tolist())
    out["gt_path"] = gt_path.tolist()
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(RUN_DIR, "config.yaml"))
    _, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
    ck = torch.load(os.path.join(RUN_DIR, f"checkpoint_step_{CKPT_STEP}.pt"),
                    map_location=DEVICE, weights_only=False)
    nn_diffusion.model.load_state_dict(ck["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
    nn_diffusion.eval()
    xy_mean, xy_std = dock_stats()

    results = {}
    for ep_idx in EPISODES:
        r = rollout_all(nn_diffusion, load_episode(ep_idx), ck["action_min"], ck["action_scale"],
                        xy_mean, xy_std)
        results[str(ep_idx)] = r
        line = " | ".join(f"{k}: ADE {r[k]['ade_cm']:.1f} FDE {r[k]['fde_cm']:.1f}" for k in STRATS)
        print(f"ep{ep_idx:3d} ({r['dock_ranking_used_frac']*100:.0f}% dock-ranked) {line}", flush=True)

    med = {k: dict(ade_cm=float(np.median([results[e][k]["ade_cm"] for e in results])),
                   fde_cm=float(np.median([results[e][k]["fde_cm"] for e in results])))
           for k in STRATS}
    for k in STRATS:
        print(f"MEDIAN {k}: ADE {med[k]['ade_cm']:.1f} cm | FDE {med[k]['fde_cm']:.1f} cm")

    json.dump({"episodes": {e: {k: {kk: vv for kk, vv in r[k].items() if kk != "path"}
                                for k in STRATS} | {"dock_ranking_used_frac": r["dock_ranking_used_frac"]}
                            for e, r in results.items()},
               "median": med, "theta_cal_deg": float(np.degrees(THETA_CAL))},
              open(os.path.join(OUT_DIR, "mpc_rank.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, len(EPISODES), figsize=(4.6 * len(EPISODES), 4.8))
    for ax, ep_idx in zip(np.atleast_1d(axes), EPISODES):
        r = results[str(ep_idx)]
        gt = np.array(r["gt_path"])
        ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.4, alpha=0.6, label="GT")
        for k, col in zip(STRATS, ("#9ecae1", "#31a354", "#e6550d")):
            p = np.array(r[k]["path"])
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.1, alpha=0.9,
                    label=f"{k} (FDE {r[k]['fde_cm']:.1f})")
        ax.scatter(*gt[0, :2], c="limegreen", s=60, zorder=5)
        ax.scatter(*gt[-1, :2], c="black", s=50, marker="X", zorder=5)
        ax.set_title(f"ep {ep_idx}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=7)
    plt.suptitle("sample aggregation: mean vs medoid vs dock-ranked (auxw2)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mpc_rank.png"), dpi=110)
    print("saved test/out/mpc_rank.png")


if __name__ == "__main__":
    main()
