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
from cleandiffuser.rollout_core import RolloutController  # noqa: E402

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
# 500 truncates the ~1900-step test episodes: FDE is then "error at step 500",
# not the docking-endpoint error. EVAL_MAX_STEPS=0 -> roll to the episode end.
MAX_STEPS = int(os.environ.get("EVAL_MAX_STEPS", "500")) or 10**9
N_SAMPLES = 8
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))   # sampling-steps sweep (Phase 1)
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
    # room1 features when the cache carries them (merged 2-camera cache);
    # None for the plain bottom-only caches -> rollout() skips dino_feat1.
    dino1 = cf["dino_top"][s:e] if "dino_top" in cf else None
    f.close(), cf.close()
    return enc, lid, nlid, dino, dino1


# Action chunking / temporal-consistency (2026-07-21, user-requested).
# Defaults reproduce the exact legacy per-frame-resample behavior byte-for-
# byte -- every previously published eval number in ablation_study_2026-07.md
# stays reproducible. Opt in via env vars for a NEW eval variant (see
# EVAL_CHUNK_K/EVAL_WARM_START docstring note below); results get their own
# EVAL_TAG so they land in separate JSON files, never overwriting old ones.
#   EVAL_CHUNK_K=1        (default) resample every frame, exactly like before.
#   EVAL_CHUNK_K=K>1       resample every K frames, execute that plan's [0:K]
#                          raw steps verbatim (no cross-time blending needed --
#                          one coherent sampled trajectory is already smooth).
#   EVAL_WARM_START=1      instead of pure-noise prior, warm-start the next
#                          sample from the PREVIOUS plan shifted by K steps
#                          (its actual continuation) via cleandiffuser's native
#                          SDEdit-style warm_start_reference/forward_level --
#                          this is the "real" temporal-consistency fix; the
#                          legacy per-index TRAJ_EMA blend (kept below,
#                          unchanged) mixes DIFFERENT absolute timesteps and is
#                          not a substitute for it.
#   EVAL_WARM_LEVEL=0.3    fraction of the full noise schedule to re-add to the
#                          warm-started reference before re-denoising (same
#                          knob as cleandiffuser's own default).
EXEC_CHUNK_K = int(os.environ.get("EVAL_CHUNK_K", "1"))
WARM_START = os.environ.get("EVAL_WARM_START", "0") == "1"
WARM_LEVEL = float(os.environ.get("EVAL_WARM_LEVEL", "0.3"))
_LEGACY_ROLLOUT = (EXEC_CHUNK_K == 1 and not WARM_START)


def rollout(nn_diffusion, solver, ep, act_min, act_scale, use_ema):
    """scripts/inference_ema_v2.py open-loop protocol, cache-fed."""
    enc, lid, nlid, dino, dino1 = ep
    ep_steps = min(len(enc) - HORIZON + 1, MAX_STEPS)
    a_min, a_scale = torch.as_tensor(act_min), torch.as_tensor(act_scale)
    goal = torch.from_numpy(np.ascontiguousarray(dino[-1])).float().to(DEVICE)
    goal = goal.view(1, 1, 196, 768).repeat(N_SAMPLES, 1, 1, 1)
    goal1 = None
    if dino1 is not None:
        goal1 = torch.from_numpy(np.ascontiguousarray(dino1[-1])).float().to(DEVICE)
        goal1 = goal1.view(1, 1, 196, 768).repeat(N_SAMPLES, 1, 1, 1)
    # Goal frame's own scan (episode's last row, same frame `goal` above uses).
    # Always included: the condition net only consumes it when
    # use_goal_lidar=True; omitting it for a goal-lidar model would silently
    # drop a token modality it was trained to always receive (see
    # test/dist_binned_error.py H5Batcher for the same fix on the aux-eval side).
    goal_lidar_pts = torch.from_numpy(lid[-1]).unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1)
    goal_lidar_npts = torch.tensor([nlid[-1]], device=DEVICE).repeat(N_SAMPLES)

    torch.manual_seed(0)
    ctrl = RolloutController(
        nn_diffusion, solver=solver, sample_steps=SAMPLE_STEPS, use_ema=use_ema,
        n_samples=N_SAMPLES, horizon=HORIZON, w_cfg=1, agg="mean",
        traj_ema_alpha=TRAJ_EMA_ALPHA, warm_start=WARM_START, warm_level=WARM_LEVEL,
        device=DEVICE)
    selected = []
    t = 0
    while t < ep_steps:
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
            "goal_lidar_points": goal_lidar_pts,
            "goal_lidar_npoints": goal_lidar_npts,
        }
        if dino1 is not None:
            feats1 = torch.from_numpy(dino1[rows[::VISION_STRIDE]].astype(np.float32)).to(DEVICE)
            context["dino_feat1"] = feats1.unsqueeze(0).repeat(N_SAMPLES, 1, 1, 1, 1).view(N_SAMPLES, -1, 196, 768)
            context["goal_feat1"] = goal1

        k = 1 if _LEGACY_ROLLOUT else min(EXEC_CHUNK_K, ep_steps - t, HORIZON)
        plan = ctrl.plan(context, act_min, act_scale, chunk_shift=k)
        if _LEGACY_ROLLOUT:
            selected.append(plan.ema[0, :])                    # EMA on, step 0 only
        else:
            selected.extend(list(plan.current[0:k]))           # raw chunk, no EMA
        t += k

    ai = np.array(selected)
    gt = enc[:ep_steps]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
    disp = np.hypot(*(gt_path[:, :2] - ai_path[:, :2]).T)
    path_len = float(np.sum(np.hypot(*np.diff(gt_path[:, :2], axis=0).T)))

    # Asymmetric forward-speed error (user request 2026-07-14): plain vel_rmse
    # penalizes a policy that drives FASTER than the demo just as much as one
    # that drives slower or wrong-direction -- but faster-in-the-same-direction
    # is exactly what AWR/residual-RL experiments are trying to achieve, so it
    # should score as neutral (0), not as an error. Only vx (forward progress)
    # gets this treatment; wz (turning) stays symmetric since faster turning
    # near the dock is an alignment risk, not a progress win.
    vx_p, vx_g = ai[:, 0], gt[:, 0]
    faster_same_dir = (np.sign(vx_p) == np.sign(vx_g)) & (np.abs(vx_p) >= np.abs(vx_g)) & (vx_g != 0)
    vx_err = np.where(faster_same_dir, 0.0, vx_p - vx_g)
    wz_err = ai[:, 1] - gt[:, 1]                                    # symmetric, unchanged

    return dict(
        vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())),
        vel_progress_rmse=float(np.sqrt(np.mean(np.concatenate([vx_err, wz_err]) ** 2))),
        speedup_frac=float(faster_same_dir.mean()),                 # how often AWR/RL behavior shows up
        ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100),
        path_len_cm=path_len * 100,
        fde_over_path=float(disp[-1] / max(path_len, 1e-9)),
        exec_chunk_k=EXEC_CHUNK_K, warm_start=WARM_START, warm_level=WARM_LEVEL if WARM_START else None,
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
