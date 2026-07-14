"""Open-loop rollout visualizer — watch what the policy is "thinking" per step.

Renders an mp4 per episode showing, at every rendered step:
  left  : the room2 camera frame the model sees (+ goal image inset)
  right : top-down view —
          - GT demo path (grey, full) + current GT robot pose (black dot)
          - executed AI path so far (green = integrated selected actions)
          - the 8 sampled candidate horizons at this step (thin orange)
          - the trajectory-EMA selected horizon (bold red)
          - current lidar scan (blue dots, sensor->robot rotation -87.9 deg)
          - ICP dock label (black star) and aux-head dock prediction (magenta x)

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=0 python test/viz_rollout.py <run_dir> --episode 6 --heldout
  # e.g. run_dir = outputs/train/flow_goal_scratch20/2026-07-12_05-19-04

Output: test/out/viz/<experiment>_ep<N>[_heldout].mp4
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "test"))

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scripts.inference_ema_v2 import build_model_from_cfg, reconstruct_pose_rk4  # noqa: E402
from utils.docking_dataset import denormalize  # noqa: E402

OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
DT = 0.0333
N_SAMPLES = 8
SAMPLE_STEPS = 20
TRAJ_EMA_ALPHA = 0.3
THETA_CAL = np.radians(-87.9)      # sensor->robot bearing rotation (test/mpc_rank.py)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(REPO, "test", "out", "viz")


def rot2d(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def integrate(vw, x0=0.0, y0=0.0, th0=0.0):
    """[T,2] (vx,wz) -> [T+1,3] pose path starting at (x0,y0,th0)."""
    path = np.zeros((len(vw) + 1, 3))
    path[0] = (x0, y0, th0)
    x, y, th = x0, y0, th0
    for i, (vx, wz) in enumerate(vw):
        x += vx * np.cos(th) * DT
        y += vx * np.sin(th) * DT
        th += wz * DT
        path[i + 1] = (x, y, th)
    return path


def to_global(pts_robot, pose):
    """[N,2] robot-frame points -> global frame given robot pose [x,y,th]."""
    return pts_robot @ rot2d(pose[2]).T + pose[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--stride", type=int, default=5, help="render every Nth step")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    h5_path = "dataset/after_0328_test.h5" if args.heldout else "dataset/after_0328_train.h5"
    cache_path = ("dataset/after_0328_test_dino_bottom.h5" if args.heldout
                  else "dataset/after_0328_train_dino_bottom.h5")

    # --- model (same loading as test/eval_run.py) ---------------------------
    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    exp = str(cfg.experiment_name)
    cks = glob.glob(os.path.join(args.run_dir, "checkpoint_step_*.pt"))
    ckpt_path = max(cks, key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    _, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    nn_diffusion.model.load_state_dict(ck["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
    nn_diffusion.eval()
    use_ema = float(cfg.get("ema_rate", 0.999)) <= 0.999
    solver = "euler" if cfg.get("diffusion_backbone") == "rectified_flow" else "ode_dpmsolver++_2M"
    aux_relative = bool(cfg.get("aux_relative", False))
    ema_cond = nn_diffusion.model_ema["condition"] if use_ema else nn_diffusion.model["condition"]

    # absolute dock-pose stats (train h5) for aux denormalization
    with h5py.File("dataset/after_0328_train.h5", "r") as f:
        p_, r_ = f["dock_pose"][:], f["reliable"][:].astype(bool)
    xy_ = p_[r_ & ~np.isnan(p_).any(1)][:, :2]
    aux_mean, aux_std = xy_.mean(0), xy_.std(0) + 1e-6

    # --- episode data --------------------------------------------------------
    f = h5py.File(h5_path, "r")
    ends = f["episode_ends"][:].astype(int)
    s = 0 if args.episode == 0 else ends[args.episode - 1]
    e = ends[args.episode]
    enc = f["encoder"][s:e].astype(np.float32)
    lid = f["lidar_points"][s:e].astype(np.float32)
    nlid = f["lidar_npoints"][s:e].astype(np.int64)
    pose_lbl = f["dock_pose"][s:e].astype(np.float32)
    rel_lbl = f["reliable"][s:e].astype(bool)
    with h5py.File(cache_path, "r") as cf:
        dino = cf["dino_bottom"][s:e]
    goal_img = f["image_bottom"][e - 1 - s + s].transpose(1, 2, 0)  # RGB HWC

    ep_steps = min(len(enc) - HORIZON + 1, args.max_steps)
    render_ts = list(range(0, ep_steps, args.stride))
    imgs = {t: f["image_bottom"][s + t].transpose(1, 2, 0) for t in render_ts}

    a_min, a_scale = torch.as_tensor(ck["action_min"]), torch.as_tensor(ck["action_scale"])
    goal_feat = torch.from_numpy(np.ascontiguousarray(dino[-1])).float().to(DEVICE)
    goal_feat = goal_feat.view(1, 1, 196, 768).repeat(N_SAMPLES, 1, 1, 1)
    goal_lidar_pts = torch.from_numpy(lid[-1]).unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1)
    goal_lidar_npts = torch.tensor([nlid[-1]], device=DEVICE).repeat(N_SAMPLES)

    gt_path = reconstruct_pose_rk4(enc[:ep_steps, 0], enc[:ep_steps, 1], dt=DT)

    # --- rollout, recording per-step artifacts at render steps ---------------
    torch.manual_seed(0)
    prev_ema, selected, frames_data = None, [], {}
    for t in range(ep_steps):
        rows = np.clip(np.arange(t - OBS_HORIZON + 1, t + 1), 0, None)
        vel = (2.0 * (torch.as_tensor(enc[rows]) - a_min) / a_scale - 1.0).float()
        feats = torch.from_numpy(dino[rows[::VISION_STRIDE]].astype(np.float32)).to(DEVICE)
        context = {
            "velocity": vel.unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1),
            "dino_feat2": feats.unsqueeze(0).repeat(N_SAMPLES, 1, 1, 1, 1).view(N_SAMPLES, -1, 196, 768),
            "goal_feat2": goal_feat,
            "goal_mask": torch.ones(N_SAMPLES, device=DEVICE),
            "lidar_points": torch.from_numpy(lid[t]).unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1),
            "lidar_npoints": torch.tensor([nlid[t]], device=DEVICE).repeat(N_SAMPLES),
            "goal_lidar_points": goal_lidar_pts,
            "goal_lidar_npoints": goal_lidar_npts,
        }
        with torch.no_grad():
            prior = torch.randn(N_SAMPLES, HORIZON, 2, device=DEVICE)
            out = nn_diffusion.sample(solver=solver, w_cfg=1, prior=prior, condition_cfg=context,
                                      n_samples=N_SAMPLES, sample_steps=SAMPLE_STEPS, use_ema=use_ema)
            out = out[0] if isinstance(out, tuple) else out
            res = denormalize(out.cpu().numpy(), ck["action_scale"], ck["action_min"])
            aux = ema_cond._aux_pred[0].cpu().numpy() if ema_cond._aux_pred is not None else None

        current = res.mean(axis=0)
        ema = current if prev_ema is None else TRAJ_EMA_ALPHA * current + (1 - TRAJ_EMA_ALPHA) * prev_ema
        prev_ema = ema.copy()
        selected.append(ema[0, :])

        if t in imgs:
            frames_data[t] = dict(samples=res.copy(), ema=ema.copy(), aux=aux,
                                  exec_so_far=np.array(selected).copy())

    # --- render ---------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"_heldout" if args.heldout else ""
    out_mp4 = os.path.join(OUT_DIR, f"{exp}_ep{args.episode}{tag}.mp4")
    R_cal = rot2d(THETA_CAL)

    fig = plt.figure(figsize=(12.8, 5.4), dpi=100)
    axL = fig.add_axes([0.02, 0.05, 0.42, 0.9])
    axR = fig.add_axes([0.50, 0.08, 0.48, 0.86])
    writer = None
    for t in render_ts:
        d = frames_data[t]
        axL.clear(), axR.clear()
        # left: camera + goal inset
        axL.imshow(imgs[t]); axL.axis("off")
        axL.set_title(f"room2 camera @ step {t}  ({exp})", fontsize=10)
        gh, gw = goal_img.shape[0] // 4, goal_img.shape[1] // 4
        axL.imshow(goal_img, extent=(imgs[t].shape[1] - gw, imgs[t].shape[1], gh, 0))
        axL.text(imgs[t].shape[1] - gw, gh + 14, "goal", fontsize=7, color="w",
                 bbox=dict(fc="k", alpha=.6, pad=1))

        # right: top-down (global = GT dead-reckoned frame)
        pose_t = gt_path[t]
        axR.plot(gt_path[:, 0], gt_path[:, 1], color="0.6", lw=1.2, ls="--", label="GT demo path")
        ex = d["exec_so_far"]
        ai_path = reconstruct_pose_rk4(ex[:, 0], ex[:, 1], dt=DT)
        axR.plot(ai_path[:, 0], ai_path[:, 1], color="#2ca02c", lw=1.6, label="AI executed (open-loop)")
        for k in range(N_SAMPLES):
            ph = integrate(d["samples"][k], *pose_t)
            axR.plot(ph[:, 0], ph[:, 1], color="#ff9d45", lw=0.7, alpha=0.5,
                     label="samples (x8)" if k == 0 else None)
        ph = integrate(d["ema"], *pose_t)
        axR.plot(ph[:, 0], ph[:, 1], color="#d62728", lw=2.0, label="selected horizon (2s)")
        # lidar + dock label/pred (sensor->robot rotation, then to global)
        scan = lid[t][:nlid[t]] @ R_cal.T
        g = to_global(scan, pose_t)
        axR.scatter(g[:, 0], g[:, 1], s=2, c="#1f77b4", alpha=0.5, label="lidar")
        if rel_lbl[t] and not np.isnan(pose_lbl[t]).any():
            dk = to_global((pose_lbl[t, :2] @ R_cal.T)[None], pose_t)[0]
            axR.scatter(*dk, marker="*", s=140, c="k", label="dock (ICP label)")
        if d["aux"] is not None and not aux_relative:
            pred_xy = d["aux"][:2] * aux_std + aux_mean
            dp = to_global((pred_xy @ R_cal.T)[None], pose_t)[0]
            axR.scatter(*dp, marker="x", s=90, c="m", label="dock (aux pred)")
        axR.scatter(*pose_t[:2], s=50, c="k", zorder=5)
        axR.set_aspect("equal", adjustable="datalim")
        axR.grid(alpha=0.3, ls="--")
        axR.legend(fontsize=7, loc="upper right")
        axR.set_title(f"top-down | step {t}/{ep_steps}", fontsize=10)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if writer is None:
            writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (buf.shape[1], buf.shape[0]))
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved {out_mp4} ({len(render_ts)} frames)")


if __name__ == "__main__":
    main()
