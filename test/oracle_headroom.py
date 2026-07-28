"""Stage-0 go/no-go gate for a trajectory-ranking critic.

A selector can only help if the candidate set CONTAINS a good trajectory. If the
policy systematically under-commands (the 2026-07-27 field diagnosis: 10/13
failures were "도커와 부딪힘", and the one success commanded 51% more |vx| and
41% more |wz| than the failures), then all M samples under-command and no
amount of ranking fixes it.

So before building any critic, measure the headroom:

    J(tau)  = w_th * wrap(theta_now - dpsi)^2  +  w_x * (x_H - x_goal)^2
    oracle  = min_m J(tau^(m))          <- best a PERFECT selector could do
    mean    = mean_m J(tau^(m))         <- what random selection gets
    first   = J(tau^(1))                <- what single-sample execution gets
    regret  = mean - oracle             <- the prize a critic is competing for

Interpretation:
  oracle << mean  -> selection has real headroom; the critic plan is worth building
  oracle ~= mean  -> every candidate is equally bad; fix the policy/data instead

Cost math is identical to test/eval_align_rgeo.py (same counterfactual identity
theta_dock(t+H) = theta_dock(t) - integral(w dt), verified corr 0.991 on demos),
so the numbers are directly comparable to the align_deg/xpos_mm already reported.

ICP CAVEAT: dock_pose is used ONLY to score candidates offline, never as a policy
input -- the same discipline the critic plan calls for.

Run:  EVAL_H5=dataset/after_0328_test.h5 EVAL_STATS_H5=dataset/after_0328_train.h5 \
      python test/oracle_headroom.py outputs/eval60k_r2cam_geo
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda key: {"runtime.cwd": REPO}[key])

from scripts.inference_ema_v2 import build_model_from_cfg  # noqa: E402
from utils.modular_dataset import ModularDockingDataset  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "test"))
from eval_run_rgeo import _resolve_sensor_files  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DT = 0.0333
HORIZON = 60
NEAR_M = 0.6              # score only the precision zone, like eval_align_rgeo
M_CANDIDATES = int(os.environ.get("N_CAND", "16"))
N_BLOCKS, BLOCK = 15, 256
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))
W_TH, W_X = 1.0, 1.0       # cost weights; ranking conclusions are scale-free

EVAL_H5 = os.environ.get("EVAL_H5", os.path.join(REPO, "dataset/after_0328_test.h5"))
STATS_H5 = os.environ.get("EVAL_STATS_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))


def _stack(batch):
    return {k: torch.stack([o[k] for o in batch]).to(DEVICE) for k in batch[0]}


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def candidate_costs(act, theta_now, dock_xy, x_goal):
    """act [M,H,2] denormalized -> (J [M], yaw_res_deg [M], x_res_mm [M])."""
    M, H, _ = act.shape
    dpsi = act[:, :, 1].sum(axis=1) * DT
    yaw_res = _wrap(theta_now - dpsi)

    x_res = np.empty(M)
    for m in range(M):
        px = py = pth = 0.0
        for k in range(H):
            v, w = float(act[m, k, 0]), float(act[m, k, 1])
            px += v * np.cos(pth) * DT
            py += v * np.sin(pth) * DT
            pth += w * DT
        c, s = np.cos(-pth), np.sin(-pth)
        rx, ry = dock_xy[0] - px, dock_xy[1] - py
        x_res[m] = (c * rx - s * ry) - x_goal
    J = W_TH * yaw_res ** 2 + W_X * x_res ** 2
    return J, np.degrees(np.abs(yaw_res)), np.abs(x_res) * 1000.0


@torch.no_grad()
def main(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    exp = str(cfg.experiment_name)
    ckpt = max(glob.glob(os.path.join(run_dir, "checkpoint_step_*.pt")),
               key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    _, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    nn_diffusion.model.load_state_dict(ck["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
    nn_diffusion.eval()
    solver = "euler" if cfg.get("diffusion_backbone") == "rectified_flow" else "ode_dpmsolver++_2M"
    act_min, act_scale = ck["action_min"], ck["action_scale"]

    sensors = _resolve_sensor_files(OmegaConf.to_container(cfg.sensors, resolve=True), EVAL_H5)
    ds = ModularDockingDataset(h5_path=EVAL_H5, sensors=sensors, horizon=cfg.horizon,
                               obs_horizon=cfg.obs_horizon, action_key=cfg.get("action_key", "encoder"),
                               train_h5_path=STATS_H5, action_norm=cfg.get("action_norm", "minmax"))
    with h5py.File(EVAL_H5, "r") as f:
        raw = f["dock_pose"][:].astype(np.float32)
        pose = np.nan_to_num(raw)
        rel = f["reliable"][:].astype(bool) & ~np.isnan(raw).any(axis=1)
        ends = f["episode_ends"][:]
    n_rows = len(pose)
    x_goal_all = np.zeros(n_rows, np.float32)
    s = 0
    for e in ends:
        e = int(e)
        iv = np.where(rel[s:e])[0]
        x_goal_all[s:e] = pose[s + iv[-1], 0] if len(iv) else 0.0
        s = e
    row2di = {int(t): i for i, t in enumerate(ds.index_map)}

    print(f"[{exp}] {os.path.basename(ckpt)} | M={M_CANDIDATES} candidates | solver={solver}", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    starts = np.sort(rng.choice(n_rows - BLOCK, size=N_BLOCKS, replace=False))

    Js, orc, mns, fst, yaw_o, yaw_m, x_o, x_m = [], [], [], [], [], [], [], []
    for st in starts:
        rows = np.arange(st, st + BLOCK)
        keep = (np.hypot(pose[rows, 0], pose[rows, 1]) < NEAR_M) & rel[rows]
        rows = np.array([r for r in rows[keep] if int(r) in row2di], np.int64)
        if not len(rows):
            continue
        for r in rows:
            obs = _stack([ds[row2di[int(r)]]["obs"]])
            rep = {k: v.repeat(M_CANDIDATES, *([1] * (v.dim() - 1))) for k, v in obs.items()}
            prior = torch.randn(M_CANDIDATES, HORIZON, 2, device=DEVICE)
            out = nn_diffusion.sample(solver=solver, w_cfg=1, prior=prior, condition_cfg=rep,
                                      n_samples=M_CANDIDATES, sample_steps=SAMPLE_STEPS, use_ema=True)
            out = out[0] if isinstance(out, tuple) else out
            act = ((out.cpu().numpy() + 1.0) / 2.0 * act_scale + act_min)   # [M,H,2]
            J, yd, xm = candidate_costs(act, pose[r, 2], pose[r, :2], x_goal_all[r])
            b = int(J.argmin())
            Js.append(J); orc.append(J[b]); mns.append(J.mean()); fst.append(J[0])
            yaw_o.append(yd[b]); yaw_m.append(yd.mean()); x_o.append(xm[b]); x_m.append(xm.mean())

    orc, mns, fst = map(np.array, (orc, mns, fst))
    yaw_o, yaw_m, x_o, x_m = map(np.array, (yaw_o, yaw_m, x_o, x_m))
    n = len(orc)
    print(f"\nscored {n} near-dock frames x {M_CANDIDATES} candidates\n")
    print(f"{'selector':<22}{'median J':>12}{'yaw (deg)':>12}{'xpos (mm)':>12}")
    print(f"{'first sample':<22}{np.median(fst):12.5f}{'':>12}{'':>12}")
    print(f"{'random (= mean)':<22}{np.median(mns):12.5f}{np.median(yaw_m):12.3f}{np.median(x_m):12.2f}")
    print(f"{'ORACLE (best of M)':<22}{np.median(orc):12.5f}{np.median(yaw_o):12.3f}{np.median(x_o):12.2f}")
    red = (1 - np.median(orc) / np.median(mns)) * 100
    print(f"\noracle vs random: J -{red:.1f}%   yaw {np.median(yaw_m):.3f} -> {np.median(yaw_o):.3f} deg   "
          f"xpos {np.median(x_m):.2f} -> {np.median(x_o):.2f} mm")
    print(f"regret a perfect critic would recover (median J): {np.median(mns - orc):.5f}")
    verdict = ("HEADROOM: selection is worth building" if red >= 30 else
               "MARGINAL: selection buys little" if red >= 15 else
               "NO HEADROOM: candidates are all alike -- fix the policy/data, not the selector")
    print(f"\nVERDICT: {verdict}")
    out_p = os.path.join(REPO, "test/out/rgeo", f"{exp}_oracle_headroom.json")
    json.dump(dict(run_dir=run_dir, ckpt=os.path.basename(ckpt), n_frames=int(n), M=M_CANDIDATES,
                   J_first=float(np.median(fst)), J_random=float(np.median(mns)),
                   J_oracle=float(np.median(orc)), reduction_pct=float(red),
                   yaw_random_deg=float(np.median(yaw_m)), yaw_oracle_deg=float(np.median(yaw_o)),
                   xpos_random_mm=float(np.median(x_m)), xpos_oracle_mm=float(np.median(x_o)),
                   verdict=verdict), open(out_p, "w"), indent=1)
    print(f"-> {out_p}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
