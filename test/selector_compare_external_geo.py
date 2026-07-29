"""Corrected version of selector_compare.py (2026-07-28, user-flagged bug).

The original selector_compare.py ran on `r2cam_geo`, whose `sensors:` spec
ALREADY includes `geometry` as a TRUNK INPUT. So its "67% recovery" result
never isolated "selection-only" benefit from "the trunk already conditions on
geometry" -- both were entangled in the same checkpoint. It cannot support the
claim "put geometry in the selector, not the trunk."

This script tests that claim properly: the policy is `r2cam_nogoal`, whose
`sensors:` spec has NO goal and NO geometry (confirmed: wheel, rgb_history,
rgb_history_top, lidar only). Candidates come from a model that has NEVER seen
a Reloc3r token. The geometry token used for SELECTION is read directly from
the offline Reloc3r cache (`geometry_bottom` in the *_reloc3r_bottom.h5 files,
row-aligned 1:1 with the main dataset) -- completely decoupled from the
policy's forward pass. This is what "selection-only, not input" actually means.

Run:  EVAL_H5=dataset/after_0328_test.h5 EVAL_STATS_H5=dataset/after_0328_train.h5 \
      python test/selector_compare_external_geo.py outputs/ck_r2cam_nogoal_60k
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
from oracle_headroom import candidate_costs, _wrap, NEAR_M, N_BLOCKS, BLOCK  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DT, HORIZON = 0.0333, 60
M = int(os.environ.get("N_CAND", "16"))
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))
EVAL_H5 = os.environ.get("EVAL_H5", os.path.join(REPO, "dataset/after_0328_test.h5"))
STATS_H5 = os.environ.get("EVAL_STATS_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))


def _geometry_cache_path(eval_h5):
    """Row-aligned Reloc3r geometry cache for whichever split EVAL_H5 is."""
    stem = re.sub(r"\.h5$", "", os.path.basename(eval_h5))
    return os.path.join(os.path.dirname(eval_h5), f"{stem}_reloc3r_bottom.h5")


@torch.no_grad()
def main(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    exp = str(cfg.experiment_name)
    sensors = _resolve_sensor_files(OmegaConf.to_container(cfg.sensors, resolve=True), EVAL_H5)
    assert "geometry" not in sensors, (
        f"{exp} has `geometry` in its sensors spec -- this script requires a policy that "
        f"NEVER sees the geometry token, or the selection-only claim is not being tested. "
        f"Use r_nogoal / r2cam_nogoal, not r_geo / r2cam_geo.")
    print(f"[{exp}] confirmed geometry-free policy. sensors={list(sensors)}", flush=True)

    ckpt = max(glob.glob(os.path.join(run_dir, "checkpoint_step_*.pt")),
               key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    _, nn = build_model_from_cfg(cfg, DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    nn.model.load_state_dict(ck["model_state_dict"])
    nn.model_ema.load_state_dict(ck["ema_state_dict"])
    nn.eval()
    solver = "euler" if cfg.get("diffusion_backbone") == "rectified_flow" else "ode_dpmsolver++_2M"
    a_min, a_scale = ck["action_min"], ck["action_scale"]

    ds = ModularDockingDataset(h5_path=EVAL_H5, sensors=sensors, horizon=cfg.horizon,
                               obs_horizon=cfg.obs_horizon, action_key=cfg.get("action_key", "encoder"),
                               train_h5_path=STATS_H5, action_norm=cfg.get("action_norm", "minmax"))
    with h5py.File(EVAL_H5, "r") as f:
        raw = f["dock_pose"][:].astype(np.float32)
        pose, rel = np.nan_to_num(raw), f["reliable"][:].astype(bool) & ~np.isnan(raw).any(axis=1)
        ends = f["episode_ends"][:]
    n_rows = len(pose)
    xg = np.zeros(n_rows, np.float32)
    s = 0
    for e in ends:
        e = int(e)
        iv = np.where(rel[s:e])[0]
        xg[s:e] = pose[s + iv[-1], 0] if len(iv) else 0.0
        s = e
    row2di = {int(t): i for i, t in enumerate(ds.index_map)}

    geo_path = _geometry_cache_path(EVAL_H5)
    with h5py.File(geo_path, "r") as gf:
        geometry_bottom = gf["geometry_bottom"][:]                     # [N,4], row-aligned to EVAL_H5
    assert len(geometry_bottom) == n_rows, (
        f"{geo_path} has {len(geometry_bottom)} rows but {EVAL_H5} has {n_rows}")
    print(f"[{exp}] external geometry from {os.path.basename(geo_path)} "
          f"(NOT part of the policy's condition_cfg)", flush=True)

    print(f"[{exp}] {os.path.basename(ckpt)} | M={M}", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    starts = np.sort(rng.choice(n_rows - BLOCK, size=N_BLOCKS, replace=False))

    KEYS = ("first", "meanagg", "hand", "oracle")
    res = {k: [] for k in KEYS}; yaw = {k: [] for k in KEYS}
    psi_g, th_icp = [], []
    for st in starts:
        rows = np.arange(st, st + BLOCK)
        keep = (np.hypot(pose[rows, 0], pose[rows, 1]) < NEAR_M) & rel[rows]
        rows = np.array([r for r in rows[keep] if int(r) in row2di], np.int64)
        for r in rows:
            item = ds[row2di[int(r)]]["obs"]                            # NO "geometry" key here
            obs = {k: v.unsqueeze(0).to(DEVICE) for k, v in item.items()}
            rep = {k: v.repeat(M, *([1] * (v.dim() - 1))) for k, v in obs.items()}
            out = nn.sample(solver=solver, w_cfg=1, prior=torch.randn(M, HORIZON, 2, device=DEVICE),
                            condition_cfg=rep, n_samples=M, sample_steps=SAMPLE_STEPS, use_ema=True)
            out = out[0] if isinstance(out, tuple) else out
            act = ((out.cpu().numpy() + 1.0) / 2.0 * a_scale + a_min)   # [M,H,2] -- policy never saw geometry

            J, yd, _xm = candidate_costs(act, pose[r, 2], pose[r, :2], xg[r])

            # selection-only geometry: read straight from the cache, bypassing the model entirely
            g = geometry_bottom[int(r)].astype(np.float32)
            psi = float(np.arctan2(g[2], g[3]))
            psi_g.append(psi); th_icp.append(float(pose[r, 2]))
            dpsi = act[:, :, 1].sum(axis=1) * DT
            J_geo = _wrap(psi - dpsi) ** 2

            mean_act = act.mean(axis=0, keepdims=True)
            Jm, ydm, _ = candidate_costs(mean_act, pose[r, 2], pose[r, :2], xg[r])

            res["first"].append(J[0]);            yaw["first"].append(yd[0])
            res["meanagg"].append(Jm[0]);         yaw["meanagg"].append(ydm[0])
            h = int(J_geo.argmin())
            res["hand"].append(J[h]);             yaw["hand"].append(yd[h])
            o = int(J.argmin())
            res["oracle"].append(J[o]);           yaw["oracle"].append(yd[o])

    for k in KEYS:
        res[k] = np.array(res[k]); yaw[k] = np.array(yaw[k])
    n = len(res["first"])
    base, orc = np.median(res["meanagg"]), np.median(res["oracle"])
    print(f"\nscored {n} near-dock frames x {M} candidates | policy sensors={list(sensors)}\n")
    print(f"{'selector':<32}{'median J':>12}{'yaw (deg)':>12}{'recovery':>11}")
    labels = dict(first="first sample", meanagg="mean-agg (deployed today)",
                  hand="hand (EXTERNAL geometry, selection-only)",
                  oracle="ORACLE (ICP, upper bound)")
    for k in KEYS:
        rec = (base - np.median(res[k])) / max(base - orc, 1e-12) * 100
        print(f"{labels[k]:<32}{np.median(res[k]):12.5f}{np.median(yaw[k]):12.3f}{rec:10.0f}%")
    c = float(np.corrcoef(np.array(psi_g), np.array(th_icp))[0, 1])
    print(f"\ncorr(psi_geometry, theta_icp) = {c:+.3f}")
    print(f"\nThis policy NEVER received `geometry` in condition_cfg. Any recovery above")
    print(f"comes ONLY from using the external Reloc3r token to pick among candidates that")
    print(f"a geometry-blind policy already produced -- the clean test of the selection-only claim.")
    out_p = os.path.join(REPO, "test/out/rgeo", f"{exp}_selector_compare_external_geo.json")
    json.dump({k: dict(median_J=float(np.median(res[k])), median_yaw_deg=float(np.median(yaw[k])))
               for k in KEYS} | dict(n_frames=int(n), M=M, corr_psi_theta=c,
                                     ckpt=os.path.basename(ckpt), policy_sensors=list(sensors),
                                     geometry_source="external_cache_not_policy_input"),
              open(out_p, "w"), indent=1)
    print(f"-> {out_p}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
