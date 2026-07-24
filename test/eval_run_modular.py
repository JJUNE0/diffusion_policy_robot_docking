"""Held-out eval for MODULAR-fusion runs (use_modular_fusion=true).

The legacy eval stack (test/dist_binned_error.py::H5Batcher, test/
eval_openloop_metrics.py::rollout) hardcodes the legacy condition dict
(dino_feat2/goal_feat2/velocity/...), so it cannot score a model trained on the
config-driven modular path (ModularSensorFusionCondition). This file is the
modular analogue: it builds the condition ctx by REUSING ModularDockingDataset
(so per-sensor reads + normalization are bit-identical to training), while
reusing the aux/align scorers and rollout metric math from the legacy files
unchanged (zero regression risk to published legacy numbers).

Metrics + JSON schema match test/eval_run.py so results drop into the same
docs/ablation_study_2026-07.md comparison. Env vars mirror the legacy tooling:
  EVAL_H5     (held-out h5, default train)     EVAL_STATS_H5 (norm space, default train)
  EVAL_EPISODES, EVAL_TAG, EVAL_STEPS, EVAL_MAX_STEPS  (same as legacy)
Sensor sidecar caches (`file:` in the saved config) are auto-remapped from the
train cache to the eval cache by swapping the dataset stem (after_0328_train ->
after_0328_test) so the same saved config scores either split.

Run:  EVAL_H5=dataset/after_0328_test.h5 EVAL_TAG=heldout \
      python test/eval_run_modular.py outputs/train/reloc3r_rot/<timestamp>
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scripts.inference_ema_v2 import build_model_from_cfg, reconstruct_pose_rk4  # noqa: E402
from cleandiffuser.rollout_core import RolloutController  # noqa: E402
from utils.modular_dataset import ModularDockingDataset  # noqa: E402
from eval_run import aux_eval, align_eval, BIN_EDGES, OUT_DIR  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HORIZON, OBS_HORIZON, VISION_STRIDE, DT = 60, 30, 6, 0.0333
N_SAMPLES = 8
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))
TRAJ_EMA_ALPHA = 0.3
MAX_STEPS = int(os.environ.get("EVAL_MAX_STEPS", "500")) or 10 ** 9

EVAL_H5 = os.environ.get("EVAL_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))
STATS_H5 = os.environ.get("EVAL_STATS_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))
EVAL_EPISODES = [int(x) for x in os.environ.get("EVAL_EPISODES", "0,50,110").split(",")]


def _stem(h5_path):
    """after_0328_train / after_0328_test — used to remap sidecar caches."""
    b = os.path.basename(h5_path)
    m = re.match(r"(.*?)(?:_dino_\w+|_reloc3r\w*)?\.h5$", b)
    return m.group(1) if m else b[:-3]


def _resolve_sensor_files(sensors, eval_h5):
    """Resolve ${hydra:runtime.cwd} -> repo root and swap the train-cache stem to
    the eval-cache stem so a config saved on the train split scores the test
    split. Sensors that read from the main h5 (no `file:`) are left untouched."""
    train_stem, eval_stem = _stem(STATS_H5), _stem(eval_h5)
    out = {}
    for name, spec in sensors.items():
        spec = dict(spec)
        f = spec.get("file")
        if f is not None:
            f = f.replace("${hydra:runtime.cwd}", REPO)
            if not os.path.isabs(f):
                f = os.path.join(REPO, f)
            if train_stem != eval_stem:
                f = os.path.join(os.path.dirname(f),
                                 os.path.basename(f).replace(train_stem, eval_stem))
            spec["file"] = f
        out[name] = spec
    return out


class ModularEvalSource:
    """Modular analogue of dist_binned_error.H5Batcher: same pose/target/binning
    machinery, but the condition ctx is produced by ModularDockingDataset (exact
    train-time per-sensor reads + normalization). Only absolute dock-pose targets
    are supported (the modular configs here never use aux_relative)."""

    def __init__(self, cfg, eval_h5=EVAL_H5, stats_h5=STATS_H5):
        import h5py
        sensors = OmegaConf.to_container(cfg.sensors, resolve=True)
        sensors = _resolve_sensor_files(sensors, eval_h5)
        self.mds = ModularDockingDataset(
            h5_path=eval_h5, sensors=sensors, horizon=HORIZON, obs_horizon=OBS_HORIZON,
            action_key=cfg.get("action_key", "encoder"), train_h5_path=stats_h5)
        # bijection: usable frame row t  <->  dataset index i  (index_map holds
        # exactly the frames with a full horizon ahead, == the `ok` rows below).
        self.row_to_didx = {int(t): i for i, t in enumerate(self.mds.index_map)}

        f = h5py.File(eval_h5, "r")
        self.enc = f["encoder"][:].astype(np.float32)
        self.episode_ends = f["episode_ends"][:]
        pose = f["dock_pose"][:].astype(np.float32)
        rel = f["reliable"][:].astype(bool)
        self.pose = np.nan_to_num(pose)
        self.rel_valid = rel & ~np.isnan(pose).any(axis=1)
        f.close()

        with h5py.File(stats_h5, "r") as sf:
            s_pose = sf["dock_pose"][:].astype(np.float32)
            s_rel = sf["reliable"][:].astype(bool)
            s_xy = s_pose[s_rel & ~np.isnan(s_pose).any(axis=1)][:, :2]
            self.dock_xy_mean = s_xy.mean(0)
            self.dock_xy_std = s_xy.std(0) + 1e-6
        self.rel_xy_std = self.dock_xy_std          # aux_relative unsupported here

        n = int(self.episode_ends[-1])
        self.n_rows = n
        self.ep_end = np.zeros(n, np.int64)
        self.ok = np.zeros(n, bool)
        s = 0
        for e in self.episode_ends:
            self.ep_end[s:e] = e
            self.ok[s:e - HORIZON + 1] = True
            s = e

    def _collate(self, rows):
        obs_list = [self.mds[self.row_to_didx[int(t)]]["obs"] for t in rows]
        ctx = {}
        for k in obs_list[0]:
            ctx[k] = torch.stack([o[k] for o in obs_list]).to(DEVICE)
        return ctx

    def batch(self, idxs):
        idxs = np.asarray(idxs)
        ctx = self._collate(idxs)
        xy_n = (self.pose[idxs, :2] - self.dock_xy_mean) / self.dock_xy_std
        tgt = np.concatenate(
            [xy_n, np.sin(self.pose[idxs, 2:3]), np.cos(self.pose[idxs, 2:3])], 1)
        reliable = self.rel_valid[idxs]
        dock_d = np.hypot(self.pose[idxs, 0], self.pose[idxs, 1])
        return (ctx, torch.from_numpy(tgt.astype(np.float32)).to(DEVICE), reliable,
                torch.from_numpy(dock_d.astype(np.float32)).to(DEVICE))


@torch.no_grad()
def modular_rollout(nn_diffusion, solver, src, ep_idx, act_min, act_scale, use_ema):
    """Open-loop rollout mirroring eval_openloop_metrics.rollout's LEGACY path
    (per-frame resample, EMA step-0), but with ctx from ModularDockingDataset."""
    ends = src.episode_ends.astype(int)
    s = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
    e = int(ends[ep_idx])
    enc = src.enc[s:e]
    ep_steps = min(len(enc) - HORIZON + 1, MAX_STEPS)
    torch.manual_seed(0)
    ctrl = RolloutController(
        nn_diffusion, solver=solver, sample_steps=SAMPLE_STEPS, use_ema=use_ema,
        n_samples=N_SAMPLES, horizon=HORIZON, w_cfg=1, agg="mean",
        traj_ema_alpha=TRAJ_EMA_ALPHA, warm_start=False, warm_level=0.3, device=DEVICE)

    selected = []
    for t in range(ep_steps):
        obs = src.mds[src.row_to_didx[s + t]]["obs"]
        context = {k: v.unsqueeze(0).to(DEVICE).repeat(
            N_SAMPLES, *([1] * v.dim())) for k, v in obs.items()}
        plan = ctrl.plan(context, act_min, act_scale, chunk_shift=1)
        selected.append(plan.ema[0, :])

    ai = np.array(selected)
    gt = enc[:ep_steps]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
    disp = np.hypot(*(gt_path[:, :2] - ai_path[:, :2]).T)
    path_len = float(np.sum(np.hypot(*np.diff(gt_path[:, :2], axis=0).T)))
    vx_p, vx_g = ai[:, 0], gt[:, 0]
    faster = (np.sign(vx_p) == np.sign(vx_g)) & (np.abs(vx_p) >= np.abs(vx_g)) & (vx_g != 0)
    vx_err = np.where(faster, 0.0, vx_p - vx_g)
    wz_err = ai[:, 1] - gt[:, 1]
    return dict(
        vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())),
        vel_progress_rmse=float(np.sqrt(np.mean(np.concatenate([vx_err, wz_err]) ** 2))),
        speedup_frac=float(faster.mean()),
        ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100),
        path_len_cm=path_len * 100, fde_over_path=float(disp[-1] / max(path_len, 1e-9)),
        steps=int(ep_steps))


def main(run_dir):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    if not cfg.get("use_modular_fusion", False):
        raise SystemExit("This run is NOT modular; use test/eval_run.py instead.")
    exp = str(cfg.experiment_name)
    cks = glob.glob(os.path.join(run_dir, "checkpoint_step_*.pt"))
    ckpt_path = max(cks, key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    nn_condition, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    nn_diffusion.model.load_state_dict(ck["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
    nn_diffusion.eval()
    use_ema = float(cfg.get("ema_rate", 0.999)) <= 0.999
    solver = "euler" if cfg.get("diffusion_backbone") == "rectified_flow" else "ode_dpmsolver++_2M"
    print(f"[{exp}] {os.path.basename(ckpt_path)} | use_ema={use_ema} | solver={solver} | "
          f"EVAL_H5={os.path.basename(EVAL_H5)}", flush=True)

    result = dict(run_dir=run_dir, ckpt=os.path.basename(ckpt_path), use_ema=use_ema,
                  bin_edges=BIN_EDGES.tolist())
    src = ModularEvalSource(cfg)
    result["aux"] = aux_eval(nn_condition, cfg, src)
    if result["aux"]:
        print(f"[{exp}] aux near(<0.6m) median {result['aux']['near_median_mm']:.1f} mm "
              f"(p90 {result['aux']['near_p90_mm']:.1f})", flush=True)
    result["align"] = align_eval(nn_diffusion, solver, use_ema, src,
                                 ck["action_min"], ck["action_scale"])
    if result["align"]:
        a = result["align"]
        print(f"[{exp}] align(<0.6m) policy {a['policy_median_deg']:.2f} deg "
              f"(p90 {a['policy_p90_deg']:.2f}) | demo median {a['demo_median_deg']:.2f} "
              f"/ p25 {a['demo_p25_deg']:.2f} | n={a['n_frames']}", flush=True)

    rolls = {}
    for ep_idx in EVAL_EPISODES:
        r = modular_rollout(nn_diffusion, solver, src, ep_idx,
                            ck["action_min"], ck["action_scale"], use_ema)
        rolls[str(ep_idx)] = r
        print(f"[{exp}] ep{ep_idx}: ADE {r['ade_cm']:.1f} cm | FDE {r['fde_cm']:.1f} cm | "
              f"velRMSE {r['vel_rmse']:.4f}", flush=True)
    result["openloop"] = rolls
    result["summary"] = dict(
        ade_cm=float(np.median([r["ade_cm"] for r in rolls.values()])),
        fde_cm=float(np.median([r["fde_cm"] for r in rolls.values()])),
        ade_cm_mean=float(np.mean([r["ade_cm"] for r in rolls.values()])),
        fde_cm_mean=float(np.mean([r["fde_cm"] for r in rolls.values()])),
        vel_rmse=float(np.median([r["vel_rmse"] for r in rolls.values()])),
        near_mm=result["aux"]["near_median_mm"] if result["aux"] else None,
        align_deg=result["align"]["policy_median_deg"] if result["align"] else None)

    tag = os.environ.get("EVAL_TAG", "")
    out = os.path.join(OUT_DIR, f"{exp}{'_' + tag if tag else ''}.json")
    json.dump(result, open(out, "w"), indent=1)
    s = result["summary"]
    print(f"[{exp}] SUMMARY: near {s['near_mm']} mm | ADE {s['ade_cm']:.1f} cm | "
          f"FDE med {s['fde_cm']:.1f} / mean {s['fde_cm_mean']:.1f} cm -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
