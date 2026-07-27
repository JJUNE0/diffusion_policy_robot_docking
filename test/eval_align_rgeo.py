"""Counterfactual TERMINAL-ALIGNMENT eval for the token-sequence architecture
(R-NoGoal / R-Goal / R-Geo), i.e. the axis the paper's claim actually lives on.

Why this exists: test/eval_run_rgeo.py reports ADE/FDE/velRMSE, which score
imitation fidelity over the whole 500-step approach and are dominated by the
approach phase. The paper's claim is about last-centimeter alignment. Those two
axes demonstrably disagree — old_baseline_100k wins ADE/FDE/velRMSE but loses
align_deg (2.10 vs 1.28) and xpos_mm (12.19 vs 8.62) to s20_batch256. So the
three R-* arms had never been measured on the axis being claimed.

Math is copied verbatim from test/eval_run.py::align_eval so the numbers are
directly comparable to the legacy runs in test/out/weekend/*.json. Only the
CONTEXT CONSTRUCTION differs: ModularDockingDataset (config-driven, token
sequence) instead of the DINO-hardcoded H5Batcher.

  theta_dock(t+H) = theta_dock(t) - integral(wz dt)

is exact (translation does not rotate the frame), so a policy commanding dpsi
over its horizon is left with |theta_now - dpsi| of misalignment. Scored on
near-dock frames, that IS the policy's docking-alignment quality — no
compounding, no imitation reference.

ICP CAVEAT: theta_now / x_goal come from the ICP dock_pose labels. Training is
ICP-free (that is the point, see reloc3r_0725.md); ICP enters here ONLY as a
measurement instrument, and the same instrument scores policy and demo, so the
POLICY-vs-DEMO gap is fair even though absolute degrees inherit ICP error.
Report as "ICP-free training, ICP-instrumented evaluation".

Run:  EVAL_H5=dataset/after_0328_test.h5 EVAL_STATS_H5=dataset/after_0328_train.h5 \
      EVAL_TAG=heldout python test/eval_align_rgeo.py outputs/train/r_geo/<timestamp>
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

# NOT `from test.eval_run_rgeo import ...`: Python's stdlib owns the name `test`,
# so the repo's test/ dir never wins that import. Put test/ itself on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_run_rgeo import _resolve_sensor_files  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DT = 0.0333
HORIZON = 60
# identical to test/eval_run.py::align_eval so legacy numbers stay comparable
ALIGN_NEAR_M = 0.6
ALIGN_N_SAMPLES = 4
N_BLOCKS = 15
BLOCK = 256
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))
OUT_DIR = os.path.join(REPO, "test/out/rgeo")

EVAL_H5 = os.environ.get("EVAL_H5", os.path.join(REPO, "dataset/after_0328_test.h5"))
STATS_H5 = os.environ.get("EVAL_STATS_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))


def _stack(batch_obs):
    """[{k: tensor}] -> {k: [B, ...]} on DEVICE."""
    keys = batch_obs[0].keys()
    return {k: torch.stack([o[k] for o in batch_obs]).to(DEVICE) for k in keys}


@torch.no_grad()
def align_eval(nn_diffusion, solver, use_ema, ds, act_min, act_scale,
               pose, rel_valid, enc, row_to_didx, episode_ends):
    rng = np.random.default_rng(0)
    # Seed the DIFFUSION PRIOR too, not just frame selection. Without this the
    # same arm re-scored drifts ~0.04 deg run-to-run (measured: r_goal 1.46 ->
    # 1.50), which is the same order as some arm-vs-arm gaps — so unseeded runs
    # cannot be compared to each other at all.
    torch.manual_seed(0)
    n_rows = len(enc)
    starts = np.sort(rng.choice(n_rows - BLOCK, size=N_BLOCKS, replace=False))
    res_pol, res_demo, xerr_pol, xerr_demo = [], [], [], []
    _rows_seen = []

    # per-row docked forward-x: x of each episode's last reliable frame
    x_goal_all = np.zeros(n_rows, np.float32)
    s = 0
    for e in episode_ends:
        e = int(e)
        iv = np.where(rel_valid[s:e])[0]
        x_goal_all[s:e] = pose[s + iv[-1], 0] if len(iv) else 0.0
        s = e

    for st in starts:
        rows = np.arange(st, st + BLOCK)
        dock_d = np.hypot(pose[rows, 0], pose[rows, 1])
        # precision zone + trustworthy ICP label + a usable dataset window
        keep = (dock_d < ALIGN_NEAR_M) & rel_valid[rows]
        rows = rows[keep]
        rows = np.array([r for r in rows if int(r) in row_to_didx], dtype=np.int64)
        if not len(rows):
            continue
        _rows_seen.append(rows)

        obs = _stack([ds[row_to_didx[int(r)]]["obs"] for r in rows])
        B = len(rows)
        rep = {k: v.repeat_interleave(ALIGN_N_SAMPLES, dim=0) for k, v in obs.items()}
        prior = torch.randn(B * ALIGN_N_SAMPLES, HORIZON, 2, device=DEVICE)
        out = nn_diffusion.sample(solver=solver, w_cfg=1, prior=prior, condition_cfg=rep,
                                  n_samples=B * ALIGN_N_SAMPLES, sample_steps=SAMPLE_STEPS,
                                  use_ema=use_ema)
        out = out[0] if isinstance(out, tuple) else out
        act = ((out.cpu().numpy() + 1.0) / 2.0 * act_scale + act_min)
        act = act.reshape(B, ALIGN_N_SAMPLES, HORIZON, 2).mean(axis=1)     # [B,H,2]

        theta_now = pose[rows, 2]
        dpsi_pol = act[:, :, 1].sum(axis=1) * DT
        dpsi_demo = np.stack([enc[r:r + HORIZON, 1].sum() * DT for r in rows])
        res_pol.append(np.abs(theta_now - dpsi_pol))
        res_demo.append(np.abs(theta_now - dpsi_demo))

        # Counterfactual forward-x: integrate SE(2) motion under the policy's
        # (vx,wz) from the ICP start pose, express the dock in the frame at t+H.
        for arr, sink in ((act, xerr_pol), (None, xerr_demo)):
            for j, r in enumerate(rows):
                a_seq = arr[j] if arr is not None else enc[r:r + HORIZON]
                dx, dy = pose[r, 0], pose[r, 1]
                px, py, pth = 0.0, 0.0, 0.0
                for k in range(HORIZON):
                    v, w = float(a_seq[k, 0]), float(a_seq[k, 1])
                    px += v * np.cos(pth) * DT
                    py += v * np.sin(pth) * DT
                    pth += w * DT
                c, sN = np.cos(-pth), np.sin(-pth)
                rx = dx - px, dy - py
                x_H = c * rx[0] - sN * rx[1]
                sink.append(abs(x_H - x_goal_all[r]))

    if not res_pol:
        return None
    p = np.degrees(np.concatenate(res_pol))
    d = np.degrees(np.concatenate(res_demo))
    xp = np.array(xerr_pol) * 1000.0
    xd = np.array(xerr_demo) * 1000.0
    # Per-frame residuals, not just medians. Frame SELECTION depends only on
    # dock_pose/reliable/rng-seed — never on the model — so all arms score the
    # exact same 480 frames in the same order. That makes arm-vs-arm a PAIRED
    # comparison (Wilcoxon), which is the only way a 1.24-vs-1.44 deg median
    # gap can be shown to be real rather than frame-sampling noise.
    align_eval.per_frame = dict(policy_deg=p, demo_deg=d, policy_mm=xp, demo_mm=xd,
                                rows=np.concatenate([np.asarray(r) for r in _rows_seen]))
    return dict(policy_median_deg=float(np.median(p)), policy_p90_deg=float(np.percentile(p, 90)),
                demo_median_deg=float(np.median(d)), demo_p25_deg=float(np.percentile(d, 25)),
                demo_p90_deg=float(np.percentile(d, 90)), n_frames=int(len(p)),
                x_policy_median_mm=float(np.median(xp)), x_policy_p90_mm=float(np.percentile(xp, 90)),
                x_demo_median_mm=float(np.median(xd)), x_demo_p25_mm=float(np.percentile(xd, 25)))


def main(run_dir):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    if not cfg.get("use_token_sequence_fusion", False):
        raise SystemExit("Not an R-* token-sequence run; use test/eval_run.py::align_eval instead.")
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

    sensors = _resolve_sensor_files(OmegaConf.to_container(cfg.sensors, resolve=True), EVAL_H5)
    ds = ModularDockingDataset(
        h5_path=EVAL_H5, sensors=sensors, horizon=cfg.horizon, obs_horizon=cfg.obs_horizon,
        action_key=cfg.get("action_key", "encoder"), train_h5_path=STATS_H5,
        action_norm=cfg.get("action_norm", "minmax"))
    for nm, got, want in (("action_min", ds.action_min, ck["action_min"]),
                          ("action_scale", ds.action_scale, ck["action_scale"])):
        if not np.allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-4):
            raise SystemExit(f"{nm} mismatch: dataset {np.asarray(got)} vs checkpoint {np.asarray(want)}")

    with h5py.File(EVAL_H5, "r") as f:
        raw_pose = f["dock_pose"][:].astype(np.float32)
        pose = np.nan_to_num(raw_pose)
        rel_valid = f["reliable"][:].astype(bool) & ~np.isnan(raw_pose).any(axis=1)
        enc = f["encoder"][:].astype(np.float32)
        episode_ends = f["episode_ends"][:]

    row_to_didx = {int(t): i for i, t in enumerate(ds.index_map)}
    print(f"[{exp}] {os.path.basename(ckpt_path)} | solver={solver} | use_ema={use_ema} | "
          f"EVAL_H5={os.path.basename(EVAL_H5)}", flush=True)

    align = align_eval(nn_diffusion, solver, use_ema, ds, ck["action_min"], ck["action_scale"],
                       pose, rel_valid, enc, row_to_didx, episode_ends)
    if align is None:
        raise SystemExit("no near-dock frames with reliable ICP labels found")

    tag = os.environ.get("EVAL_TAG", "")
    out = os.path.join(OUT_DIR, f"{exp}{'_' + tag if tag else ''}_align.json")
    json.dump(dict(run_dir=run_dir, ckpt=os.path.basename(ckpt_path), align=align),
              open(out, "w"), indent=1)
    np.savez(out.replace(".json", "_perframe.npz"), **align_eval.per_frame)
    print(f"[{exp}] ALIGN: policy {align['policy_median_deg']:.2f} deg (p90 "
          f"{align['policy_p90_deg']:.2f}) vs demo {align['demo_median_deg']:.2f} "
          f"(p25 {align['demo_p25_deg']:.2f}) | xpos {align['x_policy_median_mm']:.1f} mm "
          f"vs demo {align['x_demo_median_mm']:.1f} | n={align['n_frames']} -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
