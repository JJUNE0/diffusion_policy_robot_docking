"""Corrected held-out eval for outputs/checkpoint_step_100000.pt (the "old
baseline" seed checkpoint that all graft_auxfb_* runs used as init_from),
using the ARCHITECTURE ACTUALLY CONFIRMED BY THE CHECKPOINT'S OWN STATE_DICT
(2026-07-26 forensic check), not either candidate config file:
  - legacy SensorFusionConditionNetwork + DiT1d (pooled AdaLN)
  - use_room1=True (room1_resampler + room2_resampler both present)
  - NO goal / lidar / aux_pose branches (absent from state_dict)
  - diffusion_backbone=ddpm (rectified_flow+ode_dpmsolver++_2M produced
    garbage predictions; ddpm+ode_dpmsolver++_2M gave sane ones)
  - horizon=60 (no error-magnitude cliff at step 16 in a 60-step rollout,
    contradicting the horizon=16 claim in configs/robot/old_bassline_smr.yaml)

Same metric math / output schema as test/eval_run_rgeo.py (ADE/FDE/velRMSE,
reconstruct_pose_rk4, same 10 held-out episodes) so the JSON is directly
comparable to test/out/rgeo/r_{nogoal,goal,geo}_heldout.json.

Run:  CUDA_VISIBLE_DEVICES=0 python test/eval_run_old_baseline_100k.py
"""
import json
import os
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.inference_ema_v2 import build_model_from_cfg, reconstruct_pose_rk4  # noqa: E402
from cleandiffuser.rollout_core import RolloutController  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "test"))
import terminal_metric as tm  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT_PATH = os.path.join(REPO, "outputs/checkpoint_step_100000.pt")
TEST_H5 = os.path.join(REPO, "dataset/after_0328_test.h5")
DINO_CACHE = os.path.join(REPO, "dataset/after_0328_test_dino_bottom.h5")  # has BOTH dino_bottom+dino_top
OUT = os.path.join(REPO, "test/out/rgeo/old_baseline_100k_heldout_v2.json")

OBS_HORIZON, VISION_STRIDE, HORIZON, DT = 30, 6, 60, 0.0333
N_SAMPLES = 8
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))
TRAJ_EMA_ALPHA = 0.3
MAX_STEPS = int(os.environ.get("EVAL_MAX_STEPS", "500")) or 10 ** 9
EVAL_EPISODES = [int(x) for x in os.environ.get("EVAL_EPISODES", "0,1,2,3,4,5,6,7,8,9").split(",")]


def episode_bounds(ep_idx):
    with h5py.File(TEST_H5, "r") as f:
        ends = f["episode_ends"][:].astype(int)
    return (0 if ep_idx == 0 else ends[ep_idx - 1]), ends[ep_idx]


def load_episode(ep_idx):
    f = h5py.File(TEST_H5, "r")
    ends = f["episode_ends"][:].astype(int)
    s = 0 if ep_idx == 0 else ends[ep_idx - 1]
    e = ends[ep_idx]
    enc = f["encoder"][s:e].astype(np.float32)
    f.close()
    cf = h5py.File(DINO_CACHE, "r")
    dino2 = cf["dino_bottom"][s:e]           # room2
    dino1 = cf["dino_top"][s:e] if "dino_top" in cf else None  # room1
    cf.close()
    return enc, dino2, dino1


@torch.no_grad()
def rollout(nn_diffusion, solver, ep, act_min, act_scale, use_ema, rem=None):
    enc, dino2, dino1 = ep
    ep_steps = min(len(enc) - HORIZON + 1, MAX_STEPS)
    torch.manual_seed(0)
    ctrl = RolloutController(
        nn_diffusion, solver=solver, sample_steps=SAMPLE_STEPS, use_ema=use_ema,
        n_samples=N_SAMPLES, horizon=HORIZON, w_cfg=1, agg="mean",
        traj_ema_alpha=TRAJ_EMA_ALPHA, warm_start=False, warm_level=0.3, device=DEVICE)

    selected = []
    for t in range(ep_steps):
        rows = np.clip(np.arange(t - OBS_HORIZON + 1, t + 1), 0, None)
        vel = (2.0 * (torch.as_tensor(enc[rows]) - act_min) / act_scale - 1.0).float()
        feats2 = torch.from_numpy(dino2[rows[::VISION_STRIDE]].astype(np.float32)).to(DEVICE)
        context = {
            "velocity": vel.unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1),
            "dino_feat2": feats2.unsqueeze(0).repeat(N_SAMPLES, 1, 1, 1, 1).view(N_SAMPLES, -1, 196, 768),
        }
        if dino1 is not None:
            feats1 = torch.from_numpy(dino1[rows[::VISION_STRIDE]].astype(np.float32)).to(DEVICE)
            context["dino_feat1"] = feats1.unsqueeze(0).repeat(N_SAMPLES, 1, 1, 1, 1).view(N_SAMPLES, -1, 196, 768)
        plan = ctrl.plan(context, act_min, act_scale, chunk_shift=1)
        selected.append(plan.ema[0, :])

    ai = np.array(selected)
    gt = enc[:ep_steps]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
    disp = np.hypot(*(gt_path[:, :2] - ai_path[:, :2]).T)
    path_len = float(np.sum(np.hypot(*np.diff(gt_path[:, :2], axis=0).T)))
    out = dict(
        vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())),
        ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100),
        path_len_cm=path_len * 100, fde_over_path=float(disp[-1] / max(path_len, 1e-9)),
        steps=int(ep_steps))
    # Steering symmetry + terminal mobility. This checkpoint is the ONLY model
    # that ever docked successfully in the field (ablation_study_2026-07.md
    # §2.12), and steering bias is the only offline metric whose readings have
    # matched field outcomes (terminal_metric.turn_symmetry docstring: >0.10 was
    # fatal in every observed case). Measuring the successful model's own bias
    # gives the threshold to screen candidates against — without it, picking a
    # demo model is guesswork.
    out["_wz"] = (ai[:, 1], gt[:, 1])
    if rem is not None:
        out["_term"] = (rem[:ep_steps], np.abs(ai[:, 0]) * 1000.0, np.abs(gt[:, 0]) * 1000.0)
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cfg = OmegaConf.load(os.path.join(REPO, "configs/robot/smr.yaml"))
    # Overridden to match what the CHECKPOINT'S OWN state_dict proved true,
    # not smr.yaml's own defaults (use_room1 defaults False there) nor
    # old_bassline_smr.yaml (missing/wrong on every field checked so far).
    cfg.use_room1 = True
    cfg.use_goal = False
    cfg.use_lidar_points = False
    cfg.use_aux_pose = False
    cfg.use_modular_fusion = False
    cfg.diffusion_backbone = "ddpm"

    nn_condition, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = nn_diffusion.model.load_state_dict(ck["model_state_dict"], strict=True)
    nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"], strict=True)
    nn_diffusion.eval()
    act_min, act_scale = ck["action_min"], ck["action_scale"]
    solver = "ode_dpmsolver++_2M"
    use_ema = True
    print(f"[old_baseline_100k] step={ck.get('step')} | solver={solver} | "
          f"episodes={EVAL_EPISODES}", flush=True)

    with h5py.File(TEST_H5, "r") as f:
        ends_all = f["episode_ends"][:]
        rem_all = (tm.remaining_mm(f["dock_pose"][:], f["reliable"][:], ends_all)
                   if "dock_pose" in f else None)

    rolls, term_parts, wz_parts = {}, [], []
    for ep_idx in EVAL_EPISODES:
        ep = load_episode(ep_idx)
        s0, _e0 = episode_bounds(ep_idx)
        r = rollout(nn_diffusion, solver, ep, act_min, act_scale, use_ema,
                    rem=None if rem_all is None else rem_all[s0:])
        term_parts.append(r.pop("_term", None))
        wz_parts.append(r.pop("_wz"))
        rolls[str(ep_idx)] = r
        print(f"[old_baseline_100k] ep{ep_idx}: ADE {r['ade_cm']:.1f} cm | FDE {r['fde_cm']:.1f} cm | "
              f"velRMSE {r['vel_rmse']:.4f}", flush=True)

    terminal = None
    parts = [p for p in term_parts if p is not None]
    if parts:
        got = tm.band_table(*[np.concatenate(c) for c in zip(*parts)])
        if got:
            terminal = tm.summarize(*got)
            tm.report("old_baseline_100k", terminal)
    sym = tm.turn_symmetry(*[np.concatenate(c) for c in zip(*wz_parts)])
    tm.report_symmetry("old_baseline_100k", sym)

    summary = dict(
        ade_cm=float(np.median([r["ade_cm"] for r in rolls.values()])),
        fde_cm=float(np.median([r["fde_cm"] for r in rolls.values()])),
        ade_cm_mean=float(np.mean([r["ade_cm"] for r in rolls.values()])),
        fde_cm_mean=float(np.mean([r["fde_cm"] for r in rolls.values()])),
        vel_rmse=float(np.median([r["vel_rmse"] for r in rolls.values()])),
        term_vx_ratio=terminal["term_vx_ratio"] if terminal else None,
        term_idle_frac=terminal["term_idle_frac"] if terminal else None,
        turn_right_frac=sym["policy_right_frac"], turn_bias=sym["bias"])
    result = dict(
        run_dir="outputs/checkpoint_step_100000.pt",
        ckpt="checkpoint_step_100000.pt",
        note="Re-eval with architecture/backbone empirically confirmed from the "
             "checkpoint's own state_dict (2026-07-26), NOT from configs/robot/"
             "old_bassline_smr.yaml (contradicted on dataset path, checkpoint_dir, "
             "use_room1, and diffusion_backbone) nor smr.yaml (contradicted on "
             "use_room1 and diffusion_backbone). See docs/0725_reloc3r_test/ "
             "reloc3r/ chat log for the forensic trail.",
        openloop=rolls, terminal=terminal, symmetry=sym, summary=summary)
    json.dump(result, open(OUT, "w"), indent=1)
    print(f"[old_baseline_100k] SUMMARY: ADE {summary['ade_cm']:.1f} cm | FDE med {summary['fde_cm']:.1f} / "
          f"mean {summary['fde_cm_mean']:.1f} cm | velRMSE {summary['vel_rmse']:.4f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
