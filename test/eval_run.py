"""Generic per-run evaluation for the weekend ablation queue.

Scores one training run on the two axes the user cares about:
  * precision    -- aux dock-pose error, distance-binned (median/p90, <0.6 m)
  * goal-reaching -- open-loop ADE/FDE on 3 fixed episodes (proxy for
                     time-to-goal until closed-loop eval exists)

Usage:  python test/eval_run.py <run_dir e.g. outputs/train/flow_goal_auxw2/...>
Writes: test/out/weekend/<experiment_name>.json (+ prints a summary line)
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "test"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scripts.inference_ema_v2 import build_model_from_cfg  # noqa: E402
from dist_binned_error import H5Batcher, BIN_EDGES, bin_stats  # noqa: E402
from eval_openloop_metrics import load_episode, rollout  # noqa: E402

# Env-overridable: EVAL_EPISODES="0,4,8" (indices into EVAL_H5's episodes)
EVAL_EPISODES = [int(x) for x in os.environ.get("EVAL_EPISODES", "0,50,110").split(",")]
N_AUX_BLOCKS = 15
BLOCK = 256
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(REPO, "test", "out", "weekend")
# Counterfactual alignment metric (see align_eval)
ALIGN_NEAR_M = 0.6          # only score frames inside the precision zone
ALIGN_N_SAMPLES = 4         # policy samples averaged per frame
HORIZON, DT = 60, 0.0333


def latest_ckpt(run_dir):
    cks = glob.glob(os.path.join(run_dir, "checkpoint_step_*.pt"))
    return max(cks, key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))


@torch.no_grad()
def aux_eval(nn_condition, cfg, src):
    """Distance-binned aux error over N_AUX_BLOCKS contiguous blocks.

    The (pred, target) comparison must happen in whatever space the model was
    actually trained on: absolute dock pose (dock_xy_std) normally, or the
    current->goal RELATIVE pose (src.rel_xy_std) when aux_relative=True — see
    utils/docking_dataset.py. Mixing the two silently produces a meaningless
    mm number (caught 2026-07-13 on flow_goal_glidar: 60mm vs the true ~5mm).
    Distance BINNING always uses the raw absolute dock distance from
    src.batch()'s dock_d, regardless of target space, so results across
    absolute/relative runs stay comparable by physical regime.
    """
    aux_relative = bool(cfg.get("aux_relative", False))
    std_t = torch.as_tensor(src.rel_xy_std if aux_relative else src.dock_xy_std, device=DEVICE)
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(src.n_rows - BLOCK, size=N_AUX_BLOCKS, replace=False))
    ds, mms = [], []
    for s in starts:
        blk = np.arange(s, s + BLOCK)
        blk = blk[src.ok[blk]]
        if not len(blk):
            continue
        ctx, tgt, rel, dock_d = src.batch(blk)
        nn_condition(ctx)
        pred = nn_condition._aux_pred
        if pred is None:
            return None
        relm = torch.from_numpy(rel).to(DEVICE)
        mm = torch.hypot(*((pred[:, :2] - tgt[:, :2]) * std_t).T) * 1000.0
        ds.append(dock_d[relm].cpu().numpy())
        mms.append(mm[relm].cpu().numpy())
    d, mm = np.concatenate(ds), np.concatenate(mms)
    med, p90, cnt = bin_stats(d, mm)
    near = d < 0.6
    return dict(bin_median=med.tolist(), bin_p90=p90.tolist(), bin_count=cnt.tolist(),
                near_median_mm=float(np.median(mm[near])), near_p90_mm=float(np.percentile(mm[near], 90)),
                supervised_median_mm=float(np.median(mm[d <= 1.1])))


@torch.no_grad()
def align_eval(nn_diffusion, solver, use_ema, src, act_min, act_scale):
    """Counterfactual FINAL-ALIGNMENT error (deg) — does the policy command the
    rotation that squares the robot up with the dock?

    Why not just "roll out and see where it stops": open-loop eval feeds GT
    observations, so the robot's real pose is always the demo's pose no matter
    what the policy commands — the endpoint tells us nothing about the policy's
    docking precision. We need a counterfactual.

    The one we can compute exactly: the dock's yaw in the robot frame changes
    ONLY when the robot rotates (translation does not rotate the frame), i.e.

        theta_dock(t+H) = theta_dock(t) - integral(wz dt)

    Verified on the demos: corr(dtheta, -dpsi) = 0.991 (0.980 near-dock).
    (Regression slope is 1.069, not 1.0 — the wheel-odometry wz under-reports
    rotation by ~7%. It biases policy and demo identically, so comparisons are
    fair, but do not read the absolute degrees as calibrated truth.)

    So for a frame with dock misalignment theta_now, a policy commanding a
    heading change dpsi over its horizon would be left with |theta_now - dpsi|.
    That residual, scored on near-dock frames, IS the policy's docking-alignment
    quality — no compounding, no imitation reference.

    Reported next to the demo's own residual:
      demo_median = what an average demo achieves -> what PLAIN BC imitates.
      demo_p25    = what the better demos achieve -> the level AWR is trying to
                    pull the policy toward (reward-weighting up-weights the good
                    tail; it cannot beat the best demo, that needs online RL).
    A working precision reward should move policy_median from ~demo_median down
    toward ~demo_p25.
    """
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(src.n_rows - BLOCK, size=N_AUX_BLOCKS, replace=False))
    # per-row docked forward-x (goal reference for the counterfactual x error):
    # x of each episode's last reliable frame, broadcast to all its rows.
    x_goal_all = np.zeros(src.n_rows, np.float32)
    _s = 0
    for _e in src.episode_ends:
        _iv = np.where(src.rel_valid[_s:_e])[0]
        x_goal_all[_s:_e] = src.pose[_s + _iv[-1], 0] if len(_iv) else 0.0
        _s = _e
    res_pol, res_demo, xerr_pol, xerr_demo = [], [], [], []
    for s in starts:
        blk = np.arange(s, s + BLOCK)
        blk = blk[src.ok[blk]]
        if not len(blk):
            continue
        ctx, _, rel, dock_d = src.batch(blk)
        # precision zone + a trustworthy ICP label only
        keep = (dock_d.cpu().numpy() < ALIGN_NEAR_M) & rel
        if not keep.any():
            continue
        idx = np.where(keep)[0]
        rows = blk[idx]
        x_goal_row = x_goal_all[rows]

        sub = {k: v[idx] for k, v in ctx.items()}
        B = len(idx)
        rep = {k: v.repeat_interleave(ALIGN_N_SAMPLES, dim=0) for k, v in sub.items()}
        prior = torch.randn(B * ALIGN_N_SAMPLES, HORIZON, 2, device=DEVICE)
        out = nn_diffusion.sample(solver=solver, w_cfg=1, prior=prior, condition_cfg=rep,
                                  n_samples=B * ALIGN_N_SAMPLES,
                                  sample_steps=int(os.environ.get("EVAL_STEPS", "20")), use_ema=use_ema)
        out = out[0] if isinstance(out, tuple) else out
        act = ((out.cpu().numpy() + 1.0) / 2.0 * act_scale + act_min)      # [B*N, H, 2]
        act = act.reshape(B, ALIGN_N_SAMPLES, HORIZON, 2).mean(axis=1)      # [B, H, 2]

        theta_now = src.pose[rows, 2]                                       # ICP dock yaw (rad)
        dpsi_pol = act[:, :, 1].sum(axis=1) * DT                            # policy's commanded rotation
        dpsi_demo = np.stack([src.enc[r:r + HORIZON, 1].sum() * DT for r in rows])
        res_pol.append(np.abs(theta_now - dpsi_pol))
        res_demo.append(np.abs(theta_now - dpsi_demo))

        # Counterfactual forward-x error: integrate the robot's SE(2) motion
        # under the policy's (vx,wz) from the ICP-measured start pose, express
        # the dock in the frame at t+H, and take |x - x_goal|. Position needs
        # full integration (translation + rotation compound), unlike the yaw
        # identity above. x only: y has no learnable signal (1.2x ICP noise).
        for arr, sink in ((act, xerr_pol), (None, xerr_demo)):
            for j, r in enumerate(rows):
                a_seq = arr[j] if arr is not None else src.enc[r:r + HORIZON]
                # dock pose in the CURRENT robot frame
                dx, dy, dth = src.pose[r, 0], src.pose[r, 1], src.pose[r, 2]
                px, py, pth = 0.0, 0.0, 0.0                                  # robot pose accumulator
                for k in range(HORIZON):
                    v, w = float(a_seq[k, 0]), float(a_seq[k, 1])
                    px += v * np.cos(pth) * DT
                    py += v * np.sin(pth) * DT
                    pth += w * DT
                # dock in the frame at t+H = inverse(robot motion) applied to dock
                c, sN = np.cos(-pth), np.sin(-pth)
                rx = dx - px, dy - py
                x_H = c * rx[0] - sN * rx[1]                                 # forward-x of dock at t+H
                sink.append(abs(x_H - x_goal_row[j]))

    if not res_pol:
        return None
    p = np.degrees(np.concatenate(res_pol))
    d = np.degrees(np.concatenate(res_demo))
    xp = np.array(xerr_pol) * 1000.0                                         # mm
    xd = np.array(xerr_demo) * 1000.0
    return dict(policy_median_deg=float(np.median(p)), policy_p90_deg=float(np.percentile(p, 90)),
                demo_median_deg=float(np.median(d)), demo_p25_deg=float(np.percentile(d, 25)),
                demo_p90_deg=float(np.percentile(d, 90)), n_frames=int(len(p)),
                x_policy_median_mm=float(np.median(xp)), x_policy_p90_mm=float(np.percentile(xp, 90)),
                x_demo_median_mm=float(np.median(xd)), x_demo_p25_mm=float(np.percentile(xd, 25)))


def main(run_dir):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    exp = str(cfg.experiment_name)
    ckpt_path = latest_ckpt(run_dir)
    nn_condition, nn_diffusion = build_model_from_cfg(cfg, DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    nn_diffusion.model.load_state_dict(ck["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
    nn_diffusion.eval()
    # runs trained with ema_rate <= 0.999 have a healthy EMA -> evaluate it;
    # anything with the old 0.9999 rate must be scored on raw weights
    use_ema = float(cfg.get("ema_rate", 0.999)) <= 0.999
    solver = "euler" if cfg.get("diffusion_backbone") == "rectified_flow" else "ode_dpmsolver++_2M"
    print(f"[{exp}] {os.path.basename(ckpt_path)} | use_ema={use_ema} | solver={solver}", flush=True)

    result = dict(run_dir=run_dir, ckpt=os.path.basename(ckpt_path), use_ema=use_ema,
                  bin_edges=BIN_EDGES.tolist())

    src = H5Batcher(aux_relative=bool(cfg.get("aux_relative", False)))
    result["aux"] = aux_eval(nn_condition, cfg, src)
    if result["aux"]:
        print(f"[{exp}] aux near(<0.6m) median {result['aux']['near_median_mm']:.1f} mm "
              f"(p90 {result['aux']['near_p90_mm']:.1f})", flush=True)
    # docking-ALIGNMENT quality (the axis the precision AWR reward targets).
    # aux near_mm above measures PERCEPTION (does the model know where the dock
    # is); this measures CONTROL (does it command the rotation that squares up).
    result["align"] = align_eval(nn_diffusion, solver, use_ema, src,
                                 ck["action_min"], ck["action_scale"])
    if result["align"]:
        a = result["align"]
        print(f"[{exp}] align(<0.6m) policy {a['policy_median_deg']:.2f} deg "
              f"(p90 {a['policy_p90_deg']:.2f}) | demo median {a['demo_median_deg']:.2f} "
              f"[BC target] / p25 {a['demo_p25_deg']:.2f} [AWR target] | n={a['n_frames']}", flush=True)
        print(f"[{exp}] xpos (<0.6m) policy {a['x_policy_median_mm']:.1f} mm "
              f"(p90 {a['x_policy_p90_mm']:.1f}) | demo median {a['x_demo_median_mm']:.1f} "
              f"/ p25 {a['x_demo_p25_mm']:.1f}", flush=True)
    del src

    rolls = {}
    for ep_idx in EVAL_EPISODES:
        ep = load_episode(ep_idx)
        r = rollout(nn_diffusion, solver, ep, ck["action_min"], ck["action_scale"], use_ema)
        rolls[str(ep_idx)] = {k: v for k, v in r.items() if not k.endswith("_path")}
        print(f"[{exp}] ep{ep_idx}: ADE {r['ade_cm']:.1f} cm | FDE {r['fde_cm']:.1f} cm | "
              f"velRMSE {r['vel_rmse']:.4f} | progRMSE {r['vel_progress_rmse']:.4f} | "
              f"speedup {r['speedup_frac']*100:.0f}%", flush=True)
    result["openloop"] = rolls
    result["summary"] = dict(
        ade_cm=float(np.median([r["ade_cm"] for r in rolls.values()])),
        fde_cm=float(np.median([r["fde_cm"] for r in rolls.values()])),
        vel_rmse=float(np.median([r["vel_rmse"] for r in rolls.values()])),
        vel_progress_rmse=float(np.median([r["vel_progress_rmse"] for r in rolls.values()])),
        speedup_frac=float(np.median([r["speedup_frac"] for r in rolls.values()])),
        near_mm=result["aux"]["near_median_mm"] if result["aux"] else None,
        align_deg=result["align"]["policy_median_deg"] if result["align"] else None,
        align_demo_deg=result["align"]["demo_median_deg"] if result["align"] else None,
        align_demo_p25_deg=result["align"]["demo_p25_deg"] if result["align"] else None,
        xpos_mm=result["align"]["x_policy_median_mm"] if result["align"] else None,
        xpos_demo_mm=result["align"]["x_demo_median_mm"] if result["align"] else None)

    tag = os.environ.get("EVAL_TAG", "")
    out = os.path.join(OUT_DIR, f"{exp}{'_' + tag if tag else ''}.json")
    json.dump(result, open(out, "w"), indent=1)
    s = result["summary"]
    print(f"[{exp}] SUMMARY: near {s['near_mm']} mm | ADE {s['ade_cm']:.1f} cm | "
          f"FDE {s['fde_cm']:.1f} cm -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
