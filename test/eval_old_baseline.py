"""Evaluate the OLD working baseline (outputs/checkpoint_step_100000.pt) with
the same metric suite used for the new-generation runs, so the two are
directly comparable in docs/ablation_study_2026-07.md.

Architecture (confirmed from the checkpoint's own state_dict keys — see
`describe_arch()`; NOT read from any config file, since none ships with this
checkpoint): 2-camera (room1+room2) DINO vision + velocity history only.
NO goal conditioning, NO lidar branch, NO aux pose head. Backbone is the DDPM
SDE (postech_config.yaml: solver ode_dpmsolver++_2M, ema_rate 0.9999,
inference_sampling_steps 100). ema_rate 0.9999 over 100,000 steps -> tau=10k
steps, so unlike the 4230-step bug generation this EMA IS converged
(0.9999^100000 ~= 4.5e-5 residual) -> use_ema=True is trustworthy here.

Metrics produced (mirrors eval_run.py / eval_openloop_metrics.py):
  * vel_rmse, ADE, FDE (cm)          -- open-loop rollout, live dual-camera DINO
  * align_deg, xpos_mm (counterfactual, near-dock <0.6m) -- control-precision
    proxy. Unlike the new models this one has NO aux head, so the perception
    metric "near_mm" (predicted-vs-ICP dock pose) CANNOT be computed -- there
    is nothing to compare. The counterfactual metrics below do NOT need an aux
    head: they only need (a) the ICP dock_pose LABEL (always in the h5,
    independent of what the model conditions on) and (b) the policy's own
    rolled-out (vx,wz). So they are the only precision-axis numbers available
    for this model, and are reported next to the demo's own residual exactly
    like eval_run.py's align_eval.

Run (from repo root):  CUDA_VISIBLE_DEVICES=<free> python test/eval_old_baseline.py
Outputs: test/out/weekend/old_baseline_100k.json, test/out/weekend/old_baseline_100k_heldout.json
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cleandiffuser.diffusion import ContinuousDiffusionSDE  # noqa: E402
from cleandiffuser.nn_diffusion import DiT1d  # noqa: E402
from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork  # noqa: E402
from dino.dino_detector import DinoBatchDetector  # noqa: E402
from utils.docking_dataset import denormalize  # noqa: E402
from scripts.inference_ema_v2 import reconstruct_pose_rk4  # noqa: E402

CKPT = os.path.join(REPO, "outputs", "checkpoint_step_100000.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(REPO, "test", "out", "weekend")

OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
DT = 0.0333
N_SAMPLES = 8
SAMPLE_STEPS = 100          # matches postech_config.yaml inference_sampling_steps
SOLVER = "ode_dpmsolver++_2M"

# open-loop rollout episodes (cheap subset: full DDPM@100 sampling + live dual
# DINO is far more expensive per step than the new rectified-flow@20 models).
# train h5 has 145 episodes, held-out (after_0328_test.h5) only 10 -> separate
# index lists (matches EVAL_EPISODES=0..9 convention used elsewhere for heldout).
EPISODES_TRAIN = [0, 50, 110]
EPISODES_HELDOUT = [0, 3, 6]
MAX_STEPS = 250

# counterfactual align/xpos: near-dock reliable frames, subsampled for cost
ALIGN_NEAR_M = 0.6
ALIGN_N_FRAMES = 150
ALIGN_BATCH = 16


def describe_arch(sd):
    branches = {
        "room1 (2nd camera)": any(k.startswith("condition.room1_resampler") for k in sd),
        "room2 (camera)": any(k.startswith("condition.room2_resampler") for k in sd),
        "goal conditioning": any("goal_resampler" in k for k in sd),
        "goal-lidar": any("goal_lidar" in k for k in sd),
        "lidar branch": any(k.startswith("condition.lidar_resampler") for k in sd),
        "aux pose head": any("aux_" in k for k in sd),
    }
    print("=== architecture (from state_dict keys) ===")
    for k, v in branches.items():
        print(f"  {k:20} {'O' if v else 'X'}")
    return branches


def build_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"]
    describe_arch(sd)
    nn_condition = SensorFusionConditionNetwork(
        state_dim=2, obs_horizon=OBS_HORIZON, vision_horizon=5,
        d_model=384, nhead=6, num_layers=2, dropout=0.1,
        num_image_latents=16, velocity_dim=2, velocity_dropout_prob=0.0,
        use_goal=False, use_lidar_points=False, use_aux_pose=False, use_room1=True,
    ).to(DEVICE)
    nn_diffusion_model = DiT1d(in_dim=2, emb_dim=384, d_model=384,
                               n_heads=6, depth=12, dropout=0.0).to(DEVICE)
    model = ContinuousDiffusionSDE(nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
                                   ema_rate=0.9999, device=DEVICE)
    model.model.load_state_dict(sd)
    model.model_ema.load_state_dict(ckpt["ema_state_dict"])
    model.eval()
    return model, np.asarray(ckpt["action_min"], np.float32), np.asarray(ckpt["action_scale"], np.float32)


@torch.no_grad()
def encode_dino(dino, imgs_uint8):
    """[T,3,H,W] uint8 -> [1,T,196,768] float, live DINO encode (no cache for room1)."""
    t = torch.from_numpy(imgs_uint8).float().to(DEVICE) / 255.0
    feat, _, _ = dino.get_heatmap(t)
    return feat.view(1, imgs_uint8.shape[0], 196, 768)


@torch.no_grad()
def rollout(model, dino, act_min, act_scale, h5_path, ep_idx):
    f = h5py.File(h5_path, "r")
    ends = f["episode_ends"][:].astype(int)
    s = 0 if ep_idx == 0 else ends[ep_idx - 1]
    e = ends[ep_idx]
    enc = f["encoder"][s:e].astype(np.float32)
    img1 = f["image_top"][s:e]
    img2 = f["image_bottom"][s:e]
    f.close()

    ep_steps = min(len(enc) - HORIZON + 1, MAX_STEPS)
    a_min, a_scale = torch.as_tensor(act_min), torch.as_tensor(act_scale)

    torch.manual_seed(0)
    selected = []
    for t in range(ep_steps):
        rows = np.clip(np.arange(t - OBS_HORIZON + 1, t + 1), 0, None)
        vel = (2.0 * (torch.as_tensor(enc[rows]) - a_min) / a_scale - 1.0).float()
        sparse = rows[::VISION_STRIDE]
        f1 = encode_dino(dino, img1[sparse]).repeat(N_SAMPLES, 1, 1, 1)
        f2 = encode_dino(dino, img2[sparse]).repeat(N_SAMPLES, 1, 1, 1)
        context = {
            "velocity": vel.unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1),
            "dino_feat1": f1, "dino_feat2": f2,
        }
        prior = torch.randn(N_SAMPLES, HORIZON, 2, device=DEVICE)
        out = model.sample(solver=SOLVER, w_cfg=1, prior=prior, condition_cfg=context,
                           n_samples=N_SAMPLES, sample_steps=SAMPLE_STEPS, use_ema=True)
        out = out[0] if isinstance(out, tuple) else out
        res = denormalize(out.cpu().numpy(), act_scale, act_min)
        selected.append(res.mean(axis=0)[0, :])
        if (t + 1) % 50 == 0:
            print(f"    ep{ep_idx} step {t + 1}/{ep_steps}", flush=True)

    ai = np.array(selected)
    gt = enc[:ep_steps]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
    disp = np.hypot(*(gt_path[:, :2] - ai_path[:, :2]).T)
    return dict(vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())),
               ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100), steps=int(ep_steps))


@torch.no_grad()
def align_eval(model, dino, act_min, act_scale, h5_path):
    """Counterfactual final-alignment (deg) + forward-x (mm) error, near-dock.
    Same identity as test/eval_run.py:align_eval, self-contained here since
    this model has no lidar/aux head to reuse that helper's context builder."""
    f = h5py.File(h5_path, "r")
    ends = f["episode_ends"][:].astype(int)
    pose = f["dock_pose"][:].astype(np.float32)
    rel = f["reliable"][:].astype(bool)
    valid = rel & ~np.isnan(pose).any(axis=1)
    pose = np.nan_to_num(pose)
    dist = np.hypot(pose[:, 0], pose[:, 1])
    n = ends[-1]
    ok = np.zeros(n, bool)
    s = 0
    for e in ends:
        ok[s:e - HORIZON] = True
        s = e
    cand = np.where(valid & (dist < ALIGN_NEAR_M) & ok)[0]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(cand, size=min(ALIGN_N_FRAMES, len(cand)), replace=False))

    x_goal_all = np.zeros(n, np.float32)
    s = 0
    for e in ends:
        idx = np.where(valid[s:e])[0]
        x_goal_all[s:e] = pose[s + idx[-1], 0] if len(idx) else 0.0
        s = e

    a_min, a_scale = torch.as_tensor(act_min), torch.as_tensor(act_scale)
    res_pol, xerr_pol = [], []
    for bi in range(0, len(rows), ALIGN_BATCH):
        batch_rows = rows[bi:bi + ALIGN_BATCH]
        img1 = f["image_top"]; img2 = f["image_bottom"]; enc = f["encoder"]
        acts = []
        for r in batch_rows:
            rr = np.clip(np.arange(r - OBS_HORIZON + 1, r + 1), 0, None)
            vel = (2.0 * (torch.as_tensor(enc[rr].astype(np.float32)) - a_min) / a_scale - 1.0).float()
            sparse = rr[::VISION_STRIDE]
            f1 = encode_dino(dino, img1[sparse]).repeat(N_SAMPLES, 1, 1, 1)
            f2 = encode_dino(dino, img2[sparse]).repeat(N_SAMPLES, 1, 1, 1)
            context = {"velocity": vel.unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, 1, 1),
                      "dino_feat1": f1, "dino_feat2": f2}
            prior = torch.randn(N_SAMPLES, HORIZON, 2, device=DEVICE)
            out = model.sample(solver=SOLVER, w_cfg=1, prior=prior, condition_cfg=context,
                               n_samples=N_SAMPLES, sample_steps=SAMPLE_STEPS, use_ema=True)
            out = out[0] if isinstance(out, tuple) else out
            act = denormalize(out.cpu().numpy(), act_scale, act_min).mean(axis=0)  # [H,2]
            acts.append(act)
        print(f"    align batch {bi + len(batch_rows)}/{len(rows)}", flush=True)

        for j, r in enumerate(batch_rows):
            theta_now = pose[r, 2]
            dpsi = acts[j][:, 1].sum() * DT
            res_pol.append(abs(theta_now - dpsi))
            dx, dy = pose[r, 0], pose[r, 1]
            px = py = pth = 0.0
            for k in range(HORIZON):
                v, w = float(acts[j][k, 0]), float(acts[j][k, 1])
                px += v * np.cos(pth) * DT
                py += v * np.sin(pth) * DT
                pth += w * DT
            c, sN = np.cos(-pth), np.sin(-pth)
            rx0, rx1 = dx - px, dy - py
            x_H = c * rx0 - sN * rx1
            xerr_pol.append(abs(x_H - x_goal_all[r]))
    f.close()
    p = np.degrees(np.array(res_pol))
    xp = np.array(xerr_pol) * 1000.0
    return dict(policy_median_deg=float(np.median(p)), policy_p90_deg=float(np.percentile(p, 90)),
                x_policy_median_mm=float(np.median(xp)), x_policy_p90_mm=float(np.percentile(xp, 90)),
                n_frames=int(len(p)))


def run_suite(model, dino, act_min, act_scale, h5_path, tag, episodes):
    print(f"\n===== {tag} ({h5_path}) =====")
    rolls = {}
    for ep in episodes:
        r = rollout(model, dino, act_min, act_scale, h5_path, ep)
        rolls[str(ep)] = r
        print(f"  ep{ep}: ADE {r['ade_cm']:.1f} cm | FDE {r['fde_cm']:.1f} cm | velRMSE {r['vel_rmse']:.4f}")
    align = align_eval(model, dino, act_min, act_scale, h5_path)
    print(f"  align(<0.6m): {align['policy_median_deg']:.2f} deg (p90 {align['policy_p90_deg']:.2f}) "
          f"| xpos {align['x_policy_median_mm']:.1f} mm (p90 {align['x_policy_p90_mm']:.1f}) "
          f"| n={align['n_frames']}")
    summary = dict(
        ade_cm=float(np.median([r["ade_cm"] for r in rolls.values()])),
        fde_cm=float(np.median([r["fde_cm"] for r in rolls.values()])),
        vel_rmse=float(np.median([r["vel_rmse"] for r in rolls.values()])),
        align_deg=align["policy_median_deg"], xpos_mm=align["x_policy_median_mm"],
        near_mm=None,   # no aux head -- perception-precision not measurable
    )
    print(f"  SUMMARY: ADE {summary['ade_cm']:.1f} | FDE {summary['fde_cm']:.1f} | "
          f"align {summary['align_deg']:.2f}deg | xpos {summary['xpos_mm']:.1f}mm")
    out = dict(run_dir="outputs/checkpoint_step_100000.pt", openloop=rolls, align=align, summary=summary)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUT_DIR, f"old_baseline_100k{'_heldout' if 'test' in tag else ''}.json"), "w"), indent=1)
    return summary


def main():
    import sys as _sys
    model, act_min, act_scale = build_model()
    dino = DinoBatchDetector(device=DEVICE)
    if "--heldout-only" not in _sys.argv:
        run_suite(model, dino, act_min, act_scale,
                 os.path.join(REPO, "dataset", "after_0328_train.h5"), "train", EPISODES_TRAIN)
    run_suite(model, dino, act_min, act_scale,
             os.path.join(REPO, "dataset", "after_0328_test.h5"), "heldout(test)", EPISODES_HELDOUT)


if __name__ == "__main__":
    main()
