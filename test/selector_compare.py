"""Stage-0.5: how much of the 94% oracle headroom can a DEPLOYABLE selector get?

test/oracle_headroom.py showed that among M=16 diffusion candidates there is
almost always an excellent one (median yaw 0.584 deg vs 2.470 for a random pick
-- better than the demonstrations' own 1.028). But the oracle uses ICP dock_pose,
which the robot does not have. This script asks the question that actually
matters for deployment:

    Can a selector built ONLY from deployment-available signals recover that gap?

The one deployment-available alignment signal is the Reloc3r geometry token the
policy already receives: g = [dx_body, dy_body, sin(psi), cos(psi)], where
(dx_body, dy_body) is the unit BEARING (direction, not distance -- Reloc3r
translation is scale-ambiguous) from the current pose to the goal, and psi is
the relative YAW needed to face the dock squarely. These are two different
physical quantities. The original handcrafted cost (`hand`, preserved below)
only used psi:

    J_yaw(tau) = wrap(psi_geo - sum(w_t) * dt)^2

That ignores lateral (dy_body) misalignment entirely, which the project's own
field notes call out as the dominant real failure mode (docking fails on
lateral peg-in-hole offset, not depth). This version ADDS a bearing term
without removing the yaw one: candidate m's SE(2) endpoint (px, py) is
integrated from its own (v, w) sequence, and its net travel direction is
compared against the goal bearing:

    beta          = atan2(dy_body, dx_body)              # goal bearing, body frame
    heading(tau)  = atan2(py(tau), px(tau))               # candidate's own travel direction
    J_lat(tau)    = wrap(beta - heading(tau))^2
    J_xy(tau)     = J_yaw(tau) + J_lat(tau)                # new: `geoxy` selector

Both terms are actually already exercised in the offline oracle upper bound
(oracle_headroom.py's candidate_costs computes both a yaw residual and a
FORWARD-x residual against ICP dock_pose), but that oracle deliberately never
used ICP's y-component: see test/eval_run.py's own note, "y has no learnable
signal (1.2x ICP noise)". So this script also reports an ICP-y residual
purely as a DIAGNOSTIC (never as a selection target) -- read it as noisy, not
as ground truth.

Selectors compared, all on the identical candidate sets:
  first      - execute sample 0 (no selection)
  mean-agg   - what the control node does TODAY (RolloutController agg='mean');
               note this AVERAGES diverse candidates rather than choosing one
  hand       - argmin J_yaw   (original, preserved: yaw only)
  geoxy      - argmin J_xy    (new: yaw + lateral bearing, still ICP-free)
  ORACLE     - argmin J_icp   (upper bound, ICP yaw+forward-x supervised)

Everything is scored with the SAME ICP-based J (yaw+forward-x) as
oracle_headroom.py for the headline "recovery %" column, so `hand` and `geoxy`
stay directly comparable to the earlier single-selector run. The ICP-y
diagnostic is a separate, clearly-labeled column.

Also reports corr(psi_geo, theta_icp) and corr(beta_geo, beta_icp) -- if the
geometry token does not track the true dock yaw/bearing, no geometry-based
selector can work, and that is worth knowing before building a learned critic
on the same input.

Run:  EVAL_H5=dataset/after_0328_test.h5 EVAL_STATS_H5=dataset/after_0328_train.h5 \
      python test/selector_compare.py outputs/eval60k_r2cam_geo
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


def endpoints(act):
    """act [M,H,2] denormalized (v,w) -> (px,py,pth) [M] each, the SE(2)
    endpoint of each candidate integrated from the current (body) frame.
    Same RK-free forward-Euler integration oracle_headroom.candidate_costs
    uses internally for its forward-x term, exposed here so both the yaw and
    the new lateral-bearing cost can reuse one integration pass."""
    M, H, _ = act.shape
    px = np.zeros(M); py = np.zeros(M); pth = np.zeros(M)
    for m in range(M):
        x = y = th = 0.0
        for k in range(H):
            v, w = float(act[m, k, 0]), float(act[m, k, 1])
            x += v * np.cos(th) * DT
            y += v * np.sin(th) * DT
            th += w * DT
        px[m], py[m], pth[m] = x, y, th
    return px, py, pth


def icp_lateral_residual_mm(px, py, pth, dock_xy):
    """Diagnostic only (never a selection target): the omitted lateral (y)
    term from oracle_headroom.candidate_costs, i.e. the component that
    function's own x_res drops. ICP's y carries ~1.2x noise relative to
    signal (test/eval_run.py), so treat this as indicative, not exact."""
    M = len(px)
    y_res = np.empty(M)
    for m in range(M):
        c, s = np.cos(-pth[m]), np.sin(-pth[m])
        rx, ry = dock_xy[0] - px[m], dock_xy[1] - py[m]
        y_res[m] = s * rx + c * ry
    return np.abs(y_res) * 1000.0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DT, HORIZON = 0.0333, 60
M = int(os.environ.get("N_CAND", "16"))
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))
EVAL_H5 = os.environ.get("EVAL_H5", os.path.join(REPO, "dataset/after_0328_test.h5"))
STATS_H5 = os.environ.get("EVAL_STATS_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))


@torch.no_grad()
def main(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    exp = str(cfg.experiment_name)
    sensors = _resolve_sensor_files(OmegaConf.to_container(cfg.sensors, resolve=True), EVAL_H5)
    if "geometry" not in sensors:
        raise SystemExit(f"{exp} has no geometry sensor; the handcrafted cost needs it.")
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

    print(f"[{exp}] {os.path.basename(ckpt)} | M={M}", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    starts = np.sort(rng.choice(n_rows - BLOCK, size=N_BLOCKS, replace=False))

    # oracle_y: argmin over ICP's OWN y (lateral) residual, ICP-only, NOT
    # deployable. Answers a different question than `geoxy`/`hand`: is there
    # even a low-lateral-error candidate IN THE SET at all (a pure headroom
    # check), independent of whether any deployable signal (Reloc3r bearing,
    # or anything else) could find it. If oracle_y's ICP-y stays close to
    # mean-agg's, the candidate set itself carries no exploitable lateral
    # diversity here -- no selector, ICP-based or not, would help. If it drops
    # a lot, headroom exists and the bottleneck is specifically that no
    # deployable signal (corr(beta_geo, beta_icp)=0.497) can find it -- in
    # which case a BETTER deployable signal (e.g. live ICP against the dock
    # template, or LiDAR-based lateral estimation) is worth pursuing.
    KEYS = ("first", "meanagg", "hand", "geoxy", "oracle_y", "oracle")
    res = {k: [] for k in KEYS}; yaw = {k: [] for k in KEYS}; latmm = {k: [] for k in KEYS}
    psi_g, th_icp, beta_g, beta_icp = [], [], [], []
    for st in starts:
        rows = np.arange(st, st + BLOCK)
        keep = (np.hypot(pose[rows, 0], pose[rows, 1]) < NEAR_M) & rel[rows]
        rows = np.array([r for r in rows[keep] if int(r) in row2di], np.int64)
        for r in rows:
            item = ds[row2di[int(r)]]["obs"]
            obs = {k: v.unsqueeze(0).to(DEVICE) for k, v in item.items()}
            rep = {k: v.repeat(M, *([1] * (v.dim() - 1))) for k, v in obs.items()}
            out = nn.sample(solver=solver, w_cfg=1, prior=torch.randn(M, HORIZON, 2, device=DEVICE),
                            condition_cfg=rep, n_samples=M, sample_steps=SAMPLE_STEPS, use_ema=True)
            out = out[0] if isinstance(out, tuple) else out
            act = ((out.cpu().numpy() + 1.0) / 2.0 * a_scale + a_min)          # [M,H,2]

            J, yd, _xm = candidate_costs(act, pose[r, 2], pose[r, :2], xg[r])
            px, py, pth = endpoints(act)
            y_diag = icp_lateral_residual_mm(px, py, pth, pose[r, :2])         # diagnostic only

            # deployment-available signals from the geometry token itself
            g = item["geometry"].cpu().numpy()                                # [dx,dy,sin psi,cos psi]
            psi = float(np.arctan2(g[2], g[3]))                                # yaw-to-goal (preserved)
            beta = float(np.arctan2(g[1], g[0]))                               # NEW: lateral bearing-to-goal
            psi_g.append(psi); th_icp.append(float(pose[r, 2]))
            beta_g.append(beta); beta_icp.append(float(np.arctan2(pose[r, 1], pose[r, 0])))

            dpsi = act[:, :, 1].sum(axis=1) * DT
            J_yaw = _wrap(psi - dpsi) ** 2                                     # preserved cost
            heading = np.arctan2(py, px)
            J_lat = _wrap(beta - heading) ** 2                                 # NEW cost term
            J_xy = J_yaw + J_lat                                               # NEW combined cost

            mean_act = act.mean(axis=0, keepdims=True)
            Jm, ydm, _ = candidate_costs(mean_act, pose[r, 2], pose[r, :2], xg[r])
            mpx, mpy, mpth = endpoints(mean_act)
            ym_diag = icp_lateral_residual_mm(mpx, mpy, mpth, pose[r, :2])

            picks = dict(first=0, meanagg=0, hand=int(J_yaw.argmin()),
                        geoxy=int(J_xy.argmin()), oracle_y=int(np.abs(y_diag).argmin()),
                        oracle=int(J.argmin()))
            for k, idx in picks.items():
                if k == "meanagg":
                    res[k].append(Jm[0]); yaw[k].append(ydm[0]); latmm[k].append(ym_diag[0])
                else:
                    res[k].append(J[idx]); yaw[k].append(yd[idx]); latmm[k].append(y_diag[idx])

    for k in KEYS:
        res[k] = np.array(res[k]); yaw[k] = np.array(yaw[k]); latmm[k] = np.array(latmm[k])
    n = len(res["first"])
    base, orc = np.median(res["meanagg"]), np.median(res["oracle"])
    print(f"\nscored {n} near-dock frames x {M} candidates\n")
    print(f"{'selector':<28}{'median J':>12}{'yaw (deg)':>12}{'ICP-y (mm)*':>13}{'recovery':>11}")
    labels = dict(first="first sample", meanagg="mean-agg (deployed today)",
                  hand="hand (yaw only, preserved)", geoxy="geoxy (yaw + lateral, NEW)",
                  oracle_y="oracle_y (ICP-y only, NOT deployable)",
                  oracle="ORACLE (ICP, upper bound)")
    for k in KEYS:
        rec = (base - np.median(res[k])) / max(base - orc, 1e-12) * 100
        print(f"{labels[k]:<28}{np.median(res[k]):12.5f}{np.median(yaw[k]):12.3f}"
              f"{np.median(latmm[k]):13.1f}{rec:10.0f}%")
    print("* ICP-y is a diagnostic only -- ICP's lateral component carries ~1.2x noise "
          "relative to signal, it is not used to pick any candidate above.")
    cy = float(np.corrcoef(np.array(psi_g), np.array(th_icp))[0, 1])
    cb = float(np.corrcoef(np.array(beta_g), np.array(beta_icp))[0, 1])
    print(f"\ncorr(psi_geometry, theta_icp)  = {cy:+.3f}  (yaw-to-goal signal quality)")
    print(f"corr(beta_geometry, beta_icp)  = {cb:+.3f}  (bearing signal quality; noisier "
          f"expected, ICP y feeds both sides)")
    out_p = os.path.join(REPO, "test/out/rgeo", f"{exp}_selector_compare.json")
    json.dump({k: dict(median_J=float(np.median(res[k])), median_yaw_deg=float(np.median(yaw[k])),
                       median_icp_y_mm_diagnostic=float(np.median(latmm[k]))) for k in KEYS}
              | dict(n_frames=int(n), M=M, corr_psi_theta=cy, corr_beta_theta=cb,
                     ckpt=os.path.basename(ckpt)), open(out_p, "w"), indent=1)
    print(f"-> {out_p}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
