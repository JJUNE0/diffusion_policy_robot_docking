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
        near_mm=result["aux"]["near_median_mm"] if result["aux"] else None)

    tag = os.environ.get("EVAL_TAG", "")
    out = os.path.join(OUT_DIR, f"{exp}{'_' + tag if tag else ''}.json")
    json.dump(result, open(out, "w"), indent=1)
    s = result["summary"]
    print(f"[{exp}] SUMMARY: near {s['near_mm']} mm | ADE {s['ade_cm']:.1f} cm | "
          f"FDE {s['fde_cm']:.1f} cm -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
