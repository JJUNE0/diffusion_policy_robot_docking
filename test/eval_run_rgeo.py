"""Held-out open-loop eval for the ReLoc3R arms -- TokenSequenceFusionCondition
+ DiTCrossAttn1d.

No aux/align metrics here (no aux head exists for this architecture, per
spec) -- ADE/FDE/velRMSE plus the terminal-band table, reusing
ModularDockingDataset so observation construction/normalization is
bit-identical to training.

Run:  EVAL_H5=dataset/after_0328_test.h5 EVAL_STATS_H5=dataset/after_0328_train.h5 \
      EVAL_TAG=heldout python test/eval_run_rgeo.py outputs/train/r_relfeat_only/<timestamp>
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda key: {"runtime.cwd": REPO}[key])

from utils.inference import build_model_from_cfg, reconstruct_pose_rk4  # noqa: E402
from cleandiffuser.rollout_core import RolloutController  # noqa: E402
from utils.modular_dataset import ModularDockingDataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import terminal_metric as tm  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DT = 0.0333
N_SAMPLES = 8
SAMPLE_STEPS = int(os.environ.get("EVAL_STEPS", "20"))
TRAJ_EMA_ALPHA = 0.3
# Full-episode eval (2026-07-27, user request): the old MAX_STEPS=500 cap
# silently truncated every held-out episode to 20-34% of its length (episodes
# run 1417-2616 rows), so ADE/FDE never saw the terminal 2/3 of any approach.
# Fixed at effectively "no cap" (episodes here max out at ~2600).
MAX_STEPS = int(os.environ.get("EVAL_MAX_STEPS", "100000"))
# Action/execution horizon (user request 2026-07-27): replan every K steps and
# execute that plan's raw [0:K] chunk verbatim (no cross-time EMA blend needed
# -- one coherent sampled trajectory is already smooth internally). This is
# the same EXEC_CHUNK_K pattern as the k2/k8/k16 chunking sweep
# ablation. K=32 also makes full-episode eval CHEAPER than the old per-step
# (K=1) 500-cap protocol: ~611 vs ~5000 diffusion sampler calls across the
# held-out split. The trained prediction length (`horizon`, from ds/cfg) is
# unchanged -- only how much of each sampled plan gets executed before
# resampling changes.
EXEC_CHUNK_K = int(os.environ.get("EVAL_CHUNK_K", "32"))
OUT_DIR = os.path.join(REPO, "test/out/rgeo")

EVAL_H5 = os.environ.get("EVAL_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))
STATS_H5 = os.environ.get("EVAL_STATS_H5", os.path.join(REPO, "dataset/after_0328_train.h5"))
EVAL_EPISODES = [int(x) for x in os.environ.get("EVAL_EPISODES", "0,1,2").split(",")]


def _stem(h5_path):
    b = os.path.basename(h5_path)
    m = re.match(r"(.*?)(?:_dino_\w+|_reloc3r\w*)?\.h5$", b)
    return m.group(1) if m else b[:-3]


def _resolve_sensor_files(sensors, eval_h5):
    train_stem, eval_stem = _stem(STATS_H5), _stem(eval_h5)
    out = {}
    for name, spec in sensors.items():
        spec = dict(spec)
        # A run trained off a memfd RAM cache has that `/proc/<pid>/fd/N` path
        # baked into the config saved beside its checkpoint. At eval time the
        # daemon is gone (getsize raises FileNotFoundError), and if some
        # unrelated process happens to hold that fd the mapping would be the
        # WRONG rows -- training-split features silently scored as held-out.
        # Neither is ever what an evaluator wants, so always read from the h5.
        if spec.pop("cache_mmap", None) is not None:
            print(f"[eval] sensor '{name}': dropping train-time cache_mmap; "
                  f"reading features from the h5", flush=True)
        f = spec.get("file")
        if f is not None:
            f = f.replace("${hydra:runtime.cwd}", REPO)
            if not os.path.isabs(f):
                f = os.path.join(REPO, f)
            if train_stem != eval_stem:
                f = os.path.join(os.path.dirname(f), os.path.basename(f).replace(train_stem, eval_stem))
            spec["file"] = f
        out[name] = spec
    return out


@torch.no_grad()
def rollout(nn_diffusion, solver, ds, ep_idx, act_min, act_scale, use_ema, episode_ends,
            enc_all, rem_all=None):
    ends = episode_ends.astype(int)
    s = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
    e = int(ends[ep_idx])
    enc = enc_all[s:e]
    horizon = ds.horizon
    ep_steps = min(len(enc) - horizon + 1, MAX_STEPS)
    torch.manual_seed(0)
    ctrl = RolloutController(
        nn_diffusion, solver=solver, sample_steps=SAMPLE_STEPS, use_ema=use_ema,
        n_samples=N_SAMPLES, horizon=horizon, w_cfg=1, agg="mean",
        traj_ema_alpha=TRAJ_EMA_ALPHA, warm_start=False, warm_level=0.3, device=DEVICE)

    row_to_didx = {int(t): i for i, t in enumerate(ds.index_map) if ds.ep_start_map[i] == s}
    selected = []
    t = 0
    while t < ep_steps:
        di = row_to_didx.get(s + t)
        if di is None:
            break
        obs = ds[di]["obs"]
        context = {k: v.unsqueeze(0).to(DEVICE).repeat(N_SAMPLES, *([1] * v.dim())) for k, v in obs.items()}
        k = min(EXEC_CHUNK_K, ep_steps - t, horizon)
        plan = ctrl.plan(context, act_min, act_scale, chunk_shift=k)
        selected.extend(list(plan.current[0:k]))            # raw chunk, no EMA
        t += k

    ai = np.array(selected)
    gt = enc[:len(ai)]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=DT)
    disp = np.hypot(*(gt_path[:, :2] - ai_path[:, :2]).T)
    path_len = float(np.sum(np.hypot(*np.diff(gt_path[:, :2], axis=0).T)))
    out = dict(
        vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())),
        ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100),
        path_len_cm=path_len * 100, fde_over_path=float(disp[-1] / max(path_len, 1e-9)),
        exec_chunk_k=EXEC_CHUNK_K, steps=int(len(ai)))
    if rem_all is not None:
        # Terminal band: |vx| the policy commands vs the demo's, binned by how
        # much approach is left. This is the ranking metric — see terminal_metric.
        out["_term"] = (rem_all[s:s + len(ai)],
                        np.abs(ai[:, 0]) * 1000.0, np.abs(gt[:, 0]) * 1000.0)
    out["_wz"] = (ai[:, 1], gt[:, 1])       # steering-bias detector
    return out


def main(run_dir):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
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

    sensors = OmegaConf.to_container(cfg.sensors, resolve=True)
    sensors = _resolve_sensor_files(sensors, EVAL_H5)
    ds = ModularDockingDataset(
        h5_path=EVAL_H5, sensors=sensors, horizon=cfg.horizon, obs_horizon=cfg.obs_horizon,
        action_key=cfg.get("action_key", "encoder"), train_h5_path=STATS_H5,
        action_norm=cfg.get("action_norm", "minmax"))
    # The velocity-history observation is normalized by the DATASET while the
    # plan is denormalized by the CHECKPOINT. If the two normalizers disagree
    # (e.g. cfg lost action_norm) the model silently sees shifted inputs and the
    # numbers look plausible but mean nothing — same failure class as the
    # aux_relative stats bug in ablation_study section 2.9. Fail loudly instead.
    for nm, got, want in (("action_min", ds.action_min, ck["action_min"]),
                          ("action_scale", ds.action_scale, ck["action_scale"])):
        if not np.allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-4):
            raise SystemExit(f"{nm} mismatch: dataset {np.asarray(got)} vs checkpoint "
                             f"{np.asarray(want)} — action_norm='{cfg.get('action_norm','minmax')}' "
                             f"does not match how this checkpoint was trained.")

    import h5py
    with h5py.File(EVAL_H5, "r") as f:
        episode_ends = f["episode_ends"][:]
        enc_all = f["encoder"][:].astype(np.float32)
        rem_all = (tm.remaining_mm(f["dock_pose"][:], f["reliable"][:], episode_ends)
                   if "dock_pose" in f else None)

    rolls, term_parts, wz_parts = {}, [], []
    for ep_idx in EVAL_EPISODES:
        r = rollout(nn_diffusion, solver, ds, ep_idx, ck["action_min"], ck["action_scale"],
                   use_ema, episode_ends, enc_all, rem_all)
        term_parts.append(r.pop("_term", None))
        wz_parts.append(r.pop("_wz"))
        rolls[str(ep_idx)] = r
        print(f"[{exp}] ep{ep_idx}: ADE {r['ade_cm']:.1f} cm | FDE {r['fde_cm']:.1f} cm | "
              f"velRMSE {r['vel_rmse']:.4f}", flush=True)

    terminal = None
    parts = [p for p in term_parts if p is not None]
    if parts:
        got = tm.band_table(*[np.concatenate(c) for c in zip(*parts)])
        if got:
            terminal = tm.summarize(*got)
            tm.report(exp, terminal)

    sym = tm.turn_symmetry(*[np.concatenate(c) for c in zip(*wz_parts)])
    tm.report_symmetry(exp, sym)

    summary = dict(
        ade_cm=float(np.median([r["ade_cm"] for r in rolls.values()])),
        fde_cm=float(np.median([r["fde_cm"] for r in rolls.values()])),
        ade_cm_mean=float(np.mean([r["ade_cm"] for r in rolls.values()])),
        fde_cm_mean=float(np.mean([r["fde_cm"] for r in rolls.values()])),
        vel_rmse=float(np.median([r["vel_rmse"] for r in rolls.values()])),
        # primary ranking axis — see test/terminal_metric.py
        term_vx_ratio=terminal["term_vx_ratio"] if terminal else None,
        term_idle_frac=terminal["term_idle_frac"] if terminal else None,
        term_vx_mms=terminal["term_vx_mms"] if terminal else None,
        turn_right_frac=sym["policy_right_frac"], turn_bias=sym["bias"])
    result = dict(run_dir=run_dir, ckpt=os.path.basename(ckpt_path), openloop=rolls,
                  terminal=terminal, symmetry=sym, summary=summary)
    tag = os.environ.get("EVAL_TAG", "")
    out = os.path.join(OUT_DIR, f"{exp}{'_' + tag if tag else ''}.json")
    json.dump(result, open(out, "w"), indent=1)
    tr = f"{summary['term_vx_ratio']:.2f}" if summary["term_vx_ratio"] is not None else "n/a"
    ti = f"{summary['term_idle_frac']*100:.0f}%" if summary["term_idle_frac"] is not None else "n/a"
    print(f"[{exp}] SUMMARY: termRatio {tr} / parked {ti} [primary] | ADE {summary['ade_cm']:.1f} cm | "
          f"FDE med {summary['fde_cm']:.1f} / mean {summary['fde_cm_mean']:.1f} cm | "
          f"velRMSE {summary['vel_rmse']:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
