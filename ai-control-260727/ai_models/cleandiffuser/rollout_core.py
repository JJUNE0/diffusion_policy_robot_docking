"""Single source of truth for the open-loop rollout inner step.

Before 2026-07-21 the "sample N trajectories -> aggregate -> EMA-smooth ->
(optionally) chunk/warm-start" logic was copy-pasted in THREE places that
kept drifting apart:
  * test/eval_openloop_metrics.py         (h5 + DINO-cache fed, offline eval)
  * scripts/inference_ema_v2.py           (dataset-fed, offline single-episode viz)
  * ai-control-260715/.../run_postech_docking_demo.py  (live-stream deployment)
They differ ONLY in how each frame's `context` dict is built (data source) and
in a few knobs (mean vs medoid aggregation, wall-clock EMA reset for the live
node). The sampling/aggregation/EMA/warm-start core is identical -- and is what
this module owns. Each call site keeps its own context-building loop and just
delegates to `RolloutController.plan()`.

Reproducibility contract: with the defaults (agg="mean", chunk handled by the
caller, warm_start=False, traj_ema_alpha as before) the RNG draw order and the
numeric output are byte-for-byte identical to the pre-refactor inline code, so
every published number in docs/ablation_study_2026-07.md stays reproducible.
The controller ALWAYS draws the `prior` (even when warm-starting, matching the
old code where the draw happened unconditionally), so the seeded sequence is
preserved.

The controller computes BOTH the pre-EMA aggregated plan (`current`) and the
EMA-smoothed plan (`ema`) every call; each caller picks which to execute:
  * legacy per-frame eval  -> execute `ema[0]`      (EMA on, step 0 only)
  * action-chunking eval   -> execute `current[0:K]` (raw chunk, no EMA)
  * live deployment        -> execute `ema[0:inference_size]` (EMA on, integrate)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class RolloutPlan:
    current: np.ndarray   # aggregated denormalized plan [H, 2], PRE-EMA
    ema: np.ndarray       # EMA-smoothed denormalized plan [H, 2]
    raw_mean: np.ndarray  # aggregated NORMALIZED plan [H, 2] (warm-start ref / raw samples)
    samples: np.ndarray   # per-sample denormalized plans [N, H, 2] (for spread plots)


def _denormalize(norm_action, act_scale, act_min):
    """Identical to utils.docking_dataset.denormalize; inlined to keep this
    module import-light and bundle-portable (the deployment copy has no repo
    utils on its path)."""
    return (norm_action + 1.0) / 2.0 * act_scale + act_min


class RolloutController:
    """Stateful holder for the cross-frame rollout core (EMA + warm-start).

    One instance per episode/rollout. Call `plan(context, ...)` once per
    resample; it samples, aggregates, updates the EMA + warm-start state, and
    returns a RolloutPlan. The caller owns the outer loop and cadence.
    """

    def __init__(self, nn_diffusion, *, solver, sample_steps, use_ema, n_samples,
                 horizon, w_cfg=1, agg="mean", traj_ema_alpha=0.3,
                 warm_start=False, warm_level=0.3, ema_reset_sec=None, device=None):
        self.nn = nn_diffusion
        self.solver = solver
        self.sample_steps = int(sample_steps)
        self.use_ema = bool(use_ema)
        self.n_samples = int(n_samples)
        self.horizon = int(horizon)
        self.w_cfg = w_cfg
        assert agg in ("mean", "medoid"), f"unknown agg '{agg}'"
        self.agg = agg
        self.traj_ema_alpha = float(traj_ema_alpha)
        self.warm_start = bool(warm_start)
        self.warm_level = float(warm_level)
        self.ema_reset_sec = ema_reset_sec        # None = never reset (offline)
        self.device = device
        self._prev_ema = None                     # [H,2] denormalized
        self._prev_raw_plan = None                # [H,2] normalized (warm-start ref)
        self._last_call = None                    # wall-clock, for ema_reset_sec

    def _aggregate(self, res):
        """res: [N,H,2] DENORMALIZED -> [H,2]. mean = smooth average; medoid =
        the sample closest to the others (decisive; keeps sharp turns, matches
        the deployment / test/mpc_rank.py rationale). Aggregation is done in
        DENORMALIZED space to match the deployment's medoid exactly (the medoid
        argmin is scale-sensitive, so normalized-vs-denorm would pick different
        samples); for mean it commutes with the affine denorm, so eval numbers
        are unchanged."""
        if self.n_samples == 1:
            return res[0]
        if self.agg == "mean":
            return res.mean(axis=0)
        d2 = ((res[:, None] - res[None]) ** 2).sum(axis=(2, 3))
        return res[d2.sum(axis=1).argmin()]

    def plan(self, context, act_min, act_scale, *, chunk_shift=1, now=None):
        """Sample + aggregate + EMA once. `chunk_shift` is how many steps the
        NEXT frame will have advanced (used to shift the warm-start reference);
        pass your execution K in chunk mode, 1 otherwise. `now` (wall clock)
        enables ema_reset_sec gap-resets for the live node."""
        device = self.device
        with torch.no_grad():
            prior = torch.randn(self.n_samples, self.horizon, 2, device=device)
            warm_ref = None
            if self.warm_start and self._prev_raw_plan is not None:
                shift = min(int(chunk_shift), self.horizon - 1)
                warm = np.empty_like(self._prev_raw_plan)
                warm[:self.horizon - shift] = self._prev_raw_plan[shift:]
                warm[self.horizon - shift:] = self._prev_raw_plan[-1]
                warm_ref = torch.from_numpy(warm).float().to(device).unsqueeze(0).repeat(self.n_samples, 1, 1)
            out = self.nn.sample(
                solver=self.solver, w_cfg=self.w_cfg, prior=prior, condition_cfg=context,
                n_samples=self.n_samples, sample_steps=self.sample_steps, use_ema=self.use_ema,
                warm_start_reference=warm_ref, warm_start_forward_level=self.warm_level)
            out = out[0] if isinstance(out, tuple) else out
            raw = out.cpu().numpy()                       # [N,H,2] normalized

        samples = _denormalize(raw, act_scale, act_min)    # [N,H,2] denorm
        current = self._aggregate(samples)                 # [H,2] denorm (mean|medoid)
        # warm-start reference lives in the model's normalized action space;
        # inverse of _denormalize. (Only consumed when warm_start=True.)
        raw_mean = 2.0 * (current - act_min) / act_scale - 1.0

        if self._prev_ema is None or (
                self.ema_reset_sec is not None and now is not None
                and self._last_call is not None and now - self._last_call > self.ema_reset_sec):
            ema = current
        else:
            a = self.traj_ema_alpha
            ema = a * current + (1 - a) * self._prev_ema

        self._prev_ema = ema.copy()
        self._prev_raw_plan = raw_mean
        self._last_call = now
        return RolloutPlan(current=current, ema=ema, raw_mean=raw_mean, samples=samples)
