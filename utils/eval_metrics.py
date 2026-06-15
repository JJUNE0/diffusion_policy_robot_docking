"""
Offline trajectory-quality metrics for the docking diffusion policy.

This module compares the model's open-loop predicted motion against the recorded
ground-truth episode from the eval/validation dataset. It does NOT run a physics
simulator or a real robot: predicted velocity commands [v, w] are integrated into
an XY path (via reconstruct_pose_rk4 in the caller) and compared against the GT
path that the demonstration actually traced.

"Docking success" for an episode = the predicted final position lands within
`fde_threshold_m` (default 0.3 m = 30 cm) of the GT final (docked) position, and
optionally within `heading_threshold_deg` of the GT final heading. The success
rate is the fraction of evaluated episodes that succeed.

Public API (consumed by scripts/inference_ema.py):
    - compute_episode_metrics(...)  -> dict of per-episode metrics
    - aggregate_episode_metrics(...) -> dict of dataset-level means + success_rate
    - format_metric_textbox(metrics, keys) -> str for matplotlib annotation
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) in radians to (-pi, pi]."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _jerk(seq: np.ndarray) -> float:
    """Smoothness proxy: RMS of the discrete second difference of a 1-D sequence.

    A larger value means a more jerky (less smooth) command sequence. Computed in
    raw per-step units (no dt) since this is only used for relative comparison.
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.shape[0] < 3:
        return 0.0
    d2 = np.diff(seq, n=2)
    return float(np.sqrt(np.mean(d2 ** 2)))


def _pos_errors(gt_path: np.ndarray, ai_path: np.ndarray) -> np.ndarray:
    """Per-point Euclidean XY distance between two [N+1, >=2] pose arrays."""
    gt_xy = np.asarray(gt_path, dtype=np.float64)[:, :2]
    ai_xy = np.asarray(ai_path, dtype=np.float64)[:, :2]
    n = min(len(gt_xy), len(ai_xy))
    return np.linalg.norm(ai_xy[:n] - gt_xy[:n], axis=1)


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------
def compute_episode_metrics(
    *,
    gt_path: np.ndarray,
    ai_path: np.ndarray,
    gt_v: np.ndarray,
    gt_w: np.ndarray,
    pred_v: np.ndarray,
    pred_w: np.ndarray,
    fde_threshold_m: float = 0.3,
    heading_threshold_deg: Optional[float] = None,
    sample_v_std: Optional[np.ndarray] = None,
    sample_w_std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute trajectory + velocity metrics for a single episode.

    Args:
        gt_path / ai_path: [N+1, 3] (x, y, theta) reconstructed pose paths.
        gt_v / gt_w:       [N] ground-truth per-step linear / angular velocities.
        pred_v / pred_w:   [N] predicted (EMA-selected) per-step velocities.
        fde_threshold_m:   success radius on the final position error (meters).
        heading_threshold_deg: optional success gate on the final heading error.
        sample_v_std / sample_w_std: optional [N] cross-sample std (diversity).

    Returns:
        dict with position (ade, fde, max_pos_err), heading (heading_mae_deg,
        heading_final_err_deg), success flag, and per-channel velocity error /
        smoothness metrics.
    """
    gt_path = np.asarray(gt_path, dtype=np.float64)
    ai_path = np.asarray(ai_path, dtype=np.float64)

    # ---- position errors ----
    pos_err = _pos_errors(gt_path, ai_path)
    ade = float(pos_err.mean()) if pos_err.size else 0.0
    fde = float(pos_err[-1]) if pos_err.size else 0.0
    max_pos_err = float(pos_err.max()) if pos_err.size else 0.0

    # ---- heading errors (theta = column 2) ----
    n = min(len(gt_path), len(ai_path))
    if n > 0 and gt_path.shape[1] >= 3 and ai_path.shape[1] >= 3:
        head_err = np.abs(_wrap_to_pi(ai_path[:n, 2] - gt_path[:n, 2]))
        heading_mae_deg = float(np.degrees(head_err.mean()))
        heading_final_err_deg = float(np.degrees(head_err[-1]))
    else:
        heading_mae_deg = 0.0
        heading_final_err_deg = 0.0

    # ---- success: final position within threshold (+ optional heading gate) ----
    success = fde <= float(fde_threshold_m)
    if heading_threshold_deg is not None:
        success = success and (heading_final_err_deg <= float(heading_threshold_deg))

    # ---- velocity errors / smoothness ----
    gt_v = np.asarray(gt_v, dtype=np.float64)
    gt_w = np.asarray(gt_w, dtype=np.float64)
    pred_v = np.asarray(pred_v, dtype=np.float64)
    pred_w = np.asarray(pred_w, dtype=np.float64)
    nv = min(len(gt_v), len(pred_v))
    nw = min(len(gt_w), len(pred_w))

    v_err = pred_v[:nv] - gt_v[:nv]
    w_err = pred_w[:nw] - gt_w[:nw]
    v_mae = float(np.mean(np.abs(v_err))) if nv else 0.0
    v_rmse = float(np.sqrt(np.mean(v_err ** 2))) if nv else 0.0
    w_mae = float(np.mean(np.abs(w_err))) if nw else 0.0
    w_rmse = float(np.sqrt(np.mean(w_err ** 2))) if nw else 0.0

    v_jerk = _jerk(pred_v)
    w_jerk = _jerk(pred_w)
    v_jerk_ratio = float(v_jerk / (_jerk(gt_v) + _EPS))
    w_jerk_ratio = float(w_jerk / (_jerk(gt_w) + _EPS))

    metrics: Dict[str, float] = {
        "ade": ade,
        "fde": fde,
        "max_pos_err": max_pos_err,
        "heading_mae_deg": heading_mae_deg,
        "heading_final_err_deg": heading_final_err_deg,
        "success": bool(success),
        "v_mae": v_mae,
        "v_rmse": v_rmse,
        "w_mae": w_mae,
        "w_rmse": w_rmse,
        "v_jerk": v_jerk,
        "w_jerk": w_jerk,
        "v_jerk_ratio": v_jerk_ratio,
        "w_jerk_ratio": w_jerk_ratio,
    }

    if sample_v_std is not None and np.asarray(sample_v_std).size:
        metrics["v_sample_std"] = float(np.mean(np.asarray(sample_v_std, dtype=np.float64)))
    if sample_w_std is not None and np.asarray(sample_w_std).size:
        metrics["w_sample_std"] = float(np.mean(np.asarray(sample_w_std, dtype=np.float64)))

    return metrics


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------
# Numeric per-episode keys that get a dataset-level "{key}_mean".
_AGGREGATE_KEYS = (
    "ade",
    "fde",
    "max_pos_err",
    "heading_mae_deg",
    "heading_final_err_deg",
    "v_mae",
    "v_rmse",
    "w_mae",
    "w_rmse",
    "v_jerk",
    "w_jerk",
    "v_jerk_ratio",
    "w_jerk_ratio",
    "v_sample_std",
    "w_sample_std",
)


def aggregate_episode_metrics(per_episode_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a list of per-episode metric dicts into dataset-level numbers.

    Produces "{key}_mean" for each numeric metric, plus "success_rate"
    (fraction of episodes that succeeded) and "n_episodes".
    """
    n_episodes = len(per_episode_metrics)
    out: Dict[str, float] = {"n_episodes": int(n_episodes)}
    if n_episodes == 0:
        out["success_rate"] = 0.0
        return out

    for key in _AGGREGATE_KEYS:
        vals = [float(m[key]) for m in per_episode_metrics if key in m and m[key] is not None]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))

    successes = [bool(m.get("success", False)) for m in per_episode_metrics]
    out["success_rate"] = float(np.mean(successes))
    out["n_success"] = int(np.sum(successes))

    return out


# ---------------------------------------------------------------------------
# Plot annotation
# ---------------------------------------------------------------------------
# (label, "{:fmt}", unit) for known metric keys.
_LABELS = {
    "ade": ("ADE", "{:.3f}", "m"),
    "fde": ("FDE", "{:.3f}", "m"),
    "max_pos_err": ("Max pos err", "{:.3f}", "m"),
    "heading_mae_deg": ("Heading MAE", "{:.1f}", "deg"),
    "heading_final_err_deg": ("Heading final", "{:.1f}", "deg"),
    "v_mae": ("v MAE", "{:.3f}", "m/s"),
    "v_rmse": ("v RMSE", "{:.3f}", "m/s"),
    "w_mae": ("w MAE", "{:.3f}", "rad/s"),
    "w_rmse": ("w RMSE", "{:.3f}", "rad/s"),
    "v_jerk": ("v jerk", "{:.4f}", ""),
    "w_jerk": ("w jerk", "{:.4f}", ""),
    "v_jerk_ratio": ("v jerk ratio", "{:.2f}", "x"),
    "w_jerk_ratio": ("w jerk ratio", "{:.2f}", "x"),
    "v_sample_std": ("v sample std", "{:.4f}", ""),
    "w_sample_std": ("w sample std", "{:.4f}", ""),
    "success": ("Success", None, ""),
}


def format_metric_textbox(metrics: Dict[str, float], keys: Sequence[str]) -> str:
    """Render selected metrics as a multi-line string for a matplotlib text box."""
    lines = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        label, fmt, unit = _LABELS.get(key, (key, "{}", ""))

        if key == "success":
            text = "YES" if bool(value) else "NO"
        elif fmt is None:
            text = str(value)
        else:
            try:
                text = fmt.format(float(value))
            except (TypeError, ValueError):
                text = str(value)
            if unit:
                text = f"{text} {unit}"

        lines.append(f"{label}: {text}")
    return "\n".join(lines)
