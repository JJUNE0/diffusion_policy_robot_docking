"""Plot training convergence from metrics.jsonl.

Reads a metrics.jsonl file (one JSON object per line) and plots
denoising loss, aux loss, and dock-pose error (mm) vs step.

Can be used as:
  1) CLI:    python scripts/plot_training.py <metrics.jsonl> [out.png]
  2) Import: from scripts.plot_training import plot_from_jsonl
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _smooth(y, k=9):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def plot_from_jsonl(jsonl_path: str, out_path: str | None = None):
    """Read metrics.jsonl and save a training convergence plot.

    Args:
        jsonl_path: Path to the metrics.jsonl file.
        out_path:   Where to save the PNG. Defaults to
                    ``<same dir as jsonl>/train_convergence.png``.
    """
    if not os.path.exists(jsonl_path):
        print(f"No metrics file found at {jsonl_path}, skipping plot.")
        return

    steps, losses = [], []
    aux_losses, aux_mms = [], []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            steps.append(d["step"])
            losses.append(d.get("loss"))
            aux_losses.append(d.get("aux_loss"))
            aux_mms.append(d.get("aux_pose_mm"))

    if len(steps) == 0:
        print("metrics.jsonl is empty, skipping plot.")
        return

    steps = np.array(steps)
    losses = np.array(losses, dtype=float)

    has_aux = aux_losses[0] is not None and aux_mms[0] is not None
    n_panels = 3 if has_aux else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    axes = np.atleast_1d(axes)

    # Panel 1: Denoising loss
    axes[0].plot(steps, losses, ".", alpha=0.2, color="tab:blue", markersize=2)
    axes[0].plot(steps, _smooth(losses), color="tab:blue", linewidth=1.5, label="denoising loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("denoising loss")
    axes[0].set_title("Policy (denoising) convergence")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if has_aux:
        aux_arr = np.array(aux_losses, dtype=float)
        mm_arr = np.array(aux_mms, dtype=float)

        # Panel 2: Aux loss
        axes[1].plot(steps, aux_arr, ".", alpha=0.2, color="tab:orange", markersize=2)
        axes[1].plot(steps, _smooth(aux_arr), color="tab:orange", linewidth=1.5, label="aux loss")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("aux loss")
        axes[1].set_title("Aux Pose Loss")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        # Panel 3: Dock pose error (mm)
        axes[2].plot(steps, mm_arr, ".", alpha=0.2, color="tab:red", markersize=2)
        axes[2].plot(steps, _smooth(mm_arr), color="tab:red", linewidth=1.5, label="dock pose err (mm)")
        axes[2].axhline(10, ls="--", color="gray", alpha=0.7, label="1 cm success tol")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("dock pose error [mm]")
        axes[2].set_title("ICP aux head (precision) — train")
        axes[2].set_ylim(0, max(20, float(np.percentile(mm_arr, 95))))
        axes[2].grid(alpha=0.3)
        axes[2].legend()

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(jsonl_path), "train_convergence.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    summary = f"saved {out_path}  ({len(steps)} points, final loss {losses[-1]:.4f}"
    if has_aux:
        summary += f", dock {mm_arr[-1]:.1f}mm)"
    else:
        summary += ")"
    print(summary)


def main():
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "metrics.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    plot_from_jsonl(jsonl_path, out_path)


if __name__ == "__main__":
    main()
