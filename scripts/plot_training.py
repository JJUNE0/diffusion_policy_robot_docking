"""Plot training convergence from metrics.jsonl.

Reads a metrics.jsonl file (one JSON object per line) and plots the
denoising loss vs step.

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

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            steps.append(d["step"])
            losses.append(d.get("loss"))

    if len(steps) == 0:
        print("metrics.jsonl is empty, skipping plot.")
        return

    steps = np.array(steps)
    losses = np.array(losses, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.plot(steps, losses, ".", alpha=0.2, color="tab:blue", markersize=2)
    ax.plot(steps, _smooth(losses), color="tab:blue", linewidth=1.5, label="denoising loss")
    ax.set_xlabel("step")
    ax.set_ylabel("denoising loss")
    ax.set_title("Policy (denoising) convergence")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(jsonl_path), "train_convergence.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"saved {out_path}  ({len(steps)} points, final loss {losses[-1]:.4f})")


def main():
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "metrics.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    plot_from_jsonl(jsonl_path, out_path)


if __name__ == "__main__":
    main()
