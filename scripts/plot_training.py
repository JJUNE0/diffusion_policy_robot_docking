"""Plot single-model training convergence from a train log.

Parses lines like:  Step 1500 | Loss: 0.42 | aux 0.08 | dock 31.5mm
and plots denoising loss, aux loss, and dock-pose error (mm) vs step.

  python scripts/plot_training.py results/train_single.log [out.png]
"""

import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINE = re.compile(r"Step\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)(?:\s+\|\s+aux\s+([\d.]+)\s+\|\s+dock\s+([\d.]+)mm)?")


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "results/train_single.log"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/train_convergence.png"
    step, loss, aux, dock = [], [], [], []
    for ln in open(log):
        m = LINE.search(ln)
        if not m:
            continue
        step.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        if m.group(3):
            aux.append(float(m.group(3)))
            dock.append(float(m.group(4)))
    step = np.array(step)

    def smooth(y, k=9):
        y = np.asarray(y, float)
        if len(y) < k:
            return y
        return np.convolve(y, np.ones(k) / k, mode="same")

    has_aux = len(aux) == len(step) and len(aux) > 0
    fig, axes = plt.subplots(1, 2 if has_aux else 1, figsize=(12 if has_aux else 6, 4.5))
    axes = np.atleast_1d(axes)

    axes[0].plot(step, loss, ".", alpha=.25, color="tab:blue")
    axes[0].plot(step, smooth(loss), color="tab:blue", label="denoising loss")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("denoising loss")
    axes[0].set_title("Policy (denoising) convergence"); axes[0].grid(alpha=.3); axes[0].legend()

    if has_aux:
        ax = axes[1]
        ax.plot(step, dock, ".", alpha=.25, color="tab:red")
        ax.plot(step, smooth(dock), color="tab:red", label="dock pose err (mm)")
        ax.axhline(10, ls="--", color="gray", label="1 cm success tol")
        ax.set_xlabel("step"); ax.set_ylabel("dock pose error [mm]")
        ax.set_title("ICP aux head (precision) — train"); ax.grid(alpha=.3); ax.legend()
        ax.set_ylim(0, max(20, np.percentile(dock, 95)))

    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print(f"saved {out}  ({len(step)} points, final loss {loss[-1]:.3f}"
          + (f", dock {dock[-1]:.1f}mm)" if has_aux else ")"))


if __name__ == "__main__":
    main()
