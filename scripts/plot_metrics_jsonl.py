#!/usr/bin/env python3
"""
Plot training metrics from one or more metrics.jsonl files (Logger output).

Example:
    python scripts/plot_metrics_jsonl.py \\
      outputs/results/velocity_vision_velocity/2026-05-13_12-15-23/metrics.jsonl \\
      outputs/results/velocity_vision_velocity/2026-05-13_13-47-28/metrics.jsonl \\
      --labels run_a run_b \\
      -o outputs/results/velocity_vision_velocity/metrics_compare.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, list]:
    steps: list[int] = []
    loss: list[float | None] = []
    grad_norm: list[float | None] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" not in d:
                continue
            steps.append(int(d["step"]))
            loss.append(d.get("loss"))
            gv = d.get("grad_norm")
            grad_norm.append(float(gv) if gv is not None else None)
    return {"step": steps, "loss": loss, "grad_norm": grad_norm}


def plot_runs(runs: list[tuple[str, Path]], out_path: Path | None, smooth: int):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
        layout="constrained",
    )
    ax_loss, ax_grad = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    any_grad = False
    for idx, (label, path) in enumerate(runs):
        data = load_jsonl(path)
        color = colors[idx % len(colors)]
        steps = data["step"]
        loss_raw = data["loss"]

        s_loss = [(s, float(l)) for s, l in zip(steps, loss_raw, strict=False) if l is not None]
        if not s_loss:
            continue
        sx, ly = zip(*s_loss, strict=False)
        ax_loss.plot(sx, ly, alpha=0.25, color=color, lw=0.8)

        if smooth > 1:
            pairs = list(zip(sx, ly, strict=False))
            sma_s, sma_y = [], []
            window_vals: list[float] = []
            for _, (s_val, y_val) in enumerate(pairs):
                window_vals.append(y_val)
                if len(window_vals) > smooth:
                    window_vals.pop(0)
                sma_s.append(s_val)
                sma_y.append(sum(window_vals) / len(window_vals))
            ax_loss.plot(sma_s, sma_y, color=color, lw=1.5, label=f"{label} (MA{smooth})")
        else:
            ax_loss.plot(sx, ly, color=color, lw=1.2, label=label)

        gn = [(s, float(g)) for s, g in zip(steps, data["grad_norm"], strict=False) if g is not None]
        if gn:
            any_grad = True
            gx, gy = zip(*gn, strict=False)
            ax_grad.plot(gx, gy, alpha=0.35, color=color, lw=0.7)

    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_grad.set_xlabel("step")
    ax_grad.set_ylabel("grad_norm")
    ax_grad.grid(True, alpha=0.3)
    if not any_grad:
        ax_grad.text(
            0.5,
            0.5,
            "grad_norm is null for all points",
            ha="center",
            va="center",
            transform=ax_grad.transAxes,
        )

    fig.suptitle("metrics.jsonl comparison", fontsize=12, fontweight="bold")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path.resolve()}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from metrics.jsonl (compare multiple runs)")
    parser.add_argument("jsonl", nargs="+", help="paths to metrics.jsonl")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="legend labels (same count as paths; default: parent folder name of each jsonl)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="output figure (.png/.pdf); omit to show interactive window",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="moving average window for smoothed loss (1 = overlay raw only)",
    )
    args = parser.parse_args()

    paths = [Path(x).expanduser().resolve() for x in args.jsonl]
    for path in paths:
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)

    labels = args.labels
    if labels is None:
        labels = [p.parent.name for p in paths]
    elif len(labels) != len(paths):
        print("--labels count must match number of jsonl files", file=sys.stderr)
        sys.exit(1)

    runs = list(zip(labels, paths, strict=True))
    plot_runs(runs, args.out, max(1, args.smooth))


if __name__ == "__main__":
    main()
