"""Compare eval metrics across models from test/out/weekend/*.json.

Reads every ``<experiment>_<tag>.json`` written by test/eval_run.py and renders:
  1. one bar panel per metric, all models side by side (fixed model order across
     panels so a model is traceable panel-to-panel; every metric lower=better)
  2. per-episode endpoint error (FDE) lines, one per model — shows whether a
     model wins everywhere or only on easy episodes

Endpoint-error stats (mean/median over episodes) are recomputed here from the
per-episode ``openloop`` entries, so JSONs written before fde_cm_mean existed in
the summary still plot correctly.

Usage (from repo root):
  python test/plot_model_metrics.py                      # all *_heldout.json
  python test/plot_model_metrics.py --tag heldout --models s20_nogoal,flow_goal_adv
  python test/plot_model_metrics.py --metrics fde_cm_mean,near_mm

Output: test/out/model_metrics_<tag>.png (+ a printed table)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
WEEKEND_DIR = os.path.join(OUT_DIR, "weekend")

# label, unit, summary key (None -> recomputed from per-episode openloop data)
METRICS = {
    "fde_cm_mean": ("endpoint error FDE (mean over eps)", "cm"),
    "fde_cm": ("endpoint error FDE (median over eps)", "cm"),
    "ade_cm": ("whole-path error ADE (median)", "cm"),
    "vel_rmse": ("velocity RMSE", ""),
    "near_mm": ("aux dock-pose err <0.6m (median)", "mm"),
    "align_deg": ("final-alignment err <0.6m (median)", "deg"),
    "xpos_mm": ("final forward-x err <0.6m (median)", "mm"),
}
DEFAULT_METRICS = ["fde_cm_mean", "fde_cm", "ade_cm", "vel_rmse", "near_mm", "align_deg"]

# validated categorical palette (dataviz reference, light mode, fixed order)
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]


def load_runs(tag, model_filter):
    runs = {}
    for path in sorted(glob.glob(os.path.join(WEEKEND_DIR, f"*_{tag}.json"))):
        name = os.path.basename(path)[: -len(f"_{tag}.json")]
        if model_filter and name not in model_filter:
            continue
        d = json.load(open(path))
        s = dict(d.get("summary", {}))
        eps = d.get("openloop", {})
        fde = [r["fde_cm"] for r in eps.values()]
        ade = [r["ade_cm"] for r in eps.values()]
        if fde:  # recompute so pre-fde_cm_mean JSONs are covered
            s["fde_cm_mean"], s["fde_cm"] = float(np.mean(fde)), float(np.median(fde))
            s["ade_cm_mean"], s["ade_cm"] = float(np.mean(ade)), float(np.median(ade))
        runs[name] = dict(summary=s, per_ep_fde={int(e): r["fde_cm"] for e, r in eps.items()})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="heldout", help="JSON suffix to collect (default: heldout)")
    ap.add_argument("--models", default="", help="comma list to restrict (default: all found)")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help=f"comma list among {list(METRICS)}")
    args = ap.parse_args()

    model_filter = {m for m in args.models.split(",") if m}
    metrics = [m for m in args.metrics.split(",") if m]
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise SystemExit(f"unknown metrics {unknown}; choose from {list(METRICS)}")

    runs = load_runs(args.tag, model_filter)
    if not runs:
        raise SystemExit(f"no *_{args.tag}.json found in {WEEKEND_DIR}")

    # one fixed model order everywhere: best endpoint-error mean first
    order = sorted(runs, key=lambda n: runs[n]["summary"].get("fde_cm_mean", float("inf")))

    # printed table
    hdr = f"{'model':28s} {'n_eps':>5s} " + " ".join(f"{m:>12s}" for m in metrics)
    print(hdr + "\n" + "-" * len(hdr))
    for name in order:
        s = runs[name]["summary"]
        cells = " ".join(f"{s[m]:12.2f}" if s.get(m) is not None else f"{'n/a':>12s}" for m in metrics)
        print(f"{name:28s} {len(runs[name]['per_ep_fde']):5d} {cells}")
    n_eps = {len(r["per_ep_fde"]) for r in runs.values()}
    if len(n_eps) > 1:
        print(f"WARNING: models were scored on different episode counts {sorted(n_eps)} "
              f"-- means/medians are not directly comparable across them")

    ncol = 2
    nrow = (len(metrics) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 0.42 * len(order) * nrow + 2.2 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    y = np.arange(len(order))
    for ax, m in zip(axes.ravel(), metrics):
        label, unit = METRICS[m]
        vals = np.array([runs[n]["summary"].get(m) if runs[n]["summary"].get(m) is not None
                         else np.nan for n in order], dtype=float)
        ax.barh(y, np.nan_to_num(vals), height=0.62, color="#2a78d6", zorder=3)
        for yi, v in zip(y, vals):
            ax.text(0 if np.isnan(v) else v, yi, "  n/a" if np.isnan(v) else f" {v:.3g}",
                    va="center", fontsize=7.5, color="#52514e", zorder=4)
        ax.set_yticks(y, order, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{label}{f' [{unit}]' if unit else ''}  (lower = better)", fontsize=10)
        ax.grid(True, axis="x", ls="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)

    # per-episode FDE panel: line per model (validated palette holds 8 slots)
    shown = order[:len(PALETTE)]
    fig2, ax2 = plt.subplots(figsize=(9, 4.4))
    fig2.patch.set_facecolor("#fcfcfb")
    for name, col in zip(shown, PALETTE):
        per = runs[name]["per_ep_fde"]
        eps = sorted(per)
        lbl = name if len(per) == max(len(r["per_ep_fde"]) for r in runs.values()) \
            else f"{name} ({len(per)} eps only)"
        ax2.plot(eps, [per[e] for e in eps], "-o", color=col, lw=2, ms=5, label=lbl)
    if len(order) > len(shown):
        ax2.set_title(f"per-episode endpoint error — best {len(shown)} of {len(order)} models "
                      f"by mean FDE", fontsize=11)
    else:
        ax2.set_title("per-episode endpoint error (FDE)", fontsize=11)
    ax2.set_xlabel("test episode")
    ax2.set_ylabel("FDE [cm]")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.legend(fontsize=8, framealpha=0.9)

    os.makedirs(OUT_DIR, exist_ok=True)
    for f, suffix in ((fig, ""), (fig2, "_per_episode")):
        f.tight_layout()
        png = os.path.join(OUT_DIR, f"model_metrics_{args.tag}{suffix}.png")
        f.savefig(png, dpi=130, facecolor=f.get_facecolor(), bbox_inches="tight")
        print(f"saved {png}")


if __name__ == "__main__":
    main()
