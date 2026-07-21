"""Render the inference-hyperparameter (action-chunking K x warm-start) sweep
as small multiples: 3 metrics (rows) x 3 models (cols). Each panel plots the
metric vs K in {2,8,16} for warm-start OFF vs ON, with the K=1 legacy default
(EMA, resample-every-frame) as a horizontal reference. Writes a light-mode PNG.

Palette: dataviz default categorical, validated (validate_palette.js) --
  warm OFF = blue #2a78d6, warm ON = orange #eb6834 (2-way categorical);
  model identity is the column (labeled), metric is the row (labeled).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/work/.postech/diffusion_policy_robot_docking"
OUT_DIR = os.path.join(REPO, "test", "out", "weekend")
PNG = os.path.join(REPO, "docs", "img", "ksweep.png")

MODELS = [
    ("graft_goallidar", "goallidar"),
    ("graft_goalimg_lidar", "goalimg_lidar"),
    ("graft_lidar_goalimg", "lidar_goalimg"),
]
METRICS = [("ade_cm", "ADE (cm)"), ("fde_cm", "FDE (cm)"), ("vel_rmse", "velRMSE (x1000)")]
KS = [2, 8, 16]

# dataviz default categorical (light), validated
C_OFF, C_ON, C_REF = "#2a78d6", "#eb6834", "#8a8a86"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e6e2", "#fcfcfb"


def load(model, tag):
    fn = os.path.join(OUT_DIR, f"{model}_{tag}.json" if tag else f"{model}.json")
    if not os.path.exists(fn):
        return None
    return json.load(open(fn))["summary"]


def val(summary, key):
    if summary is None or summary.get(key) is None:
        return None
    return summary[key] * (1000.0 if key == "vel_rmse" else 1.0)


plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, axes = plt.subplots(len(METRICS), len(MODELS), figsize=(11, 8.5), sharex=True)

for r, (mkey, mlabel) in enumerate(METRICS):
    for c, (model, mshort) in enumerate(MODELS):
        ax = axes[r][c]
        base = val(load(model, "heldout"), mkey)                     # K=1 legacy
        y_off = [val(load(model, f"heldout_k{k}_warm0"), mkey) for k in KS]
        y_on = [val(load(model, f"heldout_k{k}_warm1"), mkey) for k in KS]

        if base is not None:
            ax.axhline(base, color=C_REF, lw=1.4, ls="--", zorder=1)
            ax.text(KS[-1], base, f" K1={base:.1f}", color=C_REF, va="center",
                    ha="left", fontsize=7.5)
        ax.plot(KS, y_off, "-o", color=C_OFF, lw=2.0, ms=6, zorder=3, label="warm OFF")
        ax.plot(KS, y_on, "-s", color=C_ON, lw=2.0, ms=6, zorder=3, label="warm ON")
        for xs, ys, col in ((KS, y_off, C_OFF), (KS, y_on, C_ON)):
            for x, y in zip(xs, ys):
                if y is not None:
                    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=7, color=col)

        ax.set_xticks(KS)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if r == 0:
            ax.set_title(mshort, color=INK, fontsize=11, pad=8, fontweight="bold")
        if c == 0:
            ax.set_ylabel(mlabel, color=INK, fontsize=10)
        if r == len(METRICS) - 1:
            ax.set_xlabel("action-chunk K", color=MUTED)

handles, labels = axes[0][0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=C_REF, ls="--", lw=1.4))
labels.append("K=1 legacy default (EMA)")
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), fontsize=9)
fig.suptitle("Inference hyperparameter sweep — action-chunking K x warm-start (held-out, 10 ep)",
             y=1.035, fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(PNG, dpi=150, bbox_inches="tight")
print(f"wrote {PNG}")

# also dump a compact table for the doc / accessibility
print("\nmodel,metric,K1,k2_off,k2_on,k8_off,k8_on,k16_off,k16_on")
for model, mshort in MODELS:
    for mkey, mlabel in METRICS:
        row = [f"{val(load(model,'heldout'),mkey)}"]
        for k in KS:
            row.append(f"{val(load(model,f'heldout_k{k}_warm0'),mkey)}")
            row.append(f"{val(load(model,f'heldout_k{k}_warm1'),mkey)}")
        print(f"{mshort},{mlabel}," + ",".join(row))
