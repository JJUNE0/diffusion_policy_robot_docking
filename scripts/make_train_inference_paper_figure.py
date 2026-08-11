#!/usr/bin/env python3
"""Render the shared-core, overlap-checked VoDOCK paper figure."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


W, H = 2000, 800
INK, MUTED = "#172733", "#5C6E79"
TRAIN, TRAIN_BG = "#126B94", "#EDF7FB"
LIVE, LIVE_BG = "#34775D", "#EEF7F1"
REL, REL_BG = "#A66325", "#FCF3E9"
SHARED, SHARED_BG = "#625198", "#F4F1FA"
LOSS, LOSS_BG = "#A4444C", "#FCEFF0"
CORE_BG = "#FAF9FD"
FONT = fm.FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_B = fm.FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass
class TextGroup:
    name: str
    patch: FancyBboxPatch
    texts: tuple


def rounded_patch(ax, x, y, w, h, edge, face, linewidth=1.5, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=12",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_card(
    ax, x, y, w, h, title, detail, edge, face, groups, labels,
    title_size=11.5, detail_size=9.2,
    title_frac=0.36, detail_frac=0.68,
):
    patch = rounded_patch(ax, x, y, w, h, edge, face)
    title_y = y + h * title_frac
    detail_y = y + h * detail_frac
    title_text = ax.text(
        x + w / 2, title_y, title,
        ha="center", va="center",
        fontproperties=FONT_B, fontsize=title_size, color=INK, zorder=5,
    )
    detail_text = ax.text(
        x + w / 2, detail_y, detail,
        ha="center", va="center",
        fontproperties=FONT, fontsize=detail_size, color=MUTED,
        linespacing=1.35, zorder=5,
    )
    groups.append(TextGroup(title, patch, (title_text, detail_text)))
    labels.extend((title_text, detail_text))
    return patch


def add_tag(
    ax, x, y, w, h, label, edge, face, groups, labels,
    fontsize=7.5, text_color=None,
):
    patch = rounded_patch(ax, x, y, w, h, edge, face, linewidth=1.1, zorder=6)
    item = ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontproperties=FONT_B, fontsize=fontsize,
        color=text_color or edge, linespacing=1.2, zorder=7,
    )
    groups.append(TextGroup(label, patch, (item,)))
    labels.append(item)
    return patch


def add_text(ax, labels, *args, **kwargs):
    item = ax.text(*args, **kwargs)
    labels.append(item)
    return item


def arrow(ax, x1, y1, x2, y2, color, dashed=False, zorder=4):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.6, color=color,
        linestyle=(0, (5, 4)) if dashed else "-",
        shrinkA=3, shrinkB=3, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def validate_layout(fig, groups, labels):
    """Reject text outside its card and all label-to-label collisions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    errors = []

    for group in groups:
        outer = group.patch.get_window_extent(renderer).padded(-3)
        for item in group.texts:
            inner = item.get_window_extent(renderer)
            if (
                inner.x0 < outer.x0
                or inner.x1 > outer.x1
                or inner.y0 < outer.y0
                or inner.y1 > outer.y1
            ):
                errors.append(f"text outside '{group.name}': {item.get_text()!r}")

    for first, second in combinations(labels, 2):
        a = first.get_window_extent(renderer).padded(2)
        b = second.get_window_extent(renderer).padded(2)
        if a.overlaps(b):
            errors.append(
                f"label collision: {first.get_text()!r} <> {second.get_text()!r}"
            )

    if errors:
        raise RuntimeError("Layout validation failed:\n  " + "\n  ".join(errors))


def draw(out_dir=Path("docs/figures")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    fig.patch.set_facecolor("white")
    groups, labels = [], []

    # Section headers.
    for x, label in (
        (105, "MODE"),
        (405, "COMMON CONTEXT"),
        (1135, "SHARED CORE"),
        (1835, "RESULT"),
    ):
        add_text(
            ax, labels, x, 42, label,
            ha="center", va="center",
            fontproperties=FONT_B, fontsize=9.2, color=MUTED, zorder=8,
        )

    # Mode-specific sources feed one shared input contract.
    add_tag(
        ax, 35, 168, 140, 76, "TRAIN\nrecorded",
        TRAIN, TRAIN_BG, groups, labels, fontsize=8.3,
    )
    add_tag(
        ax, 35, 520, 140, 76, "INFER\nlive",
        LIVE, LIVE_BG, groups, labels, fontsize=8.3,
    )

    add_card(
        ax, 240, 245, 330, 250,
        "Context window", "RGB ×5 + goal\nwheel (vx,wz) ×60",
        TRAIN, "#FFFFFF", groups, labels,
        title_size=12.0, detail_size=9.5,
    )

    # One shared frozen visual module and one shared learned policy.
    rounded_patch(ax, 640, 90, 1000, 500, SHARED, CORE_BG, linewidth=1.2, zorder=0)

    add_card(
        ax, 700, 165, 360, 205,
        "Frozen ReLoc3R", "current ↔ goal",
        REL, REL_BG, groups, labels,
        title_size=11.5, detail_size=9.2,
        title_frac=0.28, detail_frac=0.50,
    )
    add_tag(
        ax, 715, 310, 160, 44, "TRAIN cache",
        TRAIN, "#FFFFFF", groups, labels, fontsize=6.8,
    )
    add_tag(
        ax, 885, 310, 160, 44, "INFER live",
        LIVE, "#FFFFFF", groups, labels, fontsize=6.8,
    )

    add_card(
        ax, 1210, 230, 370, 235,
        "Docking policy", "condition fusion + DiT",
        SHARED, SHARED_BG, groups, labels,
        title_size=11.5, detail_size=9.0,
        title_frac=0.25, detail_frac=0.48,
    )
    add_tag(
        ax, 1225, 390, 160, 44, "TRAIN · θ",
        TRAIN, "#FFFFFF", groups, labels, fontsize=6.8,
    )
    add_tag(
        ax, 1395, 390, 160, 44, r"INFER · $\bar{\theta}$",
        LIVE, "#FFFFFF", groups, labels, fontsize=6.8,
    )
    add_text(
        ax, labels, 1395, 525, r"$\bar{\theta}$ = frozen averaged weights",
        ha="center", va="center",
        fontproperties=FONT, fontsize=7.6, color=SHARED, zorder=8,
    )

    # Shared input branches: images use ReLoc3R; wheel history bypasses it.
    arrow(ax, 175, 206, 240, 318, TRAIN)
    arrow(ax, 175, 558, 240, 426, LIVE)
    arrow(ax, 570, 315, 700, 255, REL)
    arrow(ax, 1060, 255, 1210, 305, SHARED)
    arrow(ax, 570, 430, 1210, 410, SHARED)

    add_text(
        ax, labels, 626, 270, "RGB + goal",
        ha="center", va="center",
        fontproperties=FONT, fontsize=7.2, color=REL, zorder=8,
    )
    add_text(
        ax, labels, 875, 447, "wheel history",
        ha="center", va="center",
        fontproperties=FONT, fontsize=7.2, color=SHARED, zorder=8,
    )
    add_text(
        ax, labels, 1135, 258, "dec1 / dec2",
        ha="center", va="center",
        fontproperties=FONT, fontsize=7.2, color=SHARED, zorder=8,
    )

    # Only the objective/execution path differs after the shared architecture.
    add_card(
        ax, 1680, 135, 300, 185,
        "Train one batch", "sample t · one call\nMSE → update",
        LOSS, LOSS_BG, groups, labels,
        title_size=10.4, detail_size=8.2,
    )
    add_card(
        ax, 1680, 475, 300, 185,
        "Infer one plan", "policy calls ×30\nplan ×60 → publish ×32",
        LIVE, LIVE_BG, groups, labels,
        title_size=10.4, detail_size=8.2,
    )
    arrow(ax, 1580, 305, 1680, 228, TRAIN)
    arrow(ax, 1580, 410, 1680, 568, LIVE)

    # Closed-loop execution returns to the live input.
    loop_y = 720
    ax.plot(
        [1832, 1832, 105, 105],
        [660, loop_y, loop_y, 606],
        color=LIVE, lw=1.5, ls=(0, (5, 4)), zorder=1,
    )
    arrow(ax, 105, 606, 105, 596, LIVE, dashed=True, zorder=2)
    add_text(
        ax, labels, W / 2, 758, "observe  →  replan",
        ha="center", va="center",
        fontproperties=FONT_B, fontsize=8.3, color=LIVE, zorder=8,
    )

    validate_layout(fig, groups, labels)

    stem = out_dir / "vodock_train_inference_paper"
    fig.savefig(
        stem.with_suffix(".png"), dpi=240,
        bbox_inches="tight", pad_inches=0.10, facecolor="white",
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight", pad_inches=0.10, facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


if __name__ == "__main__":
    outputs = draw()
    print("layout validation: PASS")
    for output in outputs:
        print(output)
