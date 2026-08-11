#!/usr/bin/env python3
"""r_relfeat_only 아키텍처 도면 -> PNG.

좌표계는 SVG와 동일한 860x800 (y는 아래로 증가). 모든 수치는
outputs/train/r_relfeat_only/.../r_relfeat_only_step_16940.pt 에서 집계한 값.

  python make_arch_png.py            # ko + en 둘 다
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

W, H = 860, 800

INK, INK2 = "#14202b", "#4d5f6e"
RULE, FROZEN, LIVE = "#c9d4dc", "#76858f", "#15618f"
TAP, TAP_SOFT, SURF2 = "#a8571a", "#f6ebe0", "#eaeff3"

SANS = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
SANSB = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf")
MONO = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf")

L = {
    "ko": dict(
        frozen_hdr="FROZEN  ·  오프라인 사전계산  ·  gradient 없음",
        cur="현재 프레임 ×5", cur_s="stride 12 · 약 1.6 s 창",
        goal="목표 프레임 ×1", goal_s="goal image",
        enc="ReLoc3R ViT-L encoder", enc_s="24 blocks · 두 스트림이 weight 공유",
        dec="ReLoc3R ViT-B decoder · 12 blocks",
        dec_s="self-attn + cross-attn · 두 스트림이 서로를 참조",
        d1_s="현재 → 목표를 참조", d2_s="목표 → 현재를 참조",
        tap="↑ TAP · pose head 이전 · 이 arm의 유일한 시각 신호",
        ph_s="avgpool → pose [12]", unused="미사용",
        boundary="HDF5 로 사전계산 저장 · 226 GB",
        train_hdr="TRAINABLE  ·  54,596,354 params",
        wheel="휠 오도메트리", wheel_s="MLP + time emb",
        e1="dec1 encoder", e2="dec2 encoder",
        enc_s2="proj → Perceiver ×16", enc_s3="독립 weight",
        tok="tok",
        concat="concat → 220 tokens  +  modality emb (3×384)  +  sinusoidal pos",
        fus="융합 Transformer · 4 layers · d=384 · 6 heads",
        fus_s="6,503,424 · padding mask는 여기서만 적용",
        cflow="C [220 × 384]  →  cross-attn 의 K, V",
        dit="DiTCrossAttn1d · 12 × DiTCrossAttnBlock",
        dit_s="44,338,176 (81.2 %) · dropout 0.0",
        xt_s="잡음 섞인 행동 + 확산 스텝",
        act_s="(v, ω) × 60 step · 2.0 s · minmax 정규화",
    ),
    "en": dict(
        frozen_hdr="FROZEN  ·  PRECOMPUTED OFFLINE  ·  NO GRADIENT",
        cur="current frames ×5", cur_s="stride 12 · ~1.6 s window",
        goal="goal frame ×1", goal_s="goal image",
        enc="ReLoc3R ViT-L encoder", enc_s="24 blocks · weights shared across streams",
        dec="ReLoc3R ViT-B decoder · 12 blocks",
        dec_s="self-attn + cross-attn · streams attend into each other",
        d1_s="current, after attending goal", d2_s="goal, after attending current",
        tap="↑ TAP · pre-head · the only vision signal in this arm",
        ph_s="avgpool → pose [12]", unused="not used",
        boundary="precomputed to HDF5 · 226 GB",
        train_hdr="TRAINABLE  ·  54,596,354 params",
        wheel="wheel odometry", wheel_s="MLP + time emb",
        e1="dec1 encoder", e2="dec2 encoder",
        enc_s2="proj → Perceiver ×16", enc_s3="independent weights",
        tok="tok",
        concat="concat → 220 tokens  +  modality emb (3×384)  +  sinusoidal pos",
        fus="fusion Transformer · 4 layers · d=384 · 6 heads",
        fus_s="6,503,424 · padding mask applied only here",
        cflow="C [220 × 384]  →  K, V of cross-attn",
        dit="DiTCrossAttn1d · 12 × DiTCrossAttnBlock",
        dit_s="44,338,176 (81.2 %) · dropout 0.0",
        xt_s="noisy action + diffusion step",
        act_s="(v, ω) × 60 steps · 2.0 s · minmax normalized",
    ),
}


def box(ax, x, y, w, h, ec, fc="none", lw=1.0, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=2))


def txt(ax, x, y, s, size=9, color=INK, mono=False, bold=False, ha="center"):
    fp = MONO if mono else (SANSB if bold else SANS)
    ax.text(x, y, s, fontproperties=fp, fontsize=size, color=color,
            ha=ha, va="center", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.2, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 3)) if dashed else "solid", zorder=3))


def draw(lang):
    t = L[lang]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis()
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---------------- FROZEN ----------------
    txt(ax, 56, 20, t["frozen_hdr"], 7.2, FROZEN, mono=True, ha="left")

    box(ax, 56, 34, 240, 48, FROZEN)
    txt(ax, 176, 53, t["cur"], 9)
    txt(ax, 176, 70, t["cur_s"], 7.4, INK2, mono=True)
    box(ax, 316, 34, 240, 48, FROZEN)
    txt(ax, 436, 53, t["goal"], 9)
    txt(ax, 436, 70, t["goal_s"], 7.4, INK2, mono=True)

    arrow(ax, 176, 82, 176, 103, FROZEN)
    arrow(ax, 436, 82, 436, 103, FROZEN)

    box(ax, 56, 104, 500, 46, FROZEN)
    txt(ax, 306, 122, t["enc"], 9)
    txt(ax, 306, 139, t["enc_s"], 7.4, INK2, mono=True)

    arrow(ax, 176, 150, 176, 171, FROZEN)
    arrow(ax, 436, 150, 436, 171, FROZEN)

    box(ax, 56, 172, 500, 56, FROZEN)
    txt(ax, 306, 192, t["dec"], 9)
    txt(ax, 306, 211, t["dec_s"], 7.4, INK2, mono=True)

    arrow(ax, 176, 228, 176, 251, TAP, lw=1.4)
    arrow(ax, 436, 228, 436, 251, TAP, lw=1.4)

    box(ax, 56, 252, 240, 56, TAP, TAP_SOFT, lw=1.5)
    txt(ax, 176, 272, "dec1  [196 × 768]", 8.5, INK, mono=True)
    txt(ax, 176, 291, t["d1_s"], 7.4, INK2, mono=True)
    box(ax, 316, 252, 240, 56, TAP, TAP_SOFT, lw=1.5)
    txt(ax, 436, 272, "dec2  [196 × 768]", 8.5, INK, mono=True)
    txt(ax, 436, 291, t["d2_s"], 7.4, INK2, mono=True)

    txt(ax, 56, 326, t["tap"], 7.6, TAP, mono=True, ha="left")

    # bypassed pose head
    ax.plot([556, 612], [280, 280], color=FROZEN, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.plot([580, 592], [272, 288], color=FROZEN, lw=1.3, zorder=3)
    ax.plot([592, 580], [272, 288], color=FROZEN, lw=1.3, zorder=3)
    box(ax, 620, 252, 224, 56, FROZEN, dashed=True)
    txt(ax, 732, 272, "PoseHead", 8.5, INK2)
    txt(ax, 732, 291, t["ph_s"], 7.4, INK2, mono=True)
    txt(ax, 732, 326, t["unused"], 7.6, FROZEN, mono=True)

    # boundary
    ax.plot([40, 152], [350, 350], color=RULE, lw=1.1, ls=(0, (5, 5)), zorder=1)
    ax.plot([460, 572], [350, 350], color=RULE, lw=1.1, ls=(0, (5, 5)), zorder=1)
    txt(ax, 306, 350, t["boundary"], 7.4, INK2, mono=True)

    # ---------------- TRAINABLE ----------------
    txt(ax, 56, 380, t["train_hdr"], 7.2, LIVE, mono=True, ha="left")

    box(ax, 56, 394, 158, 78, RULE, SURF2)
    txt(ax, 135, 411, t["wheel"], 8.6)
    txt(ax, 135, 428, "[60 × 2]", 7.4, INK2, mono=True)
    txt(ax, 135, 444, t["wheel_s"], 7.4, INK2, mono=True)
    txt(ax, 135, 460, "172,032", 7.6, LIVE, mono=True)

    for x0, cx, name, sub in ((227, 306, t["e1"], t["enc_s2"]), (398, 477, t["e2"], t["enc_s3"])):
        box(ax, x0, 394, 158, 78, TAP, TAP_SOFT, lw=1.5)
        txt(ax, cx, 411, name, 8.4, INK, mono=True)
        txt(ax, cx, 428, "[5 × 196 × 768]", 7.4, INK2, mono=True)
        txt(ax, cx, 444, sub, 7.4, INK2, mono=True)
        txt(ax, cx, 460, "1,494,144", 7.6, LIVE, mono=True)

    for cx, n in ((135, 60), (306, 80), (477, 80)):
        arrow(ax, cx, 472, cx, 503)
        txt(ax, cx + 7, 488, f"{n} {t['tok']}", 7.2, INK2, mono=True, ha="left")

    box(ax, 56, 506, 500, 34, LIVE, SURF2, lw=1.5)
    txt(ax, 306, 523, t["concat"], 8.0, INK, mono=True)

    arrow(ax, 306, 540, 306, 563)

    box(ax, 56, 566, 500, 52, LIVE, SURF2, lw=1.5)
    txt(ax, 306, 586, t["fus"], 9)
    txt(ax, 306, 605, t["fus_s"], 7.4, INK2, mono=True)

    arrow(ax, 306, 618, 306, 647)
    txt(ax, 316, 635, t["cflow"], 7.4, INK2, mono=True, ha="left")

    box(ax, 56, 650, 500, 56, LIVE, SURF2, lw=1.5)
    txt(ax, 306, 670, t["dit"], 9)
    txt(ax, 306, 689, t["dit_s"], 7.4, INK2, mono=True)

    box(ax, 620, 650, 224, 56, RULE, SURF2)
    txt(ax, 732, 670, "x_t [60 × 2] + t", 8.4, INK, mono=True)
    txt(ax, 732, 689, t["xt_s"], 7.4, INK2, mono=True)
    arrow(ax, 618, 678, 560, 678)
    txt(ax, 589, 668, "Q", 7.4, INK2, mono=True)

    arrow(ax, 306, 706, 306, 727)

    box(ax, 56, 730, 500, 46, LIVE, SURF2, lw=1.5)
    txt(ax, 306, 750, "action [60 × 2]", 9, INK, mono=True)
    txt(ax, 306, 768, t["act_s"], 7.4, INK2, mono=True)

    out = f"docs/figures/relfeat_arch_{lang}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for lang in ("ko", "en"):
        print("saved", draw(lang))
