#!/usr/bin/env python3
"""Render a code-derived, chronological training diagram for r_relfeat_only."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager as fm


W, H = 1200, 1720
INK = "#152432"
MUTED = "#526574"
LINE = "#B8C7D1"
BLUE = "#0E668F"
BLUE_BG = "#EAF5FA"
ORANGE = "#A95618"
ORANGE_BG = "#FBF0E6"
GREEN = "#31705A"
GREEN_BG = "#EAF5EF"
GRAY_BG = "#F3F6F8"
RED = "#A33C3C"

FONT = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
FONT_B = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf")
MONO = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf")


def box(ax, x, y, w, h, title, detail="", *, edge=LINE, face="white", mono=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=10",
        linewidth=1.7, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + 28, title, ha="center", va="center",
            fontproperties=MONO if mono else FONT_B, fontsize=11.5, color=INK, zorder=3)
    if detail:
        ax.text(x + w / 2, y + h - 25, detail, ha="center", va="center",
                fontproperties=MONO, fontsize=8.6, color=MUTED, linespacing=1.45, zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, text=None, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.8, color=color, linestyle="--" if dashed else "-", zorder=4))
    if text:
        ax.text((x1 + x2) / 2 + 10, (y1 + y2) / 2, text, ha="left", va="center",
                fontproperties=MONO, fontsize=8.2, color=color, zorder=5)


def phase(ax, y, number, title, subtitle, color):
    ax.text(50, y, f"{number:02d}", fontproperties=MONO, fontsize=11, color=color,
            ha="left", va="center")
    ax.text(100, y, title, fontproperties=FONT_B, fontsize=12, color=INK,
            ha="left", va="center")
    ax.text(1170, y, subtitle, fontproperties=MONO, fontsize=8.3, color=MUTED,
            ha="right", va="center")
    ax.plot([50, 1170], [y + 18, y + 18], color=LINE, lw=1.0, zorder=1)


def draw(out="docs/figures/r_relfeat_only_training_timeline_ko.png"):
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(50, 48, "r_relfeat_only · 학습 아키텍처", fontproperties=FONT_B,
            fontsize=22, color=INK, ha="left", va="center")
    ax.text(50, 80, "코드 실행 순서 기준 · 한 gradient step의 데이터 흐름",
            fontproperties=FONT, fontsize=11, color=MUTED, ha="left", va="center")

    phase(ax, 125, 0, "학습 전 1회: ReLoc3R 특징 사전계산", "frozen · no gradient", ORANGE)
    box(ax, 70, 165, 250, 92, "History RGB Hᵢ × 5",
        "image_bottom\nindices [0,12,24,36,48]", edge=ORANGE, face=ORANGE_BG)
    box(ax, 360, 165, 210, 92, "Goal RGB G × 1",
        "episode 마지막 frame", edge=ORANGE, face=ORANGE_BG)
    box(ax, 650, 165, 470, 92, "ReLoc3R encoder + bidirectional decoder",
        "ViT-L encoder → ViT-B decoder 12 blocks\npose head 직전 dec1 / dec2 tap", edge=ORANGE, face=ORANGE_BG)
    arrow(ax, 320, 211, 642, 211, color=ORANGE)
    arrow(ax, 570, 225, 642, 225, color=ORANGE)
    box(ax, 280, 290, 300, 82, "dec1ᵢ  [196 × 768]",
        "history가 goal을 참조한 stream", edge=ORANGE, face="white", mono=True)
    box(ax, 620, 290, 300, 82, "dec2ᵢ  [196 × 768]",
        "goal이 history를 참조한 stream", edge=ORANGE, face="white", mono=True)
    arrow(ax, 800, 257, 770, 282, color=ORANGE)
    arrow(ax, 950, 257, 880, 282, color=ORANGE)
    box(ax, 360, 400, 480, 66, "HDF5 sidecar cache",
        "reloc3r_dec1_bottom · reloc3r_dec2_bottom · float16", edge=ORANGE, face=ORANGE_BG)
    arrow(ax, 430, 372, 500, 394, color=ORANGE)
    arrow(ax, 770, 372, 700, 394, color=ORANGE)

    phase(ax, 510, 1, "배치 샘플링과 정규화", "batch=256 · 847 steps/epoch", BLUE)
    box(ax, 70, 550, 300, 92, "Wheel history",
        "encoder [B, 60, 2]\naction 통계로 정규화", edge=BLUE, face=BLUE_BG)
    box(ax, 450, 550, 300, 92, "Cached relations",
        "dec1, dec2 [B, 5, 196, 768]", edge=ORANGE, face=ORANGE_BG)
    box(ax, 830, 550, 300, 92, "Target action x₀",
        "future [B, 60, 2]\nminmax → [-1, 1]", edge=GREEN, face=GREEN_BG)

    phase(ax, 685, 2, "조건 시퀀스 C 생성", "trainable", BLUE)
    box(ax, 70, 725, 300, 100, "Motion encoder",
        "MLP + temporal embedding\n60 tokens × 384", edge=BLUE, face=BLUE_BG)
    box(ax, 450, 725, 300, 100, "dec1 encoder",
        "768→384 projection + Perceiver\n5 × 16 = 80 tokens", edge=BLUE, face=BLUE_BG)
    box(ax, 830, 725, 300, 100, "dec2 encoder",
        "독립 weight · projection + Perceiver\n5 × 16 = 80 tokens", edge=BLUE, face=BLUE_BG)
    arrow(ax, 220, 642, 220, 717, color=BLUE)
    arrow(ax, 600, 642, 600, 717, color=BLUE)
    arrow(ax, 730, 642, 980, 717, color=BLUE)
    box(ax, 250, 860, 700, 98, "TokenSequenceFusionCondition",
        "concat 60+80+80 = C [B, 220, 384]\nmodality embedding + sinusoidal position → Transformer 4 layers / 6 heads",
        edge=BLUE, face=BLUE_BG)
    arrow(ax, 220, 825, 430, 852, color=BLUE)
    arrow(ax, 600, 825, 600, 852, color=BLUE)
    arrow(ax, 980, 825, 770, 852, color=BLUE)

    phase(ax, 1000, 3, "DDPM score-matching forward", "ContinuousDiffusionSDE", GREEN)
    box(ax, 70, 1040, 300, 100, "Random t, ε",
        "t ~ Uniform(t_min, t_max)\nε ~ N(0, I)", edge=GREEN, face=GREEN_BG)
    box(ax, 450, 1040, 300, 100, "Forward noising",
        "xₜ = α(t)x₀ + σ(t)ε\nxₜ [B, 60, 2]", edge=GREEN, face=GREEN_BG)
    arrow(ax, 370, 1090, 442, 1090, color=GREEN)
    arrow(ax, 980, 642, 710, 1032, color=GREEN, text="x₀")
    box(ax, 830, 1040, 300, 100, "DiTCrossAttn1d",
        "12 blocks · 6 heads · non-causal\naction self-attn → C cross-attn → MLP", edge=BLUE, face=BLUE_BG)
    arrow(ax, 750, 1090, 822, 1090, color=GREEN, text="xₜ, t")
    arrow(ax, 600, 958, 980, 1032, color=BLUE, text="C → K,V")

    phase(ax, 1185, 4, "Loss와 파라미터 갱신", "aux loss 없음", RED)
    box(ax, 70, 1225, 310, 94, "Noise prediction",
        "ε̂θ = DiT(xₜ, t, C)\n[B, 60, 2]", edge=RED, face="#FBEDED")
    arrow(ax, 980, 1140, 380, 1260, color=RED)
    box(ax, 445, 1225, 310, 94, "Denoising loss",
        "L = mean[(ε̂θ − ε)²]\n전체 horizon/action 평균", edge=RED, face="#FBEDED")
    arrow(ax, 380, 1272, 437, 1272, color=RED)
    arrow(ax, 220, 1140, 500, 1217, color=GREEN, text="ε target")
    box(ax, 820, 1225, 310, 94, "Backward + AdamW",
        "zero_grad → backward → clip_grad_norm\noptimizer.step · lr_scheduler.step", edge=RED, face="#FBEDED")
    arrow(ax, 755, 1272, 812, 1272, color=RED)
    box(ax, 445, 1355, 310, 80, "EMA update",
        "θ_EMA ← 0.999 θ_EMA + 0.001 θ", edge=GREEN, face=GREEN_BG)
    arrow(ax, 975, 1319, 730, 1347, color=GREEN)

    phase(ax, 1478, 5, "반복 및 체크포인트", "20 epochs · 16,940 steps", BLUE)
    box(ax, 120, 1518, 300, 92, "다음 mini-batch",
        "drop_last=true\n동일한 01→04 반복", edge=BLUE, face=GRAY_BG)
    box(ax, 450, 1518, 300, 92, "Checkpoint",
        "매 1,000 step + 최종\nmodel · EMA · optimizer · config", edge=BLUE, face=GRAY_BG)
    box(ax, 780, 1518, 300, 92, "학습 결과",
        "action trajectory generator\n[60,2] ≈ 2.0 s @ 30 Hz", edge=BLUE, face=GRAY_BG)

    ax.text(50, 1665,
            "제외됨: DINO · image_top/usb0 · LiDAR · ICP auxiliary pose · 4-D geometry token · PoseHead",
            fontproperties=MONO, fontsize=8.5, color=MUTED, ha="left", va="center")
    ax.text(1150, 1695, "source: repository code/config · generated deterministically with matplotlib",
            fontproperties=MONO, fontsize=7.5, color=MUTED, ha="right", va="center")

    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(draw())
