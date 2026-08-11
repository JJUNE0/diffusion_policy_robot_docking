#!/usr/bin/env python3
"""Code-derived training/inference comparison for the r_relfeat_only policy.

The figure deliberately separates the shared learned network from operations
that exist only in offline training or in the real-robot deployment runtime.
It writes both Korean review figures and English paper-ready PNG/PDF figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


W, H = 1800, 1260
INK = "#142433"
MUTED = "#526675"
RULE = "#B8C7D1"
TRAIN = "#0E668F"
TRAIN_BG = "#EAF5FA"
LIVE = "#2F7459"
LIVE_BG = "#EAF5EF"
FROZEN = "#A65A1D"
FROZEN_BG = "#FBF0E6"
LOSS = "#A43F46"
LOSS_BG = "#FBEDEF"
SHARED = "#5D4C91"
SHARED_BG = "#F0ECFA"
SURFACE = "#F4F7F9"

FONT = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
FONT_B = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf")
MONO = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf")


TEXT = {
    "en": {
        "title": "r_relfeat_only: training and real-robot inference",
        "subtitle": "Chronological code path; identical learned modules are aligned across both columns",
        "train": "OFFLINE TRAINING",
        "train_sub": "B=256 · 20 epochs · 16,940 gradient steps",
        "infer": "DEPLOYMENT INFERENCE",
        "infer_sub": "N=1 · EMA weights · 30 reverse steps",
        "stage0": "0  CURRENT--GOAL RELATIONAL FEATURES",
        "tr_input": "Dataset image history + episode goal",
        "tr_input_d": "goal = final frame of each episode\nRGB indices [0,12,24,36,48] from 60-frame window",
        "in_input": "Live synchronized 60-frame window",
        "in_input_d": "orbbec0 RGB + wheel (vx,wz) at 30 Hz\nconfigured docked-position goal image",
        "tr_reloc": "Frozen ReLoc3R · offline precompute",
        "tr_reloc_d": "shared ViT-L encoder\n12 bidirectional decoder blocks",
        "in_reloc": "Frozen ReLoc3R · live extraction",
        "in_reloc_d": "encode goal once at initialization\nencode 5 current frames on every replan",
        "tr_feat": "HDF5 dec1 / dec2 cache",
        "tr_feat_d": "each [B,5,196,768] · stored float16\npose head is not executed",
        "in_feat": "Live dec1 / dec2 tensors",
        "in_feat_d": "each [1,5,196,768] · fp16 round-trip\ncurrent<->goal cross-attention occurs here",
        "same_reloc": "same frozen weights · same last dec_norm output",
        "stage1": "1  SHARED CONDITION NETWORK",
        "tr_wheel": "Wheel history",
        "tr_wheel_d": "[B,60,2]\nmin--max normalized",
        "in_wheel": "Wheel history",
        "in_wheel_d": "[1,60,2]\ncheckpoint action_min/scale",
        "rel_enc": "Independent dec1 / dec2 encoders",
        "rel_enc_d": "768->384 projection + Perceiver\n5 x 16 tokens per stream",
        "motion": "Motion encoder",
        "motion_d": "MLP + temporal embedding\n60 tokens",
        "fusion": "TokenSequenceFusionCondition",
        "fusion_d": "concat 60+80+80 = C [*,220,384]\nmodality + position embeddings · Transformer 4L / 6H",
        "same_theta": "same learned parameters theta",
        "stage2": "2  DIFFUSION PROCESS",
        "clean": "Clean target action x0",
        "clean_d": "future [B,60,2]\nmin--max normalized",
        "noise": "Forward noising",
        "noise_d": "t ~ U · eps ~ N(0,I)\nxt = alpha(t)x0 + sigma(t)eps",
        "dit_tr": "DiT noise prediction",
        "dit_tr_d": "12 blocks · non-causal action self-attn\naction queries cross-attend to C",
        "prior": "Gaussian prior xT",
        "prior_d": "[1,60,2] ~ N(0,I)",
        "solver": "DPM-Solver++(2M)",
        "solver_d": "30 reverse-denoising steps\nEMA DiT predicts eps at every step",
        "traj": "Normalized trajectory x0_hat",
        "traj_d": "[1,60,2]",
        "same_dit": "same DiT architecture; inference reads EMA copy",
        "stage3": "3  OPTIMIZATION vs. ROBOT EXECUTION",
        "loss": "Noise MSE",
        "loss_d": "mean[(eps_hat - eps)^2]\nno auxiliary loss",
        "update": "Backpropagation + AdamW",
        "update_d": "zero_grad -> backward -> clip -> step\nEMA: 0.999 old + 0.001 online",
        "denorm": "Denormalize + select",
        "denorm_d": "affine inverse to physical (vx,wz)\nN=1: mean/medoid aggregation is a no-op",
        "smooth": "Continuity smoothing",
        "smooth_d": "gamma=0.2 against shifted previous plan\ntrajectory EMA path is bypassed (plan.current)",
        "send": "32-step execution buffer",
        "send_d": "integrate to timestamped CommandStep prefix\n32-step packet every 0.1 s · replace on new inference",
        "loop": "new synchronized observation -> replan -> replace buffer",
        "footer": "r_relfeat_only contract only: wheel + one orbbec0 camera + dec1/dec2; no DINO, LiDAR, PoseHead, or explicit geometry token",
    },
    "ko": {
        "title": "r_relfeat_only: 학습과 실기 inference 통합 흐름",
        "subtitle": "코드 실행 순서 기준 · 양쪽에서 동일한 학습 모듈을 같은 높이에 배치",
        "train": "오프라인 학습",
        "train_sub": "B=256 · 20 epochs · 16,940 gradient steps",
        "infer": "실제 로봇 INFERENCE",
        "infer_sub": "N=1 · EMA weights · reverse 30 steps",
        "stage0": "0  현재--GOAL 관계 특징",
        "tr_input": "Dataset image history + episode goal",
        "tr_input_d": "goal = episode 마지막 frame\n60-frame window의 RGB index [0,12,24,36,48]",
        "in_input": "실시간 동기화 60-frame window",
        "in_input_d": "orbbec0 RGB + wheel (vx,wz), 30 Hz\n도킹 완료 위치에서 촬영한 goal image",
        "tr_reloc": "Frozen ReLoc3R · 사전계산",
        "tr_reloc_d": "weight-shared ViT-L encoder\n양방향 decoder 12 blocks",
        "in_reloc": "Frozen ReLoc3R · live 계산",
        "in_reloc_d": "시작할 때 goal을 1회 encode\n매 replan마다 현재 frame 5장을 encode",
        "tr_feat": "HDF5 dec1 / dec2 cache",
        "tr_feat_d": "각 [B,5,196,768] · float16 저장\nPoseHead는 실행하지 않음",
        "in_feat": "Live dec1 / dec2 tensors",
        "in_feat_d": "각 [1,5,196,768] · fp16 round-trip\n현재<->goal cross-attention이 여기서 발생",
        "same_reloc": "동일한 frozen weight · 동일한 마지막 dec_norm 출력",
        "stage1": "1  공통 CONDITION NETWORK",
        "tr_wheel": "Wheel history",
        "tr_wheel_d": "[B,60,2]\nmin--max 정규화",
        "in_wheel": "Wheel history",
        "in_wheel_d": "[1,60,2]\ncheckpoint action_min/scale",
        "rel_enc": "독립 dec1 / dec2 encoder",
        "rel_enc_d": "768->384 projection + Perceiver\nstream마다 5 x 16 tokens",
        "motion": "Motion encoder",
        "motion_d": "MLP + temporal embedding\n60 tokens",
        "fusion": "TokenSequenceFusionCondition",
        "fusion_d": "concat 60+80+80 = C [*,220,384]\nmodality + position emb · Transformer 4L / 6H",
        "same_theta": "동일한 학습 파라미터 theta",
        "stage2": "2  DIFFUSION PROCESS",
        "clean": "정답 action x0",
        "clean_d": "future [B,60,2]\nmin--max 정규화",
        "noise": "Forward noising",
        "noise_d": "t ~ U · eps ~ N(0,I)\nxt = alpha(t)x0 + sigma(t)eps",
        "dit_tr": "DiT noise prediction",
        "dit_tr_d": "12 blocks · action non-causal self-attn\naction query가 C에 cross-attention",
        "prior": "Gaussian prior xT",
        "prior_d": "[1,60,2] ~ N(0,I)",
        "solver": "DPM-Solver++(2M)",
        "solver_d": "reverse denoising 30 steps\n매 step EMA DiT가 eps 예측",
        "traj": "정규화 trajectory x0_hat",
        "traj_d": "[1,60,2]",
        "same_dit": "같은 DiT 구조 · inference는 EMA 복사본 사용",
        "stage3": "3  최적화 vs. 실제 로봇 실행",
        "loss": "Noise MSE",
        "loss_d": "mean[(eps_hat - eps)^2]\nauxiliary loss 없음",
        "update": "Backpropagation + AdamW",
        "update_d": "zero_grad -> backward -> clip -> step\nEMA: 0.999 old + 0.001 online",
        "denorm": "역정규화 + sample 선택",
        "denorm_d": "실제 (vx,wz) 단위로 affine inverse\nN=1이면 mean/medoid 집계 결과 동일",
        "smooth": "Continuity smoothing",
        "smooth_d": "이전 plan을 1-step shift 후 gamma=0.2 blend\ntrajectory EMA는 우회하고 plan.current 실행",
        "send": "32-step 실행 buffer",
        "send_d": "timestamp가 포함된 CommandStep prefix로 적분\n0.1 s마다 32-step packet · 새 inference 시 교체",
        "loop": "새 동기화 관측 -> replan -> 실행 buffer 교체",
        "footer": "r_relfeat_only 계약: wheel + orbbec0 한 대 + dec1/dec2만 사용 · DINO/LiDAR/PoseHead/explicit geometry 없음",
    },
}


def rounded(ax, x, y, w, h, title, detail="", *, edge=RULE, face="white", lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=10",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + 26, title, ha="center", va="center",
            fontproperties=FONT_B, fontsize=10.7, color=INK, zorder=3)
    if detail:
        ax.text(x + w / 2, y + h - 26, detail, ha="center", va="center",
                fontproperties=MONO, fontsize=7.8, color=MUTED, linespacing=1.45, zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.55, color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=4))


def stage(ax, y, label):
    ax.text(55, y, label, ha="left", va="center", fontproperties=MONO,
            fontsize=8.6, color=MUTED, zorder=3)
    ax.plot([55, 1745], [y + 18, y + 18], color=RULE, lw=0.9, zorder=1)


def bridge(ax, y, label, color=SHARED):
    ax.plot([795, 1005], [y, y], color=color, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(900, y - 10, label, ha="center", va="bottom", fontproperties=MONO,
            fontsize=7.0, color=color, zorder=3)


def draw(lang, out_dir=Path("docs/figures")):
    t = TEXT[lang]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(55, 44, t["title"], ha="left", va="center", fontproperties=FONT_B,
            fontsize=21, color=INK)
    ax.text(55, 75, t["subtitle"], ha="left", va="center", fontproperties=FONT,
            fontsize=9.5, color=MUTED)

    rounded(ax, 55, 100, 740, 62, t["train"], t["train_sub"], edge=TRAIN, face=TRAIN_BG, lw=2.0)
    rounded(ax, 1005, 100, 740, 62, t["infer"], t["infer_sub"], edge=LIVE, face=LIVE_BG, lw=2.0)

    stage(ax, 193, t["stage0"])
    rounded(ax, 55, 225, 350, 88, t["tr_input"], t["tr_input_d"], edge=FROZEN, face=FROZEN_BG)
    rounded(ax, 445, 225, 350, 88, t["tr_reloc"], t["tr_reloc_d"], edge=FROZEN, face=FROZEN_BG)
    arrow(ax, 405, 269, 437, 269, color=FROZEN)
    rounded(ax, 55, 340, 740, 78, t["tr_feat"], t["tr_feat_d"], edge=FROZEN, face="white")
    arrow(ax, 620, 313, 620, 332, color=FROZEN)

    rounded(ax, 1005, 225, 350, 88, t["in_input"], t["in_input_d"], edge=LIVE, face=LIVE_BG)
    rounded(ax, 1395, 225, 350, 88, t["in_reloc"], t["in_reloc_d"], edge=FROZEN, face=FROZEN_BG)
    arrow(ax, 1355, 269, 1387, 269, color=LIVE)
    rounded(ax, 1005, 340, 740, 78, t["in_feat"], t["in_feat_d"], edge=FROZEN, face="white")
    arrow(ax, 1570, 313, 1570, 332, color=FROZEN)
    bridge(ax, 379, t["same_reloc"], FROZEN)

    stage(ax, 456, t["stage1"])
    rounded(ax, 55, 490, 225, 84, t["tr_wheel"], t["tr_wheel_d"], edge=TRAIN, face=TRAIN_BG)
    rounded(ax, 310, 490, 485, 84, t["rel_enc"], t["rel_enc_d"], edge=SHARED, face=SHARED_BG)
    arrow(ax, 410, 418, 500, 482, color=FROZEN)
    rounded(ax, 55, 600, 225, 84, t["motion"], t["motion_d"], edge=SHARED, face=SHARED_BG)
    rounded(ax, 310, 600, 485, 84, t["fusion"], t["fusion_d"], edge=SHARED, face=SHARED_BG)
    arrow(ax, 168, 574, 168, 592, color=SHARED)
    arrow(ax, 552, 574, 552, 592, color=SHARED)

    rounded(ax, 1005, 490, 225, 84, t["in_wheel"], t["in_wheel_d"], edge=LIVE, face=LIVE_BG)
    rounded(ax, 1260, 490, 485, 84, t["rel_enc"], t["rel_enc_d"], edge=SHARED, face=SHARED_BG)
    arrow(ax, 1390, 418, 1350, 482, color=FROZEN)
    rounded(ax, 1005, 600, 225, 84, t["motion"], t["motion_d"], edge=SHARED, face=SHARED_BG)
    rounded(ax, 1260, 600, 485, 84, t["fusion"], t["fusion_d"], edge=SHARED, face=SHARED_BG)
    arrow(ax, 1118, 574, 1118, 592, color=SHARED)
    arrow(ax, 1502, 574, 1502, 592, color=SHARED)
    bridge(ax, 642, t["same_theta"])

    stage(ax, 722, t["stage2"])
    rounded(ax, 55, 755, 210, 85, t["clean"], t["clean_d"], edge=TRAIN, face=TRAIN_BG)
    rounded(ax, 295, 755, 245, 85, t["noise"], t["noise_d"], edge=TRAIN, face=TRAIN_BG)
    rounded(ax, 570, 755, 225, 85, t["dit_tr"], t["dit_tr_d"], edge=SHARED, face=SHARED_BG)
    arrow(ax, 265, 798, 287, 798, color=TRAIN)
    arrow(ax, 540, 798, 562, 798, color=TRAIN)
    arrow(ax, 552, 684, 682, 747, color=SHARED)

    rounded(ax, 1005, 755, 210, 85, t["prior"], t["prior_d"], edge=LIVE, face=LIVE_BG)
    rounded(ax, 1245, 755, 250, 85, t["solver"], t["solver_d"], edge=LIVE, face=LIVE_BG)
    rounded(ax, 1525, 755, 220, 85, t["traj"], t["traj_d"], edge=LIVE, face=LIVE_BG)
    arrow(ax, 1215, 798, 1237, 798, color=LIVE)
    arrow(ax, 1495, 798, 1517, 798, color=LIVE)
    arrow(ax, 1502, 684, 1370, 747, color=SHARED)
    bridge(ax, 818, t["same_dit"])

    stage(ax, 878, t["stage3"])
    rounded(ax, 55, 912, 350, 88, t["loss"], t["loss_d"], edge=LOSS, face=LOSS_BG)
    rounded(ax, 445, 912, 350, 88, t["update"], t["update_d"], edge=LOSS, face=LOSS_BG)
    arrow(ax, 682, 840, 300, 904, color=LOSS)
    arrow(ax, 405, 956, 437, 956, color=LOSS)

    rounded(ax, 1005, 912, 225, 88, t["denorm"], t["denorm_d"], edge=LIVE, face=LIVE_BG)
    rounded(ax, 1260, 912, 225, 88, t["smooth"], t["smooth_d"], edge=LIVE, face=LIVE_BG)
    rounded(ax, 1515, 912, 230, 88, t["send"], t["send_d"], edge=LIVE, face=LIVE_BG)
    arrow(ax, 1635, 840, 1117, 904, color=LIVE, rad=0.08)
    arrow(ax, 1230, 956, 1252, 956, color=LIVE)
    arrow(ax, 1485, 956, 1507, 956, color=LIVE)

    # Receding-horizon feedback loop within the deployment column.
    arrow(ax, 1630, 1002, 1630, 1070, color=LIVE)
    ax.plot([1630, 1100, 1030, 1030], [1070, 1070, 1070, 313], color=LIVE,
            lw=1.55, ls=(0, (5, 4)), zorder=1)
    arrow(ax, 1030, 313, 1030, 321, color=LIVE, dashed=True)
    ax.text(1330, 1087, t["loop"], ha="center", va="center", fontproperties=MONO,
            fontsize=7.7, color=LIVE)

    # Training loop / artifact handoff.
    ax.plot([620, 620, 90, 90], [1000, 1070, 1070, 1000], color=TRAIN,
            lw=1.55, ls=(0, (5, 4)), zorder=1)
    arrow(ax, 90, 1000, 90, 992, color=TRAIN, dashed=True)
    ax.text(355, 1087, "next mini-batch · checkpoint stores model + EMA + action stats",
            ha="center", va="center", fontproperties=MONO, fontsize=7.5, color=TRAIN)

    rounded(ax, 250, 1125, 1300, 54, t["footer"], "", edge=RULE, face=SURFACE, lw=1.1)
    ax.text(1745, 1218, "derived from repository code/config · deterministic matplotlib output",
            ha="right", va="center", fontproperties=MONO, fontsize=7.0, color=MUTED)

    stem = out_dir / f"vodock_train_inference_{lang}"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12,
                facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


if __name__ == "__main__":
    for language in ("ko", "en"):
        for output in draw(language):
            print(output)
