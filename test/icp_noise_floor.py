"""Teacher (ICP label) noise-floor measurement from the packed h5 (2026-07-10).

The raw episode folders (dataset/after_0328/dock/...) are not on this machine,
so scripts/icp_real_data.py cannot run as-is. This adapts its two GT-free
measurements to the h5's stored lidar_points (nearest-cluster crop, <=256 pts —
exactly the point set the aux head sees):

  A. Docked repeatability — trailing frames of each episode (robot parked at the
     dock): re-run ICP per frame seeded from the stored label; the pose scatter
     is the label *jitter* at the docked position.
  B. Recovery vs distance — take a reliable frame, apply a KNOWN SE(2) offset to
     its scan, re-run ICP seeded from the stale (un-offset) label, and measure
     how far the recovered pose is from the known answer. Real dock geometry +
     real noise at every approach distance -> the teacher noise floor the aux
     head regresses onto, binned exactly like test/dist_binned_error.py.

Run (from repo root):  python test/icp_noise_floor.py
Outputs: test/out/icp_noise_floor.png, test/out/icp_noise_floor.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from endgame import ICPConfig, ICPMatcher, make_template  # noqa: E402
from endgame.se2 import compose, pose_distance, transform_points  # noqa: E402
from scripts.icp_real_data import denoise  # noqa: E402

H5 = "dataset/after_0328_train.h5"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BIN_EDGES = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.65])
DOCKED_WINDOW = 15          # trailing frames per episode for repeatability
RECOVERY_PER_BIN = 40
# label acceptance gates used by scripts/label_subgoals.py, plus tighter probes
RMS_GATES = [0.025, 0.010, 0.005]
RELIABLE_INLIER = 0.5


def load_h5():
    f = h5py.File(H5, "r")
    return dict(
        pose=f["dock_pose"][:].astype(np.float64),
        rel=f["reliable"][:].astype(bool),
        ends=f["episode_ends"][:].astype(int),
        z_pts=f["lidar_points"], z_n=f["lidar_npoints"],
    )


def scan_at(d, t):
    n = int(d["z_n"][t])
    return denoise(d["z_pts"][t][:n].astype(np.float64))


def docked_repeatability(d, matcher):
    """Per-episode pose scatter over the trailing (parked) reliable frames."""
    stats = []
    s = 0
    for e in d["ends"]:
        idx = [t for t in range(max(s, e - DOCKED_WINDOW), e) if d["rel"][t]]
        s = e
        if len(idx) < 8:
            continue
        poses, rmss, inls = [], [], []
        for t in idx:
            sc = scan_at(d, t)
            if len(sc) < 8:
                continue
            res = matcher.match(sc, d["pose"][t])
            if res.inlier_ratio >= RELIABLE_INLIER:
                poses.append(res.pose)
                rmss.append(res.rms_residual_m)
                inls.append(res.inlier_ratio)
        if len(poses) < 8:
            continue
        poses = np.array(poses)
        stats.append(dict(
            std_x_mm=float(poses[:, 0].std() * 1000), std_y_mm=float(poses[:, 1].std() * 1000),
            std_th_deg=float(np.degrees(poses[:, 2].std())),
            rms_mm=float(np.mean(rmss) * 1000), inlier=float(np.mean(inls)), n=len(poses)))
    return stats


def recovery_vs_distance(d, matcher, rng):
    """Known-offset recovery error per dock-distance bin (teacher noise floor)."""
    dist = np.hypot(d["pose"][:, 0], d["pose"][:, 1])
    out = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        cand = np.where(d["rel"] & (dist >= lo) & (dist < hi))[0]
        take = rng.choice(cand, size=min(RECOVERY_PER_BIN, len(cand)), replace=False)
        errs_t, errs_r, rmss, inls = [], [], [], []
        for t in np.sort(take):
            sc = scan_at(d, t)
            if len(sc) < 8:
                continue
            p = d["pose"][t]
            gt = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03),
                           rng.uniform(np.radians(-4), np.radians(4))])
            res = matcher.match(transform_points(gt, sc), p)   # stale seed, moved scan
            te, re_ = pose_distance(res.pose, compose(gt, p))
            errs_t.append(te * 1000.0)
            errs_r.append(np.degrees(re_))
            rmss.append(res.rms_residual_m)
            inls.append(res.inlier_ratio)
        errs_t, rmss, inls = np.array(errs_t), np.array(rmss), np.array(inls)
        gates = {f"rms<={g*1000:.0f}mm": float(((inls >= RELIABLE_INLIER) & (rmss <= g)).mean())
                 for g in RMS_GATES}
        out.append(dict(
            bin=[float(lo), float(hi)], n=len(errs_t),
            median_mm=float(np.median(errs_t)), p90_mm=float(np.percentile(errs_t, 90)),
            median_deg=float(np.median(errs_r)), rms_median_mm=float(np.median(rmss) * 1000),
            accept=gates))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    d = load_h5()
    # tracking-style matcher: good seed available -> single restart (see
    # scripts/label_subgoals.py fast matcher), template/config as the labeler
    matcher = ICPMatcher(make_template("real_dock"),
                         ICPConfig(restart_yaws=(0.0,), max_iterations=25))

    print("A. docked repeatability ...", flush=True)
    rep = docked_repeatability(d, matcher)
    jx = np.array([r["std_x_mm"] for r in rep])
    jy = np.array([r["std_y_mm"] for r in rep])
    jt = np.array([r["std_th_deg"] for r in rep])
    jr = np.hypot(jx, jy)
    print(f"  {len(rep)} episodes | xy jitter median {np.median(jr):.1f} mm "
          f"(p90 {np.percentile(jr, 90):.1f}) | theta median {np.median(jt):.2f} deg | "
          f"fit rms median {np.median([r['rms_mm'] for r in rep]):.1f} mm")

    print("B. recovery vs distance ...", flush=True)
    rec = recovery_vs_distance(d, matcher, rng)
    for r in rec:
        print(f"  {r['bin'][0]:.2f}-{r['bin'][1]:.2f} m: median {r['median_mm']:5.1f} mm | "
              f"p90 {r['p90_mm']:6.1f} mm | {r['median_deg']:.2f} deg | "
              f"fit rms {r['rms_median_mm']:.1f} mm | accept {r['accept']}")

    json.dump(dict(repeatability=rep, recovery=rec, bin_edges=BIN_EDGES.tolist()),
              open(os.path.join(OUT_DIR, "icp_noise_floor.json"), "w"), indent=1)

    # ---- plot: teacher floor vs student (model) error --------------------
    centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    med = [r["median_mm"] for r in rec]
    p90 = [r["p90_mm"] for r in rec]
    axes[0].plot(centers, med, "-o", color="#31a354", label="ICP recovery median (teacher floor)")
    axes[0].plot(centers, p90, "--", color="#31a354", alpha=0.5, label="ICP recovery p90")
    axes[0].axhline(np.median(jr), color="#756bb1", ls="-.",
                    label=f"docked jitter median ({np.median(jr):.1f} mm)")
    model_json = os.path.join(OUT_DIR, "dist_binned_error.json")
    if os.path.exists(model_json):
        m = json.load(open(model_json))["results"]["flow-raw"]
        axes[0].plot(centers, m["median"], "-o", color="#e6550d", label="aux head median (flow-raw)")
    axes[0].axhline(30, color="0.5", ls=":", label="3 cm")
    axes[0].axhline(10, color="0.7", ls=":", label="1 cm")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dock pose distance [m]")
    axes[0].set_ylabel("XY error [mm] (log)")
    axes[0].set_title("teacher (ICP) noise floor vs student (aux head)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    for g, c in zip(RMS_GATES, ("#3182bd", "#9ecae1", "#c6dbef")):
        key = f"rms<={g*1000:.0f}mm"
        axes[1].plot(centers, [r["accept"][key] * 100 for r in rec], "-o", color=c, label=key)
    axes[1].set_xlabel("dock pose distance [m]")
    axes[1].set_ylabel("frames passing gate [%]")
    axes[1].set_title("label acceptance rate vs RMS gate (inlier>=0.5)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "icp_noise_floor.png")
    plt.savefig(png, dpi=110)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
