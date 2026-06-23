"""Validate that raw-point ICP achieves mm precision on the REAL dock LiDAR.

marker_pose is unusable, so we cannot measure absolute accuracy vs ground truth.
Instead we measure three GT-free quantities on the real after_0328 scans:

  B. Static repeatability  -- the robot is parked (enc |v| ~1e-4) at the dock, so
     registering each consecutive scan to a template gives the ICP pose *jitter*.
  C. Recovery accuracy     -- apply a KNOWN SE(2) offset to a real dock scan and
     check ICP recovers it. Real geometry + real noise, synthetic-but-known GT.
  D. Aliasing              -- does the real dock geometry yield a unique pose
     (centroid-spin restarts), i.e. is the §4 distinctiveness assumption met?

Run:  python scripts/icp_real_data.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from endgame import ICPConfig, ICPMatcher, TargetTemplate  # noqa: E402
from endgame.se2 import pose_distance, transform_points  # noqa: E402

ROOT = "dataset/after_0328/dock"
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
def load_lidar(ep):
    ts, scans = [], []
    for line in open(f"{ep}/lidar.jsonl").read().splitlines():
        r = json.loads(line)
        ts.append(r["ts"])
        scans.append(np.array([[p["x"], p["y"]] for p in r["points"]], float))
    return np.array(ts), scans


def load_enc(ep):
    rows = list(csv.DictReader(open(f"{ep}/encoder.csv")))
    return (np.array([float(r["ts"]) for r in rows]),
            np.array([[float(r["vx"]), float(r["wz"])] for r in rows]))


def dock_crop(scan, rmax=1.0, xmin=-0.3):
    d = np.hypot(scan[:, 0], scan[:, 1])
    return scan[(d < rmax) & (scan[:, 0] > xmin)]


def denoise(pts, k=4, thr=0.06):
    if len(pts) < k + 1:
        return pts
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return pts[d[:, 1:].mean(1) < thr]


def voxel(pts, vs=0.008):
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.round(pts / vs).astype(int), axis=0, return_index=True)
    return pts[np.sort(idx)]


def build_template(scans_window):
    """Aggregate static scans into one dense, denoised, voxelized template."""
    agg = np.concatenate([denoise(dock_crop(s)) for s in scans_window], axis=0)
    return voxel(denoise(agg), vs=0.008)


# --------------------------------------------------------------------------- #
def analyze_episode(ep, n_window=15, n_recovery=80, rng=None):
    rng = rng or np.random.default_rng(0)
    ts, scans = load_lidar(ep)
    ets, ev = load_enc(ep)
    name = ep.split("/")[-1]

    window = scans[-n_window:]
    vmean = np.abs(ev[ets > ets[-1] - 1.5]).mean(0)
    tmpl_pts = build_template(window)
    template = TargetTemplate(name=f"real_{name}", points=tmpl_pts, is_asymmetric=True)
    matcher = ICPMatcher(template, ICPConfig())
    ident = np.array([0.0, 0.0, 0.0])

    # ---- B. static repeatability ------------------------------------------
    poses, rmss, inliers = [], [], []
    for s in window:
        sc = denoise(dock_crop(s))
        res = matcher.match(sc, ident)
        poses.append(res.pose)
        rmss.append(res.rms_residual_m)
        inliers.append(res.inlier_ratio)
    poses = np.array(poses)
    std_xy = poses[:, :2].std(0) * 1000.0           # mm
    std_th = np.degrees(poses[:, 2].std())          # deg

    # ---- C. recovery accuracy (known offset on a real scan) ---------------
    ref = denoise(dock_crop(window[-1]))
    rec_t, rec_r, ambiguous = [], [], 0
    for _ in range(n_recovery):
        gt = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06),
                       rng.uniform(np.radians(-8), np.radians(8))])
        moved = transform_points(gt, ref)
        res = matcher.match(moved, ident)
        if res.ambiguous:
            ambiguous += 1
        t_err, r_err = pose_distance(res.pose, gt)
        rec_t.append(t_err * 1000.0)
        rec_r.append(np.degrees(r_err))
    rec_t, rec_r = np.array(rec_t), np.array(rec_r)

    # ---- D. aliasing on the real dock template ----------------------------
    alias_res = matcher.match(ref, ident)

    print(f"\n=== {name}  (template {len(tmpl_pts)} pts, parked |v|={vmean[0]:.4f} m/s) ===")
    print(f"  B. static repeatability : x/y jitter = {std_xy[0]:.2f}/{std_xy[1]:.2f} mm, "
          f"theta jitter = {std_th:.3f} deg")
    print(f"     fit quality          : rms={np.mean(rmss)*1000:.2f} mm, inlier={np.mean(inliers):.2f}")
    print(f"  C. recovery error       : median {np.median(rec_t):.2f} mm / {np.median(rec_r):.3f} deg, "
          f"90th {np.percentile(rec_t,90):.2f} mm / {np.percentile(rec_r,90):.3f} deg")
    print(f"     within 2mm/0.5deg    : {np.mean((rec_t<2)&(rec_r<0.5))*100:.0f}% of {n_recovery} trials")
    print(f"  D. aliasing             : ambiguous={alias_res.ambiguous} "
          f"(recovery ambiguous rate {ambiguous}/{n_recovery})  -> "
          f"{'DISTINCTIVE' if not alias_res.ambiguous else 'ALIASED/SYMMETRIC'}")
    return dict(name=name, template=tmpl_pts, ref=ref, pose=alias_res.pose,
                std_xy=std_xy, std_th=std_th, rec_t=rec_t, rec_r=rec_r)


def main():
    rng = np.random.default_rng(0)
    eps = [f"{ROOT}/episode_001_dock", f"{ROOT}/episode_161_dock", f"{ROOT}/episode_50_dock"]
    eps = [e for e in eps if os.path.exists(e)]
    results = [analyze_episode(e, rng=rng) for e in eps]

    # overlay: template (gray) + a known-offset scan registered back (blue)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        T = r["template"]
        ax.scatter(T[:, 1], T[:, 0], s=8, c="0.6", label="template")
        reg = transform_points(r["pose"], T)
        ax.scatter(r["ref"][:, 1], r["ref"][:, 0], s=10, c="b", alpha=0.6, label="scan")
        ax.set_title(r["name"]); ax.set_aspect("equal"); ax.invert_xaxis()
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "outputs/icp_probe/real_icp_overlay.png"
    plt.savefig(out, dpi=90)
    print(f"\nsaved {out}")

    allt = np.concatenate([r["rec_t"] for r in results])
    print(f"\nOVERALL recovery on real dock geometry: "
          f"median {np.median(allt):.2f} mm, 90th {np.percentile(allt,90):.2f} mm "
          f"(n={len(allt)})")


if __name__ == "__main__":
    main()
