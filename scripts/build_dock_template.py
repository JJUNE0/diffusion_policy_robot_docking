"""Build the canonical real-dock ICP template from many docked episodes.

The dock is a U-shaped notch (user-confirmed). A single episode's docked scan is
sparse (~70 pts) and noisy, so we ICP-align the dock region of MANY episodes into
one frame and accumulate -> a clean, dense canonical template that ICP can lock
to at mm precision.

Key choices (see memories real-lidar-icp-mm-confirmed / dock-identity-sensor-frame):
* Template = notch + shoulders (R~0.45 m). The notch *floor alone* (R<0.3) is
  near-symmetric -> aliased -> fails. The shoulders provide the asymmetry.
* Narrow ICP restart band (no 180 deg seed): the notch is borderline 180-symmetric,
  and in deployment the policy hands off roughly aligned, so a full search is both
  unnecessary and the source of catastrophic flips.
* Points kept in the LiDAR sensor frame (that is how scans arrive); the ~90 deg
  sensor->robot yaw is recorded in metadata for the servo, not baked into ICP.

Output: endgame/assets/dock_template_real.npy + dock_template_real.json
Run:    python scripts/build_dock_template.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from endgame import ICPConfig, ICPMatcher, TargetTemplate  # noqa: E402
from endgame.se2 import compose, inverse, pose_distance, transform_points  # noqa: E402
from scripts.icp_real_data import denoise, load_lidar, voxel  # noqa: E402

ROOT = "dataset/after_0328/dock"
ASSET_DIR = "endgame/assets"
R_DOCK = 0.45            # notch + shoulders
WIN = 15                 # static scans aggregated per episode
# Narrow restart band for the borderline-symmetric notch (deployment-realistic).
ICP_REAL = ICPConfig.for_real_dock()


def dock_region(ep, R=R_DOCK, win=WIN):
    """Aggregate the dock-region points of one episode's static docked window."""
    try:
        _, scans = load_lidar(ep)
    except Exception:
        return None
    if len(scans) < win:
        return None
    window = scans[-win:]
    last = scans[-1]
    near = last[np.hypot(last[:, 0], last[:, 1]) < 0.8]
    if len(near) < 10:
        return None
    # robust center = nearest point of the densest near region
    center = near[np.hypot(near[:, 0], near[:, 1]).argmin()]
    pts = np.concatenate([s[np.hypot(*(s - center).T) < R] for s in window], axis=0)
    pts = denoise(pts)
    return pts if len(pts) > 20 else None


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)
    eps = sorted(glob.glob(f"{ROOT}/episode_*_dock"))
    print(f"found {len(eps)} episodes")

    # seed template from a known-clean episode
    seed_ep = f"{ROOT}/episode_161_dock"
    seed = dock_region(seed_ep)
    centroid_ref = seed.mean(0)
    template = seed - centroid_ref          # centered at dock centroid (template frame)

    contributed, skipped, residuals = 0, 0, []
    accum = [template.copy()]
    for ep in eps:
        if ep == seed_ep:
            continue
        crop = dock_region(ep)
        if crop is None:
            skipped += 1
            continue
        crop_c = crop - crop.mean(0)
        matcher = ICPMatcher(TargetTemplate("t", template, True), ICP_REAL)
        res = matcher.match(crop_c, np.zeros(3))
        if not res.converged or res.ambiguous or res.inlier_ratio < 0.7:
            skipped += 1
            continue
        aligned = transform_points(inverse(res.pose), crop_c)   # into template frame
        accum.append(aligned)
        residuals.append(res.rms_residual_m * 1000)
        contributed += 1

    canonical = voxel(denoise(np.concatenate(accum, 0), k=4, thr=0.04), vs=0.006)
    print(f"contributed {contributed} eps, skipped {skipped}, "
          f"align rms median {np.median(residuals):.1f} mm")
    print(f"canonical template: {len(canonical)} pts, "
          f"span {(canonical.max(0)-canonical.min(0))*100} cm")

    # dock pose in the seed robot/sensor frame when docked (for the servo)
    dock_target_pose = [float(centroid_ref[0]), float(centroid_ref[1]), 0.0]

    np.save(f"{ASSET_DIR}/dock_template_real.npy", canonical.astype(np.float32))
    meta = dict(
        n_episodes_contributed=contributed,
        n_points=len(canonical),
        R_dock_m=R_DOCK,
        frame="LiDAR sensor frame, centered at dock centroid",
        dock_target_pose_in_seed_frame=dock_target_pose,
        sensor_to_robot_yaw_note="~90 deg offset (user); set in config for servo, not ICP",
        restart_yaws_deg=[round(np.degrees(y), 1) for y in ICP_REAL.restart_yaws],
        is_asymmetric=False,  # borderline-symmetric notch -> rely on narrow restart band
        seed_episode=os.path.basename(seed_ep),
    )
    json.dump(meta, open(f"{ASSET_DIR}/dock_template_real.json", "w"), indent=2)
    print(f"saved {ASSET_DIR}/dock_template_real.npy (+ .json)")

    # ---- validate on held-out episodes ------------------------------------
    rng = np.random.default_rng(1)
    tmpl = TargetTemplate("dock_real", canonical, False)
    matcher = ICPMatcher(tmpl, ICP_REAL)
    holdout = [f"{ROOT}/episode_{n}_dock" for n in ("1", "2", "3", "50", "60")]
    print("\nheld-out validation (incremental ICP precision, baseline-corrected):")
    print("  P0 = how far this episode's docked pose sits from the template frame")
    print("       (real per-episode docking variation, NOT ICP error)")
    for ep in holdout:
        crop = dock_region(ep)
        if crop is None:
            print(f"  {os.path.basename(ep):20s} : no dock region"); continue
        crop_c = crop - crop.mean(0)
        base = matcher.match(crop_c, np.zeros(3)).pose      # cross-episode baseline P0
        rt = []
        for _ in range(60):
            gt = np.array([rng.uniform(-.04, .04), rng.uniform(-.04, .04),
                           rng.uniform(np.radians(-6), np.radians(6))])
            res = matcher.match(transform_points(gt, crop_c), np.zeros(3))
            # subtract the baseline so we measure incremental precision, not P0
            rt.append(pose_distance(res.pose, compose(gt, base))[0] * 1000)
        rt = np.array(rt)
        base_mm = pose_distance(base, np.zeros(3))[0] * 1000
        print(f"  {os.path.basename(ep):20s} : ICP precision med {np.median(rt):5.2f} mm "
              f"90th {np.percentile(rt,90):5.2f} mm  <2mm {np.mean(rt<2)*100:3.0f}%  | P0={base_mm:.1f} mm")

    # render
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(canonical[:, 1], canonical[:, 0], s=10, c="crimson")
    ax.set_aspect("equal"); ax.invert_xaxis(); ax.grid(alpha=.3)
    ax.set_title(f"canonical dock template ({len(canonical)} pts, {contributed} eps)")
    plt.tight_layout(); plt.savefig("outputs/icp_probe/canonical_template.png", dpi=100)
    print("saved outputs/icp_probe/canonical_template.png")


if __name__ == "__main__":
    main()
