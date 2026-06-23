"""Synthetic-scan validation for the LiDAR end-game subsystem.

No real LiDAR data lives on the ``vla`` branch yet, so this script exercises
``endgame/`` end-to-end on simulated 2D scans and doubles as a smoke test:

  1. ICP mm recovery   -- known-shape ICP recovers an SE(2) pose to mm from a
                          noisy, partially-occluded, cluttered scan.
  2. Handoff + servo   -- a scripted "wide-basin policy" drives the robot until
                          the handoff fires, then the ICP servo docks to mm.
  3. Aliasing rejection -- a symmetric target is correctly flagged ``ambiguous``
                          (CLAUDE.md §4) while the asymmetric dock is not.

Run:  python scripts/demo_endgame.py      (exits non-zero if any check fails)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from endgame import (  # noqa: E402
    EndgameConfig,
    ICPMatcher,
    TwoRegimeController,
    make_template,
)
from endgame.se2 import (  # noqa: E402
    compose,
    inverse,
    pose_distance,
    relative_pose,
    transform_points,
    wrap_angle,
)


# --------------------------------------------------------------------------- #
# Synthetic LiDAR
# --------------------------------------------------------------------------- #
def simulate_scan(target_pose_robot, template, rng,
                  noise_m=0.003, occlusion_frac=0.30, n_clutter=8, clutter_range=2.0):
    """Render a noisy/occluded/cluttered scan of ``template`` at a known pose."""
    pts = transform_points(target_pose_robot, template.points)
    # Occlusion: drop a contiguous angular wedge (as seen from the robot).
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(ang)
    drop = int(len(pts) * occlusion_frac)
    if drop > 0:
        start = rng.integers(0, len(pts))
        keep_mask = np.ones(len(pts), dtype=bool)
        keep_mask[order[np.arange(start, start + drop) % len(pts)]] = False
        pts = pts[keep_mask]
    pts = pts + rng.normal(0.0, noise_m, pts.shape)
    if n_clutter > 0:
        clutter = rng.uniform(-clutter_range, clutter_range, size=(n_clutter, 2))
        pts = np.concatenate([pts, clutter], axis=0)
    rng.shuffle(pts)
    return pts


def fmt(pose):
    return f"({pose[0]*1000:.1f}mm, {pose[1]*1000:.1f}mm, {np.degrees(pose[2]):.2f}deg)"


# --------------------------------------------------------------------------- #
# 1. ICP mm recovery
# --------------------------------------------------------------------------- #
def test_icp_recovery(rng) -> bool:
    print("\n[1] ICP mm recovery on a noisy/occluded/cluttered scan")
    template = make_template("l_notch_dock")
    matcher = ICPMatcher(template)

    gt = np.array([0.45, 0.08, np.radians(6.0)])      # true target pose in robot frame
    seed = np.array([0.40, 0.0, 0.0])                 # coarse guess (dock pose)
    scan = simulate_scan(gt, template, rng)

    res = matcher.match(scan, seed)
    t_err, r_err = pose_distance(res.pose, gt)
    print(f"    gt   = {fmt(gt)}")
    print(f"    est  = {fmt(res.pose)}   {res.note or 'ok'}")
    print(f"    pose error = {t_err*1000:.2f} mm / {np.degrees(r_err):.3f} deg   "
          f"| rms={res.mm_residual():.2f}mm inliers={res.inlier_ratio:.2f} "
          f"trustworthy={res.is_trustworthy()}")
    ok = res.is_trustworthy() and t_err < 5e-3 and r_err < np.radians(1.0)
    print(f"    => {'PASS' if ok else 'FAIL'}  (need <5mm, <1deg, trustworthy)")
    return ok


# --------------------------------------------------------------------------- #
# 2. Handoff + end-game servo (closed loop)
# --------------------------------------------------------------------------- #
def test_handoff_and_dock(rng) -> bool:
    print("\n[2] Closed-loop: wide-basin policy -> handoff -> ICP servo -> dock")
    cfg = EndgameConfig()
    template = make_template(cfg.target.template_name)
    dock_target = np.asarray(cfg.target.dock_target_pose)

    # World setup: a fixed dock; the robot starts back/offset and must approach.
    dock_world = np.array([2.0, 0.5, 0.3])
    docked_robot = compose(dock_world, inverse(dock_target))     # pose when docked
    robot = compose(docked_robot, np.array([-1.4, 0.30, np.radians(-18.0)]))  # start offset

    # Scripted "wide-basin policy": a coarse, noisy unicycle approach toward the
    # dock — a stand-in for the learned diffusion policy (which owns this regime).
    def policy_fn(obs):
        est = obs  # coarse, noisy target-pose estimate in robot frame
        g = compose(est, inverse(dock_target))
        rho = np.hypot(g[0], g[1])
        alpha = wrap_angle(np.arctan2(g[1], g[0]))
        v = float(np.clip(0.6 * rho * np.cos(alpha), -0.15, 0.15))
        w = float(np.clip(2.0 * alpha, -0.8, 0.8))
        return v, w

    ctrl = TwoRegimeController(cfg, policy_fn=policy_fn)
    dt = 0.1
    engaged_at = None
    docked = False
    for k in range(600):
        target_robot = relative_pose(robot, dock_world)           # true, for sim only
        scan = simulate_scan(target_robot, template, rng,
                             occlusion_frac=0.25, n_clutter=6)
        # The policy sees a *coarse* estimate (cm-level noise), not ground truth.
        coarse_est = target_robot + rng.normal(0, [0.03, 0.03, np.radians(3)], 3)
        info = ctrl.step(scan, policy_obs=coarse_est)

        if info.get("event") == "engaged" and engaged_at is None:
            engaged_at = k
            print(f"    t={k*dt:4.1f}s  HANDOFF fired  "
                  f"(target was {fmt(target_robot)} in robot frame)")
        # Integrate unicycle kinematics.
        v, w = info["v"], info["w"]
        robot = np.array([robot[0] + v * np.cos(robot[2]) * dt,
                          robot[1] + v * np.sin(robot[2]) * dt,
                          wrap_angle(robot[2] + w * dt)])
        if info["done"]:
            docked = True
            final = relative_pose(robot, dock_world)
            t_err, r_err = pose_distance(final, dock_target)
            print(f"    t={k*dt:4.1f}s  DOCKED  residual = "
                  f"{t_err*1000:.2f} mm / {np.degrees(r_err):.3f} deg")
            break

    early = engaged_at is not None and engaged_at > 2   # didn't fire on frame 0
    ok = docked and early and t_err < cfg.done_translation_tol_m * 1.5
    if not docked:
        print("    did not reach DONE within horizon")
    print(f"    => {'PASS' if ok else 'FAIL'}  "
          f"(handoff after approach, final residual within mm tol)")
    return ok


# --------------------------------------------------------------------------- #
# 3. Aliasing rejection (CLAUDE.md §4)
# --------------------------------------------------------------------------- #
def test_aliasing(rng) -> bool:
    print("\n[3] Aliasing: symmetric target must be flagged ambiguous")
    seed = np.array([0.40, 0.0, 0.0])

    asym = make_template("l_notch_dock")
    sym = make_template("symmetric_rect")
    gt = np.array([0.45, 0.0, 0.0])

    res_asym = ICPMatcher(asym).match(simulate_scan(gt, asym, rng, n_clutter=4), seed)
    res_sym = ICPMatcher(sym).match(simulate_scan(gt, sym, rng, n_clutter=4), seed)

    print(f"    asymmetric dock : ambiguous={res_asym.ambiguous} "
          f"trustworthy={res_asym.is_trustworthy()}  {res_asym.note or 'ok'}")
    print(f"    symmetric  shape: ambiguous={res_sym.ambiguous} "
          f"trustworthy={res_sym.is_trustworthy()}  basins={len(res_sym.restart_poses)}")
    ok = (not res_asym.ambiguous) and res_asym.is_trustworthy() \
        and res_sym.ambiguous and (not res_sym.is_trustworthy())
    print(f"    => {'PASS' if ok else 'FAIL'}  "
          f"(asym unique & trusted, sym ambiguous & rejected)")
    return ok


def main() -> int:
    rng = np.random.default_rng(0)
    results = {
        "icp_recovery": test_icp_recovery(rng),
        "handoff_dock": test_handoff_and_dock(rng),
        "aliasing": test_aliasing(rng),
    }
    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k:16s}: {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print(f"  {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
