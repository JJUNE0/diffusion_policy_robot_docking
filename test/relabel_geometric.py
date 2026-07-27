"""Operator-independent docking labels, derived from LiDAR geometry.

WHY THIS EXISTS
---------------
Success/failure in this dataset was labelled by whoever ran the session, and the
convention drifted between people and between months. The consequences are
measurable (2026-07-25 audit):

  * `after_0328` terminal poses span 43 mm in depth and 100 mm laterally, with a
    -7.2 mm / 100-episode drift across the collection period.
  * Under a criterion loose enough to reproduce the 2026-07-23 operator's calls,
    only 31% of the March demos would themselves be labelled success. Under any
    criterion strict enough to fail that session's near-misses, ~0% would.

So the labels are not merely inconsistent between people — no fixed geometric
standard reproduces them. That makes "success rate" unquotable and makes reward
weighting / data filtering rest on sand.

WHAT THIS DOES
--------------
Recomputes, for every episode, the same three numbers with the same code:

    depth    mm along the dock's approach axis (lower = further in)
    lateral  mm off the dock centre  (the peg-in-hole axis; see below)
    yaw      deg of residual misalignment

ICP against the shared `real_dock` template, which reproduces the stored
`dock_pose` labels to +-0.2 mm and agrees forward-vs-backward to ~0.5 mm. The
pass/fail criterion is then ONE config constant applied uniformly, re-runnable
in seconds when the true threshold is finally measured.

WHAT IT CANNOT DECIDE FOR YOU
-----------------------------
Where the threshold belongs. That is a physical question this data cannot
answer, because nothing in the recording pipeline observes the actual task:
there is no charging-state, current or voltage channel (sensors logged are
lidar / encoder / imu / command / latency / marker_pose). Two numbers are
genuinely unknown — the shallowest depth that still charges, and the deepest
that is safe. Until they are measured, use --sweep and quote the CURVE.

CAVEAT ON THE LATERAL REFERENCE: `--lateral-ref` decides what "centred" means.
`median` (default) uses the dataset's own median terminal x, which is only the
true peg centre if the demos were unbiased. The 2026-07-23 successes sit ~14 mm
off that median, so it probably is biased. Pin this properly with the sweep
experiment rather than trusting the default.

Usage
-----
  python test/relabel_geometric.py --h5 dataset/after_0328_train.h5 --sweep
  python test/relabel_geometric.py --records dataset/demo_0725 --depth 490 --lateral 25
  python test/relabel_geometric.py --h5 ... --out test/out/labels_train.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from endgame import ICPConfig, ICPMatcher, make_template  # noqa: E402
from endgame.se2 import inverse  # noqa: E402
from scripts.icp_real_data import denoise  # noqa: E402

RELIABLE_INLIER, RELIABLE_RMS = 0.55, 0.02
TERMINAL_FRAMES = 25            # median over the last N reliable frames
BACK_WINDOW = 120               # frames walked back from the docked frame
SWEEP = [(530, 60), (510, 40), (490, 25), (475, 20), (465, 15), (455, 12), (450, 10)]


def _matchers():
    cold = ICPMatcher(make_template("real_dock"), ICPConfig.for_real_dock())
    return cold, ICPMatcher(cold.template, ICPConfig(restart_yaws=(0.0,), max_iterations=25))


def track(scans, cold, fast):
    """Forward ICP track -> [N,3] poses and a reliability mask."""
    n = len(scans)
    pose, ok = np.full((n, 3), np.nan), np.zeros(n, bool)
    cur = None
    for i, s in enumerate(scans):
        if len(s) < 8:
            continue
        if cur is None:
            near = s[np.hypot(s[:, 0], s[:, 1]) < 2.0]
            if len(near) < 10:
                continue
            seed = near[np.hypot(near[:, 0], near[:, 1]).argmin()]
            cur, m = np.array([seed[0], seed[1], 0.0]), cold
        else:
            m = fast
        local = s[np.hypot(*(s - cur[:2]).T) < 0.7]
        if len(local) < 8:
            cur = None
            continue
        r = m.match(denoise(local), cur)
        pose[i] = r.pose
        ok[i] = (r.inlier_ratio >= RELIABLE_INLIER and r.rms_residual_m <= RELIABLE_RMS
                 and not r.ambiguous)
        if ok[i]:
            cur = r.pose
    return pose, ok


def track_backward(scans, cold, fast, window=BACK_WINDOW):
    """Cold-start at the LAST frame and walk backward — the seeding
    scripts/label_subgoals.py uses, and all we need for a terminal label.

    Forward-tracking every frame of all 145 episodes is ~225k ICP calls and does
    not finish in a sane time; this is ~145 x 120. Verified against the forward
    tracker on the 2026-07-23 human docks: the two agree to ~0.5 mm. Returns
    (pose, ok) over the window, oldest-first, or None if the last frame is not
    at the dock (then the caller falls back to forward tracking).
    """
    last = scans[-1]
    c = last[np.hypot(last[:, 0], last[:, 1]) < 0.8]
    if len(c) < 10:
        return None
    center = c[np.hypot(c[:, 0], c[:, 1]).argmin()]
    cur = np.array([center[0], center[1], 0.0])
    poses, oks = [], []
    for i in range(len(scans) - 1, max(-1, len(scans) - 1 - window), -1):
        s = scans[i]
        local = s[np.hypot(*(s - cur[:2]).T) < 0.6]
        if len(local) < 8:
            break
        r = (cold if i == len(scans) - 1 else fast).match(denoise(local), cur)
        good = (r.inlier_ratio >= RELIABLE_INLIER and r.rms_residual_m <= RELIABLE_RMS
                and not r.ambiguous)
        poses.append(r.pose); oks.append(good)
        if not good:
            break
        cur = r.pose
    if sum(oks) < 20:
        return None
    return np.array(poses[::-1]), np.array(oks[::-1], bool)


def track_terminal(scans, cold, fast):
    """Backward first (fast); fall back to a full forward track."""
    got = track_backward(scans, cold, fast)
    return got if got is not None else track(scans, cold, fast)


# Plausibility guard. A tracked dock has to look like a dock: the robot ends up
# roughly squared up and roughly half a metre out. ICP that latched onto a wall
# instead reports |yaw| of 60-110 deg and metre-scale depths (seen on 6 of the
# 2026-07-23 runs). Emitting those as labels would be worse than emitting none,
# since a wrong number is indistinguishable from a right one downstream.
PLAUSIBLE_DEPTH_MM = (300.0, 900.0)
PLAUSIBLE_YAW_DEG = 20.0


def terminal(pose, ok):
    """(depth_mm, lateral_mm, yaw_deg, deepest_mm) or None if untrackable."""
    if ok.sum() < 20:
        return None
    P = np.array([inverse(p) for p in pose[ok]])
    q = np.median(P[-TERMINAL_FRAMES:], axis=0)
    depth, lateral, yaw = q[1] * 1000, q[0] * 1000, np.degrees(q[2])
    if not (PLAUSIBLE_DEPTH_MM[0] <= depth <= PLAUSIBLE_DEPTH_MM[1]) or abs(yaw) > PLAUSIBLE_YAW_DEG:
        return None
    return float(depth), float(lateral), float(yaw), float(P[:, 1].min() * 1000)


def from_h5(path, limit=None):
    import h5py
    f = h5py.File(path, "r")
    lp, ln, ends = f["lidar_points"], f["lidar_npoints"], f["episode_ends"][:]
    cold, fast = _matchers()
    starts = np.r_[0, ends[:-1]]
    n_ep = len(ends) if limit is None else min(limit, len(ends))
    for k, (s, e) in enumerate(zip(starts[:n_ep], ends[:n_ep])):
        # Only the tail is needed for a terminal label, and one contiguous slice
        # beats per-row reads by ~18x on this h5 (see test/dist_binned_error.py).
        a = max(int(s), int(e) - BACK_WINDOW)
        blk, npts = lp[a:int(e)], ln[a:int(e)]
        scans = [blk[i][:npts[i]].astype(np.float64) for i in range(len(blk))]
        if k % 10 == 0:
            print(f"  ... episode {k}/{n_ep}", flush=True)
        yield (f"ep_{k:03d}", terminal(*track_terminal(scans, cold, fast)))


def from_records(root):
    cold, fast = _matchers()
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        p = os.path.join(d, "lidar.jsonl")
        if not os.path.exists(p):
            continue
        ts, sc = [], []
        for line in open(p):
            r = json.loads(line)
            ts.append(r["ts"])
            sc.append(np.array([[q["x"], q["y"]] for q in r["points"]], np.float64))
        scans = [sc[i] for i in np.argsort(ts)]
        yield (os.path.basename(d), terminal(*track_terminal(scans, cold, fast)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5")
    ap.add_argument("--records")
    ap.add_argument("--depth", type=float, default=490.0, help="pass if depth <= this (mm)")
    ap.add_argument("--lateral", type=float, default=25.0, help="pass if |lateral| <= this (mm)")
    ap.add_argument("--lateral-ref", default="median", choices=["median", "zero"])
    ap.add_argument("--sweep", action="store_true", help="print pass rate vs criterion instead")
    ap.add_argument("--limit", type=int, help="first N episodes only (smoke test)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not (a.h5 or a.records):
        ap.error("need --h5 or --records")

    src = from_h5(a.h5, a.limit) if a.h5 else from_records(a.records)
    rows = []
    for name, t in src:
        if t is None:
            print(f"  {name}: ICP could not track — no label emitted")
            continue
        rows.append(dict(episode=name, depth_mm=t[0], lateral_raw_mm=t[1],
                         yaw_deg=t[2], deepest_mm=t[3]))
    if not rows:
        raise SystemExit("nothing trackable")

    ref = np.median([r["lateral_raw_mm"] for r in rows]) if a.lateral_ref == "median" else 0.0
    for r in rows:
        r["lateral_mm"] = r["lateral_raw_mm"] - ref
    d = np.array([r["depth_mm"] for r in rows])
    l = np.array([r["lateral_mm"] for r in rows])
    print(f"\n{len(rows)} episodes | lateral reference = {ref:+.1f} mm ({a.lateral_ref})")
    print(f"  depth   mean {d.mean():7.1f}  std {d.std():5.1f}  range {d.min():.0f}..{d.max():.0f} mm")
    print(f"  lateral mean {l.mean():+7.1f}  std {l.std():5.1f}  range {l.min():+.0f}..{l.max():+.0f} mm")

    if a.sweep:
        print(f"\n  {'depth<=':>8}{'|lat|<=':>9}{'pass':>8}{'rate':>8}")
        print("  " + "-" * 33)
        for dt, lt in SWEEP:
            m = (d <= dt) & (np.abs(l) <= lt)
            print(f"  {dt:>8}{lt:>9}{m.sum():>8}{m.mean()*100:>7.0f}%")
        print("\n  Quote the curve, not one number, until the charging threshold is measured.")
    else:
        for r in rows:
            r["pass"] = bool(r["depth_mm"] <= a.depth and abs(r["lateral_mm"]) <= a.lateral)
        n = sum(r["pass"] for r in rows)
        print(f"\n  criterion depth<={a.depth:.0f} mm, |lateral|<={a.lateral:.0f} mm "
              f"-> {n}/{len(rows)} pass ({n/len(rows)*100:.0f}%)")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(criterion=dict(depth_mm=a.depth, lateral_mm=a.lateral,
                                      lateral_ref_mm=float(ref)), episodes=rows),
                  open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
