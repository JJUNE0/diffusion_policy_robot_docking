"""Offline-ICP sub-goal auto-labeler (Candidate 2).

marker_pose is dead, so we recover each demo's robot<->dock geometry with the
canonical ICP template (Candidate 1) and use it to annotate the demo:

  * per-frame dock-relative pose / distance, recovered by tracking ICP BACKWARD
    in time from the docked frame (where ICP is reliable) — each frame seeded
    from the next, which is robust because consecutive 12.5 Hz scans barely move.
  * the ICP-coverage onset = the farthest distance at which ICP locks reliably
    = the handoff point the policy must reach (the §3 "convergence region").
  * sub-goals placed where the trajectory crosses fixed dock distances, each
    anchored to the nearest room1/room2 camera frame (the goal feature source).
  * a GT-free success label (did it reach the docked tolerance, ICP-confirmed).

Run:  python scripts/label_subgoals.py            # a few episodes + plot
      python scripts/label_subgoals.py --all      # all episodes, save + aggregate
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

from endgame import ICPConfig, ICPMatcher, make_template  # noqa: E402
from endgame.se2 import transform_points, wrap_angle  # noqa: E402
from scripts.icp_real_data import denoise, load_lidar  # noqa: E402

ROOT = "dataset/after_0328/dock"
OUT = "dataset/after_0328/icp_labels"
# Dock-surface clearances at which to drop near sub-goals. The dock floors at
# ~40 cm (LiDAR standoff) and ICP coverage starts ~60-115 cm, so these span the
# reliable approach. Farther sub-goals (>coverage) are vision-only (future work).
SUBGOAL_DISTS = [0.9, 0.7, 0.55, 0.45]
RELIABLE_INLIER = 0.5
RELIABLE_RMS = 0.015                  # m
LOST_PATIENCE = 6                    # consecutive unreliable frames -> stop tracking


def image_table(ep, room):
    fs = sorted(glob.glob(f"{ep}/image/{room}/*.jpg"))
    ts = np.array([float(os.path.basename(f).rsplit("_", 1)[1][:-4]) for f in fs]) if fs else np.array([])
    return fs, ts


def nearest_image(fs, ts, t):
    if len(ts) == 0:
        return None
    i = int(np.abs(ts - t).argmin())
    return os.path.basename(fs[i])


def track_episode(ep, matcher):
    """Track the dock pose backward from the docked frame. Returns per-frame dict."""
    tstamps, scans = load_lidar(ep)
    n = len(scans)
    # cold start at docked (last) frame: seed at the nearest-point location
    last = scans[-1]
    c = last[np.hypot(last[:, 0], last[:, 1]) < 0.8]
    if len(c) < 10:
        return None
    center = c[np.hypot(c[:, 0], c[:, 1]).argmin()]
    seed = np.array([center[0], center[1], 0.0])

    tmpl_pts = matcher.template.points
    # During tracking each frame is seeded from the previous reliable pose, so a
    # single restart suffices (no 180 flip possible from a good seed) -> ~5x
    # faster. The full restart band is only needed for the docked cold start.
    fast = ICPMatcher(matcher.template, ICPConfig(restart_yaws=(0.0,), max_iterations=25))
    pose = np.full((n, 3), np.nan)
    dist = np.full(n, np.nan)
    yaw = np.full(n, np.nan)
    reliable = np.zeros(n, bool)
    cur = seed
    lost = 0
    for i in range(n - 1, -1, -1):
        scan = scans[i]
        # crop around the predicted dock location (robust to walls/clutter)
        local = scan[np.hypot(*(scan - cur[:2]).T) < 0.6]
        if len(local) >= 8:
            res = (matcher if i == n - 1 else fast).match(denoise(local), cur)
            ok = (res.inlier_ratio >= RELIABLE_INLIER and res.rms_residual_m <= RELIABLE_RMS
                  and not res.ambiguous)
            pose[i] = res.pose
            # physical clearance: distance from robot to the NEAREST dock surface
            # point (not the template centroid, which the long wall arms bias).
            posed = transform_points(res.pose, tmpl_pts)
            dist[i] = float(np.hypot(posed[:, 0], posed[:, 1]).min())
            yaw[i] = res.pose[2]
            reliable[i] = ok
            if ok:
                cur = res.pose
                lost = 0
            else:
                lost += 1
        else:
            lost += 1
        if lost >= LOST_PATIENCE:
            break    # dock left the field of view going backward
    return dict(ts=tstamps, pose=pose, dist=dist, yaw=yaw, reliable=reliable, n=n)


def annotate(ep, track):
    """Derive sub-goals, handoff onset, success from a tracked episode."""
    rel = track["reliable"]
    dist = track["dist"]
    idx_rel = np.where(rel)[0]
    if len(idx_rel) == 0:
        return None
    docked_i = idx_rel.max()                       # last reliable frame ~ docked
    onset_i = idx_rel.min()                         # earliest reliable frame
    handoff_dist = float(dist[onset_i])             # farthest reliable lock = convergence boundary
    docked_dist = float(dist[docked_i])

    fs1, ts1 = image_table(ep, "room1")
    fs2, ts2 = image_table(ep, "room2")
    subgoals = []
    for D in SUBGOAL_DISTS:
        # earliest reliable frame at or inside distance D (approaching)
        cand = idx_rel[dist[idx_rel] <= D]
        if len(cand) == 0:
            continue
        fi = int(cand.min())
        t = float(track["ts"][fi])
        subgoals.append(dict(target_dist=D, frame=fi, ts=t,
                             dist=float(dist[fi]), yaw=float(track["yaw"][fi]),
                             room1=nearest_image(fs1, ts1, t),
                             room2=nearest_image(fs2, ts2, t)))
    # docked sub-goal
    t = float(track["ts"][docked_i])
    subgoals.append(dict(target_dist=0.0, frame=int(docked_i), ts=t, dist=docked_dist,
                         yaw=float(track["yaw"][docked_i]),
                         room1=nearest_image(fs1, ts1, t), room2=nearest_image(fs2, ts2, t)))

    success = docked_dist < 0.50 and rel[docked_i]   # ICP-confirmed reach of the dock floor
    return dict(episode=os.path.basename(ep), n_frames=track["n"],
                handoff_onset_frame=int(onset_i), handoff_onset_dist=handoff_dist,
                docked_frame=int(docked_i), docked_dist=docked_dist,
                n_reliable=int(rel.sum()), success=bool(success), subgoals=subgoals)


def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="label every episode under --root and save npz/json")
    p.add_argument("--root", default=ROOT, help="dir of episode_*_dock folders")
    p.add_argument("--out", default=None, help="output dir for npz/json labels (default: <root>/../icp_labels)")
    p.add_argument("--template", default="real_dock",
                   help="endgame.target_model.make_template name (e.g. real_dock_5f for a "
                        "site-specific dock geometry -- see docs/0725_reloc3r_test/"
                        "results_all_2026-07-26.md §5a)")
    p.add_argument("--episodes", default=None,
                   help="comma-separated episode numbers to label (default: demo trio, or all with --all)")
    return p.parse_args(argv)


def main(cli_args=None):
    args = parse_args(cli_args if cli_args is not None else sys.argv[1:])
    run_all = args.all
    root = args.root
    out_dir = args.out or os.path.join(os.path.dirname(root.rstrip("/")), "icp_labels")

    matcher = ICPMatcher(make_template(args.template), ICPConfig.for_real_dock())
    eps = sorted(glob.glob(f"{root}/episode_*_dock"))
    if args.episodes:
        eps = [f"{root}/episode_{n}_dock" for n in args.episodes.split(",")]
    elif not run_all:
        eps = [f"{root}/episode_{n}_dock" for n in ("161", "1", "50")]
        eps = [e for e in eps if os.path.isdir(e)] or eps[:3]
    os.makedirs(out_dir, exist_ok=True)

    anns, onsets = [], []
    for ep in eps:
        track = track_episode(ep, matcher)
        if track is None:
            continue
        ann = annotate(ep, track)
        if ann is None:
            print(f"{os.path.basename(ep):20s}: no reliable ICP coverage"); continue
        anns.append(ann)
        onsets.append(ann["handoff_onset_dist"])
        if run_all:
            np.savez(f"{out_dir}/{ann['episode']}.npz",
                     pose=track["pose"], dist=track["dist"], reliable=track["reliable"], ts=track["ts"])
            json.dump(ann, open(f"{out_dir}/{ann['episode']}.json", "w"), indent=2)
        else:
            sg = ", ".join(f"{s['target_dist']}m@f{s['frame']}" for s in ann["subgoals"])
            print(f"{ann['episode']:20s}: handoff onset {ann['handoff_onset_dist']*100:4.0f}cm, "
                  f"docked {ann['docked_dist']*100:4.0f}cm, reliable {ann['n_reliable']:3d}/{ann['n_frames']:3d}, "
                  f"success={ann['success']}\n{'':22s}subgoals: {sg}")

    if anns:
        on = np.array(onsets) * 100
        print(f"\n{len(anns)} episodes | handoff-onset distance (= convergence-region boundary): "
              f"median {np.median(on):.0f} cm, range {on.min():.0f}-{on.max():.0f} cm | "
              f"success {sum(a['success'] for a in anns)}/{len(anns)}")
    if run_all:
        json.dump([a for a in anns], open(f"{out_dir}/_index.json", "w"), indent=2)
        print(f"saved annotations to {out_dir}/")

    # plot the first labeled episode's distance trajectory + sub-goals
    if not run_all and anns:
        ep = eps[0]; track = track_episode(ep, matcher); ann = annotate(ep, track)
        t0 = track["ts"][0]; tt = track["ts"] - t0
        fig, ax = plt.subplots(figsize=(10, 5))
        rel = track["reliable"]
        ax.plot(tt[rel], track["dist"][rel] * 100, ".", color="green", label="ICP reliable")
        ax.plot(tt[~rel & ~np.isnan(track["dist"])], track["dist"][~rel & ~np.isnan(track["dist"])] * 100,
                ".", color="0.7", label="ICP unreliable")
        for s in ann["subgoals"]:
            ax.axvline(track["ts"][s["frame"]] - t0, ls="--", alpha=.6)
            ax.annotate(f"{s['target_dist']}m", (track["ts"][s["frame"]] - t0, s["dist"] * 100))
        ax.axhline(ann["handoff_onset_dist"] * 100, color="r", ls=":", label="handoff onset")
        ax.set_xlabel("time [s]"); ax.set_ylabel("dock distance [cm]")
        ax.set_title(f"{ann['episode']}: dock distance + sub-goals (offline ICP)")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("outputs/icp_probe/subgoal_trajectory.png", dpi=100)
        print("saved outputs/icp_probe/subgoal_trajectory.png")


if __name__ == "__main__":
    main()
