"""Fit a fixed camera-to-body SO(3) rotation R_cb for converting Reloc3r's
camera-frame relative-pose estimates into the robot BODY frame used by
dock_pose/lidar_points (docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md).

No camera intrinsic/extrinsic calibration exists anywhere in this repo
(confirmed by exhaustive search, see docs/reloc3r.md methodology note), so
R_cb is fit empirically: pair Reloc3r's per-frame camera-frame unit
translation direction t_cam against the ICP-derived body-frame bearing
direction t_body = [cos(bearing), sin(bearing), 0] (bearing = atan2(dy, dx)
of the current->goal relative pose from dock_pose composition), then solve
the orthogonal Procrustes problem (Kabsch algorithm, SVD) for the single best
rotation R_cb minimizing sum_i || R_cb @ t_cam_i - t_body_i ||^2.

BODY FRAME DEFINITION (explicit, load-bearing assumption, UPDATED after
inspecting held-out data): "body frame" = dock_pose's native frame ROTATED by
a fixed empirically-fit yaw so that "goal in front" actually means bearing
~= 0. Native dock_pose bearing has a circular median of ~-91 deg and is FLAT
across all distance bins (checked 0-0.2m through 0.8-1.1m, all within
-88..-100 deg) -- i.e. a constant frame-rotation offset, not a trajectory-
shape artifact, and it closely matches the previously-noted-but-never-wired
-87.9 deg "sensor->robot bearing" constant in test/mpc_rank.py /
scripts/build_dock_template.py. SENSOR_TO_BODY_YAW below is fit directly from
this dataset (robust circular median of ALL reliable held-out bearings,
negated) rather than hardcoding that external figure, but the two agree to
within ~3 deg -- independent confirmation this is a real, constant, missing
calibration rather than noise. Everything downstream (dock_pose GT used for
R_cb fitting/testing, and the Reloc3r geometry token itself) is expressed in
this corrected frame.

Same R_cb is reused for rotation: R_body_rel = R_cb @ R_cam_rel @ R_cb.T,
relative yaw = atan2(R_body_rel[1,0], R_body_rel[0,0]).

Usage:
  python scripts/calibrate_reloc3r_body_frame.py
Writes: reloc3r/body_frame_calibration.json (R_cb 3x3 + fit/validation stats)
"""
import json
import os

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_H5 = os.path.join(REPO, "dataset/after_0328_train.h5")
TRAIN_CACHE = os.path.join(REPO, "dataset/after_0328_train_reloc3r_bottom.h5")
OUT = os.path.join(REPO, "reloc3r/body_frame_calibration.json")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative_to_goal(pose, g):
    """Exact port of utils/docking_dataset.py:_relative_to_goal / endgame/se2.py
    relative_pose(reference=g, target=pose). pose:[N,3], g:[3] -> [N,3]."""
    pose = np.atleast_2d(pose)
    cg, sg = np.cos(g[2]), np.sin(g[2])
    tix = -(cg * g[0] + sg * g[1])
    tiy = -(-sg * g[0] + cg * g[1])
    ct, st = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    x = pose[:, 0] + ct * tix - st * tiy
    y = pose[:, 1] + st * tix + ct * tiy
    th = wrap(pose[:, 2] - g[2])
    return np.stack([x, y, th], axis=1)


def circular_median(angles):
    return np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles)))


def rotate_xy(dx, dy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * dx - s * dy, s * dx + c * dy


def kabsch(A, B):
    """Best rotation R (3x3) minimizing sum||R@A_i - B_i||^2, A,B: [N,3]."""
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R


def build_pairs(h5path, cache_path, max_per_episode=150, seed=0):
    rng = np.random.RandomState(seed)
    f = h5py.File(h5path, "r")
    ep_ends = f["episode_ends"][:]
    reliable = f["reliable"][:]
    dock_pose = f["dock_pose"][:]
    f.close()
    cf = h5py.File(cache_path, "r")
    rot = cf["reloc3r_rot_bottom"][:]   # [N,7] = col0(3),col1(3),geo(1)
    tdir = cf["reloc3r_dir_bottom"][:]  # [N,3] unit vector
    cf.close()

    rows, ep_starts = [], np.concatenate([[0], ep_ends[:-1]])
    for ep_id, ep_end in enumerate(ep_ends):
        ep_start = int(ep_starts[ep_id])
        goal_idx = ep_end - 1
        g = dock_pose[goal_idx]
        rel_idx = np.nonzero(reliable[ep_start:ep_end])[0] + ep_start
        rel_idx = rel_idx[rel_idx != goal_idx]
        if len(rel_idx) == 0:
            continue
        if len(rel_idx) > max_per_episode:
            rel_idx = rng.choice(rel_idx, size=max_per_episode, replace=False)
        poses = dock_pose[rel_idx]
        gt = relative_to_goal(poses, g)  # [n,3] = dx,dy,dtheta body-frame
        for k, idx in enumerate(rel_idx):
            rows.append((int(idx), int(ep_id), gt[k, 0], gt[k, 1], gt[k, 2]))
    idxs = np.array([r[0] for r in rows])
    ep_ids = np.array([r[1] for r in rows])
    gt_dx = np.array([r[2] for r in rows])
    gt_dy = np.array([r[3] for r in rows])
    gt_dth = np.array([r[4] for r in rows])
    t_cam = tdir[idxs].astype(np.float64)
    col0, col1 = rot[idxs, 0:3].astype(np.float64), rot[idxs, 3:6].astype(np.float64)
    col2 = np.cross(col0, col1)
    R_cam = np.stack([col0, col1, col2], axis=-1)  # [n,3,3], columns = col0,col1,col2
    return t_cam, R_cam, gt_dx, gt_dy, gt_dth, ep_ids


def main():
    print("Building (Reloc3r camera-frame, ICP body-frame) pairs from cached features...")
    t_cam, R_cam, gt_dx, gt_dy, gt_dth, ep_ids = build_pairs(TRAIN_H5, TRAIN_CACHE)
    n = len(t_cam)
    print(f"  {n} pairs across {len(np.unique(ep_ids))} episodes")

    # episode-level train/val split (avoid leaking a frame's near-duplicate
    # neighbor across the split)
    rng = np.random.RandomState(0)
    eps = np.unique(ep_ids)
    val_eps = set(rng.choice(eps, size=max(1, len(eps) // 5), replace=False))
    is_val = np.isin(ep_ids, list(val_eps))
    tr, va = ~is_val, is_val

    # ---- fit + apply the missing sensor(dock_pose)->body(chassis) yaw -----
    # (see module docstring: constant ~-91deg offset, flat across distance,
    # independently corroborated by test/mpc_rank.py's -87.9deg constant).
    # theta (heading) differences are rotation-invariant so only (dx,dy) need
    # correcting; fit on TRAIN split only, exactly like R_cb below.
    raw_bearing_tr = np.arctan2(gt_dy[tr], gt_dx[tr])
    sensor_to_body_yaw = -circular_median(raw_bearing_tr)
    print(f"Fit SENSOR_TO_BODY_YAW = {np.degrees(sensor_to_body_yaw):.2f} deg "
          f"(raw circular median bearing was {np.degrees(-sensor_to_body_yaw):.2f} deg)")
    gt_dx, gt_dy = rotate_xy(gt_dx, gt_dy, sensor_to_body_yaw)

    bearing = np.arctan2(gt_dy, gt_dx)
    t_body_gt = np.stack([np.cos(bearing), np.sin(bearing), np.zeros(n)], axis=1)

    R_cb = kabsch(t_cam[tr], t_body_gt[tr])
    print("R_cb (camera->body):\n", R_cb)
    print("det:", np.linalg.det(R_cb))

    # ---- validate direction on held-out episodes ----
    t_body_pred = (R_cb @ t_cam[va].T).T
    pred_bearing = np.arctan2(t_body_pred[:, 1], t_body_pred[:, 0])
    dir_err_deg = np.degrees(np.abs(wrap(pred_bearing - bearing[va])))
    print(f"[val] direction error: median={np.median(dir_err_deg):.2f} deg "
          f"mean={np.mean(dir_err_deg):.2f} p90={np.percentile(dir_err_deg,90):.2f}")

    # ---- validate yaw via conjugation R_body = R_cb @ R_cam @ R_cb^T ----
    R_body = np.einsum("ij,njk,kl->nil", R_cb, R_cam[va], R_cb.T)
    yaw_pred = np.arctan2(R_body[:, 1, 0], R_body[:, 0, 0])
    yaw_err_deg = np.degrees(np.abs(wrap(yaw_pred - gt_dth[va])))
    print(f"[val] yaw error: median={np.median(yaw_err_deg):.2f} deg "
          f"mean={np.mean(yaw_err_deg):.2f} p90={np.percentile(yaw_err_deg,90):.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "R_cb": R_cb.tolist(),
            "sensor_to_body_yaw_rad": float(sensor_to_body_yaw),
            "sensor_to_body_yaw_deg": float(np.degrees(sensor_to_body_yaw)),
            "fit_method": "kabsch_svd_on_reloc3r224_translation_direction_vs_ICP_dock_pose_bearing; "
                          "dock_pose (dx,dy) pre-rotated by sensor_to_body_yaw before fitting R_cb",
            "n_fit_pairs": int(tr.sum()), "n_val_pairs": int(va.sum()),
            "val_direction_error_median_deg": float(np.median(dir_err_deg)),
            "val_direction_error_mean_deg": float(np.mean(dir_err_deg)),
            "val_yaw_error_median_deg": float(np.median(yaw_err_deg)),
            "val_yaw_error_mean_deg": float(np.mean(yaw_err_deg)),
            "body_frame_convention": "dock_pose/lidar_points native frame, ROTATED by "
                                     "sensor_to_body_yaw so bearing~=0 means goal-in-front "
                                     "(x=fwd,y=left,theta=CCW per endgame/se2.py). Apply "
                                     "rotate_xy(dx,dy,sensor_to_body_yaw) to any raw dock_pose-"
                                     "frame (dx,dy) before use; dtheta needs no correction "
                                     "(rotation-invariant). See module docstring for derivation.",
            "reloc3r_checkpoint": "siyan824/reloc3r-224",
            "preprocessing_version": "v1_2026-07-25",
        }, f, indent=2)
    print("Saved calibration to", OUT)


if __name__ == "__main__":
    main()
