"""ICP-FREE camera->body extrinsic calibration for the Reloc3r geometry token,
using ONLY wheel odometry (no dock_pose, no ICP dock template) -- user
decision 2026-07-25 (the ICP dock template is "somewhat accurate but not
complete", must not enter the pipeline; the paper contribution is precisely
"goal-relative geometry from a frozen pretrained vision model + wheel
odometry, no task-specific ICP").

Hand-eye idea (camera rigidly mounted on the robot body):
  For a short intra-episode interval [t, t+k]:
    * Reloc3r(frame_t, frame_{t+k}) gives the camera-frame relative motion
      R_cam (3x3) and unit translation direction t_cam (3-vec).
    * Wheel odometry (reconstruct_pose_rk4 over the interval's (v,w)) gives the
      BODY-frame relative motion: yaw Delta_theta and planar translation
      direction t_body = unit([dx, dy, 0])  (+x fwd, +y left, se2 convention).
  The fixed R_cb (camera->body) must satisfy, for every interval:
    * rotation-axis:  R_cb @ axis(R_cam) = sign(Delta_theta) * z_body   (turns
      pin the OUT-OF-PLANE 2 DOF -> captures the camera's downward TILT), and
    * translation:    R_cb @ t_cam = t_body                              (mostly
      forward drives pin the remaining IN-PLANE yaw -> ~90 deg correction).
  Solve R_cb by orthogonal Procrustes (Kabsch) on the stacked unit-vector
  correspondences. The user-requested ~90 deg yaw correction emerges
  automatically here because the wheel-odometry body frame IS the true chassis
  frame (+x = forward), which the raw LiDAR/dock_pose frame differs from by
  ~90 deg.

Validation is self-consistent (held-out odometry intervals) -- NO dock_pose
needed. A dock_pose cross-check is printed for information only (clearly
labeled) and never enters the fit.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_reloc3r_odometry.py
Writes: reloc3r/body_frame_calibration_odometry.json
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "reloc3r"))
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model  # noqa: E402
from utils.inference import reconstruct_pose_rk4  # noqa: E402

TRAIN_H5 = os.path.join(_REPO, "dataset/after_0328_train.h5")
CACHE = os.path.join(_REPO, "dataset/after_0328_train_reloc3r_bottom.h5")
OUT = os.path.join(_REPO, "reloc3r/body_frame_calibration_odometry.json")
DT = 0.0333
FEAT_DIM, N_PATCH = 1024, 196
_POS_CACHE = {}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def kabsch(A, B, w=None):
    """R minimizing sum w_i ||R A_i - B_i||^2. A,B: [N,3]."""
    if w is None:
        w = np.ones(len(A))
    H = (A * w[:, None]).T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R


def rot_axis_angle(R):
    """axis (unit, right-handed for +angle), angle in [0,pi]. R: [n,3,3]."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    ax = np.stack([R[:, 2, 1] - R[:, 1, 2],
                   R[:, 0, 2] - R[:, 2, 0],
                   R[:, 1, 0] - R[:, 0, 1]], axis=1)
    n = np.linalg.norm(ax, axis=1, keepdims=True)
    ax = ax / np.clip(n, 1e-9, None)
    return ax, angle


def _canonical_pos(model, b, true_shape, device):
    key = (int(true_shape[0, 0]), int(true_shape[0, 1]))
    if key not in _POS_CACHE:
        dummy = torch.zeros(1, 3, key[0], key[1], device=device)
        ts = torch.tensor([[key[0], key[1]]], device=device)
        _x, pos = model.patch_embed(dummy, true_shape=ts)
        _POS_CACHE[key] = pos.detach()
    return _POS_CACHE[key].expand(b, -1, -1).contiguous()


@torch.no_grad()
def reloc3r_pairs(model, feats, idx_a, idx_b, device, batch=48):
    """Camera-frame R_cam[n,3,3], t_cam_unit[n,3] for pairs (a=view1, b=view2);
    pose2to1 = camera-b expressed in camera-a frame (matches the precompute
    convention: view1=current, view2=goal)."""
    n = len(idx_a)
    R_out = np.zeros((n, 3, 3), np.float64)
    t_out = np.zeros((n, 3), np.float64)
    shape224 = torch.tensor([[224, 224]], dtype=torch.int64)  # [1,2] for patch_embed / _canonical_pos
    for b0 in range(0, n, batch):
        b1 = min(b0 + batch, n)
        fa = torch.from_numpy(feats[idx_a[b0:b1]].astype(np.float32)).to(device)
        fb = torch.from_numpy(feats[idx_b[b0:b1]].astype(np.float32)).to(device)
        bn = b1 - b0
        pos = _canonical_pos(model, bn, shape224, device)
        dec1, dec2 = model._decoder(fa, pos, fb, pos)
        with torch.cuda.amp.autocast(enabled=False):
            pose2 = model._downstream_head([tok.float() for tok in dec2], shape224[0])
        pose = pose2["pose"]  # [bn,4,4] = camera-b in camera-a frame
        R_out[b0:b1] = pose[:, :3, :3].cpu().numpy()
        t = pose[:, :3, 3].cpu().numpy()
        t_out[b0:b1] = t / np.clip(np.linalg.norm(t, axis=1, keepdims=True), 1e-9, None)
    return R_out, t_out


def body_motion(enc, a, b):
    """Body-frame relative motion over [a,b] via unicycle RK4. Returns
    (t_body_unit[3], delta_theta, disp_norm).

    SIGN FIX (2026-07-25, user-caught): the raw `encoder` column 0 ("vx") has
    the OPPOSITE sign from "physically moving in the direction the camera
    faces". Verified directly across 15/15 train episodes: ICP dock-distance
    monotonically DECREASES (robot visibly approaching/docking) while
    mean(vx) < 0 in every one of them (e.g. ep0: dist 0.67->0.54m, mean
    vx=-0.0073; ep10: dist 1.25->0.55m, mean vx=-0.0069). So NEGATIVE raw vx =
    TRUE forward. reconstruct_pose_rk4's unicycle model (dx/dt=v*cos(theta))
    is linear in v and theta(t) doesn't depend on v, so negating the INPUT v
    is exactly equivalent to negating the output (dx,dy) -- done here, once,
    so every caller (calibration fit AND the coordinate-convention tests)
    gets the corrected sign automatically. wz (rotation) needed no such fix
    -- checked separately (a positive-dtheta turn correctly predicted a
    positive yaw via Reloc3r+R_cb), so only translation was affected.
    """
    seg = enc[a:b]  # [k,2] = (vx, wz); vx sign corrected below
    traj = reconstruct_pose_rk4(-seg[:, 0], seg[:, 1], dt=DT)  # [(k+1),3]
    dx, dy, dth = traj[-1, 0], traj[-1, 1], wrap(traj[-1, 2])
    disp = np.hypot(dx, dy)
    tb = np.array([dx, dy, 0.0])
    tb = tb / max(disp, 1e-9)
    return tb, dth, disp


def build_correspondences(enc, feats, model, device, episode_ends, seed=0):
    rng = np.random.RandomState(seed)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    turn_a, turn_b, turn_sign, turn_w = [], [], [], []
    straight_a, straight_b, straight_body = [], [], []

    # This is a gentle-motion docking robot; large in-place turns are rare, so
    # we (a) use a longer window to accumulate more rotation, (b) keep a low
    # inclusion threshold, and (c) WEIGHT each turn sample by |Delta_theta| so
    # the (better-conditioned) larger rotations dominate the axis estimate.
    K_TURN, K_STR = 15, 15   # ~0.5s windows
    TURN_MIN = np.radians(1.5)
    for ep_id, e in enumerate(episode_ends):
        s = int(ep_starts[ep_id]); e = int(e)
        for t in range(s, e - K_STR):
            tb, dth, disp = body_motion(enc, t, t + K_TURN)
            if abs(dth) > TURN_MIN:
                turn_a.append(t); turn_b.append(t + K_TURN)
                turn_sign.append(np.sign(dth)); turn_w.append(min(abs(dth), np.radians(30)))
            if abs(dth) < np.radians(2.5) and disp > 0.03:  # near-straight, moving
                straight_a.append(t); straight_b.append(t + K_STR)
                straight_body.append(tb)

    def sub(n, cap):
        return rng.choice(n, cap, replace=False) if n > cap else np.arange(n)

    turn_a, turn_b = np.array(turn_a), np.array(turn_b)
    turn_sign, turn_w = np.array(turn_sign), np.array(turn_w)
    straight_a, straight_b = np.array(straight_a), np.array(straight_b)
    straight_body = np.array(straight_body)
    st = sub(len(turn_a), 6000); ss = sub(len(straight_a), 6000)
    turn_a, turn_b, turn_sign, turn_w = turn_a[st], turn_b[st], turn_sign[st], turn_w[st]
    straight_a, straight_b, straight_body = straight_a[ss], straight_b[ss], straight_body[ss]
    print(f"  turn pairs: {len(turn_a)} (|dtheta| median {np.degrees(np.median(turn_w)):.1f} deg), "
          f"straight pairs: {len(straight_a)}")

    R_turn, _ = reloc3r_pairs(model, feats, turn_a, turn_b, device)
    _, t_str = reloc3r_pairs(model, feats, straight_a, straight_b, device)
    ax, ang = rot_axis_angle(R_turn)
    ax_signed = ax * turn_sign[:, None]          # all should map to +z_body
    src_turn, dst_turn = ax_signed, np.tile([0, 0, 1.0], (len(ax_signed), 1))
    src_str, dst_str = t_str, straight_body
    return (src_turn, dst_turn, turn_w), (src_str, dst_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=TRAIN_H5, help="training h5 (encoder + episode_ends)")
    ap.add_argument("--cache", default=CACHE, help="reloc3r feature cache h5 (reloc3r_bottom)")
    ap.add_argument("--out", default=OUT, help="where to write the fitted calibration json")
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_reloc3r_relpose_model(model_args="224", device=device)
    model.eval()

    with h5py.File(cli.h5, "r") as f:
        enc = f["encoder"][:].astype(np.float64)
        episode_ends = f["episode_ends"][:]
    with h5py.File(cli.cache, "r") as cf:
        feats = cf["reloc3r_bottom"]  # [N,196,1024] fp16, lazy
        feats = feats[:]  # load once (~fits; float16)

    print("Building odometry<->Reloc3r correspondences...")
    (src_turn, dst_turn, turn_w), (src_str, dst_str) = build_correspondences(
        enc, feats, model, device, episode_ends)

    # held-out split (by index) for self-consistency validation
    rng = np.random.RandomState(1)
    def split(n):
        m = np.zeros(n, bool); m[rng.rand(n) < 0.2] = True; return m
    vt, vs = split(len(src_turn)), split(len(src_str))

    # weight: turns by |dtheta| (normalized to sum 1 within group), straights
    # uniform; the two groups given equal total weight so neither dominates.
    wt = turn_w[~vt] / max(turn_w[~vt].sum(), 1e-9)
    ws = np.full((~vs).sum(), 1.0 / max((~vs).sum(), 1))
    src = np.concatenate([src_turn[~vt], src_str[~vs]])
    dst = np.concatenate([dst_turn[~vt], dst_str[~vs]])
    w = np.concatenate([wt, ws])
    R_cb = kabsch(src, dst, w)
    print("R_cb (camera->body, odometry-fit):\n", R_cb, "\n det:", np.linalg.det(R_cb))

    # ---- self-consistent validation on held-out intervals ----
    ax_pred = (R_cb @ src_turn[vt].T).T
    axis_err = np.degrees(np.arccos(np.clip(ax_pred @ np.array([0, 0, 1.0]), -1, 1)))
    t_pred = (R_cb @ src_str[vs].T).T
    t_pred /= np.clip(np.linalg.norm(t_pred, axis=1, keepdims=True), 1e-9, None)
    tdir_err = np.degrees(np.arccos(np.clip((t_pred * dst_str[vs]).sum(1), -1, 1)))
    print(f"[held-out] rotation-axis->z error: median={np.median(axis_err):.2f} deg "
          f"p90={np.percentile(axis_err,90):.2f}")
    print(f"[held-out] fwd-translation dir error: median={np.median(tdir_err):.2f} deg "
          f"p90={np.percentile(tdir_err,90):.2f}")

    # ---- dock_pose cross-check (INFORMATION ONLY, not used in fit) ----
    xcheck = {}
    try:
        old = json.load(open(os.path.join(_REPO, "reloc3r/body_frame_calibration.json")))
        R_old = np.array(old["R_cb"])
        # geodesic angle between the two rotations
        Rrel = R_cb @ R_old.T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        xcheck["geodesic_deg_vs_icp_R_cb"] = float(ang)
        print(f"[x-check, info only] geodesic angle vs old ICP-fit R_cb: {ang:.2f} deg")
    except Exception as ex:
        print("x-check skipped:", ex)

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    json.dump({
        "R_cb": R_cb.tolist(),
        "fit_method": "wheel_odometry_handeye_kabsch (ICP-FREE): rotation-axis(turns)"
                      " + fwd-translation(straights) correspondences vs Reloc3r-224 camera motion",
        "uses_icp_or_dock_pose": False,
        "val_axis_to_z_error_median_deg": float(np.median(axis_err)),
        "val_fwd_translation_dir_error_median_deg": float(np.median(tdir_err)),
        "dock_pose_cross_check_info_only": xcheck,
        "body_frame_convention": "wheel-odometry chassis frame: +x forward (vx), "
                                 "+y left, theta CCW (se2.py / reconstruct_pose_rk4). "
                                 "The ~90deg vs raw-LiDAR frame is absorbed here automatically.",
        "reloc3r_checkpoint": "siyan824/reloc3r-224",
        "preprocessing_version": "v2_odometry_2026-07-25",
    }, open(cli.out, "w"), indent=2)
    print("Saved:", cli.out)


if __name__ == "__main__":
    main()
