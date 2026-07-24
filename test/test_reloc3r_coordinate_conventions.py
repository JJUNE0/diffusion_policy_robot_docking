"""Coordinate-convention unit tests for the Reloc3r body-frame geometry token
(docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md, "반드시 추가할 좌표계 단위 테스트").

Runs against REAL held-out frames (dataset/after_0328_test.h5, disjoint from
the episodes used to fit reloc3r/body_frame_calibration.json), selected by
their ICP dock_pose ground truth to represent each named geometric case. Not
synthetic -- there is no simulator here, so "goal is in front" etc. means
"a real frame where ICP says the goal bearing is within a front/left/right/
behind sector", and we check Reloc3r+calibration reproduces the right SIGN.
Real sensor data is noisy, so checks use a majority-of-sample-frames+error-
bound criterion rather than requiring every single frame to be exactly right.

Run:  python -m pytest test/test_reloc3r_coordinate_conventions.py -v
  or: python test/test_reloc3r_coordinate_conventions.py
"""
import json
import os
import sys

import h5py
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "reloc3r"))

TEST_H5 = os.path.join(REPO, "dataset/after_0328_test.h5")
TEST_CACHE = os.path.join(REPO, "dataset/after_0328_test_reloc3r_bottom.h5")
CALIB_PATH = os.path.join(REPO, "reloc3r/body_frame_calibration.json")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative_to_goal(pose, g):
    pose = np.atleast_2d(pose)
    cg, sg = np.cos(g[2]), np.sin(g[2])
    tix = -(cg * g[0] + sg * g[1])
    tiy = -(-sg * g[0] + cg * g[1])
    ct, st = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    x = pose[:, 0] + ct * tix - st * tiy
    y = pose[:, 1] + st * tix + ct * tiy
    th = wrap(pose[:, 2] - g[2])
    return np.stack([x, y, th], axis=1)


def rotate_xy(dx, dy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * dx - s * dy, s * dx + c * dy


@pytest.fixture(scope="module")
def calib():
    with open(CALIB_PATH) as f:
        c = json.load(f)
    return np.array(c["R_cb"]), float(c["sensor_to_body_yaw_rad"])


@pytest.fixture(scope="module")
def frames():
    """(t_cam[N,3], R_cam[N,3,3], gt_dx, gt_dy, gt_dth) for every reliable
    held-out frame, using the SAME construction as the calibration fit
    (utils/docking_dataset.py:_relative_to_goal)."""
    f = h5py.File(TEST_H5, "r")
    ep_ends = f["episode_ends"][:]
    reliable = f["reliable"][:]
    dock_pose = f["dock_pose"][:]
    f.close()
    cf = h5py.File(TEST_CACHE, "r")
    rot = cf["reloc3r_rot_bottom"][:]
    tdir = cf["reloc3r_dir_bottom"][:]
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
        poses = dock_pose[rel_idx]
        gt = relative_to_goal(poses, g)
        for k, idx in enumerate(rel_idx):
            rows.append((int(idx), gt[k, 0], gt[k, 1], gt[k, 2]))
    idxs = np.array([r[0] for r in rows])
    gt_dx = np.array([r[1] for r in rows])
    gt_dy = np.array([r[2] for r in rows])
    gt_dth = np.array([r[3] for r in rows])
    t_cam = tdir[idxs].astype(np.float64)
    col0, col1 = rot[idxs, 0:3].astype(np.float64), rot[idxs, 3:6].astype(np.float64)
    col2 = np.cross(col0, col1)
    R_cam = np.stack([col0, col1, col2], axis=-1)
    return t_cam, R_cam, gt_dx, gt_dy, gt_dth


def _predict(calib, t_cam, R_cam):
    t_body = (calib @ t_cam.T).T
    t_body_xy = t_body[:, :2] / np.linalg.norm(t_body[:, :2], axis=1, keepdims=True).clip(1e-8)
    R_body = np.einsum("ij,njk,kl->nil", calib, R_cam, calib.T)
    yaw = np.arctan2(R_body[:, 1, 0], R_body[:, 0, 0])
    return t_body_xy[:, 0], t_body_xy[:, 1], yaw


def test_extrinsic_is_a_valid_rotation(calib):
    """camera-to-body extrinsic 방향 확인: R_cb must be a proper rotation
    (orthogonal, det=+1), not an arbitrary linear map or a reflection."""
    R_cb, _ = calib
    assert np.allclose(R_cb @ R_cb.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R_cb), 1.0, atol=1e-6)


def test_goal_in_front_gives_positive_dx(calib, frames):
    """goal이 로봇 정면에 있을 때 dx 부호 확인. Ground-truth bearing is expressed
    in the calibrated body/chassis frame (sensor_to_body_yaw-corrected -- the
    native dock_pose frame has a constant ~-91deg offset from chassis-forward,
    see scripts/calibrate_reloc3r_body_frame.py docstring)."""
    R_cb, yaw_corr = calib
    t_cam, R_cam, gt_dx, gt_dy, gt_dth = frames
    gt_dx, gt_dy = rotate_xy(gt_dx, gt_dy, yaw_corr)
    bearing = np.arctan2(gt_dy, gt_dx)
    dist = np.hypot(gt_dx, gt_dy)
    sel = (np.abs(bearing) < np.radians(20)) & (dist > 0.15)
    assert sel.sum() >= 30, f"too few 'goal in front' frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(R_cb, t_cam[sel], R_cam[sel])
    frac_positive = float((dx_pred > 0).mean())
    assert frac_positive > 0.8, f"only {frac_positive:.0%} of 'goal in front' frames got dx>0"


def test_goal_on_left_gives_positive_dy(calib, frames):
    """goal이 로봇 왼쪽에 있을 때 dy 부호 확인 (se2.py convention: +y = left)."""
    R_cb, yaw_corr = calib
    t_cam, R_cam, gt_dx, gt_dy, gt_dth = frames
    gt_dx, gt_dy = rotate_xy(gt_dx, gt_dy, yaw_corr)
    bearing = np.arctan2(gt_dy, gt_dx)
    dist = np.hypot(gt_dx, gt_dy)
    sel = (bearing > np.radians(30)) & (bearing < np.radians(150)) & (dist > 0.15)
    assert sel.sum() >= 15, f"too few 'goal on left' frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(R_cb, t_cam[sel], R_cam[sel])
    frac_positive = float((dy_pred > 0).mean())
    assert frac_positive > 0.7, f"only {frac_positive:.0%} of 'goal on left' frames got dy>0"


def test_goal_on_right_gives_negative_dy(calib, frames):
    """goal이 로봇 오른쪽에 있을 때 dy 반대 부호 확인."""
    R_cb, yaw_corr = calib
    t_cam, R_cam, gt_dx, gt_dy, gt_dth = frames
    gt_dx, gt_dy = rotate_xy(gt_dx, gt_dy, yaw_corr)
    bearing = np.arctan2(gt_dy, gt_dx)
    dist = np.hypot(gt_dx, gt_dy)
    sel = (bearing < -np.radians(30)) & (bearing > -np.radians(150)) & (dist > 0.15)
    assert sel.sum() >= 15, f"too few 'goal on right' frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(R_cb, t_cam[sel], R_cam[sel])
    frac_negative = float((dy_pred < 0).mean())
    assert frac_negative > 0.7, f"only {frac_negative:.0%} of 'goal on right' frames got dy<0"


def test_matching_heading_gives_near_zero_yaw(calib, frames):
    """current와 goal heading이 동일할 때 yaw ~ 0. (dtheta is rotation-invariant,
    no sensor_to_body_yaw correction needed here.)"""
    R_cb, _ = calib
    t_cam, R_cam, gt_dx, gt_dy, gt_dth = frames
    sel = np.abs(gt_dth) < np.radians(2)
    assert sel.sum() >= 30, f"too few 'matching heading' frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(R_cb, t_cam[sel], R_cam[sel])
    median_abs_yaw_deg = float(np.median(np.degrees(np.abs(yaw_pred))))
    assert median_abs_yaw_deg < 5.0, f"median |yaw| = {median_abs_yaw_deg:.2f} deg, expected ~0"


def test_current_to_goal_is_inverse_of_goal_to_current():
    """current-to-goal과 goal-to-current convention 확인: running Reloc3r with
    view1/view2 swapped must give the algebraic inverse pose (live inference,
    a few real frame pairs -- checks the raw model convention, independent of
    body-frame calibration)."""
    import torch
    import PIL.Image
    from reloc3r.utils.image import _resize_pil_image, ImgNorm
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
    from reloc3r.utils.device import to_numpy

    def to_view(arr_chw_uint8, size=224):
        arr_hwc = np.ascontiguousarray(arr_chw_uint8.transpose(1, 2, 0))
        img = PIL.Image.fromarray(arr_hwc, mode="RGB")
        W1, H1 = img.size
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        W, H = img.size
        half = min(W // 2, H // 2)
        img = img.crop((W // 2 - half, H // 2 - half, W // 2 + half, H // 2 + half))
        return dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_reloc3r_relpose_model(model_args="224", device=device)
    model.eval()

    f = h5py.File(TEST_H5, "r")
    imgs = f["image_bottom"]
    a, b = imgs[500], imgs[2000]  # two arbitrary real frames, far enough apart
    f.close()

    va, vb = to_view(a), to_view(b)
    for v in (va, vb):
        v["img"] = v["img"].to(device)
        v["true_shape"] = torch.from_numpy(v["true_shape"]).to(device)

    pose_b_to_a = to_numpy(inference_relpose([va, vb], model, device))[0]  # b rel. to a
    pose_a_to_b = to_numpy(inference_relpose([vb, va], model, device))[0]  # a rel. to b

    R1, t1 = pose_b_to_a[:3, :3], pose_b_to_a[:3, 3]
    R2, t2 = pose_a_to_b[:3, :3], pose_a_to_b[:3, 3]
    t1n, t2n = t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2)

    # pose_a_to_b should be the inverse of pose_b_to_a: R2 ~= R1^T, and the
    # translation DIRECTION should reverse under that inverse rotation:
    # unit(-R1^T @ t1) ~= unit(t2).
    R_err_deg = np.degrees(np.arccos(np.clip((np.trace(R2 @ R1) - 1) / 2, -1, 1)))
    t_pred = -(R1.T @ t1n)
    t_pred /= np.linalg.norm(t_pred)
    dir_err_deg = np.degrees(np.arccos(np.clip(np.dot(t_pred, t2n), -1, 1)))

    assert R_err_deg < 15.0, f"R2 vs R1^T geodesic error {R_err_deg:.2f} deg (expected small)"
    assert dir_err_deg < 30.0, f"inverse-direction error {dir_err_deg:.2f} deg (expected small)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
