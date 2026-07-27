"""Coordinate-convention unit tests for the Reloc3r body-frame geometry token
(docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md, "반드시 추가할 좌표계 단위 테스트").

UPDATED 2026-07-25 (ICP-free): the original version of this file validated
signs against ICP dock_pose ("goal in front/left/right"). Two problems with
that: (1) user decision -- ICP/dock_pose must not be load-bearing anywhere in
this pipeline, including validation, since the dock template is "somewhat
accurate but not complete" and must not silently become a de-facto ground
truth; (2) direct visual inspection (docs/0725_reloc3r_test/reloc3r/
odometry_calibration_check.png) found the OLD ICP-fit R_cb
(body_frame_calibration.json) actually had the TRANSLATION direction
backwards (~178deg off) against real camera+odometry motion, even though it
had passed the old version of these exact tests -- i.e. the ICP-based ground
truth was not a reliable arbiter of sign correctness here.

This version validates against WHEEL ODOMETRY instead (reconstruct_pose_rk4
over short intervals) -- the same ICP-free ground truth
scripts/calibrate_reloc3r_odometry.py fits against, but on the HELD-OUT test
split (disjoint episodes from the train-split fit) for an honest check:
  * "goal in front"        -> forward-driving interval  => dx > 0
  * "goal on left/right"   -> turning-while-moving arcs  => dy sign matches
                               turn direction (turn left -> curves left, +y)
  * "heading matches"      -> near-zero-rotation interval => yaw ~= 0
  * extrinsic validity, and current<->goal inverse-convention checks: unchanged
    (never depended on ICP).

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

from scripts.calibrate_reloc3r_odometry import body_motion  # noqa: E402

TEST_H5 = os.path.join(REPO, "dataset/after_0328_test.h5")
TEST_CACHE = os.path.join(REPO, "dataset/after_0328_test_reloc3r_bottom.h5")
CALIB_PATH = os.path.join(REPO, "reloc3r/body_frame_calibration_odometry.json")
K = 45  # ~1.5s window, matches the large/visually-clear interval used in the
        # odometry_calibration_check.png visual validation


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@pytest.fixture(scope="module")
def calib():
    with open(CALIB_PATH) as f:
        c = json.load(f)
    assert c.get("uses_icp_or_dock_pose") is False, (
        "this calibration file is not declared ICP-free; refusing to validate against it")
    return np.array(c["R_cb"])


@pytest.fixture(scope="module")
def odometry_frames():
    """(t_cam[N,3], R_cam[N,3,3], gt_dx, gt_dy, gt_dth) for K-step intervals
    across every episode of the HELD-OUT test split, using wheel odometry
    (reconstruct_pose_rk4) as ground truth -- ICP-free, and independent of the
    train-split episodes the calibration itself was fit on."""
    f = h5py.File(TEST_H5, "r")
    ep_ends = f["episode_ends"][:]
    enc = f["encoder"][:].astype(np.float64)
    f.close()
    cf = h5py.File(TEST_CACHE, "r")
    rot = cf["reloc3r_rot_bottom"][:]
    tdir = cf["reloc3r_dir_bottom"][:]
    cf.close()

    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    rows = []
    for ep_id, e in enumerate(ep_ends):
        s, e = int(ep_starts[ep_id]), int(e)
        if e - s < K + 5:
            continue
        for t in range(s, e - K, 3):
            tb, dth, disp = body_motion(enc, t, t + K)
            rows.append((t, t + K, tb[0], tb[1], dth, disp))

    a_idx = np.array([r[0] for r in rows])
    b_idx = np.array([r[1] for r in rows])
    gt_dx = np.array([r[2] for r in rows])
    gt_dy = np.array([r[3] for r in rows])
    gt_dth = np.array([r[4] for r in rows])
    gt_disp = np.array([r[5] for r in rows])

    # Reloc3r pose(a->b) is precomputed only vs each frame's EPISODE GOAL, not
    # vs an arbitrary later frame b -- so reuse the cached per-frame (rotation,
    # direction)-to-goal is NOT valid here. This fixture instead needs a fresh
    # Reloc3r(a,b) call per pair, same as scripts/calibrate_reloc3r_odometry.py.
    import torch
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model
    from scripts.calibrate_reloc3r_odometry import reloc3r_pairs

    with h5py.File(TEST_H5, "r") as f:
        feats_needed = sorted(set(a_idx.tolist()) | set(b_idx.tolist()))
        # reloc3r_pairs expects a features array indexable by absolute row --
        # load only the needed rows via the ViT-L encoder directly (no cache
        # exists for arbitrary a/b pairs on the test split at this K).
        imgs = f["image_bottom"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = setup_reloc3r_relpose_model(model_args="224", device=device)
        model.eval()
        from scripts.precompute_reloc3r_cache import array_to_view
        feat_map = {}
        with torch.no_grad():
            for i0 in range(0, len(feats_needed), 48):
                chunk = feats_needed[i0:i0 + 48]
                views, shapes = [], []
                for idx in chunk:
                    v, sh = array_to_view(imgs[idx])
                    views.append(v); shapes.append(sh)
                img_b = torch.stack(views).to(device)
                shape_b = torch.from_numpy(np.stack(shapes)).to(device)
                feat, _pos, _ = model._encode_image(img_b, shape_b)
                for k, idx in enumerate(chunk):
                    feat_map[idx] = feat[k].cpu().numpy()

    n = len(a_idx)
    feat_dim, n_patch = 1024, 196
    full_feats = np.zeros((max(feat_map.keys()) + 1, n_patch, feat_dim), dtype=np.float16)
    for idx, v in feat_map.items():
        full_feats[idx] = v.astype(np.float16)
    R_cam, t_cam = reloc3r_pairs(model, full_feats, a_idx, b_idx, device)

    return t_cam, R_cam, gt_dx, gt_dy, gt_dth, gt_disp


def _predict(calib, t_cam, R_cam):
    t_body = (calib @ t_cam.T).T
    t_body_xy = t_body[:, :2] / np.linalg.norm(t_body[:, :2], axis=1, keepdims=True).clip(1e-8)
    R_body = np.einsum("ij,njk,kl->nil", calib, R_cam, calib.T)
    yaw = np.arctan2(R_body[:, 1, 0], R_body[:, 0, 0])
    return t_body_xy[:, 0], t_body_xy[:, 1], yaw


def test_extrinsic_is_a_valid_rotation(calib):
    """camera-to-body extrinsic 방향 확인: R_cb must be a proper rotation
    (orthogonal, det=+1), not an arbitrary linear map or a reflection."""
    assert np.allclose(calib @ calib.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(calib), 1.0, atol=1e-6)


def test_forward_driving_gives_positive_dx(calib, odometry_frames):
    """goal이 로봇 정면에 있을 때 dx 부호 확인 (ICP-free reframing: clean forward-
    driving intervals, where the robot's own odometry says it moved
    essentially straight ahead, must predict dx > 0)."""
    t_cam, R_cam, gt_dx, gt_dy, gt_dth, disp = odometry_frames
    sel = (np.abs(gt_dth) < np.radians(3)) & (disp > 0.15) & (gt_dx > 0)
    assert sel.sum() >= 10, f"too few clean forward-driving frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(calib, t_cam[sel], R_cam[sel])
    frac_positive = float((dx_pred > 0).mean())
    assert frac_positive > 0.7, f"only {frac_positive:.0%} of forward-driving frames got dx>0"


def test_left_turn_gives_positive_dy(calib, odometry_frames):
    """goal이 로봇 왼쪽에 있을 때 dy 부호 확인 (ICP-free reframing: a left turn
    WHILE MOVING FORWARD curves the path toward +y (left), se2.py convention).
    Must restrict to gt_dx > 0 (forward): many docking segments turn WHILE
    REVERSING (backing up to re-align), and turning left in reverse curves the
    endpoint toward -y instead -- confirmed directly (t=3942 forward-left gave
    tb=[1.00,0.08], t=9846 backward-left gave tb=[-1.00,-0.09]); mixing the two
    without this filter makes the sign assertion physically ill-posed, not a
    calibration bug (a Gram-Schmidt re-check confirmed R_cb's y-row is exactly
    z_row x_row, i.e. already a valid right-handed rotation)."""
    t_cam, R_cam, gt_dx, gt_dy, gt_dth, disp = odometry_frames
    sel = (gt_dth > np.radians(10)) & (disp > 0.1) & (gt_dx > 0)
    assert sel.sum() >= 10, f"too few forward-left-turn frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(calib, t_cam[sel], R_cam[sel])
    frac_positive = float((dy_pred > 0).mean())
    assert frac_positive > 0.6, f"only {frac_positive:.0%} of forward-left-turn frames got dy>0"


def test_right_turn_gives_negative_dy(calib, odometry_frames):
    """goal이 로봇 오른쪽에 있을 때 dy 반대 부호 확인 (ICP-free reframing: right turn
    WHILE MOVING FORWARD curves the path toward -y). See test_left_turn_gives_
    positive_dy docstring for why the gt_dx > 0 (forward-only) filter matters."""
    t_cam, R_cam, gt_dx, gt_dy, gt_dth, disp = odometry_frames
    # right turns are rare in this docking dataset (the approach is left-turn-
    # biased -- confirmed earlier: <10 total right-turn segments across all of
    # train+test combined), so this test necessarily runs on a small N; still
    # meaningful as a sign check, just with less statistical power than its
    # left-turn counterpart.
    sel = (gt_dth < -np.radians(5)) & (disp > 0.05) & (gt_dx > 0)
    assert sel.sum() >= 3, f"too few forward-right-turn frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(calib, t_cam[sel], R_cam[sel])
    frac_negative = float((dy_pred < 0).mean())
    assert frac_negative > 0.6, f"only {frac_negative:.0%} of forward-right-turn frames got dy<0"


def test_matching_heading_gives_near_zero_yaw(calib, odometry_frames):
    """current와 goal heading이 동일할 때 yaw ~ 0 (odometry dtheta near 0)."""
    t_cam, R_cam, gt_dx, gt_dy, gt_dth, disp = odometry_frames
    sel = np.abs(gt_dth) < np.radians(2)
    assert sel.sum() >= 20, f"too few 'matching heading' frames ({sel.sum()}) to test"
    dx_pred, dy_pred, yaw_pred = _predict(calib, t_cam[sel], R_cam[sel])
    median_abs_yaw_deg = float(np.median(np.degrees(np.abs(yaw_pred))))
    assert median_abs_yaw_deg < 8.0, f"median |yaw| = {median_abs_yaw_deg:.2f} deg, expected ~0"


def test_current_to_goal_is_inverse_of_goal_to_current():
    """current-to-goal과 goal-to-current convention 확인: running Reloc3r with
    view1/view2 swapped must give the algebraic inverse pose (live inference,
    a few real frame pairs -- checks the raw model convention, independent of
    body-frame calibration and of ICP)."""
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

    R_err_deg = np.degrees(np.arccos(np.clip((np.trace(R2 @ R1) - 1) / 2, -1, 1)))
    t_pred = -(R1.T @ t1n)
    t_pred /= np.linalg.norm(t_pred)
    dir_err_deg = np.degrees(np.arccos(np.clip(np.dot(t_pred, t2n), -1, 1)))

    assert R_err_deg < 15.0, f"R2 vs R1^T geodesic error {R_err_deg:.2f} deg (expected small)"
    assert dir_err_deg < 30.0, f"inverse-direction error {dir_err_deg:.2f} deg (expected small)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
