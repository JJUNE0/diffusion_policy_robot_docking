"""Live Reloc3r geometry token for the docking control node.

Produces the same 4-vector the offline pipeline caches as `geometry_bottom`:

    g = [dx_body, dy_body, sin(yaw_body), cos(yaw_body)]

where (dx, dy) is the UNIT planar bearing from the current camera pose to the
goal pose and yaw is the relative heading, both expressed in the robot's
chassis (body) frame. Distance is deliberately absent: Reloc3r's translation is
scale-ambiguous (direction only), and metric scale is the LiDAR branch's job.

This mirrors scripts/build_reloc3r_geometry_cache.py exactly:

    t_body = R_cb @ t_cam
    R_body = R_cb @ R_cam @ R_cb.T
    yaw    = atan2(R_body[1,0], R_body[0,0])

The only numerical difference from the cache is precision: the offline path
round-trips through fp16 and a 6D rotation representation, this one keeps fp32
throughout, so live values are slightly MORE accurate, never less.

Cost (measured, H100, RoPE pytorch fallback): ~61 ms per call -- one encoder
forward on the current frame plus decoder+head on the (current, goal) pair. The
goal frame is encoded once at init since it never changes. That is ~19% of the
control node's 1067 ms inference period (inference_size=32 steps at 30 Hz), so
it fits comfortably; it would NOT fit if geometry had to run per control step.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_HERE, "reloc3r_pkg")
CALIB_DEFAULT = os.path.join(_HERE, "reloc3r_calib", "body_frame_calibration_odometry.json")


def _ensure_path():
    import sys
    if _PKG not in sys.path:
        sys.path.insert(0, _PKG)


def _array_to_view(bgr_hwc_uint8, size=224):
    """Reproduce Reloc3r load_images() for size==224 byte-for-byte: RGB, resize
    the short side, center-crop square, normalize to [-1,1]. Input is the
    node's native BGR HxWx3 frame."""
    _ensure_path()
    import PIL.Image
    from reloc3r.utils.image import _resize_pil_image, ImgNorm
    rgb = np.ascontiguousarray(bgr_hwc_uint8[:, :, ::-1])
    img = PIL.Image.fromarray(rgb, mode="RGB")
    W1, H1 = img.size
    img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    W, H = img.size
    cx, cy = W // 2, H // 2
    half = min(cx, cy)
    img = img.crop((cx - half, cy - half, cx + half, cy + half))
    return ImgNorm(img), np.int32(img.size[::-1])


class Reloc3rGeometry:
    """Encodes the goal image once, then turns each current frame into the
    body-frame geometry token."""

    def __init__(self, goal_bgr, device="cuda:0", calib_path=None, size=224):
        _ensure_path()
        from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model

        calib_path = calib_path or CALIB_DEFAULT
        if not os.path.exists(calib_path):
            raise FileNotFoundError(
                f"camera->body calibration missing: {calib_path}. Without R_cb the "
                f"geometry token would be in the CAMERA frame while the policy was "
                f"trained on the BODY frame (~90 deg apart) -- refusing to guess.")
        calib = json.load(open(calib_path))
        # Same guard as scripts/build_reloc3r_geometry_cache.py: the whole point
        # of this pipeline is that no ICP/dock-template label is used anywhere.
        if calib.get("uses_icp_or_dock_pose") is not False:
            raise ValueError(
                f"{calib_path} does not declare uses_icp_or_dock_pose=False; refusing "
                f"to build geometry from a possibly ICP-derived calibration.")
        self.R_cb = np.asarray(calib["R_cb"], dtype=np.float64)
        self.device = device
        self.size = size

        self.model = setup_reloc3r_relpose_model(str(size), device)
        self.model.eval()

        gt_, gs_ = _array_to_view(goal_bgr, size=size)
        self.goal_shape = torch.from_numpy(gs_).unsqueeze(0).to(device)
        with torch.no_grad():
            gfeat, _, _ = self.model._encode_image(gt_.unsqueeze(0).to(device), self.goal_shape)
        self.goal_feat = gfeat
        # Patch positions are a deterministic function of the (constant) 224x224
        # grid, so derive them once instead of per call.
        with torch.no_grad():
            _x, pos = self.model.patch_embed(
                torch.zeros(1, 3, int(gs_[0]), int(gs_[1]), device=device),
                true_shape=self.goal_shape)
        self.pos = pos.detach()
        logger.info("Reloc3r geometry ready (device=%s, R_cb from %s, fit=%s)",
                    device, os.path.basename(calib_path), calib.get("fit_method", "?"))

    @torch.no_grad()
    def __call__(self, cur_bgr):
        """current BGR frame -> np.float32 [4] = [dx_body, dy_body, sin, cos]."""
        ct, cs = _array_to_view(cur_bgr, size=self.size)
        cshape = torch.from_numpy(cs).unsqueeze(0).to(self.device)
        cfeat, _, _ = self.model._encode_image(ct.unsqueeze(0).to(self.device), cshape)
        _d1, d2 = self.model._decoder(cfeat, self.pos, self.goal_feat, self.pos)
        pose = self.model._downstream_head([t.float() for t in d2], self.goal_shape[0])["pose"]
        R_cam = pose[0, :3, :3].double().cpu().numpy()
        t_cam = pose[0, :3, 3].double().cpu().numpy()
        n = np.linalg.norm(t_cam)
        t_cam = t_cam / max(n, 1e-8)                      # scale-ambiguous: direction only

        t_body = self.R_cb @ t_cam
        xy = t_body[:2]
        xy = xy / max(np.linalg.norm(xy), 1e-8)           # planar unit bearing
        R_body = self.R_cb @ R_cam @ self.R_cb.T
        yaw = np.arctan2(R_body[1, 0], R_body[0, 0])
        return np.array([xy[0], xy[1], np.sin(yaw), np.cos(yaw)], dtype=np.float32)


class Reloc3rRelationFeatures:
    """Live counterpart of ``precompute_reloc3r_dec_features.py``.

    The goal image is encoded once. Each call batch-encodes a sparse history
    and returns the final, dec_norm-normalized decoder streams. ``dec1`` is the
    current/history stream attending to the goal; ``dec2`` is the goal stream
    attending to each current/history frame. Both outputs are float32 tensors
    shaped [T, 196, 768], quantized through fp16 to match the training cache.
    """

    def __init__(self, goal_bgr, device="cuda:0", size=224):
        _ensure_path()
        from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model

        self.device = torch.device(device)
        self.size = size
        self.model = setup_reloc3r_relpose_model(str(size), self.device)
        self.model.eval()

        gt, gs = _array_to_view(goal_bgr, size=size)
        self.goal_shape = torch.from_numpy(gs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            goal_feat, goal_pos, _ = self.model._encode_image(
                gt.unsqueeze(0).to(self.device), self.goal_shape)
        self.goal_feat = goal_feat.detach()
        self.goal_pos = goal_pos.detach()
        logger.info("Reloc3r relation-feature extractor ready (device=%s)", self.device)

    @torch.no_grad()
    def __call__(self, cur_bgr_frames):
        if not cur_bgr_frames:
            raise ValueError("Reloc3r relation features need at least one current frame.")

        views = [_array_to_view(frame, size=self.size) for frame in cur_bgr_frames]
        images = torch.stack([view[0] for view in views]).to(self.device)
        shapes = torch.from_numpy(np.stack([view[1] for view in views])).to(self.device)
        cur_feat, cur_pos, _ = self.model._encode_image(images, shapes)

        batch = len(cur_bgr_frames)
        goal_feat = self.goal_feat.expand(batch, -1, -1).contiguous()
        goal_pos = self.goal_pos.expand(batch, -1, -1).contiguous()
        dec1, dec2 = self.model._decoder(cur_feat, cur_pos, goal_feat, goal_pos)
        # The final iterator element is the dec_norm-normalized last layer,
        # exactly what the offline cache script stores.
        out1 = list(dec1)[-1].float()
        out2 = list(dec2)[-1].float()
        return out1.half().float(), out2.half().float()


class Reloc3rPoseFeatures:
    """Live counterpart of the reloc3r_pose1/pose2 training cache (feat_dim=12,
    see e.g. r_pose_only_step_16940.config.yaml).

    Same decoder streams as Reloc3rRelationFeatures (dec1 = current-attends-to-
    goal, dec2 = goal-attends-to-current, matching reloc3r_relpose.forward()'s
    own pose1/pose2), but run through the model's own PoseHead instead of kept
    as raw 768-dim tokens. Each pose is PoseHead's 4x4 output
    (pose[:3,:3]=R, pose[:3,3]=t, pose[3,:]=[0,0,0,1] -- see
    reloc3r_pkg/reloc3r/pose_head.py:convert_pose_to_4x4) flattened row-major
    to 12 = [R00,R01,R02,R10,R11,R12,R20,R21,R22,tx,ty,tz].

    CAUTION: this packing has NOT been verified against the offline script
    that built the training cache (after_0328_train_reloc3r_bottom_head.h5) --
    that script lives in a different repo not present here. If an
    r_pose_only-family checkpoint produces bad trajectories near the dock,
    this ordering (vs. e.g. t-then-R, or an unflattened/normalized t) is the
    first thing to check.
    """

    def __init__(self, goal_bgr, device="cuda:0", size=224):
        _ensure_path()
        from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model

        self.device = torch.device(device)
        self.size = size
        self.model = setup_reloc3r_relpose_model(str(size), self.device)
        self.model.eval()

        gt, gs = _array_to_view(goal_bgr, size=size)
        self.goal_shape = torch.from_numpy(gs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            goal_feat, goal_pos, _ = self.model._encode_image(
                gt.unsqueeze(0).to(self.device), self.goal_shape)
        self.goal_feat = goal_feat.detach()
        self.goal_pos = goal_pos.detach()
        logger.info("Reloc3r pose-feature extractor ready (device=%s)", self.device)

    @staticmethod
    def _flatten(pose_4x4):
        r = pose_4x4[:, :3, :3].reshape(pose_4x4.shape[0], 9)
        t = pose_4x4[:, :3, 3]
        return torch.cat([r, t], dim=-1)

    @torch.no_grad()
    def __call__(self, cur_bgr_frames):
        if not cur_bgr_frames:
            raise ValueError("Reloc3r pose features need at least one current frame.")

        views = [_array_to_view(frame, size=self.size) for frame in cur_bgr_frames]
        images = torch.stack([view[0] for view in views]).to(self.device)
        shapes = torch.from_numpy(np.stack([view[1] for view in views])).to(self.device)
        cur_feat, cur_pos, _ = self.model._encode_image(images, shapes)

        batch = len(cur_bgr_frames)
        goal_feat = self.goal_feat.expand(batch, -1, -1).contiguous()
        goal_pos = self.goal_pos.expand(batch, -1, -1).contiguous()
        dec1, dec2 = self.model._decoder(cur_feat, cur_pos, goal_feat, goal_pos)
        with torch.cuda.amp.autocast(enabled=False):
            pose1 = self.model._downstream_head([t.float() for t in dec1], shapes[0])["pose"]
            pose2 = self.model._downstream_head([t.float() for t in dec2], self.goal_shape[0])["pose"]
        return self._flatten(pose1).float(), self._flatten(pose2).float()
