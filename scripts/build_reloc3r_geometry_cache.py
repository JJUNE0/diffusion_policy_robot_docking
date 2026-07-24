"""Build the final Reloc3r body-frame geometry cache for training
(docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md), from the ALREADY-COMPUTED
per-frame rotation/direction cache (scripts/precompute_reloc3r_cache.py +
scripts/precompute_reloc3r_direction.py) and the fitted camera->body
calibration (scripts/calibrate_reloc3r_body_frame.py). No re-inference
needed -- this is a pure numpy transform of cached arrays, so it runs in
seconds, not minutes.

Note on "latest current RGB only" (spec requirement): the underlying
per-frame reloc3r_rot_bottom/reloc3r_dir_bottom cache already stores, for
EVERY frame t, the pose between frame t (as the sole "current" input, not a
history) and its episode's static goal frame -- this is exactly the
"latest current RGB + static goal RGB, single pair" computation the spec
asks for; RGB history was never fed to Reloc3r. At train time, a sample at
sequence-index t simply looks up the ONE geometry vector for that t (its
"latest current frame"), not a temporal stack.

Writes (row-aligned 1:1 with the source h5, same convention as the DINO/
Reloc3r feature caches):
  geometry_bottom       : (N, 4) float32 [dx_body, dy_body, sin(yaw), cos(yaw)]
                           dx,dy are body-frame, UNIT-normalized (planar).
  relative_pose_bottom  : (N, 3, 4) float32  [R_cam | t_cam_unit] -- the raw
                           camera-frame estimate (translation is the
                           Reloc3r UNIT direction, not metric -- see
                           docs/reloc3r.md; kept for provenance/debugging,
                           not used at train time).
  goal_row              : (N,) int64  -- goal_image_id per frame (episode's
                           last row), for provenance.
Attrs on geometry_bottom: preprocessing_version, reloc3r_checkpoint,
sensor_to_body_yaw_deg, R_cb (flattened), source_rot_cache, source_dir_cache.

sample_id / current_frame_timestamp are NOT duplicated here: sample_id ==
the row index itself (same index space as image_bottom/dock_pose/lidar_points
in the source h5, so it's implicit), and current_frame_timestamp ==
row_index * dt (dt=0.0333s, constant sync rate, utils/preprocessing.py) --
both trivially recoverable from the row index already used as the primary
key everywhere else in this codebase, so we don't store a redundant column.

Usage:
  python scripts/build_reloc3r_geometry_cache.py --split train
  python scripts/build_reloc3r_geometry_cache.py --split test
"""
import argparse
import json
import os

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(REPO, "reloc3r/body_frame_calibration.json")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--camera", default="image_bottom", choices=["image_bottom", "image_top"])
    args = ap.parse_args()

    cam_tag = "bottom" if args.camera == "image_bottom" else "top"
    src_h5 = os.path.join(REPO, f"dataset/after_0328_{args.split}.h5")
    cache_h5 = os.path.join(REPO, f"dataset/after_0328_{args.split}_reloc3r_{cam_tag}.h5")

    with open(CALIB_PATH) as f:
        calib = json.load(f)
    R_cb = np.array(calib["R_cb"])
    sensor_to_body_yaw = float(calib["sensor_to_body_yaw_rad"])  # unused here: only
    # (dx,dy) derived from dock_pose needed this; Reloc3r's OWN camera-frame
    # output goes through R_cb directly (R_cb was fit to land in the SAME
    # already-yaw-corrected body frame -- see calibrate script).

    with h5py.File(cache_h5, "r") as f:
        n = f["reloc3r_bottom"].shape[0]
        rot = f["reloc3r_rot_bottom"][:]   # [N,7] col0(3),col1(3),geo(1)
        tdir = f["reloc3r_dir_bottom"][:]  # [N,3] unit vector, camera frame
        episode_ends = f["episode_ends"][:]

    col0, col1 = rot[:, 0:3].astype(np.float64), rot[:, 3:6].astype(np.float64)
    col2 = np.cross(col0, col1)
    R_cam = np.stack([col0, col1, col2], axis=-1)          # [N,3,3]
    t_cam = tdir.astype(np.float64)                        # [N,3]

    # body-frame direction: R_cb @ t_cam, planar (drop z), unit-normalize
    t_body = np.einsum("ij,nj->ni", R_cb, t_cam)
    xy = t_body[:, :2]
    xy_norm = np.linalg.norm(xy, axis=1, keepdims=True).clip(1e-8)
    dx_body, dy_body = (xy / xy_norm)[:, 0], (xy / xy_norm)[:, 1]

    # body-frame relative yaw via conjugation R_body = R_cb @ R_cam @ R_cb^T
    R_body = np.einsum("ij,njk,kl->nil", R_cb, R_cam, R_cb.T)
    yaw = np.arctan2(R_body[:, 1, 0], R_body[:, 0, 0])
    sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)

    geometry = np.stack([dx_body, dy_body, sin_yaw, cos_yaw], axis=1).astype(np.float32)

    # goal_row per frame (provenance)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    goal_row = np.zeros(n, dtype=np.int64)
    for ep_id, ep_end in enumerate(episode_ends):
        s = int(ep_starts[ep_id])
        goal_row[s:ep_end] = ep_end - 1

    relative_pose = np.concatenate(
        [R_cam.astype(np.float32), t_cam.astype(np.float32)[:, :, None]], axis=2)  # [N,3,4]

    with h5py.File(cache_h5, "a") as fo:
        for key in ("geometry_bottom", "relative_pose_bottom", "goal_row"):
            if key in fo:
                del fo[key]
        gds = fo.create_dataset("geometry_bottom", data=geometry, dtype="float32")
        fo.create_dataset("relative_pose_bottom", data=relative_pose, dtype="float32")
        fo.create_dataset("goal_row", data=goal_row, dtype="int64")
        gds.attrs["preprocessing_version"] = "v1_2026-07-25"
        gds.attrs["reloc3r_checkpoint"] = "siyan824/reloc3r-224"
        gds.attrs["sensor_to_body_yaw_deg"] = float(np.degrees(sensor_to_body_yaw))
        gds.attrs["R_cb_flat"] = R_cb.flatten().tolist()
        gds.attrs["source_rot_cache"] = "reloc3r_rot_bottom"
        gds.attrs["source_dir_cache"] = "reloc3r_dir_bottom"
        gds.attrs["column_order"] = "dx_body,dy_body,sin(relative_yaw_body),cos(relative_yaw_body)"

    print(f"[{args.split}] geometry_bottom: {geometry.shape}, "
          f"dx range=[{dx_body.min():.2f},{dx_body.max():.2f}] "
          f"yaw range deg=[{np.degrees(yaw).min():.1f},{np.degrees(yaw).max():.1f}]")
    print(f"Wrote geometry_bottom/relative_pose_bottom/goal_row to {cache_h5}")


if __name__ == "__main__":
    main()
