#!/usr/bin/env python3.12
"""Expose floor4_inside_front_docking.h5's first 150 episodes as a training h5
in THIS repo's camera convention, without copying any image bytes.

1. DOCK CAMERA, PINNED BY CONTENT. Every r_relfeat / r_encfeat / r_pose arm
   conditions on `*_bottom` keys, so whatever this publishes as `image_bottom`
   must be the dock-facing view (camera_orbbec-0) -- the same physical camera
   after_0328 / 0824_4f_hallway / floor4_hallway put in their image_bottom.

   The source's LABELS cannot be used to decide which array that is:
     * 2026-08-25 23:43 -- image_bottom held a blank wall, image_top held the
       dock, matching the file's `image_bottom_source = camera_usb-0` attr.
     * 2026-08-26 00:21 -- the source was edited in place and the two arrays'
       CONTENTS were swapped (image_bottom now holds the dock) while the attrs
       were left unchanged. The attrs are now simply wrong.
   Pixel statistics do not discriminate either (measured std 30.2 for the dock
   view vs 31.2 for the washed-out wall).

   So the dock array is pinned by a sha256 over fixed sample rows. That
   fingerprint was taken from the array PROVEN to be the dock by re-encoding it
   through ReLoc3R and matching it against the already-built reloc3r_bottom
   cache bit-exactly (11/11 sample rows spanning the whole range, max|diff| =
   0.0000). If the source flips again, the build raises instead of silently
   publishing the wall, and says which array now holds the dock.

   NOTE for anyone re-running the precompute: the caches under
   dataset/f4in_150ep_train_reloc3r_bottom*.h5 were built on 2026-08-26
   00:00-01:16 and are verified dock-facing. `encoder` and `episode_ends` were
   untouched by the 00:21 edit, so those caches stay row-aligned with this file.

2. EPISODE TRIM. The file holds 154 episodes. Its sibling
   floor4_hallway_front_docking.h5 documents the house split as
   "episodes 1-150 train; episodes 151-153 validation; episodes 154+ excluded"
   (train_episode_range=[1,150]); 154 = 150 + 3 + 1 matches exactly, so the
   first 150 episodes (287256 rows) are the training split.

`encoder` is MATERIALIZED (2.3 MB, read in full at startup for the minmax
action stats); the image array stays virtual.

Usage:
    python scripts/build_f4in_150ep_vds.py
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SOURCE = Path("/home/work/.postech/jiwon/diffusion_policy_robot_docking_jiwon/"
              "dataset/floor4_inside_front_docking.h5")

N_TRAIN_EPISODES = 150
# published name -> source dataset name.
#
# HISTORY, and why this is pinned by CONTENT and not by name or attr:
# when this was first built (2026-08-25 23:45) the source had
#     image_bottom = camera_usb-0 (blank wall), image_top = camera_orbbec-0 (dock)
# so the map was {"image_bottom": "image_top"}. At 2026-08-26 00:21 the source
# was edited in place and the two arrays' CONTENTS were swapped -- image_bottom
# now holds the dock -- while `image_bottom_source` was left saying
# "camera_usb-0". The attrs are therefore stale and must not be trusted.
# Simple pixel statistics do not discriminate either (measured: std 30.2 for the
# dock view vs 31.2 for the wash-out wall). So the dock camera is pinned by a
# sha256 over fixed sample rows, taken from the array that was PROVEN to be the
# dock by re-encoding it through ReLoc3R and matching it bit-exactly against the
# already-built reloc3r_bottom cache (11/11 sample rows, max|diff| = 0.0000).
VIRTUAL_MAP = {"image_bottom": "image_bottom"}
MATERIAL_KEYS = ["encoder"]

# Fingerprint of the dock-facing array. If the source flips again, the build
# fails loudly instead of silently publishing the wall.
DOCK_FINGERPRINT_ROWS = [0, 50000, 120000, 200000, 287255]
DOCK_FINGERPRINT_SHA256 = (
    "3d4e64ed69763e5253a4d3086b80b86d43adf4b29ff49d822892de3e2f530ef9")


def _fingerprint(ds, rows):
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(np.ascontiguousarray(ds[r]).tobytes())
    return h.hexdigest()


def _strings(values):
    return np.asarray(list(values), dtype=h5py.string_dtype("utf-8"))


def build(name, out_dir, source, n_episodes):
    if not source.is_file():
        raise FileNotFoundError(source)

    with h5py.File(source, "r") as src:
        all_ends = np.asarray(src["episode_ends"][:], dtype=np.int64)
        if len(all_ends) < n_episodes:
            raise ValueError(f"{source}: {len(all_ends)} episodes, need {n_episodes}")
        episode_ends = all_ends[:n_episodes].copy()
        total_rows = int(episode_ends[-1])
        src_attrs = {k: str(v) for k, v in src.attrs.items()}

        # Guard the whole point of this script BY CONTENT: whatever array we
        # publish as image_bottom must be the dock-facing one.
        dock_key = VIRTUAL_MAP["image_bottom"]
        got = _fingerprint(src[dock_key], DOCK_FINGERPRINT_ROWS)
        if got != DOCK_FINGERPRINT_SHA256:
            other = {"image_bottom": "image_top",
                     "image_top": "image_bottom"}[dock_key]
            hint = ""
            if _fingerprint(src[other], DOCK_FINGERPRINT_ROWS) == DOCK_FINGERPRINT_SHA256:
                hint = (f" The dock pixels are now in '{other}' -- the source was "
                        f"swapped again; flip VIRTUAL_MAP to {{'image_bottom': "
                        f"'{other}'}} and re-verify before rebuilding.")
            raise ValueError(
                f"{source}:{dock_key} does not match the pinned dock "
                f"fingerprint (got {got[:16]}..., want "
                f"{DOCK_FINGERPRINT_SHA256[:16]}...).{hint} Source attrs are "
                f"stale and must not be used to decide this.")

        shapes = {}
        for pub, key in VIRTUAL_MAP.items():
            ds = src[key]
            if ds.shape[0] != int(all_ends[-1]):
                raise ValueError(f"{source}:{key} has {ds.shape[0]} rows, "
                                 f"episode_ends ends at {int(all_ends[-1])}")
            shapes[pub] = (key, ds.shape, ds.dtype)
        for key in MATERIAL_KEYS:
            if src[key].shape[0] != int(all_ends[-1]):
                raise ValueError(f"{source}:{key} row count mismatch")

    out_dir.mkdir(parents=True, exist_ok=True)
    main_out = out_dir / f"{name}_train.h5"

    # NEVER h5py.File(main_out, "w") directly. h5py opens with O_TRUNC BEFORE
    # it acquires the HDF5 lock, so if any training job has this file open the
    # output is zeroed and only THEN does the lock error surface -- the reader
    # is left mapping a 0-byte file and silently trains on garbage. Build into a
    # sibling temp file and os.replace() it in: rename is atomic, and an open
    # reader keeps its original inode intact.
    fd, tmp_name = tempfile.mkstemp(prefix=main_out.name + ".", suffix=".h5",
                                    dir=str(out_dir))
    os.close(fd)
    tmp_out = Path(tmp_name)

    with h5py.File(tmp_out, "w", libver="latest") as dst:
        dst.attrs["format"] = "f4in_150ep_vds_v1"
        dst.attrs["num_episodes"] = n_episodes
        dst.attrs["num_rows"] = total_rows
        dst.attrs["source_file"] = str(source)
        dst.attrs["camera_note"] = (
            f"image_bottom is a VDS onto the SOURCE's "
            f"'{VIRTUAL_MAP['image_bottom']}', selected by matching the pinned "
            f"dock fingerprint -- NOT by the source's image_bottom_source attr, "
            f"which is stale (it still says camera_usb-0 after the 2026-08-26 "
            f"00:21 in-place content swap).")
        dst.attrs["image_bottom_source"] = "camera_orbbec-0"
        dst.attrs["virtual_map"] = json.dumps(VIRTUAL_MAP)
        dst.attrs["dock_fingerprint_sha256"] = DOCK_FINGERPRINT_SHA256
        dst.attrs["dock_fingerprint_rows"] = json.dumps(DOCK_FINGERPRINT_ROWS)
        dst.attrs["source_attrs"] = json.dumps(src_attrs, ensure_ascii=False)
        dst.create_dataset("episode_ends", data=episode_ends)

        with h5py.File(source, "r") as src:
            for key in MATERIAL_KEYS:
                dst.create_dataset(key, data=src[key][:total_rows])
            for pub, (key, src_shape, dtype) in shapes.items():
                layout = h5py.VirtualLayout(
                    shape=(total_rows, *src_shape[1:]), dtype=dtype)
                vsrc = h5py.VirtualSource(str(source), key, shape=src_shape)
                layout[0:total_rows] = vsrc[0:total_rows]
                dst.create_virtual_dataset(pub, layout)
            # provenance so a merged row can be traced back
            names = src["episode_names"].asstr()[:n_episodes]
            dst.create_dataset("source_episode_name", data=_strings(names))

    os.replace(tmp_out, main_out)

    print(f"Built {n_episodes} episodes / {total_rows} rows -> {main_out}")
    print(f"  material: {', '.join(MATERIAL_KEYS)}")
    print(f"  virtual : " + ", ".join(f"{p} <- source:{k}"
                                      for p, (k, _, _) in shapes.items()))

    # ---- read-back verification ----
    with h5py.File(main_out, "r") as m, h5py.File(source, "r") as src:
        assert m["encoder"].shape[0] == total_rows
        assert m["image_bottom"].shape[0] == total_rows
        assert np.array_equal(m["episode_ends"][:], src["episode_ends"][:n_episodes])
        dock_key = VIRTUAL_MAP["image_bottom"]
        for row in (0, total_rows // 2, total_rows - 1):
            assert np.array_equal(m["image_bottom"][row], src[dock_key][row]), row
            assert np.array_equal(m["encoder"][row], src["encoder"][row]), row
        assert _fingerprint(m["image_bottom"], DOCK_FINGERPRINT_ROWS) == \
            DOCK_FINGERPRINT_SHA256, "published image_bottom is not the dock camera"
    print(f"Verified: image_bottom == source '{dock_key}' and matches the pinned "
          f"dock fingerprint; rows aligned.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="f4in_150ep")
    ap.add_argument("--out-dir", default=str(REPO / "dataset"))
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--episodes", type=int, default=N_TRAIN_EPISODES)
    a = ap.parse_args()
    build(a.name, Path(a.out_dir).expanduser().resolve(),
          Path(a.source).expanduser().resolve(), a.episodes)
