#!/usr/bin/env python3.12
"""Expose floor4_hallway_front_docking.h5's first 150 episodes as a training h5
plus a ReLoc3R sidecar, WITHOUT copying any image or feature bytes.

Two virtual-dataset outputs:

  dataset/f4hall150_train.h5
      episode_ends  (150,)        materialized
      encoder       (251017, 2)   materialized (read in full for action stats)
      image_bottom  VDS -> source:image_bottom

  dataset/f4hall150_train_reloc3r_bottom.h5
      episode_ends    (150,)                      materialized
      reloc3r_bottom  (251017, 196, 1024) fp16    VDS -> jiwon's already-computed
                                                  encoder cache
  ...then scripts/precompute_reloc3r_dec_features.py appends the REAL
  reloc3r_dec1_bottom / reloc3r_dec2_bottom into that same file.

1. DOCK CAMERA. Every r_relfeat arm conditions on `*_bottom` keys, so whatever
   we publish as `image_bottom` must be the dock-facing camera (orbbec-0). For
   THIS source, unlike floor4_inside, no swap is needed -- the source's own
   `image_bottom` is the dock view for all 150 training episodes. Established
   three ways on 2026-08-27:
     * attrs: episodes 141-150 are labelled `bottom=camera_orbbec-0`; episodes
       1-140 and 151-154 use the legacy naming `bottom=room2`.
     * pixels: episode-final frames of ALL 150 training episodes have a
       strictly higher gradient energy in image_bottom (~7.0) than image_top
       (~3.6); by eye image_bottom holds the green dock and image_top a blank
       ceiling, in both the `room2` and the `camera_orbbec-0` episodes.
     * ReLoc3R: re-encoding source image_bottom rows through
       precompute_reloc3r_cache.encode_frames reproduces jiwon's cached
       reloc3r_bottom to fp16 rounding (max|diff| <= 0.05 over 7 rows spanning
       the range), while image_top differs by ~20. That cache is therefore the
       dock camera's, and is the one this sidecar maps.
   Pinned by sha256 anyway, so an in-place source swap (which is exactly what
   happened to floor4_inside on 2026-08-26) fails the build loudly.

2. EPISODE TRIM. The source holds 154 episodes / 258410 rows. House split is
   "episodes 1-150 train; 151-153 validation; 154+ excluded" -- 154 = 150+3+1.
   First 150 episodes = 251017 rows.

3. ENCODER CACHE REUSE. jiwon's
   dataset/floor4_hallway_front_docking_reloc3r_bottom.h5 already holds
   reloc3r_bottom for episodes 1-153 (n_done_ep=153, rows 0..256955), built by
   an `array_to_view`/`encode_frames` that is byte-identical to this repo's,
   from a copy of this source that is byte-identical too (episode_ends,
   encoder, and sampled image_bottom rows all compare equal). Mapping it saves
   the ~1.5 h ViT-L pass; only the dec1/dec2 decoder rerun is left. Its
   `reloc3r_dec_bottom` is a SINGLE stream and is NOT reused -- the relfeat arm
   needs both dec1 and dec2.

Usage:
    python3.12 scripts/build_f4hall150_vds.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SOURCE = Path("/home/work/.postech/jiwon/floor4_hallway_front_docking.h5")
ENC_CACHE = Path("/home/work/.postech/jiwon/diffusion_policy_robot_docking_jiwon/"
                 "dataset/floor4_hallway_front_docking_reloc3r_bottom.h5")

N_TRAIN_EPISODES = 150
VIRTUAL_MAP = {"image_bottom": "image_bottom"}
MATERIAL_KEYS = ["encoder"]

DOCK_FINGERPRINT_ROWS = [0, 50000, 120000, 210190, 251016]
DOCK_FINGERPRINT_SHA256 = (
    "4caf1c950cee79a8cc07689bfe99a3e84562656dedf5eaefc548532d4809fb2f")

FEAT_KEY = "reloc3r_bottom"


def _fingerprint(ds, rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(np.ascontiguousarray(ds[r]).tobytes())
    return h.hexdigest()


def _grad_energy(chw):
    g = chw.astype(np.float32).mean(0)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def _strings(values):
    return np.asarray(list(values), dtype=h5py.string_dtype("utf-8"))


def _atomic(out_path):
    """h5py opens 'w' with O_TRUNC BEFORE taking the HDF5 lock, so writing
    straight onto a file a training job holds open zeroes it and only THEN
    errors. Build into a sibling temp file and os.replace() it in."""
    fd, tmp = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".h5",
                              dir=str(out_path.parent))
    os.close(fd)
    return Path(tmp)


def build(name, out_dir, source, enc_cache, n_episodes):
    for p in (source, enc_cache):
        if not p.is_file():
            raise FileNotFoundError(p)

    with h5py.File(source, "r") as src:
        all_ends = np.asarray(src["episode_ends"][:], dtype=np.int64)
        if len(all_ends) < n_episodes:
            raise ValueError(f"{source}: {len(all_ends)} episodes, need {n_episodes}")
        episode_ends = all_ends[:n_episodes].copy()
        total_rows = int(episode_ends[-1])
        src_attrs = {k: str(v) for k, v in src.attrs.items()}

        # --- guard the dock camera BY CONTENT ---
        dock_key = VIRTUAL_MAP["image_bottom"]
        got = _fingerprint(src[dock_key], DOCK_FINGERPRINT_ROWS)
        if got != DOCK_FINGERPRINT_SHA256:
            other = {"image_bottom": "image_top", "image_top": "image_bottom"}[dock_key]
            hint = ""
            if _fingerprint(src[other], DOCK_FINGERPRINT_ROWS) == DOCK_FINGERPRINT_SHA256:
                hint = (f" The dock pixels are now in '{other}' -- the source was "
                        f"swapped in place; flip VIRTUAL_MAP and re-verify the "
                        f"encoder cache before rebuilding.")
            raise ValueError(
                f"{source}:{dock_key} does not match the pinned dock fingerprint "
                f"(got {got[:16]}..., want {DOCK_FINGERPRINT_SHA256[:16]}...)."
                f"{hint} Source attrs are mixed ('room2' for 144 episodes) and "
                f"must not be used to decide this.")

        # cheap semantic backstop: the dock view is the structured one.
        starts = np.concatenate([[0], all_ends[:-1]])
        for epi in (0, n_episodes // 2, 140, n_episodes - 1):
            r = int(all_ends[epi]) - 1
            eb, et = _grad_energy(src[dock_key][r]), _grad_energy(src["image_top"][r])
            if eb <= et:
                raise ValueError(
                    f"episode {epi + 1} (row {r}): published image_bottom is the "
                    f"FLATTER view (grad {eb:.2f} vs image_top {et:.2f}) -- that "
                    f"is the blank-ceiling camera, not the dock.")

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
        names = src["episode_names"].asstr()[:n_episodes]

    # --- validate the borrowed encoder cache covers our rows ---
    with h5py.File(enc_cache, "r") as ec:
        feat = ec[FEAT_KEY]
        if not np.array_equal(np.asarray(ec["episode_ends"][:], dtype=np.int64),
                              all_ends):
            raise ValueError(f"{enc_cache}: episode_ends differ from {source}; "
                             f"the cache was built from different data")
        if feat.attrs.get("camera") != "image_bottom":
            raise ValueError(f"{enc_cache}:{FEAT_KEY} camera attr is "
                             f"{feat.attrs.get('camera')!r}, expected 'image_bottom'")
        n_done_ep = int(feat.attrs.get("n_done_ep", 0))
        if n_done_ep < n_episodes:
            raise ValueError(f"{enc_cache}:{FEAT_KEY} only has {n_done_ep} episodes "
                             f"filled, need {n_episodes}")
        # the attr claims done; confirm the last row we will map is not zero-fill.
        if float(np.abs(feat[total_rows - 1].astype(np.float32)).mean()) == 0.0:
            raise ValueError(f"{enc_cache}:{FEAT_KEY} row {total_rows - 1} is "
                             f"zero-filled despite n_done_ep={n_done_ep}")
        feat_shape, feat_dtype = feat.shape, feat.dtype

    out_dir.mkdir(parents=True, exist_ok=True)
    main_out = out_dir / f"{name}_train.h5"
    side_out = out_dir / f"{name}_train_reloc3r_bottom.h5"

    # ---------------- main training h5 ----------------
    tmp_out = _atomic(main_out)
    with h5py.File(tmp_out, "w", libver="latest") as dst:
        dst.attrs["format"] = "f4hall150_vds_v1"
        dst.attrs["num_episodes"] = n_episodes
        dst.attrs["num_rows"] = total_rows
        dst.attrs["source_file"] = str(source)
        dst.attrs["image_bottom_source"] = "camera_orbbec-0"
        dst.attrs["camera_note"] = (
            "image_bottom is a zero-copy VDS onto the SOURCE's own image_bottom "
            "-- no swap, unlike f4in_150ep. Verified by pinned sha256, by "
            "gradient energy over all 150 episode-final frames, and by "
            "re-encoding through ReLoc3R against the borrowed encoder cache. "
            "The source's per-episode attrs say 'camera_orbbec-0' for episodes "
            "141-150 and the legacy alias 'room2' for the rest; both are the "
            "same physical dock-facing camera.")
        dst.attrs["virtual_map"] = json.dumps(VIRTUAL_MAP)
        dst.attrs["dock_fingerprint_sha256"] = DOCK_FINGERPRINT_SHA256
        dst.attrs["dock_fingerprint_rows"] = json.dumps(DOCK_FINGERPRINT_ROWS)
        dst.attrs["source_attrs"] = json.dumps(src_attrs, ensure_ascii=False)
        dst.create_dataset("episode_ends", data=episode_ends)
        with h5py.File(source, "r") as src:
            for key in MATERIAL_KEYS:
                dst.create_dataset(key, data=src[key][:total_rows])
            for pub, (key, src_shape, dtype) in shapes.items():
                layout = h5py.VirtualLayout(shape=(total_rows, *src_shape[1:]),
                                            dtype=dtype)
                vsrc = h5py.VirtualSource(str(source), key, shape=src_shape)
                layout[0:total_rows] = vsrc[0:total_rows]
                dst.create_virtual_dataset(pub, layout)
            dst.create_dataset("source_episode_name", data=_strings(names))
    os.replace(tmp_out, main_out)
    print(f"Built {n_episodes} episodes / {total_rows} rows -> {main_out}")

    # ---------------- reloc3r sidecar (encoder stream only) ----------------
    if side_out.exists():
        with h5py.File(side_out, "r") as ex:
            have = [k for k in ("reloc3r_dec1_bottom", "reloc3r_dec2_bottom") if k in ex]
        if have:
            raise SystemExit(
                f"REFUSING to rebuild {side_out}: it already holds {have}. "
                f"Rebuilding would discard the decoder precompute. Delete it "
                f"explicitly if that is what you want.")
    tmp_side = _atomic(side_out)
    with h5py.File(tmp_side, "w", libver="latest") as dst:
        dst.attrs["format"] = "f4hall150_reloc3r_vds_v1"
        dst.attrs["num_episodes"] = n_episodes
        dst.attrs["num_rows"] = total_rows
        dst.attrs["encoder_cache_source"] = str(enc_cache)
        dst.attrs["camera"] = "image_bottom"
        dst.attrs["note"] = (
            "reloc3r_bottom is a VDS onto a pre-existing encoder cache (same "
            "encode_frames/array_to_view code, byte-identical source). "
            "reloc3r_dec1_bottom / reloc3r_dec2_bottom are appended here as REAL "
            "datasets by scripts/precompute_reloc3r_dec_features.py.")
        dst.create_dataset("episode_ends", data=episode_ends)
        layout = h5py.VirtualLayout(shape=(total_rows, *feat_shape[1:]),
                                    dtype=feat_dtype)
        vsrc = h5py.VirtualSource(str(enc_cache), FEAT_KEY, shape=feat_shape)
        layout[0:total_rows] = vsrc[0:total_rows]
        vds = dst.create_virtual_dataset(FEAT_KEY, layout)
        vds.attrs["camera"] = "image_bottom"
        vds.attrs["n_done_ep"] = n_episodes
        vds.attrs["source_h5"] = str(source)
    os.replace(tmp_side, side_out)
    print(f"Built reloc3r sidecar (encoder stream, virtual) -> {side_out}")

    # ---------------- read-back verification ----------------
    with h5py.File(main_out, "r") as m, h5py.File(side_out, "r") as s, \
            h5py.File(source, "r") as src, h5py.File(enc_cache, "r") as ec:
        assert m["encoder"].shape[0] == total_rows
        assert m["image_bottom"].shape[0] == total_rows
        assert s[FEAT_KEY].shape[0] == total_rows
        assert np.array_equal(m["episode_ends"][:], all_ends[:n_episodes])
        assert np.array_equal(s["episode_ends"][:], all_ends[:n_episodes])
        for row in (0, total_rows // 2, total_rows - 1):
            assert np.array_equal(m["image_bottom"][row], src["image_bottom"][row]), row
            assert np.array_equal(m["encoder"][row], src["encoder"][row]), row
            assert np.array_equal(s[FEAT_KEY][row], ec[FEAT_KEY][row]), row
        assert _fingerprint(m["image_bottom"], DOCK_FINGERPRINT_ROWS) == \
            DOCK_FINGERPRINT_SHA256, "published image_bottom is not the dock camera"
    print("Verified: image_bottom == source dock camera, reloc3r_bottom == cache, "
          "rows aligned in both outputs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="f4hall150")
    ap.add_argument("--out-dir", default=str(REPO / "dataset"))
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--enc-cache", default=str(ENC_CACHE))
    ap.add_argument("--episodes", type=int, default=N_TRAIN_EPISODES)
    a = ap.parse_args()
    build(a.name, Path(a.out_dir).expanduser().resolve(),
          Path(a.source).expanduser().resolve(),
          Path(a.enc_cache).expanduser().resolve(), a.episodes)
