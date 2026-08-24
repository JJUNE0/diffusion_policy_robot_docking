#!/usr/bin/env python3.12
"""Combine `after_0328` (145 ep) + `0824_4f_hallway` (5 ep) into the 150-episode
`4f_hallway_150ep` dataset WITHOUT copying any image or ReLoc3R bytes.

The big arrays are exposed as HDF5 Virtual Datasets pointing back at the two
source files, so the 226 GB `after_0328_train_reloc3r_bottom.h5` is reused in
place rather than duplicated -- only the new 5 episodes ever needed GPU time
(scripts/precompute_reloc3r_cache.py + precompute_reloc3r_dec_features.py on
`dataset/0824_4f_hallway_train.h5`).

Why not scripts/build_combined_docking_vds.py: that one maps only
`reloc3r_bottom` into its sidecar and requires a matching `_reloc3r_top.h5` per
source. The r_relfeat arms read `reloc3r_dec1_bottom`/`reloc3r_dec2_bottom` and
this dataset is bottom-camera-only (`camera_orbbec-0`), so the mapping differs.

`encoder` is MATERIALIZED (real data, ~2 MB) rather than virtual: it is read in
full at startup for the minmax action stats, and a plain dataset keeps that path
identical to a physically merged file.

Episode ordering is `after_0328`'s 145 episodes first, then the 5 new ones, so
merged row indices 0..225464 keep their original meaning.

Usage:
    python scripts/build_4f_hallway_150ep_vds.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# (dataset tag, main h5, bottom reloc3r sidecar, post-head sidecar)
SOURCES = [
    ("after_0328",
     REPO / "dataset/after_0328_train.h5",
     REPO / "dataset/after_0328_train_reloc3r_bottom.h5",
     REPO / "dataset/after_0328_train_reloc3r_bottom_head.h5"),
    ("0824_4f_hallway",
     REPO / "dataset/0824_4f_hallway_train.h5",
     REPO / "dataset/0824_4f_hallway_train_reloc3r_bottom.h5",
     REPO / "dataset/0824_4f_hallway_train_reloc3r_bottom_head.h5"),
]

# Virtual (zero-copy) row datasets per output file.
MAIN_VIRTUAL = ["image_bottom"]
# Materialized row datasets (small, read in full at train startup).
MAIN_MATERIAL = ["encoder"]
SIDECAR_VIRTUAL = ["reloc3r_bottom", "reloc3r_dec1_bottom", "reloc3r_dec2_bottom"]
# Post-head taps live in their own small sidecar (the 12-D pose that r_pose_only
# conditions on, plus the 1024-D pre-fc head vector). Small enough that the
# sensor specs load them with cache_in_ram instead of a memfd.
HEAD_VIRTUAL = ["reloc3r_pose1_bottom", "reloc3r_pose2_bottom",
                "reloc3r_head1_bottom", "reloc3r_head2_bottom"]


def _strings(values):
    return np.asarray(list(values), dtype=h5py.string_dtype("utf-8"))


def _episode_layout(main_paths):
    """Per-source (episode_ends, n_rows) plus the merged episode_ends."""
    per_source = []
    merged, offset = [], 0
    for path in main_paths:
        with h5py.File(path, "r") as h5:
            ends = np.asarray(h5["episode_ends"][:], dtype=np.int64)
        per_source.append(ends)
        merged.extend((ends + offset).tolist())
        offset += int(ends[-1])
    return per_source, np.asarray(merged, dtype=np.int64), offset


def _check(paths, keys, total_rows_expected):
    """Fail loudly before writing if a key is missing or misshaped."""
    ref = {}
    for path in paths:
        with h5py.File(path, "r") as h5:
            rows = int(h5["episode_ends"][-1])
            for key in keys:
                if key not in h5:
                    raise KeyError(f"{path}: missing '{key}'")
                ds = h5[key]
                if ds.shape[0] != rows:
                    raise ValueError(
                        f"{path}:{key} has {ds.shape[0]} rows but episode_ends "
                        f"ends at {rows}")
                sig = (ds.shape[1:], ds.dtype)
                if key in ref and ref[key] != sig:
                    raise ValueError(
                        f"{path}:{key} is {sig} but a previous source had {ref[key]}")
                ref[key] = sig
    return ref


def _write_virtual(dst, key, paths, shapes, total_rows):
    trailing, dtype = shapes[key]
    layout = h5py.VirtualLayout(shape=(total_rows, *trailing), dtype=dtype)
    cursor = 0
    for path in paths:
        with h5py.File(path, "r") as h5:
            n = h5[key].shape[0]
            src_shape = h5[key].shape
        source = h5py.VirtualSource(str(path), key, shape=src_shape)
        layout[cursor:cursor + n] = source[0:n]
        cursor += n
    if cursor != total_rows:
        raise ValueError(f"{key}: mapped {cursor} rows, expected {total_rows}")
    dst.create_virtual_dataset(key, layout)


def _write_material(dst, key, paths, shapes):
    trailing, dtype = shapes[key]
    blocks = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            blocks.append(h5[key][:])
    data = np.concatenate(blocks, axis=0)
    dst.create_dataset(key, data=data, dtype=dtype)


def _provenance(dst, per_source_ends, tags):
    dataset, file_index, source_episode = [], [], []
    for i, (tag, ends) in enumerate(zip(tags, per_source_ends)):
        for ep in range(len(ends)):
            dataset.append(tag)
            file_index.append(i)
            source_episode.append(ep)
    dst.create_dataset("source_dataset", data=_strings(dataset))
    dst.create_dataset("source_file_index", data=np.asarray(file_index, np.int32))
    dst.create_dataset("source_episode", data=np.asarray(source_episode, np.int32))


def build(name, out_dir):
    tags = [s[0] for s in SOURCES]
    mains = [s[1] for s in SOURCES]
    sidecars = [s[2] for s in SOURCES]
    heads = [s[3] for s in SOURCES]
    for path in (*mains, *sidecars, *heads):
        if not path.is_file():
            raise FileNotFoundError(path)

    per_source_ends, episode_ends, total_rows = _episode_layout(mains)
    main_shapes = _check(mains, MAIN_VIRTUAL + MAIN_MATERIAL, total_rows)
    side_shapes = _check(sidecars, SIDECAR_VIRTUAL, total_rows)
    head_shapes = _check(heads, HEAD_VIRTUAL, total_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    main_out = out_dir / f"{name}_train.h5"
    side_out = out_dir / f"{name}_train_reloc3r_bottom.h5"
    head_out = out_dir / f"{name}_train_reloc3r_bottom_head.h5"

    with h5py.File(main_out, "w", libver="latest") as dst:
        dst.attrs["format"] = "4f_hallway_150ep_vds_v1"
        dst.attrs["num_episodes"] = len(episode_ends)
        dst.attrs["num_rows"] = total_rows
        dst.attrs["sources"] = json.dumps(
            [{"dataset": t, "main": str(m), "reloc3r_bottom": str(s),
              "episodes": len(e), "rows": int(e[-1])}
             for t, m, s, e in zip(tags, mains, sidecars, per_source_ends)])
        dst.create_dataset("episode_ends", data=episode_ends)
        dst.create_dataset("source_file", data=_strings(str(p) for p in mains))
        _provenance(dst, per_source_ends, tags)
        for key in MAIN_MATERIAL:
            _write_material(dst, key, mains, main_shapes)
        for key in MAIN_VIRTUAL:
            _write_virtual(dst, key, mains, main_shapes, total_rows)

    with h5py.File(side_out, "w", libver="latest") as dst:
        dst.attrs["format"] = "4f_hallway_150ep_vds_v1"
        dst.attrs["num_episodes"] = len(episode_ends)
        dst.attrs["num_rows"] = total_rows
        dst.create_dataset("episode_ends", data=episode_ends)
        dst.create_dataset("source_file", data=_strings(str(p) for p in sidecars))
        for key in SIDECAR_VIRTUAL:
            _write_virtual(dst, key, sidecars, side_shapes, total_rows)

    with h5py.File(head_out, "w", libver="latest") as dst:
        dst.attrs["format"] = "4f_hallway_150ep_vds_v1"
        dst.attrs["num_episodes"] = len(episode_ends)
        dst.attrs["num_rows"] = total_rows
        dst.create_dataset("episode_ends", data=episode_ends)
        dst.create_dataset("source_file", data=_strings(str(p) for p in heads))
        for key in HEAD_VIRTUAL:
            _write_virtual(dst, key, heads, head_shapes, total_rows)

    print(f"Built {len(episode_ends)} episodes / {total_rows} rows")
    print(f"  main:    {main_out}")
    print(f"    material: {', '.join(MAIN_MATERIAL)}")
    print(f"    virtual:  {', '.join(MAIN_VIRTUAL)}")
    print(f"  sidecar: {side_out}")
    print(f"    virtual:  {', '.join(SIDECAR_VIRTUAL)}")
    print(f"  head:    {head_out}")
    print(f"    virtual:  {', '.join(HEAD_VIRTUAL)}")

    # ---- read-back verification across the source boundary ----
    boundary = int(per_source_ends[0][-1])
    with h5py.File(main_out, "r") as m, h5py.File(side_out, "r") as s, \
            h5py.File(head_out, "r") as h:
        assert m["encoder"].shape[0] == total_rows
        assert m["image_bottom"].shape[0] == total_rows
        for key in SIDECAR_VIRTUAL:
            assert s[key].shape[0] == total_rows, key
        for key in HEAD_VIRTUAL:
            assert h[key].shape[0] == total_rows, key
        with h5py.File(mains[1], "r") as new_main:
            assert np.array_equal(m["encoder"][boundary:boundary + 4],
                                  new_main["encoder"][0:4])
            assert np.array_equal(m["image_bottom"][boundary],
                                  new_main["image_bottom"][0])
        for out, srcs, keys in ((s, sidecars, SIDECAR_VIRTUAL),
                                (h, heads, HEAD_VIRTUAL)):
            with h5py.File(srcs[1], "r") as new_src:
                for key in keys:
                    assert np.array_equal(out[key][boundary],
                                          new_src[key][0]), key
            with h5py.File(srcs[0], "r") as old_src:
                for key in keys:
                    assert np.array_equal(out[key][boundary - 1],
                                          old_src[key][boundary - 1]), key
    print(f"Verified VDS row alignment across the boundary at row {boundary}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="4f_hallway_150ep")
    ap.add_argument("--out-dir", default=str(REPO / "dataset"))
    a = ap.parse_args()
    build(a.name, Path(a.out_dir).expanduser().resolve())
