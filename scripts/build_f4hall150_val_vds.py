#!/usr/bin/env python3.12
"""Publish floor4_hallway_front_docking.h5's HELD-OUT episodes (151-154) as an
evaluation h5 -- the 4 episodes that scripts/build_f4hall150_vds.py left out of
the 150-episode training split.

  dataset/f4hall150_val.h5
      episode_ends  (4,)        re-based to 0 for this file
      encoder       (7393, 2)   materialized
      image_bottom  VDS -> source:image_bottom[251017:258410]

Names must line up with the training pair: test/eval_run_rgeo.py rewrites the
checkpoint's sensor `file` by swapping the STATS_H5 stem for the EVAL_H5 stem,
so `f4hall150_train_reloc3r_bottom.h5` -> `f4hall150_val_reloc3r_bottom.h5`.
Build that sidecar afterwards with the ordinary two-stage precompute; at 7393
rows it is a couple of minutes, so nothing is borrowed here (jiwon's encoder
cache stops at episode 153 anyway and would not cover 154).

House split note: the source documents "episodes 1-150 train; 151-153
validation; 154+ excluded". All four un-trained episodes are published here;
the evaluator reports 151-153 and 154 separately so the house split stays
readable.

Usage:
    python3.12 scripts/build_f4hall150_val_vds.py
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
SOURCE = Path("/home/work/.postech/jiwon/floor4_hallway_front_docking.h5")

FIRST_VAL_EPISODE = 150      # 0-indexed -> episode_151
LAST_VAL_EPISODE = 154       # 0-indexed exclusive
VIRTUAL_MAP = {"image_bottom": "image_bottom"}
MATERIAL_KEYS = ["encoder"]

# Same pinned dock fingerprint as the training build, so a source that gets
# swapped in place fails here too instead of evaluating against a blank wall.
DOCK_FINGERPRINT_ROWS = [0, 50000, 120000, 210190, 251016]
DOCK_FINGERPRINT_SHA256 = (
    "4caf1c950cee79a8cc07689bfe99a3e84562656dedf5eaefc548532d4809fb2f")


def _fingerprint(ds, rows):
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(np.ascontiguousarray(ds[r]).tobytes())
    return h.hexdigest()


def _grad_energy(chw):
    g = chw.astype(np.float32).mean(0)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def _strings(values):
    return np.asarray(list(values), dtype=h5py.string_dtype("utf-8"))


def build(name, out_dir, source, first_ep, last_ep):
    if not source.is_file():
        raise FileNotFoundError(source)

    with h5py.File(source, "r") as src:
        all_ends = np.asarray(src["episode_ends"][:], dtype=np.int64)
        if last_ep > len(all_ends):
            raise ValueError(f"{source}: only {len(all_ends)} episodes")
        row0 = int(all_ends[first_ep - 1])
        row1 = int(all_ends[last_ep - 1])
        n_rows = row1 - row0
        # episode_ends rebased so this file stands alone
        episode_ends = (all_ends[first_ep:last_ep] - row0).astype(np.int64)

        dock_key = VIRTUAL_MAP["image_bottom"]
        got = _fingerprint(src[dock_key], DOCK_FINGERPRINT_ROWS)
        if got != DOCK_FINGERPRINT_SHA256:
            raise ValueError(
                f"{source}:{dock_key} does not match the pinned dock fingerprint "
                f"(got {got[:16]}...). The source's cameras were swapped in place; "
                f"re-verify before evaluating.")
        # the fingerprint rows all sit in the TRAINING range, so also check the
        # held-out range directly: the dock view is the structured one.
        for epi in range(first_ep, last_ep):
            r = int(all_ends[epi]) - 1
            eb, et = _grad_energy(src[dock_key][r]), _grad_energy(src["image_top"][r])
            if eb <= et:
                raise ValueError(
                    f"episode {epi + 1} (row {r}): image_bottom is the FLATTER view "
                    f"(grad {eb:.2f} vs image_top {et:.2f}) -- not the dock camera.")

        img_shape, img_dtype = src[dock_key].shape, src[dock_key].dtype
        names = src["episode_names"].asstr()[first_ep:last_ep]
        src_attrs = {k: str(v) for k, v in src.attrs.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.h5"

    fd, tmp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".h5",
                                    dir=str(out_dir))
    os.close(fd)
    tmp_out = Path(tmp_name)

    with h5py.File(tmp_out, "w", libver="latest") as dst:
        dst.attrs["format"] = "f4hall150_val_vds_v1"
        dst.attrs["num_episodes"] = last_ep - first_ep
        dst.attrs["num_rows"] = n_rows
        dst.attrs["source_file"] = str(source)
        dst.attrs["source_row_range"] = json.dumps([row0, row1])
        dst.attrs["source_episode_range_1based"] = json.dumps([first_ep + 1, last_ep])
        dst.attrs["image_bottom_source"] = "camera_orbbec-0"
        dst.attrs["split_note"] = (
            "Held-out episodes 151-154 -- everything build_f4hall150_vds.py left "
            "out of the 150-episode training split. The source documents "
            "151-153 as validation and 154+ as excluded.")
        dst.attrs["source_attrs"] = json.dumps(src_attrs, ensure_ascii=False)
        dst.create_dataset("episode_ends", data=episode_ends)

        with h5py.File(source, "r") as src:
            for key in MATERIAL_KEYS:
                dst.create_dataset(key, data=src[key][row0:row1])
            for pub, key in VIRTUAL_MAP.items():
                layout = h5py.VirtualLayout(shape=(n_rows, *img_shape[1:]),
                                            dtype=img_dtype)
                vsrc = h5py.VirtualSource(str(source), key, shape=img_shape)
                layout[0:n_rows] = vsrc[row0:row1]
                dst.create_virtual_dataset(pub, layout)
            dst.create_dataset("source_episode_name", data=_strings(names))
            dst.create_dataset("source_row_offset", data=np.int64(row0))

    os.replace(tmp_out, out_path)
    print(f"Built episodes {first_ep + 1}-{last_ep} / {n_rows} rows -> {out_path}")
    for i, (nm, end) in enumerate(zip(names, episode_ends)):
        start = 0 if i == 0 else int(episode_ends[i - 1])
        print(f"  [{i}] {nm}: local rows [{start}:{int(end)}] len={int(end) - start}")

    with h5py.File(out_path, "r") as m, h5py.File(source, "r") as src:
        assert m["encoder"].shape[0] == n_rows
        assert m["image_bottom"].shape[0] == n_rows
        assert int(m["episode_ends"][-1]) == n_rows
        for local in (0, n_rows // 2, n_rows - 1):
            assert np.array_equal(m["image_bottom"][local],
                                  src["image_bottom"][row0 + local]), local
            assert np.array_equal(m["encoder"][local], src["encoder"][row0 + local]), local
    print("Verified: rows aligned to source[%d:%d], image_bottom is the dock camera."
          % (row0, row1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="f4hall150_val")
    ap.add_argument("--out-dir", default=str(REPO / "dataset"))
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--first-episode", type=int, default=FIRST_VAL_EPISODE)
    ap.add_argument("--last-episode", type=int, default=LAST_VAL_EPISODE)
    a = ap.parse_args()
    build(a.name, Path(a.out_dir).expanduser().resolve(),
          Path(a.source).expanduser().resolve(), a.first_episode, a.last_episode)
