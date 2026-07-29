#!/usr/bin/env python3
"""Compile editable goal images into a ReLoc3R encoder-feature snapshot.

Only changed/new image hashes are encoded; unchanged features are copied from
the previous snapshot.  The output is written to a temporary HDF5 file and
atomically replaced, so training never observes a half-written pool.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reloc3r"))

from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model  # noqa: E402
from reloc3r.utils.image import ImgNorm, _resize_pil_image  # noqa: E402
from utils.goal_pool import GoalPool  # noqa: E402

N_PATCH = 196
FEAT_DIM = 1024


def _h5_string(values):
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def _feature_key(camera: str) -> str:
    tag = "".join(c if c.isalnum() else "_" for c in camera).strip("_")
    return f"reloc3r_{tag}"


def _preprocess(path: str, size: int = 224):
    with Image.open(path) as src:
        img = src.convert("RGB")
    w1, h1 = img.size
    img = _resize_pil_image(img, round(size * max(w1 / h1, h1 / w1)))
    w, h = img.size
    cx, cy = w // 2, h // 2
    half = min(cx, cy)
    img = img.crop((cx - half, cy - half, cx + half, cy + half))
    return ImgNorm(img), np.asarray(img.size[::-1], dtype=np.int64)


@torch.no_grad()
def _encode(model, paths, device, batch_size):
    output = np.empty((len(paths), N_PATCH, FEAT_DIM), dtype=np.float16)
    for start in range(0, len(paths), batch_size):
        stop = min(start + batch_size, len(paths))
        prepared = [_preprocess(path) for path in paths[start:stop]]
        images = torch.stack([item[0] for item in prepared]).to(device)
        shapes = torch.from_numpy(np.stack([item[1] for item in prepared])).to(device)
        feat, _pos, _ = model._encode_image(images, shapes)
        output[start:stop] = feat.float().cpu().numpy().astype(np.float16)
    return output


def _old_feature_map(path: Path, cameras):
    if not path.exists():
        return {}, None
    h = h5py.File(path, "r")
    try:
        ids = [str(x) for x in h["goal_id"].asstr()[:]]
        camera_names = [str(x) for x in h["camera"].asstr()[:]]
        if camera_names != list(cameras):
            h.close()
            return {}, None
        hashes = h["image_sha256"].asstr()[:]
        mapping = {
            (goal_id, camera, str(hashes[row, col])): row
            for row, goal_id in enumerate(ids)
            for col, camera in enumerate(cameras)
        }
        return mapping, h
    except Exception:
        h.close()
        raise


def compile_pool(args):
    cameras = list(args.camera)
    with GoalPool(args.db) as pool:
        records = pool.records(
            enabled=True,
            datasets=args.dataset,
            splits=args.split,
            variants=args.variant,
        )
    incomplete = [
        f"{rec.goal_id}: missing {sorted(set(cameras) - set(rec.images))}"
        for rec in records
        if not set(cameras).issubset(rec.images)
    ]
    if incomplete:
        raise ValueError(
            "enabled goals must contain every requested camera:\n" + "\n".join(incomplete)
        )
    if not records:
        raise ValueError("goal-pool filter selected zero enabled goals")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    old_map, old_h5 = _old_feature_map(out, cameras)
    device = torch.device(args.device)
    model = None

    fd, tmp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".h5", dir=str(out.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with h5py.File(tmp, "w") as dst:
            dst.attrs["format"] = "goal_pool_reloc3r_encoder_v1"
            dst.attrs["source_db"] = str(Path(args.db).expanduser().resolve())
            dst.attrs["reloc3r_checkpoint"] = "siyan824/reloc3r-224"
            dst.create_dataset("goal_id", data=_h5_string([r.goal_id for r in records]))
            dst.create_dataset("dataset", data=_h5_string([r.dataset for r in records]))
            dst.create_dataset("split", data=_h5_string([r.split for r in records]))
            dst.create_dataset("episode", data=np.asarray([r.episode for r in records], np.int64))
            dst.create_dataset("variant", data=_h5_string([r.variant for r in records]))
            dst.create_dataset("camera", data=_h5_string(cameras))
            hashes = np.asarray(
                [[r.images[c]["sha256"] for c in cameras] for r in records],
                dtype=h5py.string_dtype("utf-8"),
            )
            dst.create_dataset("image_sha256", data=hashes)

            for camera in cameras:
                key = _feature_key(camera)
                ds = dst.create_dataset(
                    key,
                    shape=(len(records), N_PATCH, FEAT_DIM),
                    dtype="float16",
                    chunks=(1, N_PATCH, FEAT_DIM),
                )
                pending_rows, pending_paths = [], []
                for row, rec in enumerate(records):
                    cache_key = (rec.goal_id, camera, rec.images[camera]["sha256"])
                    old_row = old_map.get(cache_key)
                    if old_row is not None and old_h5 is not None and key in old_h5:
                        ds[row] = old_h5[key][old_row]
                    else:
                        pending_rows.append(row)
                        pending_paths.append(rec.images[camera]["path"])
                if pending_rows:
                    if model is None:
                        model = setup_reloc3r_relpose_model("224", device)
                        model.eval()
                    for start in range(0, len(pending_rows), args.write_batch):
                        stop = min(start + args.write_batch, len(pending_rows))
                        feat = _encode(
                            model,
                            pending_paths[start:stop],
                            device,
                            args.encode_batch,
                        )
                        for local, row in enumerate(pending_rows[start:stop]):
                            ds[row] = feat[local]
                    print(
                        f"{camera}: encoded {len(pending_rows)}, reused "
                        f"{len(records) - len(pending_rows)}"
                    )
                else:
                    print(f"{camera}: reused all {len(records)} features")
        if old_h5 is not None:
            old_h5.close()
            old_h5 = None
        os.replace(tmp, out)
    finally:
        if old_h5 is not None:
            old_h5.close()
        if tmp.exists():
            tmp.unlink()
    print(f"Compiled {len(records)} paired goals -> {out}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--camera", action="append", required=True)
    p.add_argument("--dataset", action="append")
    p.add_argument("--split", action="append")
    p.add_argument("--variant", action="append")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--encode-batch", type=int, default=32)
    p.add_argument(
        "--write-batch",
        type=int,
        default=256,
        help="number of changed rows collected before writing (encoding still uses --encode-batch)",
    )
    return p


if __name__ == "__main__":
    compile_pool(parser().parse_args())
