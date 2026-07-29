#!/usr/bin/env python3
"""Manage an editable, synchronized multi-camera goal-image pool.

Examples:
  python scripts/goal_pool.py init --db dataset/goal_pool/goal_pool.sqlite3

  python scripts/goal_pool.py import-h5 \
    --db dataset/goal_pool/goal_pool.sqlite3 \
    --h5 dataset/after_0328_train.h5 --dataset after_0328 --split train \
    --camera orbbec-0=image_bottom --camera usb-0=image_top

  python scripts/goal_pool.py add --db ... --dataset after_0328 --split train \
    --episode 0 --variant qr_removed \
    --camera orbbec-0=/path/orbbec.png --camera usb-0=/path/usb.png

Deletion is recoverable: ``remove`` disables an entry; ``restore`` enables it.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.goal_pool import GoalPool  # noqa: E402


def _mapping(values):
    out = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got '{value}'")
        key, val = value.split("=", 1)
        if not key or not val:
            raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got '{value}'")
        out[key] = val
    return out


def _save_h5_image(array, path: Path, jpeg_quality: int = 95):
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D image, got {arr.shape}")
    if arr.shape[0] in (1, 3, 4):
        arr = arr[:3].transpose(1, 2, 0)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected RGB/CHW image, got {arr.shape}")
    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(
            path,
            format="JPEG",
            quality=int(jpeg_quality),
            subsampling=0,
            optimize=True,
        )
    elif suffix == ".png":
        image.save(path, format="PNG")
    else:
        raise ValueError(f"unsupported extraction format: {path.suffix}")


def cmd_init(args):
    with GoalPool(args.db, create=True):
        pass
    print(f"Initialized goal pool: {Path(args.db).resolve()}")


def cmd_import_h5(args):
    cameras = _mapping(args.camera)
    with GoalPool(args.db, create=True) as pool, h5py.File(args.h5, "r") as f:
        if "episode_ends" not in f:
            raise KeyError(f"{args.h5}: missing episode_ends")
        for camera, source in cameras.items():
            if source not in f:
                raise KeyError(f"{args.h5}: missing {source} for camera {camera}")
        ends = f["episode_ends"][:].astype(int)
        imported = skipped = 0
        for source_episode, end in enumerate(ends):
            episode = int(args.episode_offset) + source_episode
            goal_id = pool.default_goal_id(args.dataset, args.split, episode, args.variant)
            if not args.replace and pool.conn.execute(
                "SELECT 1 FROM goals WHERE goal_id=?", (goal_id,)
            ).fetchone():
                skipped += 1
                continue
            with tempfile.TemporaryDirectory(prefix="goal_pool_import_") as td:
                paths = {}
                for camera, source in cameras.items():
                    path = Path(td) / f"{camera}.jpg"
                    _save_h5_image(
                        f[source][end - 1],
                        path,
                        jpeg_quality=args.jpeg_quality,
                    )
                    paths[camera] = path
                pool.upsert(
                    goal_id=goal_id,
                    dataset=args.dataset,
                    split=args.split,
                    episode=episode,
                    variant=args.variant,
                    images=paths,
                    enabled=True,
                    note=args.note,
                    metadata={
                        "source_h5": os.path.abspath(args.h5),
                        "source_episode": int(source_episode),
                        "source_row": int(end - 1),
                        "source_keys": cameras,
                    },
                    replace=args.replace,
                )
            imported += 1
    print(f"Imported {imported} paired goals; skipped {skipped} existing entries")


def cmd_add(args):
    images = _mapping(args.camera)
    goal_id = args.goal_id or GoalPool.default_goal_id(
        args.dataset, args.split, args.episode, args.variant
    )
    with GoalPool(args.db, create=True) as pool:
        pool.upsert(
            goal_id=goal_id,
            dataset=args.dataset,
            split=args.split,
            episode=args.episode,
            variant=args.variant,
            images=images,
            enabled=not args.disabled,
            parent_id=args.parent_id,
            note=args.note,
            replace=args.replace,
        )
    print(goal_id)


def cmd_update(args):
    images = _mapping(args.camera)
    if not images:
        raise ValueError("update requires at least one --camera NAME=PATH")
    with GoalPool(args.db) as pool:
        pool.update_images(args.goal_id, images)
    print(f"Updated {args.goal_id}")


def cmd_set_enabled(args, enabled):
    with GoalPool(args.db) as pool:
        pool.set_enabled(args.goal_id, enabled)
    print(f"{'Enabled' if enabled else 'Disabled'} {args.goal_id}")


def cmd_list(args):
    with GoalPool(args.db) as pool:
        rows = pool.records(
            enabled=True if args.enabled_only else None,
            datasets=args.dataset,
            splits=args.split,
            variants=args.variant,
        )
    for rec in rows:
        cams = ",".join(sorted(rec.images))
        state = "enabled" if rec.enabled else "disabled"
        print(
            f"{rec.goal_id}\t{state}\t{rec.dataset}\t{rec.split}\t"
            f"ep={rec.episode}\t{rec.variant}\t{cams}"
        )
    print(f"Total: {len(rows)}")


def cmd_validate(args):
    with GoalPool(args.db) as pool:
        errors = pool.validate(args.require_camera)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("Goal pool is valid")


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("init")
    q.add_argument("--db", required=True)
    q.set_defaults(func=cmd_init)

    q = sub.add_parser("import-h5")
    q.add_argument("--db", required=True)
    q.add_argument("--h5", required=True)
    q.add_argument("--dataset", required=True)
    q.add_argument("--split", required=True)
    q.add_argument("--variant", default="original")
    q.add_argument("--camera", action="append", required=True, metavar="NAME=H5_KEY")
    q.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="add this offset to source-local episode ids (for merged splits)",
    )
    q.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="[1-100]",
    )
    q.add_argument("--note", default="")
    q.add_argument("--replace", action="store_true")
    q.set_defaults(func=cmd_import_h5)

    q = sub.add_parser("add")
    q.add_argument("--db", required=True)
    q.add_argument("--goal-id")
    q.add_argument("--dataset", required=True)
    q.add_argument("--split", required=True)
    q.add_argument("--episode", required=True, type=int)
    q.add_argument("--variant", required=True)
    q.add_argument("--camera", action="append", required=True, metavar="NAME=PATH")
    q.add_argument("--parent-id")
    q.add_argument("--note", default="")
    q.add_argument("--disabled", action="store_true")
    q.add_argument("--replace", action="store_true")
    q.set_defaults(func=cmd_add)

    q = sub.add_parser("update")
    q.add_argument("--db", required=True)
    q.add_argument("--goal-id", required=True)
    q.add_argument("--camera", action="append", required=True, metavar="NAME=PATH")
    q.set_defaults(func=cmd_update)

    for name, enabled in (("remove", False), ("restore", True)):
        q = sub.add_parser(name)
        q.add_argument("--db", required=True)
        q.add_argument("--goal-id", required=True)
        q.set_defaults(func=lambda args, flag=enabled: cmd_set_enabled(args, flag))

    q = sub.add_parser("list")
    q.add_argument("--db", required=True)
    q.add_argument("--enabled-only", action="store_true")
    q.add_argument("--dataset", action="append")
    q.add_argument("--split", action="append")
    q.add_argument("--variant", action="append")
    q.set_defaults(func=cmd_list)

    q = sub.add_parser("validate")
    q.add_argument("--db", required=True)
    q.add_argument("--require-camera", action="append", default=[])
    q.set_defaults(func=cmd_validate)
    return p


if __name__ == "__main__":
    ns = parser().parse_args()
    ns.func(ns)
