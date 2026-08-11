#!/usr/bin/env python3
"""Create exhaustive JPG review pages and a CSV manifest for a goal pool."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.goal_pool import GoalPool  # noqa: E402

VARIANTS = ("original", "qr_removed", "color_changed")
CAMERAS = ("orbbec-0", "usb-0")


def build(args):
    out = Path(args.out).expanduser().resolve()
    pages = out / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    with GoalPool(args.db) as pool:
        records = pool.records(enabled=True)
    by_key = {(r.dataset, r.episode, r.variant): r for r in records}
    datasets = sorted({r.dataset for r in records})

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "goal_id", "dataset", "split", "episode", "variant", "camera",
            "path", "sha256", "width", "height", "parent_id",
        ])
        for rec in records:
            for camera in CAMERAS:
                info = rec.images[camera]
                writer.writerow([
                    rec.goal_id, rec.dataset, rec.split, rec.episode, rec.variant,
                    camera, info["path"], info["sha256"], info["width"],
                    info["height"], rec.parent_id or "",
                ])

    usb_mismatches = []
    counts = Counter()
    for dataset in datasets:
        episodes = sorted({r.episode for r in records if r.dataset == dataset})
        for episode in episodes:
            original = by_key[(dataset, episode, "original")]
            for variant in VARIANTS:
                rec = by_key[(dataset, episode, variant)]
                counts[(dataset, variant)] += 1
                if rec.images["usb-0"]["sha256"] != original.images["usb-0"]["sha256"]:
                    usb_mismatches.append(rec.goal_id)
        for page_index, start in enumerate(range(0, len(episodes), args.rows_per_page), 1):
            chunk = episodes[start:start + args.rows_per_page]
            label_h = 20
            sheet = Image.new(
                "RGB", (3 * 320, len(chunk) * (240 + label_h)), (20, 20, 20)
            )
            draw = ImageDraw.Draw(sheet)
            for row, episode in enumerate(chunk):
                y = row * (240 + label_h)
                for col, variant in enumerate(VARIANTS):
                    rec = by_key[(dataset, episode, variant)]
                    image = Image.open(rec.images["orbbec-0"]["path"]).convert("RGB")
                    sheet.paste(image, (col * 320, y))
                    draw.text(
                        (col * 320 + 4, y + 242),
                        f"{dataset} ep{episode:04d} {variant}",
                        fill="white",
                    )
            name = f"{dataset}_page_{page_index:03d}.jpg"
            sheet.save(pages / name, quality=92, subsampling=0)

    summary = {
        "enabled_goals": len(records),
        "image_files": len(records) * len(CAMERAS),
        "counts": {f"{d}/{v}": counts[(d, v)] for d in datasets for v in VARIANTS},
        "usb_hash_mismatches": usb_mismatches,
        "review_pages": len(list(pages.glob("*.jpg"))),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest: {manifest}")
    print(f"pages: {pages}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows-per-page", type=int, default=10)
    return p


if __name__ == "__main__":
    build(parser().parse_args())
