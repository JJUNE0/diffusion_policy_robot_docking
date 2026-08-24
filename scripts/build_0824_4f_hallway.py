#!/usr/bin/env python3.12
"""Convert the raw `dataset/0824_4f_hallway/*.zip` exports into the episode
layout `utils/preprocessing.py` expects.

Difference from scripts/build_front_dock_5f.py, which handles the 5th-floor
zips: these records carry `"segments": []` / `"split_by_segment": false`, so the
sensor CSV/JSONL sit at the ARCHIVE ROOT rather than under `segment_NN/`, and
there is no marker to trim the approach phase against -- the whole record is the
episode.

Camera mapping (verified by comparing extracted frames against
`after_0328_train.h5`'s own `image_bottom`/`image_top` rows):

    camera_orbbec-0 -> image/room2 == image_bottom   (dock-facing, the one the
                                                      r_relfeat arms train on)
    camera_usb-0    -> image/room1 == image_top      (upward view; only written
                                                      when --room1_camera given)

Timestamps: the exporter writes NANOSECONDS (monotonic) everywhere while the
downstream pipeline assumes SECONDS (`utils/preprocessing.py` builds its sync
grid with `target_interval = 1/target_hz` and gates matches with
`max_time_diff=0.05`). So every ts written here -- CSV column, JSONL field and
the `_<ts>.jpg` filename suffix -- is converted to seconds.

Output (per episode, matching `_collect_episode_folders`'s `dock/` glob):

    <out>/dock/episode_<rec>_dock/
        encoder.csv          ts,vx,wz   (sorted, deduped, seconds)
        lidar.jsonl          (seconds; unused unless preprocessing --use_lidar)
        command.csv          (kept for reference; unused by preprocessing)
        meta.json            provenance: zip, record_idx, labels, frame counts
        image/room2/*.jpg    every camera_orbbec-0 frame

Usage:
    python scripts/build_0824_4f_hallway.py \
        --zip_dir dataset/0824_4f_hallway \
        --out dataset/0824_4f_hallway_episodes
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import zipfile

FRAME_RE = re.compile(r"^(frame_\d+)_(\d+)\.jpg$")

# Exporter clock -> pipeline clock. See the module docstring.
NS_PER_S = 1e9


def to_seconds(ts_ns) -> float:
    return float(ts_ns) / NS_PER_S


def fmt_seconds(ts_s: float) -> str:
    """9 decimals keeps full ns resolution and round-trips through float()."""
    return f"{ts_s:.9f}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip_dir", default="dataset/0824_4f_hallway")
    p.add_argument("--out", default="dataset/0824_4f_hallway_episodes")
    p.add_argument("--room2_camera", default="camera_orbbec-0",
                   help="camera exported as image/room2 (= image_bottom, the dock-facing view)")
    p.add_argument("--room1_camera", default=None,
                   help="optional second camera exported as image/room1 (use_room1=true only)")
    p.add_argument("--require_success", action="store_true", default=True,
                   help="skip records whose metadata labels.success is not true")
    p.add_argument("--allow_failures", dest="require_success", action="store_false",
                   help="keep records regardless of labels.success")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild episodes that already exist in --out")
    return p.parse_args()


def records_in_zip(zf: zipfile.ZipFile):
    """Yield (prefix, metadata) for every record packed in one zip.

    Anchoring on `metadata.json` handles both a single record at the archive
    root (prefix "") and a nested `<session>/record_NNNNN/` layout.
    """
    for name in sorted(zf.namelist()):
        if os.path.basename(name) == "metadata.json" and name.count("/") <= 2:
            prefix = name[: -len("metadata.json")]
            yield prefix, json.loads(zf.read(name))


def copy_csv(zf, prefix, fname, out_path):
    """Copy a root-level CSV, sort by ts, drop duplicate ts, ns -> s.

    Returns the number of rows written, or 0 when the CSV is absent/empty.
    """
    try:
        raw = zf.read(f"{prefix}{fname}").decode("utf-8")
    except KeyError:
        return 0
    rows = [r for r in csv.reader(io.StringIO(raw)) if r]
    if len(rows) < 2:
        return 0
    header, body = rows[0], rows[1:]
    ts_col = header.index("ts")
    body.sort(key=lambda r: int(r[ts_col]))
    dedup, seen = [], set()
    for r in body:
        t = int(r[ts_col])
        if t in seen:
            continue
        seen.add(t)
        r[ts_col] = fmt_seconds(to_seconds(t))
        dedup.append(r)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(dedup)
    return len(dedup)


def copy_jsonl(zf, prefix, fname, out_path):
    """Copy a root-level JSONL sorted by `ts`, ns -> s. Returns rows written."""
    try:
        raw = zf.read(f"{prefix}{fname}").decode("utf-8")
    except KeyError:
        return 0
    recs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            d["ts"] = to_seconds(d["ts"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        recs.append(d)
    if not recs:
        return 0
    recs.sort(key=lambda d: d["ts"])
    with open(out_path, "w") as f:
        for d in recs:
            f.write(json.dumps(d) + "\n")
    return len(recs)


def extract_frames(zf, prefix, camera, out_dir):
    """Copy every `camera` jpg out of the zip, renaming the filename's ns suffix
    to seconds (`_get_image_timestamps` parses it as the frame's ts). Returns
    the count."""
    os.makedirs(out_dir, exist_ok=True)
    pre = f"{prefix}{camera}/frames/"
    n = 0
    for name in zf.namelist():
        if not name.startswith(pre) or not name.endswith(".jpg"):
            continue
        m = FRAME_RE.match(os.path.basename(name))
        if m is None:
            continue
        out_name = f"{m.group(1)}_{fmt_seconds(to_seconds(m.group(2)))}.jpg"
        with zf.open(name) as src, open(os.path.join(out_dir, out_name), "wb") as dst:
            shutil.copyfileobj(src, dst)
        n += 1
    return n


def build_record(zf, zip_name, prefix, meta, args, out_root):
    rec = str(meta.get("record_idx", "")).strip()
    ep_name = f"episode_{rec}_dock"
    ep_dir = os.path.join(out_root, "dock", ep_name)

    labels = meta.get("labels") or {}
    if args.require_success and labels.get("success") is not True:
        return ep_name, f"skip: labels.success={labels.get('success')!r}"

    if os.path.isdir(ep_dir):
        if not args.overwrite:
            return ep_name, "skip: already built"
        shutil.rmtree(ep_dir)
    os.makedirs(ep_dir, exist_ok=True)

    n_enc = copy_csv(zf, prefix, "encoder.csv", os.path.join(ep_dir, "encoder.csv"))
    n_lid = copy_jsonl(zf, prefix, "lidar.jsonl", os.path.join(ep_dir, "lidar.jsonl"))
    copy_csv(zf, prefix, "command.csv", os.path.join(ep_dir, "command.csv"))

    if n_enc == 0:
        shutil.rmtree(ep_dir)
        return ep_name, "skip: empty/missing encoder.csv"

    n2 = extract_frames(zf, prefix, args.room2_camera, os.path.join(ep_dir, "image", "room2"))
    n1 = 0
    if args.room1_camera:
        n1 = extract_frames(zf, prefix, args.room1_camera, os.path.join(ep_dir, "image", "room1"))

    if n2 == 0:
        shutil.rmtree(ep_dir)
        return ep_name, f"skip: no {args.room2_camera} frames"

    json.dump(
        {
            "zip": zip_name,
            "prefix": prefix,
            "record_idx": rec,
            "labels": labels,
            "duration_sec": meta.get("duration_sec"),
            "room2_camera": args.room2_camera,
            "room1_camera": args.room1_camera,
            "n_encoder_rows": n_enc,
            "n_lidar_rows": n_lid,
            "n_frames_room2": n2,
            "n_frames_room1": n1,
        },
        open(os.path.join(ep_dir, "meta.json"), "w"),
        indent=2,
        ensure_ascii=False,
    )
    return ep_name, f"ok: enc={n_enc} lidar={n_lid} room2_frames={n2}" + (
        f" room1_frames={n1}" if args.room1_camera else "")


def main():
    args = parse_args()
    out_root = args.out
    os.makedirs(os.path.join(out_root, "dock"), exist_ok=True)

    zips = sorted(f for f in os.listdir(args.zip_dir) if f.endswith(".zip"))
    n_ok = n_skip = 0
    for zname in zips:
        with zipfile.ZipFile(os.path.join(args.zip_dir, zname)) as zf:
            for prefix, meta in records_in_zip(zf):
                ep, status = build_record(zf, zname, prefix, meta, args, out_root)
                if status.startswith("ok"):
                    n_ok += 1
                else:
                    n_skip += 1
                print(f"[{zname}] {ep}: {status}", flush=True)

    print(f"\nbuilt {n_ok} episodes, skipped {n_skip} -> {out_root}/dock")


if __name__ == "__main__":
    sys.exit(main())
