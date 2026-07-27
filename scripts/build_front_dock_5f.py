"""Convert the raw `dataset/front_dock_5th_floor/*.zip` exports into the episode
layout `utils/preprocessing.py` expects.

Two rules come straight from `dataset/front_dock_5th_floor/readme.md`:

  1. only records whose `metadata.json` has `labels.success == true` are used;
  2. only the part of the record AFTER the segment marker is used — `segment_01`
     is the drive-up to the docking start pose, not the docking behaviour we
     want to imitate. Concretely we keep `segment_02` and onward (a record with
     3 segments has two markers; everything after the FIRST one is docking).

The sensor CSV/JSONL are already split per segment by the exporter, so the
segment filter is just "concatenate segment_02..N". The camera frames are NOT
split, so they are filtered by timestamp (frame filenames carry the same
nanosecond monotonic clock as encoder/lidar `ts`).

Camera mapping: `camera_orbbec-0` is the forward camera that sees the dock =
`room2` (`image_bottom`), the single camera the current recipe trains on
(`use_room1: false`). `--room1_camera` can add a second camera if wanted.

Output (per episode, named `episode_<record_idx>_dock` so that
`scripts/label_subgoals.py`'s glob picks it up):

    <out>/dock/episode_<rec>_dock/
        encoder.csv          ts,vx,wz   (segment_02..N, sorted, deduped)
        lidar.jsonl          (segment_02..N)
        command.csv          (kept for reference; unused by preprocessing)
        meta.json            provenance: zip, prefix, segments, t0
        image/room2/*.jpg    frames with ts >= t0

Usage:
    python scripts/build_front_dock_5f.py \
        --zip_dir dataset/front_dock_5th_floor \
        --out dataset/front_dock_5f
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

# First segment index to keep. segment_01 = approach to the docking start pose.
FIRST_DOCK_SEGMENT = 2

FRAME_TS_RE = re.compile(r"_(\d+)\.jpg$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip_dir", default="dataset/front_dock_5th_floor")
    p.add_argument("--out", default="dataset/front_dock_5f")
    p.add_argument("--room2_camera", default="camera_orbbec-0",
                   help="camera exported as image/room2 (= image_bottom, the dock-facing view)")
    p.add_argument("--room1_camera", default=None,
                   help="optional second camera exported as image/room1 (use_room1=true only)")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild episodes that already exist in --out")
    return p.parse_args()


def records_in_zip(zf: zipfile.ZipFile):
    """Yield (prefix, metadata) for every record packed in one zip.

    Zips are inconsistent: some hold a single record at the archive root
    (prefix ""), some a flat list of `export_NNN/`, some a nested
    `0713_5th_floor/record_NNNNN/`. Anchoring on `metadata.json` handles all.
    """
    for name in sorted(zf.namelist()):
        if os.path.basename(name) == "metadata.json" and name.count("/") <= 2:
            prefix = name[: -len("metadata.json")]
            yield prefix, json.loads(zf.read(name))


def dock_segments(names, prefix):
    """Sorted `segment_NN` dir names at/after FIRST_DOCK_SEGMENT."""
    segs = set()
    for n in names:
        if not n.startswith(prefix):
            continue
        head = n[len(prefix):].split("/")[0]
        m = re.fullmatch(r"segment_(\d+)", head)
        if m and int(m.group(1)) >= FIRST_DOCK_SEGMENT:
            segs.add(head)
    return sorted(segs)


def read_csv_rows(zf, path):
    """Return (header, rows) for a CSV inside the zip; ([], []) if absent."""
    try:
        raw = zf.read(path).decode("utf-8")
    except KeyError:
        return [], []
    rdr = csv.reader(io.StringIO(raw))
    rows = [r for r in rdr if r]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def concat_csv(zf, prefix, segs, fname, out_path):
    """Concatenate a per-segment CSV, sort by ts, drop duplicate ts.

    Returns the first ts (int) or None when there is no data.
    """
    header, rows = [], []
    for s in segs:
        h, r = read_csv_rows(zf, f"{prefix}{s}/{fname}")
        if h:
            header = h
            rows.extend(r)
    if not header or not rows:
        return None
    ts_col = header.index("ts")
    rows.sort(key=lambda r: int(r[ts_col]))
    dedup, seen = [], set()
    for r in rows:
        t = int(r[ts_col])
        if t in seen:
            continue
        seen.add(t)
        dedup.append(r)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(dedup)
    return int(dedup[0][ts_col])


def concat_jsonl(zf, prefix, segs, fname, out_path):
    """Concatenate per-segment JSONL, sorted by `ts`. Returns first ts or None."""
    lines = []
    for s in segs:
        try:
            raw = zf.read(f"{prefix}{s}/{fname}").decode("utf-8")
        except KeyError:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ts = json.loads(line)["ts"]
            except (json.JSONDecodeError, KeyError):
                continue
            lines.append((float(ts), line))
    if not lines:
        return None
    lines.sort(key=lambda x: x[0])
    with open(out_path, "w") as f:
        for _, line in lines:
            f.write(line + "\n")
    return lines[0][0]


def extract_frames(zf, prefix, camera, t0, out_dir):
    """Copy `camera` jpgs with timestamp >= t0 out of the zip. Returns count."""
    os.makedirs(out_dir, exist_ok=True)
    pre = f"{prefix}{camera}/frames/"
    n = 0
    for name in zf.namelist():
        if not name.startswith(pre) or not name.endswith(".jpg"):
            continue
        m = FRAME_TS_RE.search(name)
        if not m or float(m.group(1)) < t0:
            continue
        with zf.open(name) as src, open(os.path.join(out_dir, os.path.basename(name)), "wb") as dst:
            shutil.copyfileobj(src, dst)
        n += 1
    return n


def build_record(zf, zip_name, prefix, meta, args, out_root):
    rec = str(meta.get("record_idx", "")).strip()
    ep_name = f"episode_{rec}_dock"
    ep_dir = os.path.join(out_root, "dock", ep_name)

    success = (meta.get("labels") or {}).get("success")
    if success is not True:
        return ep_name, f"skip: labels.success={success!r}"

    segs = dock_segments(zf.namelist(), prefix)
    if not segs:
        return ep_name, f"skip: no segment_{FIRST_DOCK_SEGMENT:02d}+ (marker missing)"

    if os.path.isdir(ep_dir):
        if not args.overwrite:
            return ep_name, "skip: already built"
        shutil.rmtree(ep_dir)
    os.makedirs(ep_dir, exist_ok=True)

    enc_t0 = concat_csv(zf, prefix, segs, "encoder.csv", os.path.join(ep_dir, "encoder.csv"))
    lid_t0 = concat_jsonl(zf, prefix, segs, "lidar.jsonl", os.path.join(ep_dir, "lidar.jsonl"))
    concat_csv(zf, prefix, segs, "command.csv", os.path.join(ep_dir, "command.csv"))

    if enc_t0 is None or lid_t0 is None:
        shutil.rmtree(ep_dir)
        return ep_name, f"skip: empty encoder({enc_t0}) or lidar({lid_t0}) after the marker"

    # The marker time itself is only stored as wall-clock ISO, while frames /
    # encoder / lidar share a monotonic ns clock. The earliest post-marker
    # sensor sample is therefore the marker in that clock.
    t0 = min(float(enc_t0), float(lid_t0))

    n2 = extract_frames(zf, prefix, args.room2_camera, t0, os.path.join(ep_dir, "image", "room2"))
    n1 = 0
    if args.room1_camera:
        n1 = extract_frames(zf, prefix, args.room1_camera, t0, os.path.join(ep_dir, "image", "room1"))

    if n2 == 0:
        shutil.rmtree(ep_dir)
        return ep_name, f"skip: no {args.room2_camera} frames after t0"

    json.dump(
        {
            "zip": zip_name,
            "prefix": prefix,
            "record_idx": rec,
            "segments_used": segs,
            "t0_ns": t0,
            "duration_sec": meta.get("duration_sec"),
            "room2_camera": args.room2_camera,
            "room1_camera": args.room1_camera,
            "n_frames_room2": n2,
            "n_frames_room1": n1,
        },
        open(os.path.join(ep_dir, "meta.json"), "w"),
        indent=2,
    )
    return ep_name, f"ok: segs={','.join(s[-2:] for s in segs)} room2_frames={n2}"


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
