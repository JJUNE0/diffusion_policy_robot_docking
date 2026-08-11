#!/usr/bin/env python3
"""Build one no-split docking dataset from multiple HDF5 files.

Large camera and ReLoc3R arrays are exposed through HDF5 Virtual Datasets, so
combining sources or excluding a bad episode does not duplicate their bytes.
Input ordering defines the merged per-dataset episode numbering.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


def _strings(values: Iterable[str]):
    return np.asarray(list(values), dtype=h5py.string_dtype("utf-8"))


def _open_all(paths: Sequence[Path]):
    return [h5py.File(path, "r") for path in paths]


def _common_row_datasets(files):
    common = set(files[0].keys())
    for h5 in files[1:]:
        common &= set(h5.keys())
    common.discard("episode_ends")
    rows = [int(h5["episode_ends"][-1]) for h5 in files]
    out = []
    for key in sorted(common):
        datasets = [h5[key] for h5 in files]
        if not all(isinstance(ds, h5py.Dataset) and ds.ndim >= 1 for ds in datasets):
            continue
        if not all(ds.shape[0] == n for ds, n in zip(datasets, rows)):
            continue
        if not all(ds.shape[1:] == datasets[0].shape[1:] for ds in datasets):
            continue
        if not all(ds.dtype == datasets[0].dtype for ds in datasets):
            continue
        out.append(key)
    return out


def _parse_excludes(values):
    out = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"expected DATASET:EPISODE, got {value!r}")
        dataset, episode = value.rsplit(":", 1)
        out.add((dataset, int(episode)))
    return out


def _segments(main_paths, dataset_names, excludes):
    merged_episode = Counter()
    segments = []
    found_excludes = set()
    for file_index, (path, dataset) in enumerate(zip(main_paths, dataset_names)):
        with h5py.File(path, "r") as h5:
            ends = np.asarray(h5["episode_ends"][:], dtype=np.int64)
        start = 0
        for source_episode, end in enumerate(ends):
            dataset_episode = merged_episode[dataset]
            merged_episode[dataset] += 1
            key = (dataset, dataset_episode)
            if key in excludes:
                found_excludes.add(key)
            else:
                segments.append({
                    "file_index": file_index,
                    "dataset": dataset,
                    "dataset_episode": dataset_episode,
                    "source_episode": source_episode,
                    "start": int(start),
                    "end": int(end),
                })
            start = int(end)
    missing = excludes - found_excludes
    if missing:
        raise ValueError(f"excluded episodes not found: {sorted(missing)}")
    return segments


def _coalesced_segments(segments):
    """Merge adjacent episodes from one source into minimal VDS mappings."""
    out = []
    for segment in segments:
        segment = dict(segment)
        if (
            out
            and out[-1]["file_index"] == segment["file_index"]
            and out[-1]["end"] == segment["start"]
        ):
            out[-1]["end"] = segment["end"]
        else:
            out.append(segment)
    return out


def _write_vds(output, sources, segments, dataset_names=None):
    handles = _open_all(sources)
    try:
        keys = list(dataset_names or _common_row_datasets(handles))
        total_rows = sum(seg["end"] - seg["start"] for seg in segments)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=output.name + ".", suffix=".h5", dir=str(output.parent)
        )
        os.close(fd)
        temp = Path(temp_name)
        try:
            with h5py.File(temp, "w", libver="latest") as dst:
                dst.attrs["format"] = "combined_docking_vds_v2"
                dst.create_dataset(
                    "source_file", data=_strings(str(path.resolve()) for path in sources)
                )
                for key in keys:
                    template = handles[0][key]
                    layout = h5py.VirtualLayout(
                        shape=(total_rows, *template.shape[1:]), dtype=template.dtype
                    )
                    cursor = 0
                    virtual_sources = [
                        h5py.VirtualSource(
                            str(path.resolve()), key, shape=handles[i][key].shape
                        )
                        for i, path in enumerate(sources)
                    ]
                    for seg in _coalesced_segments(segments):
                        count = seg["end"] - seg["start"]
                        source = virtual_sources[seg["file_index"]]
                        layout[cursor:cursor + count] = source[seg["start"]:seg["end"]]
                        cursor += count
                    dst.create_virtual_dataset(key, layout)
            os.replace(temp, output)
        finally:
            if temp.exists():
                temp.unlink()
    finally:
        for h5 in handles:
            h5.close()
    return keys, total_rows


def build(args):
    main = [Path(path).expanduser().resolve() for path in args.main]
    bottom = [Path(path).expanduser().resolve() for path in args.bottom]
    top = [Path(path).expanduser().resolve() for path in args.top]
    if not (len(main) == len(bottom) == len(top) == len(args.dataset)):
        raise ValueError("--main/--bottom/--top/--dataset counts must match")
    for path in (*main, *bottom, *top):
        if not path.is_file():
            raise FileNotFoundError(path)

    excludes = _parse_excludes(args.exclude)
    segments = _segments(main, args.dataset, excludes)
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_out = output_dir / f"{args.name}.h5"
    bottom_out = output_dir / f"{args.name}_reloc3r_bottom.h5"
    top_out = output_dir / f"{args.name}_reloc3r_top.h5"

    main_keys, total_rows = _write_vds(main_out, main, segments)
    bottom_keys, bottom_rows = _write_vds(
        bottom_out, bottom, segments, dataset_names=["reloc3r_bottom"]
    )
    top_keys, top_rows = _write_vds(
        top_out, top, segments, dataset_names=["reloc3r_top"]
    )
    if not (total_rows == bottom_rows == top_rows):
        raise ValueError("main/bottom/top combined row counts differ")

    episode_ends = []
    source_dataset = []
    source_file_index = []
    source_episode = []
    source_dataset_episode = []
    cursor = 0
    for seg in segments:
        cursor += seg["end"] - seg["start"]
        episode_ends.append(cursor)
        source_dataset.append(seg["dataset"])
        source_file_index.append(seg["file_index"])
        source_episode.append(seg["source_episode"])
        source_dataset_episode.append(seg["dataset_episode"])

    with h5py.File(main_out, "r+") as dst:
        dst.create_dataset("episode_ends", data=np.asarray(episode_ends, np.int64))
        dst.create_dataset("source_dataset", data=_strings(source_dataset))
        dst.create_dataset("source_file_index", data=np.asarray(source_file_index, np.int32))
        dst.create_dataset("source_episode", data=np.asarray(source_episode, np.int32))
        dst.create_dataset(
            "source_dataset_episode", data=np.asarray(source_dataset_episode, np.int32)
        )
        dst.attrs["num_episodes"] = len(segments)
        dst.attrs["num_rows"] = total_rows
        dst.attrs["excluded_episodes"] = json.dumps(sorted(excludes))
    for output in (bottom_out, top_out):
        with h5py.File(output, "r+") as dst:
            dst.create_dataset("episode_ends", data=np.asarray(episode_ends, np.int64))
            dst.attrs["num_episodes"] = len(segments)
            dst.attrs["num_rows"] = total_rows
            dst.attrs["excluded_episodes"] = json.dumps(sorted(excludes))

    print(
        f"Built {len(segments)} episodes / {total_rows} rows; excluded={sorted(excludes)}\n"
        f"  main:   {main_out} ({', '.join(main_keys)})\n"
        f"  bottom: {bottom_out} ({', '.join(bottom_keys)})\n"
        f"  top:    {top_out} ({', '.join(top_keys)})"
    )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="all_combined")
    p.add_argument("--main", action="append", required=True)
    p.add_argument("--bottom", action="append", required=True)
    p.add_argument("--top", action="append", required=True)
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument(
        "--exclude", action="append", default=[], metavar="DATASET:EPISODE"
    )
    return p


if __name__ == "__main__":
    build(parser().parse_args())
