#!/usr/bin/env python3
"""Pin one HDF5 dataset in a process-owned memfd for shared read-only mapping.

The resulting ``mmap_path`` in the metadata JSON can be opened by unrelated
processes owned by the same user. DDP ranks and forked DataLoader workers then
share one RAM copy instead of duplicating a large NumPy cache per rank.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def _redirect_output(path: str | None):
    if not path:
        return None
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = target.open("a", buffering=1)
    os.dup2(stream.fileno(), sys.stdout.fileno())
    os.dup2(stream.fileno(), sys.stderr.fileno())
    return stream


def _write_metadata(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def main(args):
    _log_stream = _redirect_output(args.log)
    source = Path(args.source).expanduser().resolve()
    metadata = Path(args.metadata).expanduser().resolve()
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.unlink(missing_ok=True)

    with h5py.File(source, "r") as h5:
        ds = h5[args.dataset]
        shape = tuple(int(x) for x in ds.shape)
        dtype = np.dtype(ds.dtype)
        nbytes = int(ds.nbytes)

        fd = os.memfd_create(args.name, flags=0)
        os.ftruncate(fd, nbytes)
        mmap_path = f"/proc/{os.getpid()}/fd/{fd}"
        cache = np.memmap(
            mmap_path,
            mode="r+",
            dtype=dtype,
            shape=shape,
            order="C",
        )

        started = time.time()
        copied = 0
        print(
            f"memfd cache load start: {source}:{args.dataset} "
            f"{shape} {dtype} {nbytes / 1e9:.1f} GB -> {mmap_path}",
            flush=True,
        )
        for start in range(0, shape[0], args.slab_rows):
            stop = min(start + args.slab_rows, shape[0])
            cache[start:stop] = ds[start:stop]
            copied += int(cache[start:stop].nbytes)
            slab_index = start // args.slab_rows
            if (
                start == 0
                or stop == shape[0]
                or slab_index % args.log_every == 0
            ):
                print(
                    f"memfd cache {stop}/{shape[0]} rows, "
                    f"{copied / 1e9:.1f} GB, {time.time() - started:.1f}s",
                    flush=True,
                )
        cache.flush()

    payload = {
        "pid": os.getpid(),
        "fd": fd,
        "mmap_path": mmap_path,
        "source": str(source),
        "dataset": args.dataset,
        "shape": list(shape),
        "dtype": dtype.str,
        "nbytes": nbytes,
        "ready": True,
        "loaded_seconds": time.time() - started,
    }
    _write_metadata(metadata, payload)
    print(
        f"memfd cache ready: metadata={metadata} mmap_path={mmap_path}",
        flush=True,
    )

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopping:
        signal.pause()
    metadata.unlink(missing_ok=True)
    print("memfd cache daemon stopped", flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--name", default="h5_dataset_cache")
    p.add_argument("--slab-rows", type=int, default=256)
    p.add_argument("--log-every", type=int, default=32)
    p.add_argument("--log")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
