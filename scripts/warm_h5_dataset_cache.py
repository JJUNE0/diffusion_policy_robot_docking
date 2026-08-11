#!/usr/bin/env python3
"""Sequentially read one HDF5 dataset to populate the shared OS page cache.

The read buffer is bounded by ``--slab-rows``; the complete dataset is not
copied into a process-private NumPy array.  Consequently, multiple training
processes can reuse the same kernel page-cache pages without duplicating tens
of gigabytes per process.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--slab-rows", type=int, default=512)
    parser.add_argument("--log-every-gb", type=float, default=5.0)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if args.slab_rows < 1:
        raise ValueError("--slab-rows must be positive")

    started = time.monotonic()
    with h5py.File(source, "r") as h5:
        dataset = h5[args.dataset]
        total_rows = int(dataset.shape[0])
        total_bytes = int(dataset.nbytes)
        next_log = 0
        checksum = 0.0

        print(
            f"page-cache warm start: {source}:{args.dataset} "
            f"shape={dataset.shape} dtype={dataset.dtype} "
            f"size={total_bytes / 2**30:.2f} GiB",
            flush=True,
        )

        for start in range(0, total_rows, args.slab_rows):
            stop = min(start + args.slab_rows, total_rows)
            slab = dataset[start:stop]
            # Touch the returned pages before releasing the bounded user-space
            # buffer. The backing file pages remain reusable in the OS cache.
            checksum += float(slab.reshape(-1)[0])
            copied = stop * total_bytes // total_rows
            if copied >= next_log or stop == total_rows:
                elapsed = time.monotonic() - started
                rate = copied / max(elapsed, 1e-9)
                eta = (total_bytes - copied) / max(rate, 1.0)
                print(
                    f"page-cache warm {stop}/{total_rows} rows, "
                    f"{copied / 1e9:.1f}/{total_bytes / 1e9:.1f} GB, "
                    f"{rate / 1e6:.1f} MB/s, eta={eta / 60:.1f} min",
                    flush=True,
                )
                next_log = copied + int(args.log_every_gb * 1e9)

    elapsed = time.monotonic() - started
    print(
        f"page-cache warm complete: {args.dataset}, "
        f"{total_bytes / 1e9:.1f} GB in {elapsed / 60:.1f} min "
        f"({total_bytes / max(elapsed, 1e-9) / 1e6:.1f} MB/s), "
        f"checksum={checksum:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
