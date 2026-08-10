#!/usr/bin/env python3
"""Parallel-copy a contiguous HDF5 VDS into one process-owned shared memfd."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np


def _redirect_output(path):
    if not path:
        return None
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = target.open("a", buffering=1)
    os.dup2(stream.fileno(), sys.stdout.fileno())
    os.dup2(stream.fileno(), sys.stderr.fileno())
    return stream


def _write_metadata(path, payload):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _mapping_tasks(source, dataset, task_rows):
    with h5py.File(source, "r") as h5:
        ds = h5[dataset]
        if not ds.is_virtual:
            raise ValueError(f"{source}:{dataset} is not a virtual dataset")
        shape = tuple(int(x) for x in ds.shape)
        dtype = np.dtype(ds.dtype)
        mappings = ds.virtual_sources()
    tasks = []
    for mapping in mappings:
        vlo, vhi = mapping.vspace.get_select_bounds()
        slo, shi = mapping.src_space.get_select_bounds()
        if tuple(vlo[1:]) != tuple(slo[1:]) or tuple(vhi[1:]) != tuple(shi[1:]):
            raise ValueError("only row-wise VDS mappings are supported")
        count = int(vhi[0] - vlo[0] + 1)
        if count != int(shi[0] - slo[0] + 1):
            raise ValueError("VDS source and destination row counts differ")
        for offset in range(0, count, task_rows):
            length = min(task_rows, count - offset)
            tasks.append((
                str(mapping.file_name), str(mapping.dset_name),
                int(vlo[0]) + offset, int(slo[0]) + offset, length,
            ))
    expected = sum(task[4] for task in tasks)
    if expected != shape[0]:
        raise ValueError(f"VDS tasks cover {expected} rows, expected {shape[0]}")
    return shape, dtype, tasks


def _worker(task_queue, progress_queue, mmap_path, shape, dtype_str, slab_rows):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    handles = {}
    try:
        dtype = np.dtype(dtype_str)
        dest = np.memmap(mmap_path, mode="r+", dtype=dtype, shape=shape, order="C")
        while True:
            task = task_queue.get()
            if task is None:
                break
            file_name, dset_name, vstart, sstart, count = task
            if file_name not in handles:
                handles[file_name] = h5py.File(file_name, "r")
            src = handles[file_name][dset_name]
            for offset in range(0, count, slab_rows):
                length = min(slab_rows, count - offset)
                dest[vstart + offset : vstart + offset + length] = src[
                    sstart + offset : sstart + offset + length
                ]
                progress_queue.put(("progress", length))
        progress_queue.put(("done", os.getpid()))
    except BaseException:
        progress_queue.put(("error", os.getpid(), traceback.format_exc()))
        raise
    finally:
        for handle in handles.values():
            handle.close()


def main(args):
    _stream = _redirect_output(args.log)
    source = Path(args.source).expanduser().resolve()
    metadata = Path(args.metadata).expanduser().resolve()
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.unlink(missing_ok=True)
    shape, dtype, tasks = _mapping_tasks(source, args.dataset, args.task_rows)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize

    fd = os.memfd_create(args.name, flags=0)
    os.ftruncate(fd, nbytes)
    mmap_path = f"/proc/{os.getpid()}/fd/{fd}"
    # Touch the mapping only in workers. Parent owns fd for the daemon lifetime.
    started = time.time()
    print(
        f"parallel memfd cache start: {source}:{args.dataset} {shape} {dtype} "
        f"{nbytes / 1e9:.1f} GB -> {mmap_path}; workers={args.workers}, "
        f"tasks={len(tasks)}",
        flush=True,
    )

    context = mp.get_context("fork")
    task_queue = context.Queue()
    progress_queue = context.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(args.workers):
        task_queue.put(None)
    workers = [
        context.Process(
            target=_worker,
            args=(task_queue, progress_queue, mmap_path, shape, dtype.str, args.slab_rows),
        )
        for _ in range(args.workers)
    ]

    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        for process in workers:
            if process.is_alive():
                process.terminate()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for process in workers:
        process.start()

    copied_rows = 0
    done = 0
    next_log_bytes = 0
    error = None
    while done < len(workers) and not stopping:
        try:
            message = progress_queue.get(timeout=1.0)
        except queue.Empty:
            failed = [p for p in workers if p.exitcode not in (None, 0)]
            if failed:
                error = f"cache workers failed: {[(p.pid, p.exitcode) for p in failed]}"
                break
            continue
        if message[0] == "progress":
            copied_rows += int(message[1])
            copied_bytes = copied_rows * row_bytes
            if copied_bytes >= next_log_bytes or copied_rows == shape[0]:
                elapsed = time.time() - started
                rate = copied_bytes / max(elapsed, 1e-6)
                eta = (nbytes - copied_bytes) / max(rate, 1.0)
                print(
                    f"parallel memfd cache {copied_rows}/{shape[0]} rows, "
                    f"{copied_bytes / 1e9:.1f}/{nbytes / 1e9:.1f} GB, "
                    f"{rate / 1e6:.1f} MB/s, eta {eta / 60:.1f}m",
                    flush=True,
                )
                next_log_bytes = copied_bytes + int(args.log_every_gb * 1e9)
        elif message[0] == "done":
            done += 1
        elif message[0] == "error":
            error = message[2]
            break

    if error or stopping:
        for process in workers:
            if process.is_alive():
                process.terminate()
    for process in workers:
        process.join()
    if stopping:
        metadata.unlink(missing_ok=True)
        return 130
    if error or any(process.exitcode != 0 for process in workers):
        metadata.unlink(missing_ok=True)
        raise RuntimeError(error or f"worker exit codes: {[p.exitcode for p in workers]}")
    if copied_rows != shape[0]:
        raise RuntimeError(f"copied {copied_rows} rows, expected {shape[0]}")

    payload = {
        "pid": os.getpid(), "fd": fd, "mmap_path": mmap_path,
        "source": str(source), "dataset": args.dataset,
        "shape": list(shape), "dtype": dtype.str, "nbytes": nbytes,
        "ready": True, "workers": args.workers,
        "loaded_seconds": time.time() - started,
    }
    _write_metadata(metadata, payload)
    print(f"parallel memfd cache ready: metadata={metadata} mmap_path={mmap_path}", flush=True)

    stopping = False
    def daemon_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, daemon_stop)
    signal.signal(signal.SIGTERM, daemon_stop)
    while not stopping:
        signal.pause()
    metadata.unlink(missing_ok=True)
    print("parallel memfd cache daemon stopped", flush=True)
    return 0


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--name", default="h5_vds_cache")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--task-rows", type=int, default=8192)
    p.add_argument("--slab-rows", type=int, default=256)
    p.add_argument("--log-every-gb", type=float, default=2.0)
    p.add_argument("--log")
    return p


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
