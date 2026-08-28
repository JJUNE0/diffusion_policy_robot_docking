#!/bin/bash
# floor4_hallway 150ep -- r_relfeat_only_now_f4hall150, alone on the box.
#
# Same arm as r_relfeat_only_now / *_f4in150: only the dataset changed.
#   dataset/f4hall150_train.h5                 150 ep / 251017 rows
#   dataset/f4hall150_train_reloc3r_bottom.h5  reloc3r_dec1/dec2_bottom
#
# RAM: dec1+dec2 pin 196*768*2 B * 2 streams = 602 KB/row -> 251017 rows =
# 141 GiB of a 300 GiB 0-swap host. Comfortable; it runs alone anyway.
#
# Speed: num_workers=8 with 16-sample microbatches. num_workers=0 leaves the
# GPU at ~18% because ONE thread gathers ~770 MB of fp16 per batch. Workers
# cannot ship a whole 256-batch through this container's 2 GB /dev/shm, but
# microbatch=16 x 8 workers x prefetch 1 keeps only ~0.75 GB in flight and
# _RebatchedDataLoader reassembles 256. This does NOT change the math:
# RandomSampler's permutation is independent of batch size, so regrouping 16
# microbatches of 16 yields exactly the 256-chunks batch_size=256 would have.
# prefetch_factor MUST be passed explicitly -- smr_rgeo.yaml sets it to null and
# _build_dataloader does int(args.get("prefetch_factor", 1)), which raises on None.
#
# smr_rgeo.yaml lists `_self_` AFTER the sensors_variant group, so the variant's
# own train_data_path is shadowed and must be re-passed on the CLI.
#
# Usage:  bash test/queue_f4hall150_relfeat.sh
#   KEEP_CACHE=1 keeps the dec memfd daemons up after training.

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
NAME=${NAME:-r_relfeat_only_now_f4hall150}
VARIANT=reloc3r_relfeat_only_now_f4hall150
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_f4hall150.log"
LOCK_FILE="$LOG_DIR/queue_f4hall150.lock"
SIDECAR="$(pwd)/dataset/f4hall150_train_reloc3r_bottom.h5"
TRAIN_DATA_PATH="$(pwd)/dataset/f4hall150_train.h5"

DEC1_META=outputs/cache/f4hall150_reloc3r_dec1_bottom_memfd.json
DEC2_META=outputs/cache/f4hall150_reloc3r_dec2_bottom_memfd.json

NUM_WORKERS=${NUM_WORKERS:-8}
MICROBATCH=${MICROBATCH:-16}

mkdir -p "$LOG_DIR" outputs/cache
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "f4hall150 relfeat is already active." >&2
    exit 1
fi

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

read_cache_path() {
    "$PY" - "$1" <<'PYCODE'
import json
import os
import sys

try:
    payload = json.load(open(sys.argv[1]))
    path = payload["mmap_path"]
    pid = int(payload["pid"])
    if payload.get("ready") and os.path.exists(f"/proc/{pid}") and os.path.exists(path):
        print(path)
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PYCODE
}

cache_pid() {
    "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["pid"])' "$1" 2>/dev/null
}

# NOTE: reloc3r_dec1/dec2_bottom are REAL datasets written by
# precompute_reloc3r_dec_features.py, so cache_h5_dataset_memfd.py is correct
# here. (Only `reloc3r_bottom` in this sidecar is virtual, and the relfeat arm
# never touches it.)
start_cache_if_needed() {
    META=$1
    DATASET=$2
    if CACHE=$(read_cache_path "$META"); then
        log "Reusing ready memfd cache for $DATASET: $CACHE"
        return
    fi
    log "Starting memfd cache daemon for $DATASET."
    "$PY" -u scripts/cache_h5_dataset_memfd.py \
        --source "$SIDECAR" \
        --dataset "$DATASET" \
        --metadata "$META" \
        --name "f4hall150_$DATASET" \
        --log "$LOG_DIR/memfd_f4hall150_${DATASET}.log" &
    disown
}

log "f4hall150 relfeat started (150 ep / 251017 rows)."

for f in "$SIDECAR" "$TRAIN_DATA_PATH"; do
    [ -f "$f" ] || { log "ERROR: missing $f"; exit 1; }
done
"$PY" - "$SIDECAR" <<'PYCODE' || exit 1
import sys
import h5py
with h5py.File(sys.argv[1], "r") as f:
    for k in ("reloc3r_dec1_bottom", "reloc3r_dec2_bottom"):
        if k not in f:
            raise SystemExit(f"ERROR: {sys.argv[1]} has no {k}; run "
                             f"scripts/precompute_reloc3r_dec_features.py first")
        done = int(f["reloc3r_dec1_bottom"].attrs.get("n_done_ep", 0))
    if done < len(f["episode_ends"]):
        raise SystemExit(f"ERROR: dec precompute only reached episode {done} of "
                         f"{len(f['episode_ends'])}")
PYCODE

start_cache_if_needed "$DEC1_META" reloc3r_dec1_bottom
start_cache_if_needed "$DEC2_META" reloc3r_dec2_bottom

while true; do
    DEC1=$(read_cache_path "$DEC1_META") || DEC1=
    DEC2=$(read_cache_path "$DEC2_META") || DEC2=
    if [ -n "$DEC1" ] && [ -n "$DEC2" ]; then break; fi
    log "Waiting for dec1/dec2 memfd caches."
    sleep 45
done
log "RAM caches ready: dec1=$DEC1 dec2=$DEC2"

TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
log "TRAIN START $NAME ($VARIANT) workers=$NUM_WORKERS microbatch=$MICROBATCH"
"$PY" -u scripts/train.py \
    --config-name smr_rgeo \
    sensors_variant="$VARIANT" \
    experiment_name="$NAME" \
    device=cuda:0 \
    train_data_path="$TRAIN_DATA_PATH" \
    num_workers="$NUM_WORKERS" \
    prefetch_factor=1 \
    +loader_microbatch_size="$MICROBATCH" \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1" \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2" \
    > "$TRAIN_LOG" 2>&1
RC=$?
log "TRAIN END $NAME (rc=$RC)."

if [ "${KEEP_CACHE:-0}" != "1" ]; then
    for META in "$DEC1_META" "$DEC2_META"; do
        P=$(cache_pid "$META") && [ -n "$P" ] && kill "$P" 2>/dev/null && \
            log "Stopped memfd daemon pid $P ($META)."
    done
fi

log "f4hall150 relfeat DONE (rc=$RC)."
exit "$RC"
