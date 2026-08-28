#!/bin/bash
# floor4_inside 150ep -- PHASE 1: r_relfeat_only_now_f4in150 ALONE, with the
# whole box behind it (the arm that has to be done by tomorrow morning).
#
# Why it runs alone: all three arms' RAM caches together would be
# dec1+dec2 (161 GiB) + reloc3r_bottom (107 GiB) = 268 GiB of a 300 GiB
# 0-swap host, leaving ~22 GiB. Phase 1 pins only dec1+dec2, so relfeat gets
# both the RAM headroom and the whole H100. Phase 2
# (test/queue_f4in150_phase2_pose_encfeat.sh) runs pose + encfeat afterwards.
#
# Speed: num_workers=8 with 16-sample microbatches. num_workers=0 (what the
# 4f150 runs used) leaves the GPU at ~18% because ONE thread gathers ~770 MB of
# fp16 per batch; that costs ~2.0 s of a 2.5 s step. Workers cannot ship a whole
# 256-batch through this container's 2 GB /dev/shm -- measured: batch_size=256
# with num_workers=8 dies with "unable to allocate shared memory", and so do
# microbatch=32 and workers=16 -- but microbatch=16 x 8 workers x prefetch 1
# keeps only ~0.75 GB in flight and moves a 256-batch in 0.219 s.
#
# This does NOT change the math: RandomSampler's permutation is independent of
# batch size, so regrouping 16 microbatches of 16 yields exactly the 256-chunks
# batch_size=256 would have produced, and steps/epoch is 1087 either way.
# prefetch_factor MUST be passed explicitly -- smr_rgeo.yaml sets it to null and
# _build_dataloader does int(args.get("prefetch_factor", 1)), which would raise
# on None.
#
# Usage:  bash test/queue_f4in150_phase1_relfeat.sh
#   KEEP_CACHE=1 keeps the dec memfd daemons up (phase 2 does not need them).

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
NAME=${NAME:-r_relfeat_only_now_f4in150}
VARIANT=reloc3r_relfeat_only_now_f4in150
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_f4in150_phase1.log"
LOCK_FILE="$LOG_DIR/queue_f4in150_phase1.lock"
SIDECAR="$(pwd)/dataset/f4in_150ep_train_reloc3r_bottom.h5"
TRAIN_DATA_PATH="$(pwd)/dataset/f4in_150ep_train.h5"

DEC1_META=outputs/cache/f4in150_reloc3r_dec1_bottom_memfd.json
DEC2_META=outputs/cache/f4in150_reloc3r_dec2_bottom_memfd.json

NUM_WORKERS=${NUM_WORKERS:-8}
MICROBATCH=${MICROBATCH:-16}

mkdir -p "$LOG_DIR" outputs/cache
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "f4in150 phase 1 is already active." >&2
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

# NOTE: this sidecar is a REAL h5 (written by precompute_reloc3r_*), not a VDS,
# so it needs cache_h5_dataset_memfd.py -- cache_h5_vds_memfd_parallel.py
# refuses a non-virtual dataset.
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
        --name "f4in150_$DATASET" \
        --log "$LOG_DIR/memfd_f4in150_${DATASET}.log" &
    disown
}

log "f4in150 PHASE 1 started (relfeat alone, 150 ep / 287256 rows)."

for f in "$SIDECAR" "$TRAIN_DATA_PATH"; do
    [ -f "$f" ] || { log "ERROR: missing $f"; exit 1; }
done

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

log "f4in150 PHASE 1 DONE (rc=$RC)."
exit "$RC"
