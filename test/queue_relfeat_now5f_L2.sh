#!/bin/bash
# L2 ablation queue: condition_num_layers=2 (fusion transformer 4 -> 2 layers)
# vs the "_now" (current-frame-included, 0.5s-spaced) arms already queued in
# test/queue_relfeat_now5f.sh. Skips the tokmatch pair to save time (per user
# request) -- covers full/dec1/dec2/pose only:
#
#   r_relfeat_only_now_L2      vs r_relfeat_only_now      (Phase 3 of the main queue)
#   r_relfeat_dec1_only_now_L2 vs r_relfeat_dec1_only_now (Phase 1, already done)
#   r_relfeat_dec2_only_now_L2 vs r_relfeat_dec2_only_now (Phase 1)
#   r_pose_only_now_L2         vs r_pose_only_now         (Phase 3)
#
# ONE variable changed per job (condition_num_layers only); sensors_variant,
# indexing, seed, optimizer etc. all identical to the matching 4-layer "_now"
# arm, so each pair isolates fusion depth cleanly.
#
# Runs strictly SEQUENTIALLY (not paired) and gates each launch on:
#   1) the main queue's Phase 2 (tokmatch pair, ~21GiB each) being finished --
#      avoids 3-way GPU memory contention with that pair.
#   2) a real-time free-GPU-memory floor, so it also waits out whatever the
#      main queue happens to be running (e.g. Phase 3) at launch time instead
#      of assuming a fixed schedule.
# This reuses the SAME shared memfd RAM caches as the main queue (dec1_bottom
# / dec2_bottom) -- no re-caching needed.

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_relfeat_now5f_L2.log"
MAIN_QUEUE_LOG="$LOG_DIR/queue_relfeat_now5f.log"
LOCK_FILE="$LOG_DIR/queue_relfeat_now5f_L2.lock"
DEC1_CACHE_METADATA=outputs/cache/reloc3r_dec1_bottom_memfd.json
DEC2_CACHE_METADATA=outputs/cache/reloc3r_dec2_bottom_memfd.json
FREE_MB_THRESHOLD=30000

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "An L2 queue is already active." >&2
    exit 1
fi

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$QUEUE_LOG"
}

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

log "L2 queue started."

log "Waiting for main queue's Phase 2 (tokmatch pair) to finish."
while ! grep -q "Phase complete: r_relfeat_dec1_only_tokmatch_now" "$MAIN_QUEUE_LOG" 2>/dev/null; do
    sleep 60
done
log "Main queue Phase 2 finished."

while true; do
    DEC1_CACHE=$(read_cache_path "$DEC1_CACHE_METADATA") || DEC1_CACHE=
    DEC2_CACHE=$(read_cache_path "$DEC2_CACHE_METADATA") || DEC2_CACHE=
    if [ -n "$DEC1_CACHE" ] && [ -n "$DEC2_CACHE" ]; then
        break
    fi
    log "Waiting for dec1/dec2 memfd cache metadata to be ready."
    sleep 30
done
log "Shared RAM caches confirmed: dec1=$DEC1_CACHE, dec2=$DEC2_CACHE."

wait_for_memory() {
    while true; do
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            | head -n 1 | tr -d ' ')
        if [ "$FREE_MB" -ge "$FREE_MB_THRESHOLD" ]; then
            return
        fi
        log "Waiting for GPU memory: ${FREE_MB} MiB free; ${FREE_MB_THRESHOLD} MiB required."
        sleep 60
    done
}

run_one() {
    NAME=$1
    VARIANT=$2
    shift 2
    EXTRA_OVERRIDES=("$@")
    TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
    wait_for_memory
    log "TRAIN START $NAME ($VARIANT, condition_num_layers=2) overrides=${EXTRA_OVERRIDES[*]:-none}"
    "$PY" -u scripts/train.py \
        --config-name smr_rgeo \
        sensors_variant="$VARIANT" \
        experiment_name="$NAME" \
        device=cuda:0 \
        condition_num_layers=2 \
        "${EXTRA_OVERRIDES[@]}" \
        > "$TRAIN_LOG" 2>&1
    RC=$?
    log "TRAIN END $NAME (rc=$RC)."
}

run_one r_relfeat_only_now_L2 reloc3r_relfeat_only_now \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE" "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

run_one r_relfeat_dec1_only_now_L2 reloc3r_relfeat_dec1_only_now \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE"

run_one r_relfeat_dec2_only_now_L2 reloc3r_relfeat_dec2_only_now \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

run_one r_pose_only_now_L2 reloc3r_pose_only_now

log "L2 QUEUE ALL DONE."
