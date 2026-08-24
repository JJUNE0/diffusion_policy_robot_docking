#!/bin/bash
# floor_multi (floor4_hallway + floor4_inside + floor5, 80 episodes each,
# 411890 rows) retraining queue -- full / dec1_only / dec2_only / pose_only,
# all "_now" (current-frame-included, 0.5s-spaced) indexing. Runs STRICTLY
# SEQUENTIALLY (one job at a time, never parallel): host RAM is already
# ~230GiB committed to the shared dec1_bottom/dec2_bottom memfd caches
# (outputs/cache/floor_multi_reloc3r_dec{1,2}_bottom_memfd.json), leaving only
# a few GiB of true free RAM on this 250GiB/0-swap box -- running jobs one at
# a time keeps peak host-RAM overhead to a single training process instead of
# stacking multiple concurrent ones.

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_relfeat_now5f_floor_multi.log"
LOCK_FILE="$LOG_DIR/queue_relfeat_now5f_floor_multi.lock"
DEC1_CACHE_METADATA=outputs/cache/floor_multi_reloc3r_dec1_bottom_memfd.json
DEC2_CACHE_METADATA=outputs/cache/floor_multi_reloc3r_dec2_bottom_memfd.json
TRAIN_DATA_PATH="$(pwd)/dataset/floor_multi_train.h5"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A floor_multi queue is already active." >&2
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

log "floor_multi queue started."

DEC1_CACHE=$(read_cache_path "$DEC1_CACHE_METADATA") || { log "ERROR: dec1 memfd cache not ready"; exit 1; }
DEC2_CACHE=$(read_cache_path "$DEC2_CACHE_METADATA") || { log "ERROR: dec2 memfd cache not ready"; exit 1; }
log "Shared RAM caches confirmed: dec1=$DEC1_CACHE, dec2=$DEC2_CACHE."

run_one() {
    NAME=$1
    VARIANT=$2
    shift 2
    EXTRA_OVERRIDES=("$@")
    TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
    log "TRAIN START $NAME ($VARIANT) overrides=${EXTRA_OVERRIDES[*]:-none}"
    "$PY" -u scripts/train.py \
        --config-name smr_rgeo \
        sensors_variant="$VARIANT" \
        experiment_name="$NAME" \
        device=cuda:0 \
        train_data_path="$TRAIN_DATA_PATH" \
        "${EXTRA_OVERRIDES[@]}" \
        > "$TRAIN_LOG" 2>&1
    RC=$?
    log "TRAIN END $NAME (rc=$RC)."
}

run_one r_relfeat_only_now_floor_multi reloc3r_relfeat_only_now_floor_multi \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE" "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

run_one r_relfeat_dec1_only_now_floor_multi reloc3r_relfeat_dec1_only_now_floor_multi \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE"

run_one r_relfeat_dec2_only_now_floor_multi reloc3r_relfeat_dec2_only_now_floor_multi \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

run_one r_pose_only_now_floor_multi reloc3r_pose_only_now_floor_multi

log "floor_multi QUEUE ALL DONE."
