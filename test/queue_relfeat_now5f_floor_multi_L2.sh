#!/bin/bash
# L2 ablation (condition_num_layers=2) for the floor_multi arms -- same 4
# variants (full/dec1/dec2/pose) as test/queue_relfeat_now5f_floor_multi.sh,
# same "_now" indexing, only the fusion transformer depth changes.
#
# Waits for the 4-layer floor_multi queue to fully finish before starting:
# host RAM is razor-thin on this box (~230GiB of 250GiB already pinned by the
# shared dec1_bottom/dec2_bottom memfd caches), so only ONE training process
# may run at a time -- never in parallel with the 4-layer queue.

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_relfeat_now5f_floor_multi_L2.log"
MAIN_QUEUE_LOG="$LOG_DIR/queue_relfeat_now5f_floor_multi.log"
LOCK_FILE="$LOG_DIR/queue_relfeat_now5f_floor_multi_L2.lock"
DEC1_CACHE_METADATA=outputs/cache/floor_multi_reloc3r_dec1_bottom_memfd.json
DEC2_CACHE_METADATA=outputs/cache/floor_multi_reloc3r_dec2_bottom_memfd.json
TRAIN_DATA_PATH="$(pwd)/dataset/floor_multi_train.h5"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A floor_multi L2 queue is already active." >&2
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

log "floor_multi L2 queue started."

log "Waiting for the 4-layer floor_multi queue to finish (RAM is too tight to run in parallel)."
while ! grep -q "floor_multi QUEUE ALL DONE" "$MAIN_QUEUE_LOG" 2>/dev/null; do
    sleep 60
done
log "4-layer floor_multi queue finished."

DEC1_CACHE=$(read_cache_path "$DEC1_CACHE_METADATA") || { log "ERROR: dec1 memfd cache not ready"; exit 1; }
DEC2_CACHE=$(read_cache_path "$DEC2_CACHE_METADATA") || { log "ERROR: dec2 memfd cache not ready"; exit 1; }
log "Shared RAM caches confirmed: dec1=$DEC1_CACHE, dec2=$DEC2_CACHE."

run_one() {
    NAME=$1
    VARIANT=$2
    shift 2
    EXTRA_OVERRIDES=("$@")
    TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
    log "TRAIN START $NAME ($VARIANT, condition_num_layers=2) overrides=${EXTRA_OVERRIDES[*]:-none}"
    "$PY" -u scripts/train.py \
        --config-name smr_rgeo \
        sensors_variant="$VARIANT" \
        experiment_name="$NAME" \
        device=cuda:0 \
        train_data_path="$TRAIN_DATA_PATH" \
        condition_num_layers=2 \
        "${EXTRA_OVERRIDES[@]}" \
        > "$TRAIN_LOG" 2>&1
    RC=$?
    log "TRAIN END $NAME (rc=$RC)."
}

run_one r_relfeat_only_now_floor_multi_L2 reloc3r_relfeat_only_now_floor_multi \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE" "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

run_one r_relfeat_dec1_only_now_floor_multi_L2 reloc3r_relfeat_dec1_only_now_floor_multi \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE"

run_one r_relfeat_dec2_only_now_floor_multi_L2 reloc3r_relfeat_dec2_only_now_floor_multi \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

run_one r_pose_only_now_floor_multi_L2 reloc3r_pose_only_now_floor_multi

log "floor_multi L2 QUEUE ALL DONE."
