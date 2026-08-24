#!/bin/bash
# Retrain the real-robot-validated r_relfeat_only_now arm on 4f_hallway_150ep
# (after_0328's 145 episodes + the 5 new 0824_4f_hallway episodes = 150 ep /
# 238688 rows). Same condition, same network, same optimizer as
# selected_models/1_r_relfeat_only_now/2026-08-15_01-23-27 -- only the dataset
# grew, via configs/robot/sensors_variant/reloc3r_relfeat_only_now_4f150.yaml.
#
# Both dataset files are VDS (scripts/build_4f_hallway_150ep_vds.py), so the
# dec1/dec2 RAM caches are loaded with scripts/cache_h5_vds_memfd_parallel.py,
# which walks the per-source VDS mappings with a worker pool. Pinned RAM is
# ~72 GiB per stream / ~144 GiB total on this 300 GiB host.
#
# Usage:  bash test/queue_relfeat_only_now_4f150.sh
#   The memfd daemons outlive this script (they must, to keep the RAM mapping
#   alive) but are torn down at the end unless KEEP_CACHE=1.

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
NAME=${NAME:-r_relfeat_only_now_4f150}
VARIANT=reloc3r_relfeat_only_now_4f150
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_relfeat_only_now_4f150.log"
LOCK_FILE="$LOG_DIR/queue_relfeat_only_now_4f150.lock"
SIDECAR="$(pwd)/dataset/4f_hallway_150ep_train_reloc3r_bottom.h5"
# smr_rgeo.yaml lists `_self_` AFTER the sensors_variant group, so the variant's
# own train_data_path is shadowed by the base config's -- it has to be overridden
# on the CLI (same as test/queue_relfeat_now5f_floor_multi.sh does).
TRAIN_DATA_PATH="$(pwd)/dataset/4f_hallway_150ep_train.h5"
DEC1_CACHE_METADATA=outputs/cache/4f150_reloc3r_dec1_bottom_memfd.json
DEC2_CACHE_METADATA=outputs/cache/4f150_reloc3r_dec2_bottom_memfd.json
CACHE_WORKERS=${CACHE_WORKERS:-12}

mkdir -p "$LOG_DIR" outputs/cache
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A 4f150 queue is already active." >&2
    exit 1
fi

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
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

cache_pid() {
    "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["pid"])' "$1" 2>/dev/null
}

start_cache_if_needed() {
    METADATA=$1
    DATASET=$2
    if CACHE=$(read_cache_path "$METADATA"); then
        log "Reusing already-ready memfd cache for $DATASET: $CACHE"
        return
    fi
    log "Starting parallel VDS memfd cache daemon for $DATASET."
    "$PY" -u scripts/cache_h5_vds_memfd_parallel.py \
        --source "$SIDECAR" \
        --dataset "$DATASET" \
        --metadata "$METADATA" \
        --name "4f150_$DATASET" \
        --workers "$CACHE_WORKERS" \
        --log "$LOG_DIR/memfd_4f150_${DATASET}.log" &
    disown
}

log "4f150 queue started (dataset=4f_hallway_150ep, 150 ep / 238688 rows)."

start_cache_if_needed "$DEC1_CACHE_METADATA" reloc3r_dec1_bottom
start_cache_if_needed "$DEC2_CACHE_METADATA" reloc3r_dec2_bottom

while true; do
    DEC1_CACHE=$(read_cache_path "$DEC1_CACHE_METADATA") || DEC1_CACHE=
    DEC2_CACHE=$(read_cache_path "$DEC2_CACHE_METADATA") || DEC2_CACHE=
    if [ -n "$DEC1_CACHE" ] && [ -n "$DEC2_CACHE" ]; then
        break
    fi
    log "Waiting for dec1/dec2 memfd caches to finish loading."
    sleep 60
done
log "Shared RAM caches ready: dec1=$DEC1_CACHE, dec2=$DEC2_CACHE."

TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
log "TRAIN START $NAME ($VARIANT) -> $TRAIN_LOG"
"$PY" -u scripts/train.py \
    --config-name smr_rgeo \
    sensors_variant="$VARIANT" \
    experiment_name="$NAME" \
    device=cuda:0 \
    train_data_path="$TRAIN_DATA_PATH" \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE" \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE" \
    > "$TRAIN_LOG" 2>&1
RC=$?
log "TRAIN END $NAME (rc=$RC)."

if [ "${KEEP_CACHE:-0}" != "1" ]; then
    for META in "$DEC1_CACHE_METADATA" "$DEC2_CACHE_METADATA"; do
        P=$(cache_pid "$META") && [ -n "$P" ] && kill "$P" 2>/dev/null && \
            log "Stopped memfd daemon pid $P ($META)."
    done
else
    log "KEEP_CACHE=1 -- memfd daemons left running for a follow-up run."
fi

log "4f150 QUEUE DONE (rc=$RC)."
exit "$RC"
