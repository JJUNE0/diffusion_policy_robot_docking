#!/bin/bash
# Queue the two remaining token-matched directional ablations at matched
# training settings.  The dec1-only 16-query arm is launched immediately by
# the caller; after the older dec2-only job releases its memory, these two
# token-matched arms can run alongside dec1-only on the same H100.

set -u

cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
CURRENT_PID=225475
CURRENT_EXPERIMENT=r_relfeat_dec2_only
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_relfeat_directional_parallel.log"
LOCK_FILE="$LOG_DIR/queue_relfeat_directional_parallel.lock"
DEC1_CACHE_METADATA=outputs/cache/reloc3r_dec1_bottom_memfd.json
DEC2_CACHE_METADATA=outputs/cache/reloc3r_dec2_bottom_memfd.json

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A directional-ablation queue is already active." >&2
    exit 1
fi

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$QUEUE_LOG"
}

current_job_is_active() {
    [ -r "/proc/$CURRENT_PID/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$CURRENT_PID/cmdline" | grep -Fq \
        "experiment_name=$CURRENT_EXPERIMENT"
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

log "Queue started; waiting for $CURRENT_EXPERIMENT (PID $CURRENT_PID)."
while current_job_is_active; do
    sleep 60
done
log "$CURRENT_EXPERIMENT is no longer active."

# Avoid racing another GPU job that may start while the current run exits.
# With dec1-only still resident, about 59 GiB should be free. Require 50 GiB
# before adding the two token-matched jobs (about 21 GiB each).
while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' ')
    if [ "$FREE_MB" -ge 50000 ]; then
        break
    fi
    log "Waiting for GPU memory: ${FREE_MB} MiB free; 50000 MiB required."
    sleep 60
done

while true; do
    DEC1_CACHE=$(read_cache_path "$DEC1_CACHE_METADATA") || DEC1_CACHE=
    DEC2_CACHE=$(read_cache_path "$DEC2_CACHE_METADATA") || DEC2_CACHE=
    if [ -n "$DEC1_CACHE" ] && [ -n "$DEC2_CACHE" ]; then
        break
    fi
    log "Waiting for ready dec1/dec2 memfd cache metadata."
    sleep 60
done
log "Shared RAM caches ready: dec1=$DEC1_CACHE, dec2=$DEC2_CACHE."

run_one() {
    NAME=$1
    VARIANT=$2
    SENSOR=$3
    CACHE_PATH=$4
    TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
    log "TRAIN START $NAME ($VARIANT, batch_size=256, num_epochs=20, cache=$CACHE_PATH)."
    "$PY" -u scripts/train.py \
        --config-name smr_rgeo \
        sensors_variant="$VARIANT" \
        experiment_name="$NAME" \
        device=cuda:0 \
        "sensors.${SENSOR}.cache_mmap=$CACHE_PATH" \
        > "$TRAIN_LOG" 2>&1
    RC=$?
    log "TRAIN END $NAME (rc=$RC)."
    return "$RC"
}

log "Launching the two token-matched directional ablations in parallel."
run_one r_relfeat_dec1_only_tokmatch reloc3r_relfeat_dec1_only_tokmatch \
    reloc3r_dec1 "$DEC1_CACHE" &
PID_DEC1_TM=$!
run_one r_relfeat_dec2_only_tokmatch reloc3r_relfeat_dec2_only_tokmatch \
    reloc3r_dec2 "$DEC2_CACHE" &
PID_DEC2_TM=$!

log "Worker PIDs: dec1-token-matched=$PID_DEC1_TM, dec2-token-matched=$PID_DEC2_TM."

wait "$PID_DEC1_TM"; RC_DEC1_TM=$?
wait "$PID_DEC2_TM"; RC_DEC2_TM=$?

log "Queue complete: dec1-token-matched=$RC_DEC1_TM, dec2-token-matched=$RC_DEC2_TM."

if [ "$RC_DEC1_TM" -ne 0 ] || [ "$RC_DEC2_TM" -ne 0 ]; then
    exit 1
fi
