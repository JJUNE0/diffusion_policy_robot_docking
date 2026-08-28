#!/bin/bash
# floor4_inside 150ep -- PHASE 2: r_encfeat_only_now_f4in150 + r_pose_only_now_f4in150
# in parallel, after phase 1 (relfeat) has released its 161 GiB of dec caches.
#
# RAM: only encfeat needs a shared cache -- reloc3r_bottom, 287256 x 196 x 1024
# x 2 B = ~107 GiB. Its two streams (enc_hist / enc_goal) read the SAME key, so
# they share ONE memfd; np.memmap-ing one path twice maps the same pages. The
# pose arm's sidecar is 6.6 MB and uses cache_in_ram, no daemon.
#
# Loader: same trick phase 1 used -- workers cannot ship a whole 256-batch
# through the container's 2 GB /dev/shm, so they ship microbatches and
# _RebatchedDataLoader reassembles 256. Batches are unchanged (RandomSampler's
# permutation does not depend on batch size; 1087 steps/epoch either way).
#
# encfeat carries 1024-dim features = 8.0 MB/sample, vs relfeat's 6.0 MB, so it
# uses microbatch 8 (~0.51 GB in flight for 8 workers x prefetch 1) rather than
# phase 1's 16. Measured: 6.0 MB/sample at microbatch 16 (0.75 GB) works and at
# microbatch 32 (1.5 GB) does NOT, so staying near 0.5 GB keeps clear of the
# cliff -- especially with the pose arm's workers sharing the same /dev/shm.
# floor(floor(278406/8)/32) = 1087 steps/epoch, same as every other arm.
#
# Usage:  bash test/queue_f4in150_phase2_pose_encfeat.sh

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_f4in150_phase2.log"
LOCK_FILE="$LOG_DIR/queue_f4in150_phase2.lock"
SIDECAR="$(pwd)/dataset/f4in_150ep_train_reloc3r_bottom.h5"
HEAD="$(pwd)/dataset/f4in_150ep_train_reloc3r_bottom_head.h5"
TRAIN_DATA_PATH="$(pwd)/dataset/f4in_150ep_train.h5"
ENC_META=outputs/cache/f4in150_reloc3r_bottom_memfd.json

NUM_WORKERS=${NUM_WORKERS:-8}
ENC_MICROBATCH=${ENC_MICROBATCH:-8}
POSE_MICROBATCH=${POSE_MICROBATCH:-16}

mkdir -p "$LOG_DIR" outputs/cache
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "f4in150 phase 2 is already active." >&2
    exit 1
fi

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

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

for f in "$SIDECAR" "$HEAD" "$TRAIN_DATA_PATH"; do
    [ -f "$f" ] || { log "ERROR: missing $f"; exit 1; }
done

log "f4in150 PHASE 2 started (encfeat + pose, parallel)."

# Real h5, not a VDS -> cache_h5_dataset_memfd.py (the parallel VDS loader
# refuses a non-virtual dataset).
if CACHE=$(read_cache_path "$ENC_META"); then
    log "Reusing ready memfd cache: $CACHE"
else
    log "Starting memfd cache daemon for reloc3r_bottom (~107 GiB)."
    "$PY" -u scripts/cache_h5_dataset_memfd.py \
        --source "$SIDECAR" \
        --dataset reloc3r_bottom \
        --metadata "$ENC_META" \
        --name f4in150_reloc3r_bottom \
        --log "$LOG_DIR/memfd_f4in150_reloc3r_bottom.log" &
    disown
fi

while true; do
    ENC=$(read_cache_path "$ENC_META") || ENC=
    [ -n "$ENC" ] && break
    log "Waiting for the reloc3r_bottom memfd cache."
    sleep 45
done
log "RAM cache ready: enc=$ENC"

run_arm() {
    NAME=$1; VARIANT=$2; MB=$3
    shift 3
    TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
    log "TRAIN START $NAME ($VARIANT) workers=$NUM_WORKERS microbatch=$MB"
    "$PY" -u scripts/train.py \
        --config-name smr_rgeo \
        sensors_variant="$VARIANT" \
        experiment_name="$NAME" \
        device=cuda:0 \
        train_data_path="$TRAIN_DATA_PATH" \
        num_workers="$NUM_WORKERS" \
        prefetch_factor=1 \
        +loader_microbatch_size="$MB" \
        "$@" \
        > "$TRAIN_LOG" 2>&1
    log "TRAIN END $NAME (rc=$?)."
}

run_arm r_encfeat_only_now_f4in150 reloc3r_encfeat_only_now_f4in150 "$ENC_MICROBATCH" \
    "sensors.reloc3r_enc_hist.cache_mmap=$ENC" \
    "sensors.reloc3r_enc_goal.cache_mmap=$ENC" &
PID_ENC=$!

run_arm r_pose_only_now_f4in150 reloc3r_pose_only_now_f4in150 "$POSE_MICROBATCH" &
PID_POSE=$!

log "Worker PIDs: encfeat=$PID_ENC pose=$PID_POSE"
wait "$PID_ENC"; RC_ENC=$?
wait "$PID_POSE"; RC_POSE=$?
log "Phase 2 done: encfeat=$RC_ENC pose=$RC_POSE"

if [ "${KEEP_CACHE:-0}" != "1" ]; then
    P=$(cache_pid "$ENC_META") && [ -n "$P" ] && kill "$P" 2>/dev/null && \
        log "Stopped memfd daemon pid $P."
fi

log "f4in150 PHASE 2 DONE."
exit $(( RC_ENC | RC_POSE ))
