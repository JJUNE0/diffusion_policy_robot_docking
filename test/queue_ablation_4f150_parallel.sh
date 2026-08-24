#!/bin/bash
# 4f_hallway_150ep tap-depth ablation: run ALL THREE arms CONCURRENTLY on the
# single H100. Same dataset (150 ep / 238688 rows), same network, same
# optimizer, same 220 condition tokens -- the ONLY variable is where along
# ReLoc3R the condition is tapped:
#
#   r_relfeat_only_now_4f150  pre-head decoder tokens  [196, 768] x dec1,dec2
#   r_encfeat_only_now_4f150  PRE-cross-attention ViT-L [196,1024] x hist,goal
#   r_pose_only_now_4f150     POST-head final pose      [1, 12]   x pose1,pose2
#
# Why parallel is the fast option here: a single arm leaves the H100 at 0-18%
# util (measured) because num_workers=0 makes one CPU thread gather ~770 MB of
# fp16 features per batch out of the RAM cache. The GPU is idle most of the
# wall clock, so three arms interleave into that idle time instead of queueing
# behind each other -- ~12h total instead of ~37h sequential. num_workers stays
# 0 because /dev/shm is still capped at 2 GB on this container (checked), which
# is what the smr_rgeo.yaml comment warns about.
#
# RAM: dec1 + dec2 + reloc3r_bottom memfd = 72 + 72 + 89 = ~233 GiB of 300 GiB.
# The pose arm needs no shared cache (its sidecar is 5.7 MB, cache_in_ram).
# GPU: ~28 + ~30 + ~6 GiB of 80 GiB.
#
# Usage:  bash test/queue_ablation_4f150_parallel.sh
#   KEEP_CACHE=1 leaves the memfd daemons up afterwards for a follow-up run.

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_ablation_4f150.log"
LOCK_FILE="$LOG_DIR/queue_ablation_4f150.lock"
SIDECAR="$(pwd)/dataset/4f_hallway_150ep_train_reloc3r_bottom.h5"
TRAIN_DATA_PATH="$(pwd)/dataset/4f_hallway_150ep_train.h5"
CACHE_WORKERS=${CACHE_WORKERS:-12}

DEC1_META=outputs/cache/4f150_reloc3r_dec1_bottom_memfd.json
DEC2_META=outputs/cache/4f150_reloc3r_dec2_bottom_memfd.json
ENC_META=outputs/cache/4f150_reloc3r_bottom_memfd.json

mkdir -p "$LOG_DIR" outputs/cache
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A 4f150 ablation queue is already active." >&2
    exit 1
fi

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
# Three CUDA contexts share one device; expandable segments keeps the caching
# allocator from fragmenting itself into a false OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

start_cache_if_needed() {
    META=$1
    DATASET=$2
    if CACHE=$(read_cache_path "$META"); then
        log "Reusing ready memfd cache for $DATASET: $CACHE"
        return
    fi
    log "Starting parallel VDS memfd cache daemon for $DATASET."
    "$PY" -u scripts/cache_h5_vds_memfd_parallel.py \
        --source "$SIDECAR" \
        --dataset "$DATASET" \
        --metadata "$META" \
        --name "4f150_$DATASET" \
        --workers "$CACHE_WORKERS" \
        --log "$LOG_DIR/memfd_4f150_${DATASET}.log" &
    disown
}

log "4f150 ablation queue started (3 arms, parallel)."

start_cache_if_needed "$DEC1_META" reloc3r_dec1_bottom
start_cache_if_needed "$DEC2_META" reloc3r_dec2_bottom
start_cache_if_needed "$ENC_META"  reloc3r_bottom

while true; do
    DEC1=$(read_cache_path "$DEC1_META") || DEC1=
    DEC2=$(read_cache_path "$DEC2_META") || DEC2=
    ENC=$(read_cache_path "$ENC_META")   || ENC=
    if [ -n "$DEC1" ] && [ -n "$DEC2" ] && [ -n "$ENC" ]; then break; fi
    log "Waiting for memfd caches (dec1=${DEC1:-...} dec2=${DEC2:-...} enc=${ENC:-...})."
    sleep 45
done
log "RAM caches ready: dec1=$DEC1 dec2=$DEC2 enc=$ENC"

run_arm() {
    NAME=$1
    VARIANT=$2
    shift 2
    TRAIN_LOG="$LOG_DIR/train_${NAME}.log"
    log "TRAIN START $NAME ($VARIANT) -> $TRAIN_LOG"
    "$PY" -u scripts/train.py \
        --config-name smr_rgeo \
        sensors_variant="$VARIANT" \
        experiment_name="$NAME" \
        device=cuda:0 \
        train_data_path="$TRAIN_DATA_PATH" \
        "$@" \
        > "$TRAIN_LOG" 2>&1
    log "TRAIN END $NAME (rc=$?)."
}

# Baseline (restarted): pre-head decoder tokens.
run_arm r_relfeat_only_now_4f150 reloc3r_relfeat_only_now_4f150 \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1" \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2" &
PID_RELFEAT=$!

# Arm B: pre-cross-attention ViT-L tokens. Both streams read reloc3r_bottom, so
# they share ONE memfd -- mapping the same path twice maps the same pages.
run_arm r_encfeat_only_now_4f150 reloc3r_encfeat_only_now_4f150 \
    "sensors.reloc3r_enc_hist.cache_mmap=$ENC" \
    "sensors.reloc3r_enc_goal.cache_mmap=$ENC" &
PID_ENCFEAT=$!

# Arm A: post-head 12-D pose. cache_in_ram, no shared cache needed.
run_arm r_pose_only_now_4f150 reloc3r_pose_only_now_4f150 &
PID_POSE=$!

log "Worker PIDs: relfeat=$PID_RELFEAT encfeat=$PID_ENCFEAT pose=$PID_POSE"

wait "$PID_RELFEAT"; RC_RELFEAT=$?
wait "$PID_ENCFEAT"; RC_ENCFEAT=$?
wait "$PID_POSE";    RC_POSE=$?
log "All arms done: relfeat=$RC_RELFEAT encfeat=$RC_ENCFEAT pose=$RC_POSE"

if [ "${KEEP_CACHE:-0}" != "1" ]; then
    for META in "$DEC1_META" "$DEC2_META" "$ENC_META"; do
        P=$(cache_pid "$META") && [ -n "$P" ] && kill "$P" 2>/dev/null && \
            log "Stopped memfd daemon pid $P ($META)."
    done
else
    log "KEEP_CACHE=1 -- memfd daemons left running."
fi

log "4f150 ABLATION QUEUE DONE."
exit $(( RC_RELFEAT | RC_ENCFEAT | RC_POSE ))
