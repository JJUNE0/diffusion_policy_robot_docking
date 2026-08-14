#!/bin/bash
# Queue all 6 "_now" (current-frame-included, 0.5s-spaced 5-frame vision
# history) retraining arms on the single H100, 2 jobs in parallel per phase,
# reusing ONE shared memfd RAM cache each for reloc3r_dec1_bottom /
# reloc3r_dec2_bottom (same precomputed per-frame ReLoc3R features as the
# original arms -- only which row indices get selected changed, see
# utils/modular_dataset.py::_history's new `window` override and the
# reloc3r_*_now.yaml configs).
#
# Phase 1: r_relfeat_dec1_only_now       + r_relfeat_dec2_only_now
# Phase 2: r_relfeat_dec1_only_tokmatch_now + r_relfeat_dec2_only_tokmatch_now
# Phase 3: r_relfeat_only_now            + r_pose_only_now
#   (r_pose_only_now uses its own tiny cache_in_ram head-feature file, not
#   the memfd caches, so pairing it with the full dec1+dec2 arm doesn't
#   contend for the same RAM cache.)

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_relfeat_now5f.log"
LOCK_FILE="$LOG_DIR/queue_relfeat_now5f.lock"
DEC1_CACHE_METADATA=outputs/cache/reloc3r_dec1_bottom_memfd.json
DEC2_CACHE_METADATA=outputs/cache/reloc3r_dec2_bottom_memfd.json

mkdir -p "$LOG_DIR" outputs/cache
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A now5f queue is already active." >&2
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

log "Queue started."

# ---- Start (or reuse) the two shared memfd RAM caches ----
start_cache_if_needed() {
    METADATA=$1
    DATASET=$2
    if CACHE=$(read_cache_path "$METADATA"); then
        log "Reusing already-ready memfd cache for $DATASET: $CACHE"
        return
    fi
    log "Starting memfd cache daemon for $DATASET."
    "$PY" -u scripts/cache_h5_dataset_memfd.py \
        --source dataset/after_0328_train_reloc3r_bottom.h5 \
        --dataset "$DATASET" \
        --metadata "$METADATA" \
        --name "$DATASET" \
        --log "$LOG_DIR/memfd_${DATASET}.log" &
    disown
}
start_cache_if_needed "$DEC1_CACHE_METADATA" reloc3r_dec1_bottom
start_cache_if_needed "$DEC2_CACHE_METADATA" reloc3r_dec2_bottom

while true; do
    DEC1_CACHE=$(read_cache_path "$DEC1_CACHE_METADATA") || DEC1_CACHE=
    DEC2_CACHE=$(read_cache_path "$DEC2_CACHE_METADATA") || DEC2_CACHE=
    if [ -n "$DEC1_CACHE" ] && [ -n "$DEC2_CACHE" ]; then
        break
    fi
    log "Waiting for dec1/dec2 memfd cache metadata to be ready."
    sleep 30
done
log "Shared RAM caches ready: dec1=$DEC1_CACHE, dec2=$DEC2_CACHE."

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
        "${EXTRA_OVERRIDES[@]}" \
        > "$TRAIN_LOG" 2>&1
    RC=$?
    log "TRAIN END $NAME (rc=$RC)."
    return "$RC"
}

run_pair() {
    NAME_A=$1; VARIANT_A=$2; shift 2
    # split remaining args by a literal "--" separator into A-overrides / rest
    A_OVR=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do A_OVR+=("$1"); shift; done
    shift # drop --
    NAME_B=$1; VARIANT_B=$2; shift 2
    B_OVR=("$@")

    run_one "$NAME_A" "$VARIANT_A" "${A_OVR[@]}" &
    PID_A=$!
    run_one "$NAME_B" "$VARIANT_B" "${B_OVR[@]}" &
    PID_B=$!
    log "Phase worker PIDs: $NAME_A=$PID_A, $NAME_B=$PID_B"
    wait "$PID_A"; RC_A=$?
    wait "$PID_B"; RC_B=$?
    log "Phase complete: $NAME_A=$RC_A, $NAME_B=$RC_B"
    if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then
        log "WARNING: non-zero exit in phase ($NAME_A=$RC_A, $NAME_B=$RC_B) -- continuing to next phase anyway."
    fi
}

log "Phase 1: dec1_only_now + dec2_only_now"
run_pair \
    r_relfeat_dec1_only_now reloc3r_relfeat_dec1_only_now \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE" \
    -- \
    r_relfeat_dec2_only_now reloc3r_relfeat_dec2_only_now \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

log "Phase 2: dec1_only_tokmatch_now + dec2_only_tokmatch_now"
run_pair \
    r_relfeat_dec1_only_tokmatch_now reloc3r_relfeat_dec1_only_tokmatch_now \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE" \
    -- \
    r_relfeat_dec2_only_tokmatch_now reloc3r_relfeat_dec2_only_tokmatch_now \
    "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE"

log "Phase 3: relfeat_only_now (both decoders) + pose_only_now"
run_pair \
    r_relfeat_only_now reloc3r_relfeat_only_now \
    "sensors.reloc3r_dec1.cache_mmap=$DEC1_CACHE" "sensors.reloc3r_dec2.cache_mmap=$DEC2_CACHE" \
    -- \
    r_pose_only_now reloc3r_pose_only_now

log "ALL DONE."
