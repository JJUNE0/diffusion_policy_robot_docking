#!/bin/bash
# floor4_hallway 150ep -- r_pose_only_now_f4hall150, alone on the box.
#
# Same arm as r_pose_only_now / *_f4in150: only the dataset changed.
#   dataset/f4hall150_train.h5
#   dataset/f4hall150_train_reloc3r_bottom_head.h5   (pose1/pose2_bottom, 12-D)
#
# No memfd daemon: pose1/pose2 are 6.6 MB total for 150ep and load via
# `cache_in_ram: true` at dataset init.
#
# Loader: num_workers=8 + 16-sample microbatches, same reasoning as the
# relfeat queue script -- num_workers=0 leaves the GPU underfed, and a whole
# 256-batch cannot fit through the container's 2 GB /dev/shm at once.
# smr_rgeo.yaml lists `_self_` AFTER the sensors_variant group, so the
# variant's own train_data_path is shadowed and must be re-passed on the CLI.
#
# Usage:  bash test/queue_f4hall150_pose.sh

set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
NAME=${NAME:-r_pose_only_now_f4hall150}
VARIANT=reloc3r_pose_only_now_f4hall150
LOG_DIR=outputs/logs
QUEUE_LOG="$LOG_DIR/queue_f4hall150_pose.log"
LOCK_FILE="$LOG_DIR/queue_f4hall150_pose.lock"
HEAD="$(pwd)/dataset/f4hall150_train_reloc3r_bottom_head.h5"
TRAIN_DATA_PATH="$(pwd)/dataset/f4hall150_train.h5"

NUM_WORKERS=${NUM_WORKERS:-8}
MICROBATCH=${MICROBATCH:-16}

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "f4hall150 pose is already active." >&2
    exit 1
fi

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

log "f4hall150 pose started (150 ep / 251017 rows)."

for f in "$HEAD" "$TRAIN_DATA_PATH"; do
    [ -f "$f" ] || { log "ERROR: missing $f"; exit 1; }
done
"$PY" - "$HEAD" <<'PYCODE' || exit 1
import sys
import h5py
with h5py.File(sys.argv[1], "r") as f:
    for k in ("reloc3r_pose1_bottom", "reloc3r_pose2_bottom"):
        if k not in f:
            raise SystemExit(f"ERROR: {sys.argv[1]} has no {k}; run "
                             f"scripts/precompute_reloc3r_head_features.py first")
        done = int(f["reloc3r_head1_bottom"].attrs.get(
            "n_done_shard0", f[k].shape[0]))
    if done < f[k].shape[0]:
        raise SystemExit(f"ERROR: head precompute incomplete "
                         f"({done}/{f[k].shape[0]} rows)")
PYCODE

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
    > "$TRAIN_LOG" 2>&1
RC=$?
log "TRAIN END $NAME (rc=$RC)."
log "f4hall150 pose DONE (rc=$RC)."
exit "$RC"
