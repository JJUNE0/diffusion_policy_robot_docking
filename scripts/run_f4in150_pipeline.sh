#!/bin/bash
# Chain the floor4_inside 150ep run end to end so nothing idles between stages:
#
#   [already running] ReLoc3R ViT-L encoder cache   -> reloc3r_bottom
#     -> dec1/dec2 relational cache                 -> reloc3r_dec{1,2}_bottom
#       -> PHASE 1 training: r_relfeat_only_now_f4in150 (alone, deadline arm)
#       -> (concurrently) post-head cache           -> *_head.h5, so PHASE 2's
#                                                      pose arm is unblocked
#
# The head cache runs alongside phase-1 training on purpose: it is ~25 min of
# GPU out of a multi-hour run, and doing it here means phase 2 can start the
# moment phase 1 ends instead of adding its own precompute wait.
#
# Usage:  nohup bash scripts/run_f4in150_pipeline.sh > outputs/logs/f4in150_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
SIDECAR=dataset/f4in_150ep_train_reloc3r_bottom.h5
HEAD=dataset/f4in_150ep_train_reloc3r_bottom_head.h5
ENC_LOG="$LOG_DIR/precompute_f4in_enc.log"
DEC_LOG="$LOG_DIR/precompute_f4in_dec.log"
HEAD_LOG="$LOG_DIR/precompute_f4in_head.log"

export HF_HOME=/home/work/.postech/diffusion_policy_robot_docking/.hf_cache
export WANDB_MODE=disabled

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "waiting for the encoder cache to finish"
while true; do
    if grep -q "^Done\." "$ENC_LOG" 2>/dev/null; then
        log "encoder cache complete"
        break
    fi
    if ! pgrep -f "precompute_reloc3r_cache.py.*f4in_150ep" >/dev/null; then
        log "ERROR: encoder precompute died without writing 'Done.' -- stopping"
        tail -20 "$ENC_LOG"
        exit 1
    fi
    sleep 60
done

log "starting dec1/dec2 precompute"
"$PY" -u scripts/precompute_reloc3r_dec_features.py \
    --cache "$SIDECAR" --camera image_bottom > "$DEC_LOG" 2>&1
RC=$?
if [ "$RC" -ne 0 ] || ! grep -q "^Done\." "$DEC_LOG"; then
    log "ERROR: dec precompute failed (rc=$RC)"
    tail -20 "$DEC_LOG"
    exit 1
fi
log "dec1/dec2 precompute complete"

log "launching PHASE 1 (r_relfeat_only_now_f4in150)"
nohup bash test/queue_f4in150_phase1_relfeat.sh \
    > "$LOG_DIR/queue_f4in150_phase1.stdout" 2>&1 &
PHASE1=$!
log "phase 1 pid $PHASE1"

# Give the memfd daemons a head start before adding GPU load.
sleep 300
log "starting post-head precompute (for phase 2's pose arm)"
"$PY" -u scripts/precompute_reloc3r_head_features.py \
    --dec_cache "$SIDECAR" --out "$HEAD" \
    --camera image_bottom --device cuda:0 > "$HEAD_LOG" 2>&1
log "post-head precompute rc=$?"

wait "$PHASE1"
log "PHASE 1 finished rc=$?"
log "next: bash test/queue_f4in150_phase2_pose_encfeat.sh"
