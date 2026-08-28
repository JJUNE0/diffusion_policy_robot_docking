#!/bin/bash
# floor4_hallway 150ep, end to end, nothing idling between stages:
#
#   dec1/dec2 relational precompute  -> f4hall150_train_reloc3r_bottom.h5
#     -> memfd RAM caches + training  (test/queue_f4hall150_relfeat.sh)
#
# The ViT-L encoder pass is SKIPPED: reloc3r_bottom in that sidecar is a VDS
# onto jiwon's already-computed floor4_hallway_front_docking_reloc3r_bottom.h5
# (same encode_frames code, byte-identical source, verified by re-encoding).
# That is ~1.5 h of H100 time not spent.
#
# Usage:
#   nohup bash scripts/run_f4hall150_pipeline.sh > outputs/logs/f4hall150_pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
LOG_DIR=outputs/logs
SIDECAR=dataset/f4hall150_train_reloc3r_bottom.h5
DEC_LOG="$LOG_DIR/precompute_f4hall150_dec.log"

mkdir -p "$LOG_DIR"
export HF_HOME=/home/work/.postech/diffusion_policy_robot_docking/.hf_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if "$PY" -c "
import sys, h5py
with h5py.File('$SIDECAR','r') as f:
    ok = ('reloc3r_dec1_bottom' in f and 'reloc3r_dec2_bottom' in f and
          int(f['reloc3r_dec1_bottom'].attrs.get('n_done_ep',0)) >= len(f['episode_ends']))
sys.exit(0 if ok else 1)" 2>/dev/null; then
    log "dec1/dec2 already complete -- skipping precompute"
else
    log "starting dec1/dec2 precompute (resumable, appends in place)"
    "$PY" -u scripts/precompute_reloc3r_dec_features.py \
        --cache "$SIDECAR" --camera image_bottom > "$DEC_LOG" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ] || ! grep -q "^Done\." "$DEC_LOG"; then
        log "ERROR: dec precompute failed (rc=$RC)"
        tail -20 "$DEC_LOG"
        exit 1
    fi
    log "dec1/dec2 precompute complete"
fi

log "launching training (r_relfeat_only_now_f4hall150)"
bash test/queue_f4hall150_relfeat.sh
RC=$?
log "training finished rc=$RC"
exit "$RC"
