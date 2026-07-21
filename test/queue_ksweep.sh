#!/bin/bash
# Inference hyperparameter sweep (2026-07-21, user-requested): action-chunking
# K in {2,8,16} x warm-start {off,on} on the 3 best held-out models. 18 runs.
# Pure inference (no training) via the shared rollout_core; each writes a
# uniquely-tagged heldout JSON. Runs on GPU1 (lighter training load).
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/usr/bin/python3
QLOG=outputs/queue_ksweep.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

MODELS="graft_goallidar graft_goalimg_lidar graft_lidar_goalimg"
for m in $MODELS; do
  dir=$(command ls -d outputs/train/"$m"/*/ | tail -1); dir="${dir%/}"
  for k in 2 8 16; do
    for warm in 0 1; do
      tag="heldout_k${k}_warm${warm}"
      log "START $m $tag"
      EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
        EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG="$tag" \
        EVAL_CHUNK_K=$k EVAL_WARM_START=$warm EVAL_WARM_LEVEL=0.3 \
        CUDA_VISIBLE_DEVICES=1 "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 \
        && log "END $m $tag ok" || log "FAIL $m $tag"
    done
  done
done
log "KSWEEP COMPLETE"
