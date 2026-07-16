#!/bin/bash
# Auto-eval for the manually-relaunched s20_2cam (it runs OUTSIDE queue_phase2,
# so the queue's auto-eval never fires for it). Waits for the training process,
# then evaluates with the 2-camera-aware harness:
#   train eval  -> EVAL_CACHE = the MERGED cache (train bottom cache has no dino_top)
#   heldout eval-> test bottom cache (dino_top was merged in successfully)
set -u
cd "$(dirname "$0")/.."
QLOG=outputs/queue_phase2.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }
log "watcher: waiting for s20_2cam training to finish"
while pgrep -f "experiment_name=s20_2cam" > /dev/null; do sleep 120; done
dir=$(ls -d outputs/train/s20_2cam/* | tail -1)
log "watcher: s20_2cam done -> eval ($dir)"
EVAL_CACHE=$PWD/dataset/after_0328_train_dino_merged.h5 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL s20_2cam (train)"
EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
  EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL s20_2cam heldout"
log "watcher: s20_2cam eval complete"
