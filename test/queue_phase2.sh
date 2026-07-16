#!/bin/bash
# Phase 2 (2026-07-15): isolate camera-count vs batch-size as causes of the
# online docking failure, against flow_goal_scratch20 as the single-variable
# baseline (from-scratch 20ep, everything else identical).
#
#   Exp A (2-camera):   use_room1=true  — restores the old baseline's 2nd
#                       camera, the #1 suspect (SS2.12).
#   Exp B (batch 256):  batch_size=256  — same 20-epoch data exposure, just
#                       smaller-batch updates (sharp-minima hypothesis).
#
# Room1 DINO features MUST live as a "dino_top" dataset INSIDE the same cache
# file as "dino_bottom" (utils/docking_dataset.py requires both in one file).
# The bottom-cache files are held open (read) by the currently-running
# trainings for their whole duration, so we cannot append to them yet -->
# precompute to a STAGE file first, then h5py-copy the dataset in once the
# owning training process exits and releases its file handle.
#
#   nohup setsid bash test/queue_phase2.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
QLOG=outputs/queue_phase2.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

log "waiting for room1 (image_top) DINO stage cache: train"
while pgrep -f "precompute_dino_cache.py --h5 dataset/after_0328_train.h5 --camera image_top" > /dev/null; do sleep 60; done
log "train room1 stage cache done -> building held-out room1 stage cache"
CUDA_VISIBLE_DEVICES=0 python -u scripts/precompute_dino_cache.py \
  --h5 dataset/after_0328_test.h5 --camera image_top --out dataset/after_0328_test_dino_top_STAGE.h5 \
  > outputs/precompute_dino_top_test.log 2>&1
log "held-out room1 stage cache done"

merge(){  # $1=stage_file $2=target_bottom_cache_file
  python3 -c "
import h5py
src = h5py.File('$1', 'r')
dst = h5py.File('$2', 'a')
if 'dino_top' in dst:
    print('dino_top already in $2, skip')
else:
    src.copy('dino_top', dst)
    print('merged dino_top into $2')
src.close(); dst.close()
"
}

log "waiting for GPU0 (s20_nogoal) to free, to release the train bottom-cache file handle"
while pgrep -f "experiment_name=s20_nogoal" > /dev/null; do sleep 60; done
log "s20_nogoal done -> merging room1 into train bottom cache"
merge dataset/after_0328_train_dino_top_STAGE.h5 dataset/after_0328_train_dino_bottom.h5

log "waiting for GPU1 (ddpm_goal_adv) to free, to release the held-out bottom-cache file handle"
while pgrep -f "experiment_name=ddpm_goal_adv" > /dev/null; do sleep 60; done
log "ddpm_goal_adv done -> merging room1 into held-out bottom cache"
merge dataset/after_0328_test_dino_top_STAGE.h5 dataset/after_0328_test_dino_bottom.h5
log "both merges done -> both GPUs free -> launching Exp A + Exp B in parallel"

run(){  # $1=gpu $2=name, rest=overrides
  gpu=$1; name=$2; shift 2
  log "TRAIN START $name (gpu$gpu: $*)"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/train.py experiment_name="$name" device=cuda:0 \
    num_epochs=20 save_interval=1000 "$@" > "outputs/train_$name.log" 2>&1
  rc=$?; log "TRAIN END $name (rc=$rc)"; [ $rc -ne 0 ] && { log "SKIP EVAL $name"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  CUDA_VISIBLE_DEVICES=$gpu python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=$gpu \
    python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name heldout"
}

run 0 s20_2cam use_room1=true &
run 1 s20_batch256 batch_size=256 &
wait
log "QUEUE phase2 COMPLETE"
