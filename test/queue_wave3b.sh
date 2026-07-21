#!/bin/bash
# Wave-3b (2026-07-19, user-approved): from-SCRATCH rectified-flow control,
# same graft ablation conditions as graft_flow_control (control overrides:
# no goal/lidar/aux/AWR) but WITHOUT init_from -- random init. Purpose:
# graft_flow_control warm-starts from checkpoint_step_100000.pt, a DDPM
# checkpoint whose network was trained to predict eps (noise) under a VP-SDE;
# rectified_flow's loss instead targets (x0-x1), a velocity field under linear
# interpolation (see cleandiffuser/diffusion/{diffusionsde,rectifiedflow}.py
# loss()). Same DiT1d/condition-net shapes so the state_dict loads fine, but
# the two objectives are NOT equivalent, so the loaded weights are a transfer-
# learning init, not a true same-backbone warm-start -- graft_flow_control's
# step-90 loss (~1.0, vs ~0.01-0.03 for the ddpm-warm-started cells at a
# similar step) already shows this mismatch. This scratch cell isolates
# "is rectified-flow itself worse here" from "is this particular warm-start
# transfer rocky" by removing the transfer-learning confound entirely.
#
# Hyperparameters matched to the graft cells per user request (batch_size,
# num_epochs/steps, num_workers) EXCEPT learning_rate: user chose to follow
# the existing §0 convention (scratch=1e-4, warm-start=3e-5) rather than
# force lr=3e-5, since a warm-start-tuned lr would underpower a random-init
# run within the same 16940-step budget and confound "flow is worse" with
# "lr was wrong for scratch training".
#
# GPU slot: waits for graft_gil_aux (GPU0, closest to done of the 3 GPU0
# wave2 cells at write time) to free up, keeping the 3/GPU ceiling intact.
#
#   nohup setsid bash test/queue_wave3b.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/usr/bin/python3
QLOG=outputs/queue_wave3b.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

train_once(){
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
    diffusion_backbone=rectified_flow use_room1=true use_dino_cache=false \
    learning_rate=1e-4 num_epochs=10 batch_size=128 num_workers=4 prefetch_factor=1 \
    save_interval=5000 "$@" > "outputs/train_$name.log" 2>&1
}

run(){  # $1=gpu $2=name, rest=overrides — train with shm/OOM-retry, then eval both sets
  local gpu=$1 name=$2 rc attempt; shift 2
  for attempt in 1 2 3 4; do
    log "TRAIN START $name attempt$attempt (gpu$gpu: $*)"
    train_once "$gpu" "$name" "$@"; rc=$?
    log "TRAIN END $name attempt$attempt (rc=$rc)"
    [ $rc -eq 0 ] && break
    if grep -qE "unable to allocate shared memory|CUDA out of memory" "outputs/train_$name.log"; then
      log "$name: shm/OOM race casualty, retrying in 90s"
      sleep 90
    else
      log "$name: non-shm failure, giving up"; return
    fi
  done
  [ $rc -ne 0 ] && { log "SKIP EVAL $name"; return; }
  local dir; dir=$(command ls -d outputs/train/"$name"/*/ | tail -1); dir="${dir%/}"
  EVAL_CACHE=$PWD/dataset/after_0328_train_dino_merged.h5 CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name heldout"
}

log "waiting for GPU slot held by graft_gil_aux to free up"
while pgrep -f "experiment_name=graft_gil_aux " > /dev/null; do sleep 30; done
log "graft_gil_aux ended, slot free"

run 0 graft_flow_scratch_control use_goal=false use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=false

log "QUEUE wave3b COMPLETE"
