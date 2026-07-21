#!/bin/bash
# Wave-3 (2026-07-19, user-approved): rectified-flow backbone re-run of two
# graft6 cells (control, goalimg_lidar) — same graft ablation, same warm-start
# trunk (outputs/checkpoint_step_100000.pt, itself a DDPM checkpoint), but
# diffusion_backbone=rectified_flow instead of ddpm. This is a legitimate
# cross-backbone warm-start: diffusion_backbone only swaps the diffusion-
# process wrapper (ContinuousRectifiedFlow vs ContinuousDiffusionSDE) around
# the SAME nn_diffusion_model/nn_condition network (utils/setups.py
# _select_backbone comment: "flow vs ddpm ablation = one config flag,
# orthogonal to the sensors") — model_state_dict/ema_state_dict load
# unchanged regardless of which backbone trained them.
#
# GPU SLOTS: 0/1 are already at 3 jobs each (graft6-verified ceiling) via
# queue_wave2.sh. Rather than push to 4/GPU (untested, and workers=8 already
# proved 3->4 breaks shm), each job here WAITS for its designated wave2
# slot-holder to finish before starting, so concurrency never exceeds 3/GPU.
#   GPU0 slot freed by: graft_lidar        (was ~92% done when this was written)
#   GPU1 slot freed by: graft_goallidar_aux (ditto)
#
#   graft_flow_control          rectified-flow analog of graft_g0_control
#   graft_flow_goalimg_lidar    rectified-flow analog of graft_goalimg_lidar
#
#   nohup setsid bash test/queue_wave3.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/usr/bin/python3
QLOG=outputs/queue_wave3.log
OLD=$PWD/outputs/checkpoint_step_100000.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

train_once(){
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
    init_from="$OLD" diffusion_backbone=rectified_flow use_room1=true use_dino_cache=false \
    learning_rate=3e-5 num_epochs=10 batch_size=128 num_workers=4 prefetch_factor=1 \
    save_interval=5000 "$@" > "outputs/train_$name.log" 2>&1
}

run(){  # $1=gpu $2=name, rest=overrides — train with shm-retry, then eval both sets
  local gpu=$1 name=$2 rc attempt; shift 2
  for attempt in 1 2 3 4; do
    log "TRAIN START $name attempt$attempt (gpu$gpu: $*)"
    train_once "$gpu" "$name" "$@"; rc=$?
    log "TRAIN END $name attempt$attempt (rc=$rc)"
    [ $rc -eq 0 ] && break
    if grep -qE "unable to allocate shared memory|CUDA out of memory" "outputs/train_$name.log"; then
      log "$name: shm/OOM race casualty (transient GPU contention from a slot handoff), retrying in 90s"
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

wait_slot(){  # block until $1's training process is completely gone
  local holder=$1
  log "waiting for GPU slot held by $holder to free up"
  while pgrep -f "experiment_name=$holder " > /dev/null; do sleep 30; done
  log "$holder ended, slot free"
}

(
  wait_slot graft_lidar
  run 0 graft_flow_control use_goal=false use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=false
) &

(
  wait_slot graft_goallidar_aux
  run 1 graft_flow_goalimg_lidar use_goal=true use_lidar_points=true use_aux_pose=false use_goal_lidar=true adv_weight=false
) &

wait
log "QUEUE wave3 COMPLETE"
