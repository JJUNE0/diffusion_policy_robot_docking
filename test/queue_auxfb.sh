#!/bin/bash
# Aux-pose FEEDBACK ablation (2026-07-21, user-approved). New mechanism
# (sensor_fusion_condition.py use_aux_feedback): the ICP-distilled dock-pose
# estimate [x,y,sin,cos] is projected (zero-init Linear, starts as a no-op)
# and ADDED back into the conditioning readout that the diffusion trunk sees,
# giving the last-approach trajectory an explicit dock target to converge to.
# Hypothesis (user): aux-pose feedback may BE the true form of "goal-lidar"
# (both inject dock geometry; §2.10 found goal-lidar conditioning had no
# effect, possibly because aux_pred already carries that info more precisely).
#
# Same graft recipe as §9/§10: warm-start from the old baseline
# (checkpoint_step_100000.pt, 2-cam DDPM), 10 epoch, batch 128, workers 4,
# live 2-cam DINO, aux_relative=false (absolute dock pose, matches gil_aux).
# All three carry lidar + aux_pose + aux_feedback; they differ only in the
# goal branches:
#
#   graft_auxfb_lidar    baseline + lidar + aux + FEEDBACK           (no goal)
#   graft_auxfb_goalimg  + goal-image                                (no goal-lidar)
#   graft_auxfb_full     + goal-image + goal-lidar
#
# If auxfb_lidar already matches/beats the goal-lidar cells, that supports the
# "feedback == goal-lidar" hypothesis. 3 cells / 2 GPUs (GPU0 x2, GPU1 x1),
# well under the 6x4-worker shm ceiling; serialized warmup + shm/OOM retry
# kept from queue_wave2 out of caution.
#
#   nohup setsid bash test/queue_auxfb.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/usr/bin/python3
QLOG=outputs/queue_auxfb.log
OLD=$PWD/outputs/checkpoint_step_100000.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

train_once(){
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
    init_from="$OLD" diffusion_backbone=ddpm use_room1=true use_dino_cache=false \
    learning_rate=3e-5 num_epochs=10 batch_size=128 num_workers=4 prefetch_factor=1 \
    save_interval=5000 use_lidar_points=true use_aux_pose=true use_aux_feedback=true \
    adv_weight=false "$@" > "outputs/train_$name.log" 2>&1
}

run(){  # $1=gpu $2=name, rest=overrides — train (shm/OOM-retry) then eval both
  local gpu=$1 name=$2 rc attempt; shift 2
  for attempt in 1 2 3 4; do
    log "TRAIN START $name attempt$attempt (gpu$gpu: $*)"
    train_once "$gpu" "$name" "$@"; rc=$?
    log "TRAIN END $name attempt$attempt (rc=$rc)"
    [ $rc -eq 0 ] && break
    if grep -qE "unable to allocate shared memory|CUDA out of memory" "outputs/train_$name.log"; then
      log "$name: shm/OOM race casualty, retrying in 90s"; sleep 90
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

wait_warm(){  # block until $1 prints Step 0 (or 420s cap / crash)
  local name=$1 t=0
  until grep -q "^Step 0 " "outputs/train_$name.log" 2>/dev/null; do
    sleep 10; t=$((t+10)); [ $t -ge 420 ] && { log "WARN $name not warm after ${t}s"; return; }
    grep -q "unable to allocate shared memory" "outputs/train_$name.log" 2>/dev/null && return
  done
  log "$name warm (Step 0 after ${t}s)"
}

run 0 graft_auxfb_lidar   use_goal=false use_goal_lidar=false &
wait_warm graft_auxfb_lidar
run 1 graft_auxfb_goalimg use_goal=true  use_goal_lidar=false &
wait_warm graft_auxfb_goalimg
run 0 graft_auxfb_full    use_goal=true  use_goal_lidar=true  &

wait
log "QUEUE auxfb COMPLETE"
