#!/bin/bash
# Wave-2 graft ablation launcher (2026-07-18): 6 cells, 3 per GPU, workers 4.
#
# workers=8 was requested but is PHYSICALLY IMPOSSIBLE on this box: /dev/shm
# is fixed at 2G (remount needs root; denied), each job keeps
# num_workers x prefetch_factor collated batches in shm, and 6 jobs x 8
# workers ~= double the 6x4 budget graft6 proved out — the 19:08 workers-8
# attempt crashed in STEADY STATE (not just the warmup race) as soon as the
# second job's workers spun up. 6 x 4 = the known-stable ceiling.
# Replaces the separate queue_graft_awr2.sh / queue_graft_aux4.sh launches —
# both of the awr2 cells died at first batch in the /dev/shm allocation race
# (2G shm, transient burst when jobs hit their first collate together; same
# race killed 3/6 graft6 cells on 07-17). Two defenses here:
#   * SERIALIZED WARMUP: each job must print "Step 0" (or time out) before the
#     next one starts — the burst never overlaps.
#   * AUTO-RETRY: a job whose log shows the shm RuntimeError is relaunched
#     (up to 4 attempts, 90s cooldown) instead of being silently lost.
#
# Cells (all grafted from outputs/checkpoint_step_100000.pt, 10ep, batch 128):
#   GPU0: gil_awr_p1        goalimg_lidar + AWR precision v1
#         lidar             lidar input only (no goal-lidar, no aux)
#         gil_aux           goalimg+lidar+goal-lidar+aux = g5_full minus AWR
#   GPU1: gil_awr_p2        goalimg_lidar + AWR precision_v2 (07-18 formula)
#         lidar_goalimg     goal-image + lidar (no goal-lidar)
#         goallidar_aux     lidar + goal-lidar + aux
#
#   nohup setsid bash test/queue_wave2.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/usr/bin/python3
QLOG=outputs/queue_wave2.log
OLD=$PWD/outputs/checkpoint_step_100000.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

train_once(){  # $1=gpu $2=name, rest=overrides
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
    init_from="$OLD" diffusion_backbone=ddpm use_room1=true use_dino_cache=false \
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
    if grep -q "unable to allocate shared memory" "outputs/train_$name.log"; then
      log "$name: shm race casualty, retrying in 90s"
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

wait_warm(){  # block until $1's log shows the first training step (or 420s cap)
  local name=$1 t=0
  until grep -q "^Step 0 " "outputs/train_$name.log" 2>/dev/null; do
    sleep 10; t=$((t+10))
    [ $t -ge 420 ] && { log "WARN $name not warm after ${t}s, starting next anyway"; return; }
    # if it already died (retry loop owns it), don't hold up the launch train
    grep -q "unable to allocate shared memory" "outputs/train_$name.log" 2>/dev/null && return
  done
  log "$name warm (Step 0 seen after ${t}s)"
}

run 0 graft_gil_awr_p1    use_goal=true  use_lidar_points=true use_aux_pose=false use_goal_lidar=true  adv_weight=true adv_mode=precision &
wait_warm graft_gil_awr_p1
run 1 graft_gil_awr_p2    use_goal=true  use_lidar_points=true use_aux_pose=false use_goal_lidar=true  adv_weight=true adv_mode=precision_v2 &
wait_warm graft_gil_awr_p2
run 0 graft_lidar         use_goal=false use_lidar_points=true use_aux_pose=false use_goal_lidar=false adv_weight=false &
wait_warm graft_lidar
run 1 graft_lidar_goalimg use_goal=true  use_lidar_points=true use_aux_pose=false use_goal_lidar=false adv_weight=false &
wait_warm graft_lidar_goalimg
run 0 graft_gil_aux       use_goal=true  use_lidar_points=true use_aux_pose=true  use_goal_lidar=true  adv_weight=false &
wait_warm graft_gil_aux
run 1 graft_goallidar_aux use_goal=false use_lidar_points=true use_aux_pose=true  use_goal_lidar=true  adv_weight=false &

wait
log "QUEUE wave2 COMPLETE"
