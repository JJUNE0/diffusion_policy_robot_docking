#!/bin/bash
# Aux-head axis of the graft ablation (2026-07-18, user-approved scenario).
# Four cells, all grafted from outputs/checkpoint_step_100000.pt with the
# same recipe as graft6 (10ep, batch 128, live 2-cam DINO), num_workers=8:
#
#   graft_lidar          lidar input ONLY (no goal-lidar; the existing
#                        graft_goallidar cell bundled both, so lidar's solo
#                        effect was never measured)
#   graft_lidar_goalimg  goal-image + lidar (no goal-lidar) — vs the existing
#                        graft_goalimg_lidar this isolates goal-lidar's margin
#   graft_goallidar_aux  lidar + goal-lidar + AUX HEAD — aux effect on the
#                        goallidar base; near_mm becomes measurable
#   graft_gil_aux        goal-image + lidar + goal-lidar + AUX = g5_full minus
#                        AWR — if this scores well, g5_full's degradation is
#                        pinned on AWR
#
# Runs 2 per GPU alongside the already-running gil_awr_p1/p2 (=> 3 per GPU,
# the load level verified by graft6). Starts are staggered to dodge the
# first-batch shm allocation race hit on 07-17.
#
#   nohup setsid bash test/queue_graft_aux4.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/usr/bin/python3
QLOG=outputs/queue_graft_aux4.log
OLD=$PWD/outputs/checkpoint_step_100000.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

run(){  # $1=gpu $2=name, rest=overrides
  gpu=$1; name=$2; shift 2
  log "TRAIN START $name (gpu$gpu: $*)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
    init_from="$OLD" diffusion_backbone=ddpm use_room1=true use_dino_cache=false \
    learning_rate=3e-5 num_epochs=10 batch_size=128 num_workers=8 prefetch_factor=1 \
    save_interval=5000 adv_weight=false "$@" > "outputs/train_$name.log" 2>&1
  rc=$?; log "TRAIN END $name (rc=$rc)"; [ $rc -ne 0 ] && { log "SKIP EVAL $name"; return; }
  dir=$(command ls -d outputs/train/"$name"/*/ | tail -1); dir="${dir%/}"
  EVAL_CACHE=$PWD/dataset/after_0328_train_dino_merged.h5 CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name heldout"
}

# GPU0: lidar-solo + the key g5_full-minus-AWR cell
run 0 graft_lidar         use_goal=false use_lidar_points=true use_aux_pose=false use_goal_lidar=false &
sleep 20
run 0 graft_gil_aux       use_goal=true  use_lidar_points=true use_aux_pose=true  use_goal_lidar=true  &
sleep 20
# GPU1: goal-image+lidar (no goal-lidar) + aux on the goallidar base
run 1 graft_lidar_goalimg use_goal=true  use_lidar_points=true use_aux_pose=false use_goal_lidar=false &
sleep 20
run 1 graft_goallidar_aux use_goal=false use_lidar_points=true use_aux_pose=true  use_goal_lidar=true  &

wait
log "QUEUE graft_aux4 COMPLETE"
