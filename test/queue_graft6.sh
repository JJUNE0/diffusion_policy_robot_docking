#!/bin/bash
# Graft ablation, 6 cells (2026-07-17, user-approved): all grafted from the
# PROVEN-ONLINE old baseline (outputs/checkpoint_step_100000.pt, 2-cam DDPM),
# 10 epochs each, live 2-camera DINO (no cache -- see memory h5-io-quirks),
# batch 32 / workers 4 / prefetch 1 (the only combo that doesn't hit the 1GB
# shm ceiling). Run CONCURRENTLY, 3 per GPU -- each job uses ~7GB/81GB and
# ~8% GPU util at batch 32, so 3x parallel per GPU has ample headroom.
#
#   G0            control: no new branches, no AWR
#   G0_awr        + AWR(precision) only -- isolates AWR's own effect
#   G_goalimg     + goal-image conditioning only
#   G_goallidar   + lidar branch + goal-lidar conditioning (no goal image)
#   G_goalimg_lidar  + goal-image + goal-lidar together (no aux, no AWR)
#   G5            full stack: goal-image + lidar + aux + goal-lidar + AWR(precision)
#
#   nohup setsid bash test/queue_graft6.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/home/work/yes/bin/python
QLOG=outputs/queue_graft6.log
OLD=$PWD/outputs/checkpoint_step_100000.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

run(){  # $1=gpu $2=name, rest=overrides
  gpu=$1; name=$2; shift 2
  log "TRAIN START $name (gpu$gpu: $*)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
    init_from="$OLD" diffusion_backbone=ddpm use_room1=true use_dino_cache=false \
    learning_rate=3e-5 num_epochs=10 batch_size=32 num_workers=4 prefetch_factor=1 \
    save_interval=5000 "$@" > "outputs/train_$name.log" 2>&1
  rc=$?; log "TRAIN END $name (rc=$rc)"; [ $rc -ne 0 ] && { log "SKIP EVAL $name"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  EVAL_CACHE=$PWD/dataset/after_0328_train_dino_merged.h5 CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name heldout"
}

# GPU0: control, AWR-only, goal-image-only
run 0 graft_g0_control    use_goal=false use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=false &
run 0 graft_g0_awr        use_goal=false use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=true adv_mode=precision &
run 0 graft_goalimg       use_goal=true  use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=false &

# GPU1: goal-lidar-only, goal-image+goal-lidar, full stack (G5)
run 1 graft_goallidar     use_goal=false use_lidar_points=true  use_aux_pose=false use_goal_lidar=true  adv_weight=false &
run 1 graft_goalimg_lidar use_goal=true  use_lidar_points=true  use_aux_pose=false use_goal_lidar=true  adv_weight=false &
run 1 graft_g5_full       use_goal=true  use_lidar_points=true  use_aux_pose=true  use_goal_lidar=true  adv_weight=true adv_mode=precision &

wait
log "QUEUE graft6 COMPLETE"
