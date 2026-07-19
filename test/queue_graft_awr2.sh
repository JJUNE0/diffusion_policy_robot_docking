#!/bin/bash
# AWR-formula follow-up to the graft6 ablation (2026-07-18). Base config is
# the graft_goalimg_lidar cell (best held-out FDE of graft6: goal-image +
# lidar + goal-lidar, no aux, grafted from outputs/checkpoint_step_100000.pt),
# now +AWR in two flavours to separate "AWR on a good base" from "the formula":
#
#   graft_gil_awr_p1   + AWR adv_mode=precision    (v1 — same formula that hurt
#                        graft_g0_awr: ungated align + gated speed term)
#   graft_gil_awr_p2   + AWR adv_mode=precision_v2 (07-18 formula: no speed
#                        term, align reward gated INTO the precision zone)
#
# Comparisons this enables (all vs the existing graft_goalimg_lidar json):
#   p1 vs base  -> does v1-AWR hurt the good base the way it hurt g0?
#   p2 vs p1    -> does the formula fix the wz-noise regression?
#   p2 vs base  -> is v2-AWR a net win?
#
# batch 128 / workers 4 / 10 epochs, same as graft6. One job per GPU.
#
#   nohup setsid bash test/queue_graft_awr2.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/usr/bin/python3
QLOG=outputs/queue_graft_awr2.log
OLD=$PWD/outputs/checkpoint_step_100000.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

run(){  # $1=gpu $2=name, rest=overrides
  gpu=$1; name=$2; shift 2
  log "TRAIN START $name (gpu$gpu: $*)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
    init_from="$OLD" diffusion_backbone=ddpm use_room1=true use_dino_cache=false \
    learning_rate=3e-5 num_epochs=10 batch_size=128 num_workers=4 prefetch_factor=1 \
    save_interval=5000 use_goal=true use_lidar_points=true use_aux_pose=false \
    use_goal_lidar=true adv_weight=true "$@" > "outputs/train_$name.log" 2>&1
  rc=$?; log "TRAIN END $name (rc=$rc)"; [ $rc -ne 0 ] && { log "SKIP EVAL $name"; return; }
  dir=$(command ls -d outputs/train/"$name"/*/ | tail -1); dir="${dir%/}"
  EVAL_CACHE=$PWD/dataset/after_0328_train_dino_merged.h5 CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name heldout"
}

run 0 graft_gil_awr_p1 adv_mode=precision &
sleep 20   # stagger: avoid the first-batch shm allocation race (07-17)
run 1 graft_gil_awr_p2 adv_mode=precision_v2 &

wait
log "QUEUE graft_awr2 COMPLETE"
