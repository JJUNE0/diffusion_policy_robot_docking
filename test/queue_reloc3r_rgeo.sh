#!/bin/bash
# R-NoGoal / R-Goal / R-Geo training queue (docs/0725_reloc3r_test/reloc3r/
# reloc3r_0725.md, user request 2026-07-25). Three arms, ONE config
# (configs/robot/smr_rgeo.yaml) switched purely via `sensors_variant=`
# (Hydra config group) -- everything else (network, optimizer, batch size,
# epochs, dataset, sampler) is byte-identical across arms, verified by
# test/test_reloc3r_acceptance_criteria.py::test_base_and_geometry_share_everything_except_geometry_token.
#
#   r_nogoal : sensors_variant=no_goal                    (Z_rgb, Z_lidar, Z_wheel)
#   r_goal   : sensors_variant=goal_appearance             (+ Z_goal)
#   r_geo    : sensors_variant=goal_appearance_geometry    (+ Z_reloc geometry token)
#
# Each arm scored on held-out test split via test/eval_run_rgeo.py (ADE/FDE/
# velRMSE open-loop rollout; no aux/align metrics -- no aux head in this
# architecture, per spec).
#
#   nohup setsid bash test/queue_reloc3r_rgeo.sh > outputs/queue_rgeo.log 2>&1 < /dev/null &
# Usage: queue_reloc3r_rgeo.sh <gpu_id> <log_suffix> <arm:variant> [<arm:variant> ...]
#   e.g. queue_reloc3r_rgeo.sh 0 gpu0 r_nogoal:no_goal
#        queue_reloc3r_rgeo.sh 1 gpu1 r_goal:goal_appearance r_geo:goal_appearance_geometry
set -u
cd "$(dirname "$0")/.."
PY=/tmp/claude-1100/-home-work--postech-diffusion-policy-robot-docking/88194c75-b380-43b0-8c57-c004af706396/scratchpad/reloc3r_eval/venv/bin/python
GPU_ID=${1:?gpu id required}; shift
SUFFIX=${1:?log suffix required}; shift
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU_ID
QLOG=outputs/queue_rgeo_$SUFFIX.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

run(){  # $1=experiment_name  $2=sensors_variant
  name=$1; variant=$2
  log "TRAIN START $name (sensors_variant=$variant)"
  $PY -u scripts/train.py --config-name smr_rgeo sensors_variant="$variant" \
    experiment_name="$name" device=cuda:0 \
    > "outputs/train_$name.log" 2>&1
  rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && { log "SKIP EVAL $name (train failed)"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  EVAL_H5=$PWD/dataset/after_0328_test.h5 EVAL_STATS_H5=$PWD/dataset/after_0328_train.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout \
    $PY -u test/eval_run_rgeo.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name (heldout)"
}

for arm_variant in "$@"; do
  arm="${arm_variant%%:*}"
  variant="${arm_variant##*:}"
  run "$arm" "$variant"
done
log "QUEUE ($SUFFIX) COMPLETE"
