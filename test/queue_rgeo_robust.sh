#!/bin/bash
# Robust-action-normalization wave for R-NoGoal / R-Goal / R-Geo (2026-07-25).
#
# WHY: `minmax` action normalization lets outliers set the scale -- vx spans
# +-0.27 m/s while p1/p99 are only -0.075/+0.084 -- so the 2-15 mm/s docking
# end-game occupies under 3% of the [-1,1] output range and drowns in sampler
# noise. Measured contributor to the terminal stall seen on the robot on
# 2026-07-23 (test/terminal_metric.py). `robust` puts the bounds at +-p99,
# symmetric so vx=0 maps exactly to 0: 3.36x resolution on vx, 3.31x on wz,
# 1.7% of frames clipped. Architecture unchanged -- this is a clean one-knob
# ablation against the minmax wave.
#
# WHY ALL THREE ARMS: the normalizer changes the action space the model learns
# in, so a robust r_geo is NOT comparable against a minmax r_goal/r_nogoal.
# Run the wave as a matched set or the geometry-token conclusion is confounded.
# (Same reason smr_rgeo.yaml still defaults to action_norm: minmax -- the
# existing wave was trained with it.)
#
# Everything else is byte-identical to test/queue_reloc3r_rgeo.sh.
#
#   nohup setsid bash test/queue_rgeo_robust.sh 0 gpu0 > outputs/queue_rgeo_robust.log 2>&1 < /dev/null &
# Usage: queue_rgeo_robust.sh <gpu_id> <log_suffix> [<arm:variant> ...]
#   default arms: all three, suffixed `_rb` so they never collide with the
#   minmax runs already in outputs/train/.
set -u
cd "$(dirname "$0")/.."
PY=/tmp/claude-1100/-home-work--postech-diffusion-policy-robot-docking/88194c75-b380-43b0-8c57-c004af706396/scratchpad/reloc3r_eval/venv/bin/python
GPU_ID=${1:?gpu id required}; shift
SUFFIX=${1:?log suffix required}; shift
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU_ID
QLOG=outputs/queue_rgeo_robust_$SUFFIX.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

run(){  # $1=experiment_name  $2=sensors_variant
  name=$1; variant=$2
  log "TRAIN START $name (sensors_variant=$variant, action_norm=robust)"
  $PY -u scripts/train.py --config-name smr_rgeo sensors_variant="$variant" \
    experiment_name="$name" device=cuda:0 action_norm=robust \
    > "outputs/train_$name.log" 2>&1
  rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && { log "SKIP EVAL $name (train failed)"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  # eval_run_rgeo asserts the dataset normalizer matches the checkpoint's, so a
  # config/checkpoint mismatch fails loudly here instead of silently scoring
  # the model on shifted inputs.
  EVAL_H5=$PWD/dataset/after_0328_test.h5 EVAL_STATS_H5=$PWD/dataset/after_0328_train.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout \
    $PY -u test/eval_run_rgeo.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name (heldout)"
}

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(r_nogoal_rb:no_goal r_goal_rb:goal_appearance r_geo_rb:goal_appearance_geometry)
fi
for arm_variant in "${ARMS[@]}"; do
  run "${arm_variant%%:*}" "${arm_variant##*:}"
done
log "QUEUE ($SUFFIX) COMPLETE"
