#!/bin/bash
# 3x3 seed-replication sweep for the R-NoGoal/R-Goal/R-Geo ablation
# (user request 2026-07-26).
#
# WHY: the single-seed result (r_geo align_deg -0.153 vs r_nogoal, p=1.4e-9)
# cannot be trusted. That p-value pairs over FRAMES within one checkpoint pair;
# it does not touch TRAINING-SEED variance. docs/ablation_study_2026-07.md gives
# direct evidence that run-level noise on align_deg is ~0.1 deg:
# graft_goalimg_lidar (1.373) vs graft_lidar_goalimg (1.479) differ by 0.106 deg
# while differing only in goal-lidar conditioning, which that study's own paired
# analysis concluded is inert (conclusion 5, "무효과 ... 기각"). So the measured
# effect is only ~1.5x the noise floor. 3 seeds x 3 arms is the minimum design
# that can separate them.
#
# Budget = old_baseline's exact recipe (batch 16 x 100,000 steps ~= 7.4 epochs,
# ai-control/ai_models/postech_config.yaml @ git dca73b2). That is the ONE model
# that ever succeeded on the real robot (§2.12), so this also tests whether the
# token-sequence arms behave differently under its training regime. NOTE this
# means the new runs differ from the earlier seed-0 runs in BOTH budget and
# seed — compare within this sweep only, never against the batch256/16.9k runs.
#
# save_interval=10000 (not the config's 1000): 856 MB per checkpoint x 100 saves
# x 9 runs would be 770 GB.
#
# Usage: queue_rgeo_seeds.sh <gpu_id> <arm> <sensors_variant> <seed>
set -u
cd "$(dirname "$0")/.."
GPU_ID=${1:?gpu id}; ARM=${2:?arm}; VARIANT=${3:?sensors_variant}; SEED=${4:?seed}
NAME="${ARM}_b16s${SEED}"
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU_ID
QLOG="outputs/queue_seeds_${NAME}.log"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

log "TRAIN START $NAME (variant=$VARIANT seed=$SEED gpu=$GPU_ID batch=16 steps=100000)"
python -u scripts/train.py --config-name smr_rgeo \
  sensors_variant="$VARIANT" experiment_name="$NAME" seed="$SEED" \
  batch_size=16 num_epochs=null diffusion_gradient_steps=100000 \
  save_interval=10000 device=cuda:0 \
  > "outputs/train_$NAME.log" 2>&1
rc=$?
log "TRAIN END $NAME (rc=$rc)"
[ $rc -ne 0 ] && { log "SKIP EVAL $NAME (train failed)"; exit 1; }

DIR=$(ls -d --color=never outputs/train/"$NAME"/*/ | tail -1)
log "EVAL START $NAME ($DIR)"
COMMON="EVAL_H5=$PWD/dataset/after_0328_test.h5 EVAL_STATS_H5=$PWD/dataset/after_0328_train.h5 EVAL_TAG=heldout"
env $COMMON EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 \
  python -u test/eval_run_rgeo.py "$DIR" >> "$QLOG" 2>&1 || log "EVAL FAILED $NAME (rollout)"
# The axis the paper's claim actually lives on — see test/eval_align_rgeo.py.
env $COMMON python -u test/eval_align_rgeo.py "$DIR" >> "$QLOG" 2>&1 || log "EVAL FAILED $NAME (align)"
log "DONE $NAME"
