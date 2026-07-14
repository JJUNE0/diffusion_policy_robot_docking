#!/bin/bash
# CLEAN ablation queue (2026-07-14, GPU0). Re-runs every knob ablation as a
# TRUE single-variable comparison against flow_goal_scratch20.
#
# Why re-run: the weekend versions of these ablations are confounded —
#   * cfg07 / auxw_w2 / p4 were WARM-STARTED from flow_goal_auxw, which itself
#     inherited weights trained under the OLD uniform aux loss, and carry a
#     longer effective training history (4230+4230+run) than scratch20 (8460).
#   * nolidar / nogoal were only 10 epochs vs scratch20's 20 — the modality
#     effect is entangled with half the training budget.
# Every run here: from-scratch, 20 epochs, lr 1e-4, ema 0.999, identical to
# flow_goal_scratch20 EXCEPT the one knob named in the run name.
#
# Baseline to compare against (already done): flow_goal_scratch20
#   held-out: near 5.18mm | ADE 5.2 | FDE 11.1
#
#   nohup setsid bash test/queue_gpu0_clean_ablation.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
QLOG=outputs/queue_gpu0.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

run(){  # $1=experiment_name, rest = hydra overrides
  name=$1; shift
  log "TRAIN START $name ($*)"
  python -u scripts/train.py experiment_name="$name" device=cuda:0 \
    num_epochs=20 save_interval=1000 "$@" > "outputs/train_$name.log" 2>&1
  rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && { log "SKIP EVAL $name (train failed)"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout \
    python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name heldout"
}

# --- core hypothesis (highest value): modality ablations at MATCHED 20 epochs -
# "LiDAR = 정밀 담당, goal = 접근 담당" — the study's central claim. The weekend
# versions had a 10-vs-20 epoch confound, so these are the ones that actually
# decide it.
run s20_nolidar  use_lidar_points=false
run s20_nogoal   use_goal=false
# --- hyperparameter knobs, now single-variable vs scratch20 -------------------
run s20_cfg07    goal_mask_prob=0.7      # unconditional branch -> enables w_cfg>1 later
run s20_w2       aux_weight=2.0          # precision-vs-approach tradeoff
run s20_p4       aux_dist_power=4.0      # sharper near-dock weighting
log "QUEUE GPU0 (clean ablation) COMPLETE"
