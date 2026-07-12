#!/bin/bash
# Weekend ablation queue (2026-07-10). Sequential train -> eval -> next run.
# Performance axes: goal-reaching (open-loop ADE/FDE) + precision (aux near-dock mm).
# Ordered so the highest-expected-value runs land first if the weekend runs short.
# Usage: nohup bash test/weekend_queue.sh <pid-of-running-ddpm-train> &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
QLOG=outputs/weekend_queue.log
AUXW_CKPT=$PWD/outputs/train/flow_goal_auxw/2026-07-10_15-36-56/checkpoint_step_4230.pt

log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

# 0. wait for the already-running ddpm_goal_auxw, then eval it
if [ -n "${1:-}" ]; then
  log "waiting for ddpm_goal_auxw (pid $1)"
  while kill -0 "$1" 2>/dev/null; do sleep 120; done
  log "ddpm_goal_auxw finished"
  DDPM_DIR=$(ls -d outputs/train/ddpm_goal_auxw/* 2>/dev/null | tail -1)
  [ -n "$DDPM_DIR" ] && { python -u test/eval_run.py "$DDPM_DIR" >> "$QLOG" 2>&1 || log "EVAL FAILED ddpm_goal_auxw"; }
fi

run(){  # $1=experiment_name, rest = hydra overrides
  name=$1; shift
  log "TRAIN START $name ($*)"
  python -u scripts/train.py experiment_name="$name" save_interval=1000 "$@" \
    > "outputs/train_$name.log" 2>&1
  rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && return
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name"
}

# --- warm-starts from flow_goal_auxw (current best), fine-tune lr ------------
run flow_goal_auxw2     learning_rate=3e-5 num_epochs=20 init_from="$AUXW_CKPT"
run flow_goal_cfg07     learning_rate=3e-5 goal_mask_prob=0.7 init_from="$AUXW_CKPT"
run flow_goal_auxw_w2   learning_rate=3e-5 aux_weight=2.0 init_from="$AUXW_CKPT"
run flow_goal_p4        learning_rate=3e-5 aux_dist_power=4.0 init_from="$AUXW_CKPT"
# --- from-scratch science ablations (full lr, ema 0.999 already in config) ---
run flow_goal_scratch20 num_epochs=20
run flow_goal_nolidar   use_lidar_points=false
run flow_goal_nogoal    use_goal=false
log "QUEUE COMPLETE"
