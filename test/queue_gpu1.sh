#!/bin/bash
# GPU1 queue (2026-07-13 night). Runs in parallel with glidar_abs on GPU0.
#   nohup bash test/queue_gpu1.sh > /dev/null 2>&1 &
# Two threads of inquiry:
#   (1) offline residual = advantage-weighted BC (does reward-weighting the
#       fast-approach demo segments beat plain BC? — the slowness fix)
#   (2) ddpm ablation = clean from-scratch ddpm baselines to compare backbones
#       WITHOUT the warm-start confound of the weekend's ddpm_goal_auxw
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
QLOG=outputs/queue_gpu1.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

run(){  # $1=experiment_name, rest = hydra overrides
  name=$1; shift
  log "TRAIN START $name ($*)"
  python -u scripts/train.py experiment_name="$name" device=cuda:0 save_interval=1000 "$@" \
    > "outputs/train_$name.log" 2>&1
  rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && { log "SKIP EVAL $name (train failed)"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  # train-set eval
  python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name"
  # held-out eval
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout \
    python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name heldout"
}

# (1) offline residual (advantage-weighted BC), flow, from-scratch 20ep — the
#     direct counterpart to flow_goal_scratch20 (same everything + adv_weight).
run flow_goal_adv        adv_weight=true num_epochs=20
# (2a) clean ddpm baseline (matches flow_goal_scratch20 config on ddpm backbone)
run ddpm_goal_scratch20  diffusion_backbone=ddpm num_epochs=20
# (2b) ddpm + offline residual (does AWR help the ddpm backbone too?)
run ddpm_goal_adv        diffusion_backbone=ddpm adv_weight=true num_epochs=20
log "QUEUE GPU1 COMPLETE"
