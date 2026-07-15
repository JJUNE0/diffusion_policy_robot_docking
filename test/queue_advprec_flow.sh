#!/bin/bash
# Precision-AWR focused study on flow + goal-image (2026-07-15).
# Narrowed per user: prove the precision reward on ONE cell (flow, goal-image)
# before the 2x2 backbone/goal-cond expansion. Compares:
#   (A) 학습 중 적용   = from-scratch 20ep + precision reward
#   (B) pretrained     = warm-start from scratch20 + precision reward, lr 3e-5
#   x angle-only vs angle+x-position reward
# = 4 runs. Baseline (no reward) = flow_goal_scratch20 (already done):
#   align 1.36 deg (demo p25 0.58) | xpos 20.1 mm (demo p25 7.9)
# A working reward moves policy toward the demo-p25 line.
#
# Position reward: x only. Measured on 145 demos vs ICP noise floor —
#   final x spread 1.9x (weak but real), y 1.2x (= noise). adv_w_pos targets x.
#
# Runs AFTER the current GPU0/GPU1 queues drain (does not touch them).
#   nohup setsid bash test/queue_advprec_flow.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
QLOG=outputs/queue_advprec_flow.log
S20=$PWD/outputs/train/flow_goal_scratch20/2026-07-12_05-19-04/checkpoint_step_8460.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

log "waiting for GPU0/GPU1 queues to finish"
while pgrep -f "scripts/train.py" > /dev/null; do sleep 300; done
log "GPUs free -> precision-AWR flow study"

run(){ gpu=$1; name=$2; shift 2
  log "TRAIN START $name (gpu$gpu: $*)"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/train.py experiment_name="$name" device=cuda:0 \
    adv_weight=true adv_mode=precision save_interval=1000 "$@" > "outputs/train_$name.log" 2>&1
  rc=$?; log "TRAIN END $name (rc=$rc)"; [ $rc -ne 0 ] && return
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  CUDA_VISIBLE_DEVICES=$gpu python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=$gpu \
    python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAIL $name heldout"
}

# GPU0: (A) from-scratch 20ep     GPU1: (B) warm-start from scratch20, lr 3e-5
(
  run 0 advprec_A_angle  num_epochs=20                                   # (A) 각도만
  run 0 advprec_A_posang num_epochs=20 adv_w_pos=1.0                     # (A) 각도+x위치
  log "GPU0 (A) branch done"
) &
(
  run 1 advprec_B_angle  init_from="$S20" learning_rate=3e-5 num_epochs=10                 # (B) 각도만
  run 1 advprec_B_posang init_from="$S20" learning_rate=3e-5 num_epochs=10 adv_w_pos=1.0   # (B) 각도+x위치
  log "GPU1 (B) branch done"
) &
wait
log "QUEUE advprec_flow COMPLETE"
