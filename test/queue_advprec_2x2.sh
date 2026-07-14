#!/bin/bash
# Precision-first AWR — 2x2 ablation (2026-07-14).
#   backbone  : flow (rectified flow) | ddpm
#   goal 조건 : goal-image only       | goal-image + goal-lidar (glidar)
# = 4 runs, all with adv_mode=precision, from-scratch 20ep, otherwise identical
#   to flow_goal_scratch20 -> every cell is a clean single-variable comparison.
#
# REWARD (adv_mode=precision), precision >> speed by design:
#   A = 1.0·z(정렬개선)          정밀도: horizon 동안 |yaw| 오차 감소량
#     + 1.0·z(최종정렬품질)      정밀도: 그 시연이 최종 몇 도로 도킹했나 (에피소드 크레딧)
#     + 0.3·z(접근속도)·gate(d)  속도  : dock 0.6m 안에서는 gate=0 (정밀 구간에서 속도 보상 OFF)
# 근거(145 시연 실측, ICP 노이즈 대비): 최종 x 1.9x / y 1.2x = 위치는 기구적으로 고정 →
# 배울 신호 없음. 최종 yaw 7.1x → 정밀도 신호는 사실상 '정렬(각도)'뿐.
#
# Baselines already trained (no reward, same recipe):
#   flow + goal-image      = flow_goal_scratch20     (held-out near 5.18 / FDE 11.1)
#   flow + goal-lidar      = flow_goal_glidar_abs    (held-out near 4.95 / FDE 10.0)
#   ddpm + goal-image      = ddpm_goal_scratch20     (GPU1 queue, in progress)
#
# Waits for both current queues to free their GPU, then runs 2 per GPU.
#   nohup setsid bash test/queue_advprec_2x2.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
QLOG=outputs/queue_advprec.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

# --- wait until no training process is left on the box -----------------------
log "waiting for the running queues (GPU0 clean-ablation, GPU1 adv/ddpm) to finish"
while pgrep -f "scripts/train.py" > /dev/null; do sleep 300; done
log "GPUs free -> starting the precision-AWR 2x2"

run(){  # $1=gpu, $2=name, rest=overrides
  gpu=$1; name=$2; shift 2
  log "TRAIN START $name (gpu$gpu: $*)"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/train.py experiment_name="$name" device=cuda:0 \
    adv_weight=true adv_mode=precision num_epochs=20 save_interval=1000 "$@" \
    > "outputs/train_$name.log" 2>&1
  rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && { log "SKIP EVAL $name (train failed)"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  CUDA_VISIBLE_DEVICES=$gpu python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name"
  EVAL_H5=dataset/after_0328_test.h5 EVAL_CACHE=dataset/after_0328_test_dino_bottom.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout CUDA_VISIBLE_DEVICES=$gpu \
    python -u test/eval_run.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name heldout"
}

# GPU0: flow 쪽 2개 / GPU1: ddpm 쪽 2개  (병렬)
(
  run 0 advprec_flow_goalimg                                          # flow + goal 이미지만
  run 0 advprec_flow_glidar  use_goal_lidar=true aux_relative=false   # flow + goal-lidar
  log "GPU0 branch complete"
) &
(
  run 1 advprec_ddpm_goalimg diffusion_backbone=ddpm                                        # ddpm + goal 이미지만
  run 1 advprec_ddpm_glidar  diffusion_backbone=ddpm use_goal_lidar=true aux_relative=false # ddpm + goal-lidar
  log "GPU1 branch complete"
) &
wait
log "QUEUE advprec 2x2 COMPLETE"
