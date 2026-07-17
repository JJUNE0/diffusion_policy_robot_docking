#!/bin/bash
# Watchdog for the 6-cell graft queue (2026-07-17). Polls every 2 min; if a
# cell's training process has died WITHOUT the queue script logging a clean
# "TRAIN END" (i.e. killed by OOM or crashed silently), it's relaunched with
# HALVED CPU load: num_workers 4->2, prefetch_factor 1 (unchanged, already
# minimal). GPU batch_size/epochs/step schedule stay identical so the run
# resumes the ablation on equal footing (fresh step 0 -- init_from reloads the
# ORIGINAL checkpoint_step_100000.pt trunk, not the half-trained state, since
# these runs don't checkpoint often enough at 10ep/save_interval=5000 to
# resume cleanly mid-run; a graft warm-start restart is cheap early on).
#
#   nohup setsid bash test/watchdog_graft6.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/home/work/yes/bin/python
OLD=$PWD/outputs/checkpoint_step_100000.pt
WLOG=outputs/watchdog_graft6.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$WLOG"; }

declare -A GPU=( [graft_g0_control]=0 [graft_g0_awr]=0 [graft_goalimg]=0
                 [graft_goallidar]=1 [graft_goalimg_lidar]=1 [graft_g5_full]=1 )
declare -A OVERRIDES=(
  [graft_g0_control]="use_goal=false use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=false"
  [graft_g0_awr]="use_goal=false use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=true adv_mode=precision"
  [graft_goalimg]="use_goal=true use_lidar_points=false use_aux_pose=false use_goal_lidar=false adv_weight=false"
  [graft_goallidar]="use_goal=false use_lidar_points=true use_aux_pose=false use_goal_lidar=true adv_weight=false"
  [graft_goalimg_lidar]="use_goal=true use_lidar_points=true use_aux_pose=false use_goal_lidar=true adv_weight=false"
  [graft_g5_full]="use_goal=true use_lidar_points=true use_aux_pose=true use_goal_lidar=true adv_weight=true adv_mode=precision"
)

log "watchdog armed: polling every 120s, restart with num_workers=2 on unexplained death"
while true; do
  sleep 120
  for name in "${!GPU[@]}"; do
    dir_done=$(ls -d outputs/train/"$name"/*/checkpoint_step_*.pt 2>/dev/null | grep -v "_${name}$" | tail -1)
    # already fully finished? (final checkpoint at total_gradient_steps written)
    if grep -q "^\[.*\] TRAIN END $name (rc=0)\$" outputs/queue_graft6.log outputs/watchdog_graft6.log 2>/dev/null; then
      continue
    fi
    running=$(pgrep -f "experiment_name=$name " || true)
    if [ -z "$running" ]; then
      log "$name: NOT RUNNING and no clean TRAIN END -> assuming crash/OOM, restarting (num_workers=2)"
      gpu=${GPU[$name]}
      rm -rf "outputs/train/$name"
      CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -u scripts/train.py experiment_name="$name" device=cuda:0 \
        init_from="$OLD" diffusion_backbone=ddpm use_room1=true use_dino_cache=false \
        learning_rate=3e-5 num_epochs=10 batch_size=32 num_workers=1 prefetch_factor=1 \
        save_interval=5000 ${OVERRIDES[$name]} \
        > "outputs/train_${name}.log" 2>&1 &
      disown
      log "$name: relaunched (pid $!) with num_workers=2"
    fi
  done
done
