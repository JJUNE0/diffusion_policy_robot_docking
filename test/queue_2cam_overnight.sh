#!/bin/bash
# Unattended overnight pipeline for the 2026-07-27 field demo (user request
# 2026-07-26 night): wait for the dino_top cache, smoke-test the 2-camera
# configs, train both arms, evaluate both, print a ranked summary.
#
# WHY TWO CAMERAS: the only model that ever docked successfully in the field
# (outputs/checkpoint_step_100000.pt) is also the only one with two cameras;
# every variant that failed on the robot had one. See
# configs/robot/sensors_variant/no_goal_2cam.yaml for the full field record.
#
# WHY A SMOKE GATE: a config typo would otherwise burn the entire night. The
# gate runs 3 gradient steps per arm and aborts the whole queue on failure.
#
#   nohup setsid bash test/queue_2cam_overnight.sh > outputs/queue_2cam.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
QLOG=outputs/queue_2cam.log
log(){ echo "[$(date '+%m-%d %H:%M')] $*"; }

TOP=dataset/after_0328_train_dino_top.h5
NROWS=225465

# ---- 1. wait for the dino_top precompute ----
log "waiting for $TOP to reach $NROWS rows"
while true; do
  if ! pgrep -f "precompute_dino_cache.*image_top" >/dev/null 2>&1; then
    n=$(python3 -c "
import h5py,sys
try:
    with h5py.File('$TOP','r') as f: print(f['dino_top'].shape[0])
except Exception: print(0)" 2>/dev/null)
    if [ "$n" = "$NROWS" ]; then log "cache ready ($n rows)"; break; fi
    log "ABORT: precompute process gone but cache has $n/$NROWS rows"; exit 1
  fi
  sleep 60
done

# ---- 2. smoke gate: 3 gradient steps per arm ----
for av in "r2cam_nogoal:no_goal_2cam" "r2cam_geo:goal_appearance_geometry_2cam"; do
  variant="${av##*:}"
  log "SMOKE $variant"
  CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --config-name smr_rgeo \
    sensors_variant="$variant" experiment_name="_smoke_$variant" seed=0 \
    batch_size=16 num_epochs=null diffusion_gradient_steps=3 \
    save_interval=1000000 device=cuda:0 > "outputs/smoke_$variant.log" 2>&1
  if [ $? -ne 0 ]; then
    log "ABORT: smoke failed for $variant"; tail -20 "outputs/smoke_$variant.log"; exit 1
  fi
  rm -rf "outputs/train/_smoke_$variant"
  log "SMOKE OK $variant"
done

# ---- 3. train both arms in parallel, then evaluate each ----
run(){  # $1=gpu $2=name $3=variant
  local gpu=$1 name=$2 variant=$3
  log "TRAIN START $name (gpu$gpu, batch16 x 100k)"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/train.py --config-name smr_rgeo \
    sensors_variant="$variant" experiment_name="$name" seed=0 \
    batch_size=16 num_epochs=null diffusion_gradient_steps=100000 \
    save_interval=10000 device=cuda:0 > "outputs/train_$name.log" 2>&1
  local rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && { log "SKIP EVAL $name"; return 1; }
  local dir; dir=$(ls -d --color=never outputs/train/"$name"/*/ | tail -1)
  log "EVAL $name ($dir)"
  local COMMON="EVAL_H5=$PWD/dataset/after_0328_test.h5 EVAL_STATS_H5=$PWD/dataset/after_0328_train.h5 EVAL_TAG=heldout"
  env $COMMON EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 CUDA_VISIBLE_DEVICES=$gpu \
    python -u test/eval_run_rgeo.py "$dir" 2>&1 | tail -6
  env $COMMON CUDA_VISIBLE_DEVICES=$gpu \
    python -u test/eval_align_rgeo.py "$dir" 2>&1 | tail -3
  log "DONE $name"
}

run 0 r2cam_nogoal no_goal_2cam &
P1=$!
run 1 r2cam_geo goal_appearance_geometry_2cam &
P2=$!
wait $P1 $P2

# ---- 4. ranked summary; steering bias is the screening axis ----
log "=== SUMMARY (bias is the only offline metric that has tracked field outcomes) ==="
python3 - <<'EOF'
import json, glob, os
rows=[]
for f in sorted(glob.glob('test/out/rgeo/r2cam_*_heldout.json')):
    n=os.path.basename(f).replace('_heldout.json',''); s=json.load(open(f))['summary']
    a=None
    af=f.replace('_heldout.json','_heldout_align.json')
    if os.path.exists(af): a=json.load(open(af))['align']
    rows.append((n,s,a))
print(f"{'model':<16}{'bias%p':>8}{'right%':>8}{'park%':>7}{'align':>8}{'xpos':>8}{'ADE':>7}{'FDE':>7}")
for n,s,a in sorted(rows, key=lambda r:(r[1].get('turn_bias') or 9)):
    print(f"{n:<16}{(s.get('turn_bias') or 0)*100:+8.1f}{(s.get('turn_right_frac') or 0)*100:8.1f}"
          f"{(s.get('term_idle_frac') or 0)*100:7.1f}"
          f"{(a['policy_median_deg'] if a else float('nan')):8.3f}{(a['x_policy_median_mm'] if a else float('nan')):8.2f}"
          f"{s['ade_cm_mean']:7.2f}{s['fde_cm_mean']:7.2f}")
print()
print("demo reference: right 53.6% | park 13.9% | align 1.028 | xpos 8.83")
print("screening rule (terminal_metric.py): |bias| > 0.10 was fatal in every observed field case")
EOF
log "QUEUE COMPLETE"
