#!/bin/bash
# Reloc3r rot-vs-rot+dir ablation (2026-07-22, docs/reloc3r.md + user request).
# DINO is fully replaced by Reloc3r's ViT-L encoder (user decision) -- this is
# no longer a DINO-vs-Reloc3r comparison, it isolates whether the (validated
# unreliable-at-close-range) Reloc3r DIRECTION channel helps or hurts once
# rotation + raw lidar are already in the mix, with NO aux/ICP-distilled head
# (raw lidar points only, per user request).
#
#   rot     : reloc3r encoder + rotation-to-goal only      -> smr_reloc3r_rot.yaml
#   rot+dir : reloc3r encoder + rotation + direction-to-goal -> smr_reloc3r_rotdir.yaml
#
# Both: from-scratch, 20 epochs, matched to flow_goal_scratch20 protocol.
# Waits for the TRAIN reloc3r cache AND the direction sidecar
# (scripts/precompute_reloc3r_direction.py) before the rot+dir arm.
# Each arm scored on train split + held-out test split via
# test/eval_run_modular.py (the legacy eval stack cannot score modular runs).
#
#   nohup setsid bash test/queue_reloc3r_ablation.sh > outputs/queue_reloc3r.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
PY=/tmp/claude-1100/-home-work--postech-diffusion-policy-robot-docking/88194c75-b380-43b0-8c57-c004af706396/scratchpad/reloc3r_eval/venv/bin/python
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
QLOG=outputs/queue_reloc3r.log
TRAIN_CACHE=dataset/after_0328_train_reloc3r_bottom.h5
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$QLOG"; }

wait_ep_done () {  # $1=h5 path  $2=dataset key  $3=n_episodes
  while true; do
    d=$($PY - "$1" "$2" <<'EOF' 2>/dev/null
import sys, h5py
try:
    with h5py.File(sys.argv[1], "r") as f:
        print(int(f[sys.argv[2]].attrs.get("n_done_ep", 0)))
except Exception:
    print(-1)
EOF
)
    [ "$d" = "$3" ] && break
    sleep 60
  done
}

log "waiting for reloc3r feature+rotation cache (145 episodes)..."
wait_ep_done "$TRAIN_CACHE" reloc3r_bottom 145
log "reloc3r cache ready"

run(){  # $1=experiment_name  $2..=train overrides (incl --config-name ...)
  name=$1; shift
  log "TRAIN START $name ($*)"
  # num_workers=0: the modular reloc3r sensors collate a big precomputed
  # [B,5,196,1024] float tensor per batch (~2GB @ batch_size=512); multi-worker
  # collate needs that much AGAIN per in-flight batch in /dev/shm, which is
  # container-capped at 2GB here and crashed both arms on first attempt
  # (2026-07-22 23:0x, RuntimeError: unable to allocate shared memory). Loading
  # in the main process avoids shm entirely; a bit slower per epoch but robust.
  $PY -u scripts/train.py experiment_name="$name" device=cuda:0 num_workers=0 \
    num_epochs=20 save_interval=2000 "$@" > "outputs/train_$name.log" 2>&1
  rc=$?
  log "TRAIN END $name (rc=$rc)"
  [ $rc -ne 0 ] && { log "SKIP EVAL $name (train failed)"; return; }
  dir=$(ls -d outputs/train/"$name"/* | tail -1)
  $PY -u test/eval_run_modular.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name (train)"
  EVAL_H5=$PWD/dataset/after_0328_test.h5 EVAL_STATS_H5=$PWD/dataset/after_0328_train.h5 \
    EVAL_EPISODES=0,1,2,3,4,5,6,7,8,9 EVAL_TAG=heldout \
    $PY -u test/eval_run_modular.py "$dir" >> "$QLOG" 2>&1 || log "EVAL FAILED $name (heldout)"
}

# Arm 1: rotation only (needs only reloc3r_bottom/reloc3r_rot_bottom -- ready now).
run reloc3r_rot_noaux --config-name smr_reloc3r_rot

# Arm 2: rotation + direction (needs the direction sidecar too).
log "waiting for direction cache (145 episodes)..."
wait_ep_done "$TRAIN_CACHE" reloc3r_dir_bottom 145
log "direction cache ready"
run reloc3r_rotdir_noaux --config-name smr_reloc3r_rotdir

log "QUEUE reloc3r rot-vs-rotdir ablation COMPLETE"
