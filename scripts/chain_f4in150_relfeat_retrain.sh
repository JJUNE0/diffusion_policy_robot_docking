#!/bin/bash
# Wait for the phase-2 encfeat run to finish, then retrain r_relfeat on the
# rebuilt f4in_150ep h5 (image_bottom = camera_orbbec-0, dock-facing, pinned by
# the sha256 fingerprint in scripts/build_f4in_150ep_vds.py).
#
# Sequential on purpose: relfeat pins dec1+dec2 (161 GiB) and encfeat pins
# reloc3r_bottom (107 GiB); 268 GiB together on a 300 GiB 0-swap host is the
# overlap the two-phase plan exists to avoid. Waiting also frees the whole H100.
#
# The rerun goes to a NEW timestamped output dir, so the verified 07:12 run at
# outputs/train/r_relfeat_only_now_f4in150/2026-08-26_01-04-20 is untouched and
# the two can be compared.
#
# Usage:
#   nohup bash scripts/chain_f4in150_relfeat_retrain.sh \
#     > outputs/logs/chain_f4in150_relfeat_retrain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

PY=/usr/bin/python3.12
ENC_META=outputs/cache/f4in150_reloc3r_bottom_memfd.json
ENC_LOG=outputs/logs/train_r_encfeat_only_now_f4in150.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "waiting for r_encfeat_only_now_f4in150 to finish"
while pgrep -f "experiment_name=r_encfeat_only_now_f4in150" >/dev/null; do
    sleep 120
done

if grep -q "Final checkpoint saved at step 21740" "$ENC_LOG" 2>/dev/null; then
    log "encfeat completed normally"
else
    log "WARNING: encfeat exited without its final checkpoint -- last lines:"
    tail -5 "$ENC_LOG"
    log "continuing to the relfeat retrain anyway (independent arm)"
fi

# Release encfeat's 107 GiB so relfeat's 161 GiB has the box to itself.
P=$("$PY" -c 'import json;print(json.load(open("'"$ENC_META"'"))["pid"])' 2>/dev/null) || P=
if [ -n "${P:-}" ] && [ -d "/proc/$P" ]; then
    kill "$P" 2>/dev/null && log "stopped encfeat memfd daemon pid $P (freed ~107 GiB)"
    sleep 20
fi

log "starting the relfeat retrain"
bash test/queue_f4in150_phase1_relfeat.sh
log "relfeat retrain finished rc=$?"
