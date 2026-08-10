#!/usr/bin/env bash
# Controlled pre-head vs post-head baselines for `r_relfeat_only`.
#
# Three arms, ONE variable: where ReLoc3R's bidirectional decoder stream is
# tapped. Everything else -- smr_rgeo.yaml (d_model/depth/heads/dropout/
# condition_num_layers), batch_size=256, num_epochs=20, lr=1e-4, wd=1e-5,
# ema=0.999, action_norm=minmax, seed=0, dataset, sampler, 220 condition
# tokens -- is loaded from the SAME config and is identical by construction.
#
#   r_relfeat_only   (already trained, 2026-07-28)  pre-head  [196,768]/frame/stream
#   r_posthead_only  (this script, GPU0)            post-pool [1,1024]/frame/stream
#   r_pose_only      (this script, GPU1)            final pose[1,  12]/frame/stream
#
# One GPU per arm on purpose: DDP across 2 GPUs would change the effective
# batch size relative to r_relfeat_only and break the comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

HEAD_H5=dataset/after_0328_train_reloc3r_bottom_head.h5
mkdir -p outputs/logs

# Poll the DATA, not `pgrep`: a pgrep pattern containing the script name also
# matches the waiting shell's own command line, which self-deadlocks.
echo "=== waiting for head-feature precompute to fill every row ==="
for _ in $(seq 1 120); do
  python - <<'PY' && break || sleep 30
import h5py, numpy as np, sys
f = h5py.File("dataset/after_0328_train_reloc3r_bottom_head.h5", "r")
idx = np.linspace(0, f["reloc3r_head1_bottom"].shape[0] - 1, 400).astype(int)
x = f["reloc3r_head1_bottom"][idx]
n = int((np.abs(x).sum(axis=(1, 2)) == 0).sum())
print(f"  fill check: {400 - n}/400 sampled rows non-zero", flush=True)
sys.exit(0 if n == 0 else 1)
PY
done

python - <<'PY'
import h5py, numpy as np, sys
f = h5py.File("dataset/after_0328_train_reloc3r_bottom_head.h5", "r")
ok = True
for k in ["reloc3r_head1_bottom", "reloc3r_head2_bottom",
          "reloc3r_pose1_bottom", "reloc3r_pose2_bottom"]:
    d = f[k]
    # Sample across the whole range; an unfilled shard shows up as all-zero rows.
    idx = np.linspace(0, d.shape[0] - 1, 400).astype(int)
    x = d[idx]
    zero_rows = int((np.abs(x).sum(axis=(1, 2)) == 0).sum())
    print(f"{k:24s} {str(d.shape):18s} zero_rows={zero_rows}/400  "
          f"absmax={np.abs(x).max():.3f}")
    ok &= zero_rows == 0
if not ok:
    sys.exit("precompute incomplete -- unfilled (all-zero) rows found")
print("precompute verified complete.")
PY

echo "=== smoke test (5 steps each, sequential, catches wiring errors early) ==="
for V in reloc3r_posthead_only reloc3r_pose_only; do
  python scripts/train.py --config-name smr_rgeo \
    sensors_variant=$V experiment_name=_smoke_$V \
    num_epochs=null diffusion_gradient_steps=5 save_interval=1000000 \
    device=cuda:0 > outputs/logs/_smoke_$V.log 2>&1
  echo "  smoke OK: $V"
done

echo "=== launching both full runs (20 epochs, one GPU each) ==="
nohup python scripts/train.py --config-name smr_rgeo \
  sensors_variant=reloc3r_posthead_only experiment_name=r_posthead_only \
  device=cuda:0 > outputs/logs/train_r_posthead_only.log 2>&1 &
echo "  r_posthead_only -> cuda:0 (pid $!)"
sleep 20
nohup python scripts/train.py --config-name smr_rgeo \
  sensors_variant=reloc3r_pose_only experiment_name=r_pose_only \
  device=cuda:1 > outputs/logs/train_r_pose_only.log 2>&1 &
echo "  r_pose_only -> cuda:1 (pid $!)"
wait
