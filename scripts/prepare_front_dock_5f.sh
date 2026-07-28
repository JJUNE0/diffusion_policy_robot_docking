#!/usr/bin/env bash
# Full data-prep pipeline for the 5th-floor front-dock dataset, from the raw
# episode layout produced by scripts/build_front_dock_5f.py to every cache the
# R-Geo config (configs/robot/smr_rgeo_5f.yaml) reads.
#
#   dataset/front_dock_5f/dock/episode_*_dock   (81 episodes, success-only,
#                                                post-segment-marker only)
#     -> front_dock_5f_{train,test}.h5              images/encoder/lidar points
#     -> front_dock_5f_{train,test}_dino_bottom.h5  frozen DINOv3 patch feats
#     -> front_dock_5f_{train,test}_reloc3r_bottom.h5
#            reloc3r_bottom (encoder feats) + reloc3r_rot_bottom
#            + reloc3r_dir_bottom + geometry_bottom (the R-Geo token)
#
# Split: first 71 episodes train, last 10 held out. Episodes sort by record id,
# so the held-out block is the tail of the 0713 session -- a different recording
# day, i.e. the split also measures session-level generalization.
#
# Usage:  bash scripts/prepare_front_dock_5f.sh
set -euo pipefail

cd "$(dirname "$0")/.."

N_TRAIN=71
DEV=${DEV:-cuda:0}

echo "=== [1/4] preprocessing -> h5 ==="
python -u utils/preprocessing.py \
  --data_root dataset/front_dock_5f --save_path dataset/front_dock_5f_train.h5 \
  --use_lidar --lidar_format points --lidar_crop_r 0.8 --lidar_max_points 256 \
  --no_room1 --max_episodes "$N_TRAIN"

python -u utils/preprocessing.py \
  --data_root dataset/front_dock_5f --save_path dataset/front_dock_5f_test.h5 \
  --use_lidar --lidar_format points --lidar_crop_r 0.8 --lidar_max_points 256 \
  --no_room1 --episode_start "$N_TRAIN"

for SPLIT in train test; do
  H5=dataset/front_dock_5f_${SPLIT}.h5
  RCACHE=dataset/front_dock_5f_${SPLIT}_reloc3r_bottom.h5

  echo "=== [2/4] DINO cache ($SPLIT) ==="
  python -u scripts/precompute_dino_cache.py \
    --h5 "$H5" --camera image_bottom \
    --out dataset/front_dock_5f_${SPLIT}_dino_bottom.h5 --device "$DEV"

  echo "=== [3/4] Reloc3r rotation + direction cache ($SPLIT) ==="
  python -u scripts/precompute_reloc3r_cache.py \
    --h5 "$H5" --camera image_bottom --out "$RCACHE"
  python -u scripts/precompute_reloc3r_direction.py \
    --cache "$RCACHE" --camera image_bottom

  echo "=== [4/4] Reloc3r body-frame geometry token ($SPLIT) ==="
  python -u scripts/build_reloc3r_geometry_cache.py \
    --cache "$RCACHE" --camera image_bottom
done

echo "=== done ==="
ls -la dataset/front_dock_5f_*.h5
