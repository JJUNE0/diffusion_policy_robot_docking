#!/usr/bin/env bash
# Full data-prep pipeline for the 5th-floor front-dock dataset, from the raw
# episode layout produced by scripts/build_front_dock_5f.py to every cache the
# ReLoc3R relfeat configs read.
#
#   dataset/front_dock_5f/dock/episode_*_dock   (81 episodes, success-only,
#                                                post-segment-marker only)
#     -> front_dock_5f_{train,test}.h5              images/encoder
#     -> front_dock_5f_{train,test}_reloc3r_bottom.h5
#            reloc3r_bottom (ViT-L encoder feats)
#            + reloc3r_dec1_bottom / reloc3r_dec2_bottom (the relfeat streams)
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

echo "=== [1/3] preprocessing -> h5 ==="
python -u utils/preprocessing.py \
  --data_root dataset/front_dock_5f --save_path dataset/front_dock_5f_train.h5 \
  --no_room1 --max_episodes "$N_TRAIN"

python -u utils/preprocessing.py \
  --data_root dataset/front_dock_5f --save_path dataset/front_dock_5f_test.h5 \
  --no_room1 --episode_start "$N_TRAIN"

for SPLIT in train test; do
  H5=dataset/front_dock_5f_${SPLIT}.h5
  RCACHE=dataset/front_dock_5f_${SPLIT}_reloc3r_bottom.h5

  echo "=== [2/3] ReLoc3R ViT-L encoder cache ($SPLIT) ==="
  python -u scripts/precompute_reloc3r_cache.py \
    --h5 "$H5" --camera image_bottom --out "$RCACHE"

  echo "=== [3/3] ReLoc3R dec1/dec2 relational cache ($SPLIT) ==="
  python -u scripts/precompute_reloc3r_dec_features.py \
    --cache "$RCACHE" --camera image_bottom
done

echo "=== done ==="
ls -la dataset/front_dock_5f_*.h5
