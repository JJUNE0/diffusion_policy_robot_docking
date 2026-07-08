"""Precompute frozen-DINO patch features for every frame of one camera and cache
them to an HDF5 file, row-aligned 1:1 with the source image rows.

Motivation: the DINOv3 backbone is frozen and the input frames are fixed, so the
[196, 768] features are identical every training step/epoch. Recomputing them on
the GPU each step is the dominant training bottleneck. Precomputing once lets the
training loop read features from disk and skip the backbone entirely.

This reproduces EXACTLY the backbone path of DinoBatchDetector.get_heatmap
(dino/dino_detector.py): bilinear->224 bicubic resize, ImageNet normalize,
model(x).last_hidden_state[:, 5:, :]. The discarded sim-map / OpenCV verification
path is NOT run here.

Usage (defaults match configs/robot/smr.yaml):
    python scripts/precompute_dino_cache.py \
        --h5 dataset/after_0328_train.h5 \
        --camera image_bottom \
        --out dataset/after_0328_train_dino_bottom.h5
"""
import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
N_PATCH = 196
FEAT_DIM = 768
N_SPECIAL = 5          # 1 CLS + 4 register tokens, dropped via [:, 5:, :]

# Output dataset name per camera (room1=image_top, room2=image_bottom).
OUT_KEY = {"image_bottom": "dino_bottom", "image_top": "dino_top"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="dataset/after_0328_train.h5",
                   help="input HDF5 with the image dataset")
    p.add_argument("--camera", default="image_bottom",
                   choices=list(OUT_KEY.keys()), help="which camera to encode")
    p.add_argument("--out", default=None,
                   help="output HDF5 path (default: <h5 stem>_dino_<cam>.h5)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def encode_batch(model, imgs_uint8, device, mean, std):
    """imgs_uint8: [B, 3, 240, 320] uint8 -> [B, 196, 768] float32 on CPU."""
    x = torch.from_numpy(imgs_uint8).to(device, non_blocking=True).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
    x = (x - mean) / std
    out = model(x).last_hidden_state[:, N_SPECIAL:, :]      # [B, 196, 768]
    return out.float().cpu().numpy()


def main():
    args = parse_args()
    out_key = OUT_KEY[args.camera]
    out_path = args.out or f"{os.path.splitext(args.h5)[0]}_dino_{args.camera.split('_')[1]}.h5"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading {DINO_MODEL} on {device} ...")
    model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with h5py.File(args.h5, "r") as f:
        n = f[args.camera].shape[0]
        episode_ends = f["episode_ends"][:]
    print(f"Source: {args.h5}[{args.camera}] -> {n} frames")
    print(f"Output: {out_path}[{out_key}] shape ({n},{N_PATCH},{FEAT_DIM}) fp16 "
          f"(~{n * N_PATCH * FEAT_DIM * 2 / 1e9:.1f} GB)")

    # Open output (resume if it already exists with the right shape).
    mode = "a" if os.path.exists(out_path) else "w"
    with h5py.File(out_path, mode) as fo:
        if out_key in fo:
            dset = fo[out_key]
            assert dset.shape == (n, N_PATCH, FEAT_DIM), \
                f"existing {out_key} shape {dset.shape} != expected {(n, N_PATCH, FEAT_DIM)}"
            start = int(dset.attrs.get("n_done", 0))
            print(f"Resuming from row {start}")
        else:
            dset = fo.create_dataset(
                out_key, shape=(n, N_PATCH, FEAT_DIM), dtype="float16",
                chunks=(1, N_PATCH, FEAT_DIM))
            fo.create_dataset("episode_ends", data=episode_ends)
            dset.attrs["source_h5"] = os.path.abspath(args.h5)
            dset.attrs["camera"] = args.camera
            dset.attrs["n_done"] = 0
            start = 0

        # Reopen source per batch is unnecessary; keep one handle for reads.
        with h5py.File(args.h5, "r") as f:
            z_img = f[args.camera]
            for i in range(start, n, args.batch):
                j = min(i + args.batch, n)
                feats = encode_batch(model, z_img[i:j], device, mean, std)
                dset[i:j] = feats.astype(np.float16)
                dset.attrs["n_done"] = j
                if (i // args.batch) % 20 == 0:
                    fo.flush()
                    print(f"  {j}/{n} ({100.0 * j / n:.1f}%)", flush=True)
        dset.attrs["n_done"] = n
        print("Done.")


if __name__ == "__main__":
    main()
