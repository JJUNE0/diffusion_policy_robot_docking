"""Precompute Reloc3r ViT-L encoder features for the docking dataset.

Writes one row-aligned dataset into an output h5, 1:1 with the source h5 rows:

  reloc3r_<cam> : (N, 196, 1024) float16 -- ViT-L encoder patch tokens.

This is the base cache everything else is built from: the relational dec1/dec2
streams (scripts/precompute_reloc3r_dec_features.py), the post-pose-head taps
(scripts/precompute_reloc3r_head_features.py), and the compiled goal pool
(scripts/compile_goal_pool.py) all consume it. Each frame is ViT-L-encoded
exactly ONCE.

Import note: the vendored Reloc3r package lives at <repo>/reloc3r and its inner
package is also named `reloc3r`; we sys.path.insert the OUTER dir so it wins over
the sibling repo folder (see reloc3r/VENDORING_NOTE.md).

Usage:
  python scripts/precompute_reloc3r_cache.py \
      --h5 dataset/after_0328_train.h5 --camera image_bottom \
      --out dataset/after_0328_train_reloc3r_bottom.h5 [--limit_episodes 1]
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image
import h5py

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "reloc3r"))
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model  # noqa: E402
from reloc3r.utils.image import _resize_pil_image, ImgNorm       # noqa: E402

N_PATCH = 196          # 224/16 = 14, 14*14 = 196
FEAT_DIM = 1024        # ViT-L encoder embed dim
OUT_KEY = {"image_bottom": "reloc3r_bottom", "image_top": "reloc3r_top"}


def array_to_view(arr_chw_uint8, size=224):
    """Match Reloc3r's load_images() preprocessing for size==224 exactly:
    resize short side then center-crop to 224x224, normalize to [-1,1]."""
    arr_hwc = np.ascontiguousarray(arr_chw_uint8.transpose(1, 2, 0))
    img = PIL.Image.fromarray(arr_hwc, mode="RGB")
    W1, H1 = img.size
    img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    W, H = img.size
    cx, cy = W // 2, H // 2
    half = min(cx, cy)
    img = img.crop((cx - half, cy - half, cx + half, cy + half))
    return ImgNorm(img), np.int32(img.size[::-1])  # tensor [3,224,224], (H,W)


@torch.no_grad()
def encode_frames(model, imgs_uint8, device, size=224, batch=32):
    """imgs_uint8: [n,3,240,320] uint8 -> (feats [n,196,1024] fp16 CPU,
    pos [n,196,2] long GPU-kept-per-batch, shapes [n,2])."""
    n = imgs_uint8.shape[0]
    feats = np.zeros((n, N_PATCH, FEAT_DIM), dtype=np.float16)
    tensors, shapes = [], []
    for i in range(n):
        t, s = array_to_view(imgs_uint8[i], size=size)
        tensors.append(t)
        shapes.append(s)
    shapes = np.stack(shapes)  # [n,2]
    for b0 in range(0, n, batch):
        b1 = min(b0 + batch, n)
        img = torch.stack(tensors[b0:b1]).to(device)
        ts = torch.from_numpy(shapes[b0:b1]).to(device)
        feat, _pos, _ = model._encode_image(img, ts)   # [b,196,1024]
        feats[b0:b1] = feat.float().cpu().numpy().astype(np.float16)
    return feats, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--camera", default="image_bottom", choices=list(OUT_KEY))
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit_episodes", type=int, default=0,
                    help="0 = all; >0 = only first K episodes (smoke test).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_reloc3r_relpose_model(model_args=str(args.size), device=device)
    model.eval()

    feat_key = OUT_KEY[args.camera]
    with h5py.File(args.h5, "r") as f:
        episode_ends = f["episode_ends"][:]
        n_total = f[args.camera].shape[0]

    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    n_ep = len(episode_ends)
    if args.limit_episodes > 0:
        n_ep = min(n_ep, args.limit_episodes)
    # rows we will actually fill (smoke test only fills first K episodes)
    fill_rows = int(episode_ends[n_ep - 1]) if args.limit_episodes > 0 else n_total

    mode = "a" if os.path.exists(args.out) else "w"
    with h5py.File(args.out, mode) as fo:
        if feat_key not in fo:
            fo.create_dataset(feat_key, shape=(n_total, N_PATCH, FEAT_DIM),
                              dtype="float16", chunks=(1, N_PATCH, FEAT_DIM))
            fo.create_dataset("episode_ends", data=episode_ends)
            fo[feat_key].attrs["source_h5"] = os.path.abspath(args.h5)
            fo[feat_key].attrs["camera"] = args.camera
            fo[feat_key].attrs["n_done_ep"] = 0
        feat_ds = fo[feat_key]
        start_ep = int(feat_ds.attrs.get("n_done_ep", 0))

        with h5py.File(args.h5, "r") as f:
            z_img = f[args.camera]
            for ep in range(start_ep, n_ep):
                s, e = int(ep_starts[ep]), int(episode_ends[ep])
                imgs = z_img[s:e]                       # [n_ep,3,240,320] uint8
                feats, _shapes = encode_frames(model, imgs, device, args.size, args.batch)
                feat_ds[s:e] = feats
                feat_ds.attrs["n_done_ep"] = ep + 1
                if ep % 5 == 0:
                    fo.flush()
                    print(f"  episode {ep+1}/{n_ep}  rows[{s}:{e}]  "
                          f"({100.0*(e)/max(fill_rows,1):.1f}% of fill)", flush=True)
        print(f"Done. Filled {fill_rows} rows for {n_ep} episodes -> {args.out}")


if __name__ == "__main__":
    main()
