"""Precompute Reloc3r features for the docking dataset (DINO-cache analogue).

Writes TWO row-aligned datasets into an output h5, 1:1 with the source h5 rows:

  reloc3r_<cam>      : (N, 196, 1024) float16  -- ViT-L encoder patch tokens
                       (the DINO-patch replacement; feeds a `dino_image` encoder
                       with feat_dim=1024 on the modular fusion path).
  reloc3r_rot_<cam>  : (N, 7)         float16  -- per-frame rotation-to-goal:
                       [R[:,0] (3), R[:,1] (3), geodesic_angle (1)], where R is
                       the rotation block of Reloc3r's pose2to1 for the pair
                       (frame_t, episode_goal_frame). Validated in docs/reloc3r.md
                       (rotation median 0.9 deg, r=0.93 vs ICP heading). Direction
                       (translation) is intentionally NOT stored -- lidar owns it.

Efficiency: each frame is ViT-L-encoded exactly ONCE. The rotation head reuses
those cached encoder features (only the cheaper ViT-B decoder + pose head re-run
per frame vs. its episode goal), so cost is ~1 encode/frame, not 3.

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
ROT_DIM = 7            # 6D (first two columns of R) + geodesic angle
OUT_KEY = {"image_bottom": ("reloc3r_bottom", "reloc3r_rot_bottom"),
           "image_top": ("reloc3r_top", "reloc3r_rot_top")}


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


@torch.no_grad()
def rotation_to_goal(model, feats, shapes, goal_row, device, batch=32):
    """Reuse cached encoder feats: run decoder + pose head for each frame vs the
    episode goal frame. Returns [n,7] fp16. pose2to1 = goal relative to current
    (matches docs/reloc3r.md validation)."""
    n = feats.shape[0]
    out = np.zeros((n, ROT_DIM), dtype=np.float16)
    goal_feat = torch.from_numpy(feats[goal_row].astype(np.float32))[None].to(device)  # [1,196,1024]
    goal_shape = torch.from_numpy(shapes[goal_row].astype(np.int64))[None].to(device)
    # positions are deterministic from patch grid; recompute via model.rope indices.
    # Reloc3r's _encode_image returns pos from patch_embed; recompute by a dummy
    # encode of the goal to grab its pos once (cheap, 1 frame).
    for b0 in range(0, n, batch):
        b1 = min(b0 + batch, n)
        bn = b1 - b0
        cur_feat = torch.from_numpy(feats[b0:b1].astype(np.float32)).to(device)  # [bn,196,1024]
        cur_shape = torch.from_numpy(shapes[b0:b1].astype(np.int64)).to(device)
        gf = goal_feat.expand(bn, -1, -1).contiguous()
        # positions: use patch_embed on a zero image to get canonical pos grid.
        pos = _canonical_pos(model, bn, cur_shape, device)
        gpos = _canonical_pos(model, bn, goal_shape.expand(bn, -1).contiguous(), device)
        dec1, dec2 = model._decoder(cur_feat, pos, gf, gpos)
        with torch.cuda.amp.autocast(enabled=False):
            pose2 = model._downstream_head([tok.float() for tok in dec2], goal_shape[0])
        pose = pose2["pose"]  # [bn,4,4]
        R = pose[:, :3, :3]
        col0 = R[:, :, 0]
        col1 = R[:, :, 1]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        geo = torch.arccos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)).unsqueeze(1)
        rep = torch.cat([col0, col1, geo], dim=1)  # [bn,7]
        out[b0:b1] = rep.float().cpu().numpy().astype(np.float16)
    return out


_POS_CACHE = {}


def _canonical_pos(model, b, true_shape, device):
    """Patch position indices for a [b,196,2] grid, matching what
    _encode_image would produce. Reloc3r's ManyAR_PatchEmbed derives pos from
    true_shape; we obtain it by running patch_embed on a zero image once and
    caching (shape is constant at 224x224)."""
    key = (int(true_shape[0, 0].item()), int(true_shape[0, 1].item()))
    if key not in _POS_CACHE:
        H, W = key
        dummy = torch.zeros(1, 3, H, W, device=device)
        _x, pos = model.patch_embed(dummy, true_shape=true_shape[:1])
        _POS_CACHE[key] = pos.detach()
    return _POS_CACHE[key].expand(b, -1, -1).contiguous()


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

    feat_key, rot_key = OUT_KEY[args.camera]
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
            fo.create_dataset(rot_key, shape=(n_total, ROT_DIM), dtype="float16",
                              chunks=(min(4096, n_total), ROT_DIM))
            fo.create_dataset("episode_ends", data=episode_ends)
            fo[feat_key].attrs["source_h5"] = os.path.abspath(args.h5)
            fo[feat_key].attrs["camera"] = args.camera
            fo[feat_key].attrs["n_done_ep"] = 0
        feat_ds = fo[feat_key]
        rot_ds = fo[rot_key]
        start_ep = int(feat_ds.attrs.get("n_done_ep", 0))

        with h5py.File(args.h5, "r") as f:
            z_img = f[args.camera]
            for ep in range(start_ep, n_ep):
                s, e = int(ep_starts[ep]), int(episode_ends[ep])
                imgs = z_img[s:e]                       # [n_ep,3,240,320] uint8
                feats, shapes = encode_frames(model, imgs, device, args.size, args.batch)
                goal_row = (e - 1) - s                  # goal = last frame of episode (local idx)
                rots = rotation_to_goal(model, feats, shapes, goal_row, device, args.batch)
                feat_ds[s:e] = feats
                rot_ds[s:e] = rots
                feat_ds.attrs["n_done_ep"] = ep + 1
                if ep % 5 == 0:
                    fo.flush()
                    print(f"  episode {ep+1}/{n_ep}  rows[{s}:{e}]  "
                          f"({100.0*(e)/max(fill_rows,1):.1f}% of fill)", flush=True)
        print(f"Done. Filled {fill_rows} rows for {n_ep} episodes -> {args.out}")


if __name__ == "__main__":
    main()
