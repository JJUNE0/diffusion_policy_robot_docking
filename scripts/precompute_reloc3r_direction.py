"""Add the Reloc3r translation-DIRECTION-to-goal channel to an EXISTING
reloc3r cache (scripts/precompute_reloc3r_cache.py), for the rot-vs-rot+dir
ablation (user request 2026-07-22).

Reuses the already-computed `reloc3r_<cam>` ViT-L encoder patch features (the
expensive part) and only reruns the cheap decoder + pose head per frame, so
this is fast (no re-encoding). Writes a new row-aligned dataset:

  reloc3r_dir_<cam> : (N, 3) float16 -- unit translation vector (goal relative
                      to current, camera frame). Scale-ambiguous by
                      construction (docs/reloc3r.md); direction only.

Usage:
  python scripts/precompute_reloc3r_direction.py \
      --cache dataset/after_0328_train_reloc3r_bottom.h5 --camera image_bottom
"""
import os
import sys
import argparse

import numpy as np
import torch
import h5py

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "reloc3r"))
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model  # noqa: E402

N_PATCH = 196
FEAT_DIM = 1024
DIR_DIM = 3
FEAT_KEY = {"image_bottom": "reloc3r_bottom", "image_top": "reloc3r_top"}
DIR_KEY = {"image_bottom": "reloc3r_dir_bottom", "image_top": "reloc3r_dir_top"}

_POS_CACHE = {}


def _canonical_pos(model, b, true_shape, device):
    key = (int(true_shape[0, 0].item()), int(true_shape[0, 1].item()))
    if key not in _POS_CACHE:
        H, W = key
        dummy = torch.zeros(1, 3, H, W, device=device)
        _x, pos = model.patch_embed(dummy, true_shape=true_shape[:1])
        _POS_CACHE[key] = pos.detach()
    return _POS_CACHE[key].expand(b, -1, -1).contiguous()


@torch.no_grad()
def direction_to_goal(model, feats, shape224, goal_row, device, batch=32):
    """feats: [n,196,1024] fp16 np (cached encoder output). shape224: constant
    (224,224) true_shape used for every frame (matches precompute_reloc3r_cache
    array_to_view, size=224 always -> square crop). Returns [n,3] fp16 unit dir."""
    n = feats.shape[0]
    out = np.zeros((n, DIR_DIM), dtype=np.float16)
    goal_feat = torch.from_numpy(feats[goal_row].astype(np.float32))[None].to(device)
    goal_shape = shape224[None].to(device)
    for b0 in range(0, n, batch):
        b1 = min(b0 + batch, n)
        bn = b1 - b0
        cur_feat = torch.from_numpy(feats[b0:b1].astype(np.float32)).to(device)
        cur_shape = shape224.expand(bn, -1).to(device)
        gf = goal_feat.expand(bn, -1, -1).contiguous()
        pos = _canonical_pos(model, bn, cur_shape, device)
        gpos = _canonical_pos(model, bn, goal_shape.expand(bn, -1).contiguous(), device)
        dec1, dec2 = model._decoder(cur_feat, pos, gf, gpos)
        with torch.cuda.amp.autocast(enabled=False):
            pose2 = model._downstream_head([tok.float() for tok in dec2], goal_shape[0])
        t = pose2["pose"][:, :3, 3]
        t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)
        out[b0:b1] = t.float().cpu().numpy().astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="existing reloc3r cache h5 (in-place append)")
    ap.add_argument("--camera", default="image_bottom", choices=list(FEAT_KEY))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_reloc3r_relpose_model(model_args="224", device=device)
    model.eval()
    shape224 = torch.tensor([224, 224], dtype=torch.int64)

    feat_key, dir_key = FEAT_KEY[args.camera], DIR_KEY[args.camera]
    with h5py.File(args.cache, "a") as fo:
        episode_ends = fo["episode_ends"][:]
        n_total = fo[feat_key].shape[0]
        ep_starts = np.concatenate([[0], episode_ends[:-1]])
        n_ep = len(episode_ends)

        if dir_key not in fo:
            fo.create_dataset(dir_key, shape=(n_total, DIR_DIM), dtype="float16",
                              chunks=(min(4096, n_total), DIR_DIM))
            fo[dir_key].attrs["n_done_ep"] = 0
        dir_ds = fo[dir_key]
        feat_ds = fo[feat_key]
        start_ep = int(dir_ds.attrs.get("n_done_ep", 0))

        for ep in range(start_ep, n_ep):
            s, e = int(ep_starts[ep]), int(episode_ends[ep])
            feats = feat_ds[s:e]                      # [n,196,1024] fp16, cached
            goal_row = (e - 1) - s
            dirs = direction_to_goal(model, feats, shape224, goal_row, device, args.batch)
            dir_ds[s:e] = dirs
            dir_ds.attrs["n_done_ep"] = ep + 1
            if ep % 10 == 0:
                fo.flush()
                print(f"  episode {ep+1}/{n_ep}  rows[{s}:{e}]  "
                      f"({100.0*(e)/max(n_total,1):.1f}% of fill)", flush=True)
    print(f"Done. Filled {n_total} rows for {n_ep} episodes -> {args.cache}[{dir_key}]")


if __name__ == "__main__":
    main()
