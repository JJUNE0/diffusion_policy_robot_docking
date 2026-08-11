"""Add Reloc3r's pre-head cross-attention DECODER tokens to an EXISTING
reloc3r cache (scripts/precompute_reloc3r_cache.py), for the history<->goal
relational condition branch (`reloc3r_relation` encoder,
cleandiffuser/nn_condition/modality_encoders.py). Architecture reference:
arxiv 2412.08376 (Reloc3r) -- shared ViT encoder -> cross-attention ViT
decoder -> Head -> relative pose. This script keeps the decoder's
cross-attended patch tokens themselves, from BEFORE the pose head collapses
them to a single vector.

Reuses the already-cached `reloc3r_<cam>` ViT-L encoder patch features (the
expensive part, one _encode_image pass per frame) and reruns only the
decoder -- no pose head at all. Writes two new row-aligned datasets into the
SAME cache file, appended in place and independently resumable, so it is safe
to run after the encoder cache is already complete:

  reloc3r_dec1_<cam> : (N, 196, 768) float16 -- current/history frame's
                       decoder stream after cross-attending INTO the episode
                       goal frame's stream ("goal-aware history" tokens).
  reloc3r_dec2_<cam> : (N, 196, 768) float16 -- goal frame's decoder stream
                       after cross-attending INTO the current frame's stream
                       ("current-aware goal" tokens; this is the exact stream
                       the pose head consumes and collapses -- kept here at
                       full [196,768] patch resolution instead of collapsing).

Usage:
  python scripts/precompute_reloc3r_dec_features.py \
      --cache dataset/after_0328_train_reloc3r_bottom.h5 --camera image_bottom \
      [--limit_episodes 2]
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
DEC_DIM = 768          # dec_embed_dim
FEAT_KEY = {"image_bottom": "reloc3r_bottom", "image_top": "reloc3r_top"}
DEC_KEY = {"image_bottom": ("reloc3r_dec1_bottom", "reloc3r_dec2_bottom"),
           "image_top": ("reloc3r_dec1_top", "reloc3r_dec2_top")}

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
def decoder_feats_to_goal(model, feats, shape224, goal_row, device, batch=32):
    """feats: [n,196,1024] fp16 np (cached encoder output). shape224: constant
    (224,224) true_shape used for every frame (matches precompute_reloc3r_cache
    array_to_view, size=224 always -> square crop). Returns (dec1_out,
    dec2_out), each [n,196,768] fp16 -- the dec_norm-normalized last decoder
    layer, PRE pose head."""
    n = feats.shape[0]
    dec1_out = np.zeros((n, N_PATCH, DEC_DIM), dtype=np.float16)
    dec2_out = np.zeros((n, N_PATCH, DEC_DIM), dtype=np.float16)
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
        dec1_out[b0:b1] = dec1[-1].float().cpu().numpy().astype(np.float16)
        dec2_out[b0:b1] = dec2[-1].float().cpu().numpy().astype(np.float16)
    return dec1_out, dec2_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="existing reloc3r cache h5 (in-place append)")
    ap.add_argument("--camera", default="image_bottom", choices=list(FEAT_KEY))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit_episodes", type=int, default=0,
                    help="0 = all; >0 = only first K episodes (smoke test).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_reloc3r_relpose_model(model_args="224", device=device)
    model.eval()
    shape224 = torch.tensor([224, 224], dtype=torch.int64)

    feat_key = FEAT_KEY[args.camera]
    dec1_key, dec2_key = DEC_KEY[args.camera]
    with h5py.File(args.cache, "a") as fo:
        episode_ends = fo["episode_ends"][:]
        n_total = fo[feat_key].shape[0]
        ep_starts = np.concatenate([[0], episode_ends[:-1]])
        n_ep = len(episode_ends)
        if args.limit_episodes > 0:
            n_ep = min(n_ep, args.limit_episodes)

        if dec1_key not in fo:
            fo.create_dataset(dec1_key, shape=(n_total, N_PATCH, DEC_DIM), dtype="float16",
                              chunks=(1, N_PATCH, DEC_DIM))
            fo.create_dataset(dec2_key, shape=(n_total, N_PATCH, DEC_DIM), dtype="float16",
                              chunks=(1, N_PATCH, DEC_DIM))
            fo[dec1_key].attrs["n_done_ep"] = 0
        dec1_ds = fo[dec1_key]
        dec2_ds = fo[dec2_key]
        feat_ds = fo[feat_key]
        start_ep = int(dec1_ds.attrs.get("n_done_ep", 0))

        for ep in range(start_ep, n_ep):
            s, e = int(ep_starts[ep]), int(episode_ends[ep])
            feats = feat_ds[s:e]                      # [n,196,1024] fp16, cached
            goal_row = (e - 1) - s                    # goal = last frame of episode (local idx)
            d1, d2 = decoder_feats_to_goal(model, feats, shape224, goal_row, device, args.batch)
            dec1_ds[s:e] = d1
            dec2_ds[s:e] = d2
            dec1_ds.attrs["n_done_ep"] = ep + 1
            if ep % 10 == 0:
                fo.flush()
                print(f"  episode {ep+1}/{n_ep}  rows[{s}:{e}]  "
                      f"({100.0*(e)/max(n_total,1):.1f}% of fill)", flush=True)
    print(f"Done. Filled through episode {n_ep} -> {args.cache}[{dec1_key},{dec2_key}]")


if __name__ == "__main__":
    main()
