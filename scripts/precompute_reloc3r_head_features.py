"""Add Reloc3r's POST-head taps to a SIDECAR h5, as the controlled baseline for
`r_relfeat_only` (which conditions on the PRE-head decoder tokens).

`scripts/precompute_reloc3r_dec_features.py` cached the decoder's cross-attended
patch tokens from BEFORE the pose head. This script takes those exact same
tokens and pushes them THROUGH the frozen pose head (reloc3r/reloc3r/
pose_head.py::PoseHead), keeping the two tap points that come after it:

  reloc3r_head1_<cam> : (N, 1, 1024) float16 -- dec1 stream, the head's
  reloc3r_head2_<cam> : (N, 1, 1024) float16 -- dec2 stream.
                        `more_mlps` output, i.e. the LAST vector the head
                        computes before fc_t/fc_rot regress the pose off it.
                        This is the head's own "token": 196 patch positions
                        have already been destroyed by AdaptiveAvgPool2d(1),
                        but the representation is still a wide feature vector
                        rather than a 6-DoF point estimate.
  reloc3r_pose1_<cam> : (N, 1, 12) float16 -- dec1 stream, [R(9, row-major,
  reloc3r_pose2_<cam> : (N, 1, 12) float16 -- dec2 stream.
                        post-SVD-orthogonalization) , t(3, raw fc_t output)],
                        i.e. exactly the content of the 4x4 pose Reloc3r
                        returns. Translation is scale-ambiguous by
                        construction (docs/reloc3r.md).

The head is shared between both streams in Reloc3r itself (Reloc3rRelpose.
forward calls self._downstream_head on dec1 and dec2 with the SAME
self.pose_head), so applying it per-stream here is faithful, not an extension.

Every frame is a (current, episode-goal) pair -- the pairing is already baked
into the cached dec tokens, so no goal bookkeeping is needed here.

Output goes to a SIDECAR file (not appended in place) so the 226 GB primary
reloc3r cache is never reopened for write. Sensor specs reach it via `file:`.

Usage:
  # verify the head re-implementation against Reloc3r's own head, then exit
  python scripts/precompute_reloc3r_head_features.py --verify_only \
      --dec_cache dataset/after_0328_train_reloc3r_bottom.h5

  # fill (row-sharded; run one process per GPU)
  python scripts/precompute_reloc3r_head_features.py \
      --dec_cache dataset/after_0328_train_reloc3r_bottom.h5 \
      --out dataset/after_0328_train_reloc3r_bottom_head.h5 \
      --camera image_bottom --shard 0 --num_shards 2 --device cuda:0
"""
import os
import sys
import time
import argparse

import numpy as np
import torch
import h5py

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "reloc3r"))
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model  # noqa: E402

DEC_DIM = 768
HEAD_DIM = 1024          # 4 * patch_size**2 = 4 * 16**2
POSE_DIM = 12            # R(9) + t(3)
IMG_HW = (224, 224)

DEC_KEY = {"image_bottom": ("reloc3r_dec1_bottom", "reloc3r_dec2_bottom"),
           "image_top": ("reloc3r_dec1_top", "reloc3r_dec2_top")}
HEAD_KEY = {"image_bottom": ("reloc3r_head1_bottom", "reloc3r_head2_bottom"),
            "image_top": ("reloc3r_head1_top", "reloc3r_head2_top")}
POSE_KEY = {"image_bottom": ("reloc3r_pose1_bottom", "reloc3r_pose2_bottom"),
            "image_top": ("reloc3r_pose1_top", "reloc3r_pose2_top")}


@torch.no_grad()
def head_taps(pose_head, tokens):
    """tokens: [B,196,768] float32 cuda (a cached dec stream).

    Mirrors PoseHead.forward step for step, returning the two post-head taps.
    Returns (feat [B,1024], pose12 [B,12]).
    """
    H, W = IMG_HW
    ps = pose_head.patch_size
    feat = pose_head.proj(tokens)                                   # [B,196,1024]
    feat = feat.transpose(-1, -2).view(tokens.shape[0], -1, H // ps, W // ps)
    for i in range(pose_head.num_resconv_block):
        feat = pose_head.res_conv[i](feat)                          # 1x1 convs only
    feat = pose_head.avgpool(feat)                                  # <-- 196 -> 1
    feat = feat.view(feat.size(0), -1)                              # [B,1024]
    feat = pose_head.more_mlps(feat)                                # [B,1024] TAP A
    out_t = pose_head.fc_t(feat)                                    # [B,3]
    out_r = pose_head.fc_rot(feat)                                  # [B,9]
    R = pose_head.svd_orthogonalize(out_r)                          # [B,3,3] SO(3)
    pose12 = torch.cat([R.reshape(-1, 9), out_t], dim=1)            # [B,12] TAP B
    return feat, pose12


@torch.no_grad()
def verify(model, dec_cache, camera, device, n=64):
    """Assert head_taps' pose reconstruction == Reloc3r's own head output."""
    d1k, d2k = DEC_KEY[camera]
    with h5py.File(dec_cache, "r") as f:
        toks = torch.from_numpy(f[d1k][:n].astype(np.float32)).to(device)
    shape = torch.tensor([IMG_HW], dtype=torch.int64).expand(n, -1).to(device)
    with torch.cuda.amp.autocast(enabled=False):
        ref = model._downstream_head([toks], shape)["pose"]          # [n,4,4]
    _feat, pose12 = head_taps(model.pose_head, toks)
    dR = (pose12[:, :9].reshape(-1, 3, 3) - ref[:, :3, :3]).abs().max().item()
    dt = (pose12[:, 9:] - ref[:, :3, 3]).abs().max().item()
    print(f"[verify] max|dR| = {dR:.3e}   max|dt| = {dt:.3e}  (n={n})")
    assert dR < 1e-4 and dt < 1e-4, "head re-implementation diverges from Reloc3r"
    print("[verify] OK -- taps sit on Reloc3r's exact head computation.")
    return pose12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec_cache", required=True, help="h5 holding reloc3r_dec{1,2}_<cam>")
    ap.add_argument("--out", default=None, help="sidecar h5 to create/fill")
    ap.add_argument("--camera", default="image_bottom", choices=list(DEC_KEY))
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = setup_reloc3r_relpose_model(model_args="224", device=device)
    model.eval()
    pose_head = model.pose_head

    verify(model, args.dec_cache, args.camera, device)
    if args.verify_only:
        return
    if not args.out:
        raise SystemExit("--out is required unless --verify_only")

    d1k, d2k = DEC_KEY[args.camera]
    h1k, h2k = HEAD_KEY[args.camera]
    p1k, p2k = POSE_KEY[args.camera]

    with h5py.File(args.dec_cache, "r") as fi:
        n_total = fi[d1k].shape[0]
        episode_ends = fi["episode_ends"][:]

    # Shard by contiguous row block: each process owns a disjoint slice, so the
    # writes never overlap and each can be resumed independently.
    edges = np.linspace(0, n_total, args.num_shards + 1).astype(np.int64)
    r0, r1 = int(edges[args.shard]), int(edges[args.shard + 1])
    done_attr = f"n_done_shard{args.shard}"
    print(f"[shard {args.shard}/{args.num_shards}] rows [{r0}:{r1})  of {n_total}", flush=True)

    with h5py.File(args.out, "a") as fo:
        if "episode_ends" not in fo:
            fo.create_dataset("episode_ends", data=episode_ends)
        for key, dim in ((h1k, HEAD_DIM), (h2k, HEAD_DIM), (p1k, POSE_DIM), (p2k, POSE_DIM)):
            if key not in fo:
                # (N,1,D): the singleton axis is the "patch" axis, so the SAME
                # DinoImageEncoder path that consumes [B,T,196,768] consumes
                # this as [B,T,1,D] with n_patch=1 -- no Python change needed.
                fo.create_dataset(key, shape=(n_total, 1, dim), dtype="float16",
                                  chunks=(min(8192, n_total), 1, dim))
        fo[h1k].attrs["source_dec_cache"] = os.path.basename(args.dec_cache)
        fo[h1k].attrs["reloc3r_checkpoint"] = "siyan824/reloc3r-224"
        fo[h1k].attrs["tap"] = "PoseHead.more_mlps output (post AdaptiveAvgPool2d)"
        fo[p1k].attrs["tap"] = "PoseHead final pose: R(9,row-major,post-SVD) ++ t(3,raw fc_t)"
        fo[p1k].attrs["column_order"] = "R00,R01,R02,R10,R11,R12,R20,R21,R22,tx,ty,tz"

        start = int(fo[h1k].attrs.get(done_attr, r0))
        if start >= r1:
            print(f"[shard {args.shard}] already complete.", flush=True)
            return

        t_start = time.time()
        with h5py.File(args.dec_cache, "r") as fi:
            d1, d2 = fi[d1k], fi[d2k]
            for b0 in range(start, r1, args.batch):
                b1 = min(b0 + args.batch, r1)
                for dk, hk, pk in ((d1, h1k, p1k), (d2, h2k, p2k)):
                    toks = torch.from_numpy(dk[b0:b1].astype(np.float32)).to(device)
                    feat, pose12 = head_taps(pose_head, toks)
                    fo[hk][b0:b1] = feat.cpu().numpy().astype(np.float16)[:, None, :]
                    fo[pk][b0:b1] = pose12.cpu().numpy().astype(np.float16)[:, None, :]
                fo[h1k].attrs[done_attr] = b1
                frac = (b1 - r0) / max(r1 - r0, 1)
                el = time.time() - t_start
                print(f"  [shard {args.shard}] rows {b1}/{r1}  {100*frac:.1f}%  "
                      f"elapsed {el/60:.1f}m  eta {el/max(frac,1e-9)*(1-frac)/60:.1f}m",
                      flush=True)
                fo.flush()
    print(f"[shard {args.shard}] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
