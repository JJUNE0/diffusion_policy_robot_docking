# DP-Image-Goal baseline

This baseline uses the unmodified official `real-stanford/diffusion_policy`
implementation pinned as the git submodule `third_party/diffusion_policy` at
commit `5ba07ac6661db573af695b419a7947ecb704690f`.

## Contract

- Data: the same 235 combined 4F+5F episodes used by the 1-camera RTDP run.
- Observation: five bottom-camera frames from a 60-frame window (stride 12),
  one bottom-camera goal image, and all 60 wheel `(v,w)` observations.
- Output: 60 future raw physical commands with shape `[60,2] = (v,w)`.
- Goal sampling: choose same-floor versus cross-floor first (`p=0.1` cross),
  then choose the variant within that candidate group using
  `original=0.495`, `qr_removed=0.495`, `color_changed=0.01`. These are
  hierarchical independent decisions, not competing entries in one categorical
  distribution.
- Training: one experiment seed (`42`), 20 epochs, official ResNet18 image
  encoder with GroupNorm, official conditional 1-D U-Net, DDPM epsilon loss,
  AdamW/cosine warmup, and official EMA.

The adapter uses `n_obs_steps=1` with named history keys. This preserves the five
frame positions while encoding the static goal only once instead of five times.
The official shared RGB encoder embeds each image independently, making this the
appearance-only image-goal control requested for comparison with RTDP's pairwise
relational tokens.

## Environment and run

```bash
python -m venv --system-site-packages .venv_dp_image_goal
.venv_dp_image_goal/bin/python -m pip install -r requirements_dp_image_goal.txt
bash scripts/run_dp_image_goal_full.sh
```

The launcher loads the 84-GB `image_bottom` VDS into one process-owned memfd.
Both DDP ranks and their workers map the same read-only RAM pages, avoiding
network-HDF5 random reads and duplicate per-rank caches. It then trains on both
H100s with BF16, TF32-enabled kernels, fused AdamW, per-GPU batch 512, and global
batch 1024. Logs and checkpoints are written under
`outputs/train/dp_image_goal/<UTC timestamp>/`.
