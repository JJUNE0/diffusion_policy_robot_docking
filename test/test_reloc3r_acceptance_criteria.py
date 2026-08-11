"""Acceptance-criteria tests for the ReLoc3R token-sequence arms
("Implementation Acceptance Criteria" #3, #6, #7). #1/#2/#4/#5/#8 are exercised
structurally by configs/robot/smr_rgeo.yaml + sensors_variant/reloc3r_*.yaml
(one override switches the arm, shared everything but `sensors:`) and by the
smoke-tested train runs; #9 is scripts/train.py's existing config/checkpoint
logging, unchanged.

Run:  python -m pytest test/test_reloc3r_acceptance_criteria.py -v
"""
import os
import sys

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.setups import model_setups  # noqa: E402
from utils.modular_dataset import ModularDockingDataset  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
VARIANTS = ["reloc3r_relfeat_only", "reloc3r_posthead_only", "reloc3r_pose_only"]

# ${hydra:runtime.cwd} is normally resolved by Hydra's OWN resolver, which is
# only registered inside a real @hydra.main run -- loading a saved/standalone
# config outside that context (as these tests do) raises
# UnsupportedInterpolationType otherwise. Stand in for it.
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda key: {"runtime.cwd": REPO}[key])


def _load_cfg(variant, **overrides):
    base = OmegaConf.load(os.path.join(REPO, "configs/robot/smr_rgeo.yaml"))
    group = OmegaConf.load(os.path.join(REPO, f"configs/robot/sensors_variant/{variant}.yaml"))
    # sensors_variant/*.yaml uses "# @package _global_" under real Hydra; here
    # we replicate that merge by hand (OmegaConf.load alone doesn't apply the
    # package directive -- it's a Hydra-time concept). Drop the `defaults:`
    # list key itself (Hydra-only bookkeeping, not a real config value).
    if "defaults" in base:
        del base["defaults"]
    cfg = OmegaConf.merge(base, group)
    cfg.train_data_path = os.path.join(REPO, "dataset/after_0328_train.h5")
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=True)
    return cfg


@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_forward_shapes(variant):
    """Acceptance criterion #3: condition_tokens [B,N_condition,D],
    noisy_actions [B,H,2], predicted_noise [B,H,2]."""
    torch.manual_seed(0)
    cfg = _load_cfg(variant, batch_size=4, num_workers=0, device=DEVICE)
    dataset, dataloader, nn_condition, nn_diffusion_model, nn_diffusion = model_setups(cfg)

    batch = next(iter(dataloader))
    context = {k: v.to(DEVICE) for k, v in batch["obs"].items()}
    action = batch["act"].to(DEVICE)
    B, H, A = action.shape
    assert (H, A) == (cfg.horizon, 2)

    condition_tokens = nn_condition(context)
    assert condition_tokens.dim() == 3
    assert condition_tokens.shape[0] == B
    assert condition_tokens.shape[2] == cfg.d_model

    noisy_actions = torch.randn(B, H, 2, device=DEVICE)
    t = torch.randint(0, 100, (B,), device=DEVICE)
    predicted_noise = nn_diffusion_model(noisy_actions, t, condition_tokens)
    assert predicted_noise.shape == (B, H, 2)
    assert noisy_actions.shape == (B, H, 2)


def test_no_future_timestamp_in_condition():
    """Acceptance criterion #6: sensor observations must never include a
    frame with index > t (the current step), and the action target must
    start exactly at t (not before)."""
    cfg = _load_cfg("reloc3r_relfeat_only")
    sensors = OmegaConf.to_container(cfg.sensors, resolve=True)
    ds = ModularDockingDataset(
        h5_path=cfg.train_data_path, sensors=sensors, horizon=cfg.horizon,
        obs_horizon=cfg.obs_horizon, train_h5_path=cfg.train_data_path)

    with h5py.File(cfg.train_data_path, "r") as f:
        enc_all = f["encoder"][:]

    rng = np.random.RandomState(0)
    sample_idxs = rng.choice(len(ds), size=200, replace=False)
    for i in sample_idxs:
        t = ds.index_map[i]
        ep_start = ds.ep_start_map[i]
        sample = ds[i]
        # (a) wheel-encoder history: every used row must be in [ep_start, t].
        wheel = sample["obs"]["wheel"].numpy()  # normalized, but ROW COUNT/order is what matters
        assert wheel.shape[0] == cfg.obs_horizon
        # cross-check against the raw source: the LAST row of the returned
        # window must equal frame t (not t+1 or later).
        raw_window_last = enc_all[t]
        # normalize the same way the dataset does, to compare apples-to-apples
        raw_norm_last = ds.normalize_action(raw_window_last)
        assert np.allclose(wheel[-1], raw_norm_last, atol=1e-5), (
            f"idx={i} t={t}: wheel history's last row is not frame t -- "
            f"possible future/misaligned read")
        # (b) action target must start at t: act[0] should equal the
        # normalized encoder row at t (the FIRST predicted step is the
        # current-step action, not some later one).
        act0 = sample["act"][0].numpy()
        assert np.allclose(act0, raw_norm_last, atol=1e-5), (
            f"idx={i} t={t}: action target does not start at t")


def test_relfeat_and_posthead_share_everything_except_the_tapped_stream():
    """Acceptance criterion #7: the relfeat arm (r_relfeat_only, the
    real-robot-validated model) and its controlled posthead baseline
    (r_posthead_only) must be IDENTICAL in every shared parameter/config,
    differing ONLY in WHERE the frozen ReLoc3R stream is tapped
    (dec1/dec2 [196,768] pre-pose-head vs head1/head2 [1,1024] post-pool).
    Checked two ways:
      (a) config equality on every key except `sensors`/`experiment_name`.
      (b) parameter-for-parameter equality of every module the two condition
          nets have IN COMMON (same seed -> same init -> byte-identical
          tensors), and equal DiT architecture end-to-end (the DiT never sees
          `sensors` at all, so it must be 100% identical).
    """
    cfg_relfeat = _load_cfg("reloc3r_relfeat_only")
    cfg_posthead = _load_cfg("reloc3r_posthead_only")
    ignore_keys = {"sensors", "experiment_name"}
    for key in set(cfg_relfeat.keys()) | set(cfg_posthead.keys()):
        if key in ignore_keys:
            continue
        assert cfg_relfeat.get(key) == cfg_posthead.get(key), (
            f"config key '{key}' differs between arms")

    # `wheel` is declared FIRST, identically, in both sensors_variant files, so
    # under a matching seed reset immediately before EACH full model_setups()
    # call the two arms consume identical torch-RNG draws up to the point their
    # vision sensors diverge -- i.e. this DOES validate wheel-encoder init
    # identity (unlike the DiT case below, where the condition net's own
    # variable-size components sit BETWEEN the shared encoder and the DiT).
    torch.manual_seed(7)
    cfg_relfeat_r = _load_cfg("reloc3r_relfeat_only", device=DEVICE)
    _, _, cond_relfeat, dit_relfeat, _ = model_setups(cfg_relfeat_r)
    torch.manual_seed(7)
    cfg_posthead_r = _load_cfg("reloc3r_posthead_only", device=DEVICE)
    _, _, cond_posthead, dit_posthead, _ = model_setups(cfg_posthead_r)

    # DiT (the denoiser) never sees `sensors` -- verify its ARCHITECTURE
    # (hyperparameters -> shapes -> init-seeded weights) is identical between
    # arms. NOTE: comparing dit_relfeat/dit_posthead's state_dicts directly
    # (built via the two full model_setups() calls above) is NOT a valid
    # seeded-identity check -- the condition net is constructed first and may
    # consume a different number of torch RNG draws per arm, so by the time DiT
    # construction starts the shared RNG stream can already have diverged even
    # under one global manual_seed. Rebuild DiT directly, in isolation, with a
    # fresh seed immediately before EACH instantiation instead.
    from cleandiffuser.nn_diffusion import DiTCrossAttn1d
    dit_kwargs_relfeat = dict(in_dim=2, emb_dim=cfg_relfeat_r.d_model, d_model=cfg_relfeat_r.d_model,
                              n_heads=cfg_relfeat_r.n_heads, depth=cfg_relfeat_r.depth, dropout=0.0)
    dit_kwargs_posthead = dict(in_dim=2, emb_dim=cfg_posthead_r.d_model, d_model=cfg_posthead_r.d_model,
                               n_heads=cfg_posthead_r.n_heads, depth=cfg_posthead_r.depth, dropout=0.0)
    assert dit_kwargs_relfeat == dit_kwargs_posthead, "DiT hyperparameters differ between arms"
    torch.manual_seed(123)
    dit_a = DiTCrossAttn1d(**dit_kwargs_relfeat)
    torch.manual_seed(123)
    dit_b = DiTCrossAttn1d(**dit_kwargs_posthead)
    sd_a, sd_b = dit_a.state_dict(), dit_b.state_dict()
    assert sd_a.keys() == sd_b.keys(), "DiT architecture differs between arms"
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"DiT param '{k}' differs between arms"
    # also sanity-check the ACTUAL trained-model DiTs share shapes, even though
    # their init values legitimately differ per the RNG-stream note above.
    assert dit_relfeat.state_dict().keys() == dit_posthead.state_dict().keys()
    for k in dit_relfeat.state_dict():
        assert dit_relfeat.state_dict()[k].shape == dit_posthead.state_dict()[k].shape, (
            f"DiT param '{k}' shape differs")

    # Condition net: the sensor PRESENT IN BOTH (wheel) must be
    # parameter-identical; the vision sensors are named per tap point.
    shared_sensors = set(cond_relfeat.sensor_names) & set(cond_posthead.sensor_names)
    assert shared_sensors == {"wheel"}
    for name in shared_sensors:
        sd_r = cond_relfeat.encoders[name].state_dict()
        sd_p = cond_posthead.encoders[name].state_dict()
        assert sd_r.keys() == sd_p.keys(), f"encoder '{name}' architecture differs"
        for k in sd_r:
            assert torch.equal(sd_r[k], sd_p[k]), f"encoder '{name}' param '{k}' differs"
    assert set(cond_relfeat.sensor_names) - shared_sensors == {"reloc3r_dec1", "reloc3r_dec2"}
    assert set(cond_posthead.sensor_names) - shared_sensors == {"reloc3r_head1", "reloc3r_head2"}

    # Token budget is held EXACTLY equal -- that is what makes this a
    # controlled baseline rather than a capacity comparison.
    assert (cond_relfeat.encoders["reloc3r_dec1"].n_tokens
            == cond_posthead.encoders["reloc3r_head1"].n_tokens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
