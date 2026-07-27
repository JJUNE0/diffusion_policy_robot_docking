"""Acceptance-criteria tests for R-NoGoal/R-Goal/R-Geo
(docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md, "Implementation Acceptance
Criteria" #3, #6, #7). #1/#2/#4/#5/#8 are exercised structurally by
configs/robot/smr_rgeo.yaml + sensors_variant/*.yaml (one flag switches all
three, shared everything but `sensors:`) and by the smoke-tested train runs
(2026-07-25 session); #9 is scripts/train.py's existing config/checkpoint
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
VARIANTS = ["no_goal", "goal_appearance", "goal_appearance_geometry"]

# ${hydra:runtime.cwd} is normally resolved by Hydra's OWN resolver, which is
# only registered inside a real @hydra.main run -- loading a saved/standalone
# config outside that context (as these tests, or test/eval_run_modular.py,
# do) raises UnsupportedInterpolationType otherwise. Stand in for it.
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
    cfg = _load_cfg("goal_appearance_geometry")
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


def test_base_and_geometry_share_everything_except_geometry_token():
    """Acceptance criterion #7: base (no_goal) and geometry
    (goal_appearance_geometry) models must be IDENTICAL in every shared
    parameter/config, differing only by the presence of the geometry
    sensor/token. Checked two ways:
      (a) config equality on every key except `sensors`/`sensors_variant`
          bookkeeping and `experiment_name`.
      (b) parameter-for-parameter equality of every module the two condition
          nets and DiTs have IN COMMON (same seed -> same init -> byte-
          identical tensors), and equal DiT architecture end-to-end (DiT
          never sees `sensors` at all, so it must be 100% identical).
    """
    cfg_base = _load_cfg("no_goal")
    cfg_geo = _load_cfg("goal_appearance_geometry")
    ignore_keys = {"sensors", "experiment_name"}
    for key in set(cfg_base.keys()) | set(cfg_geo.keys()):
        if key in ignore_keys:
            continue
        assert cfg_base.get(key) == cfg_geo.get(key), f"config key '{key}' differs between arms"

    # Shared encoders (wheel, rgb_history, lidar) are declared FIRST, in the
    # same relative order, in both sensors_variant/*.yaml files, so under a
    # matching seed reset immediately before EACH full model_setups() call
    # they consume identical torch-RNG draws up until the point each arm's
    # extra sensors (goal/geometry) diverge -- i.e. this DOES validate their
    # init identity (unlike the DiT case below, where the condition net's
    # own variable-size components sit BETWEEN the shared encoders and DiT).
    torch.manual_seed(7)
    cfg_base_r = _load_cfg("no_goal", device=DEVICE)
    _, _, cond_base, dit_base, _ = model_setups(cfg_base_r)
    torch.manual_seed(7)
    cfg_geo_r = _load_cfg("goal_appearance_geometry", device=DEVICE)
    _, _, cond_geo, dit_geo, _ = model_setups(cfg_geo_r)

    # DiT (the denoiser) never sees `sensors` -- verify its ARCHITECTURE
    # (hyperparameters -> shapes -> init-seeded weights) is identical between
    # arms. NOTE: comparing dit_base/dit_geo's state_dicts directly (built via
    # the two full model_setups() calls above) is NOT a valid seeded-identity
    # check -- the condition net is constructed first and consumes a
    # DIFFERENT NUMBER of torch RNG draws per arm (modality_emb size and the
    # fusion Transformer both depend on n_modalities), so by the time DiT
    # construction starts the shared torch RNG stream has already diverged
    # even under one global manual_seed. Rebuild DiT directly, in isolation,
    # with a fresh seed immediately before EACH instantiation instead.
    from cleandiffuser.nn_diffusion import DiTCrossAttn1d
    dit_kwargs_base = dict(in_dim=2, emb_dim=cfg_base_r.d_model, d_model=cfg_base_r.d_model,
                           n_heads=cfg_base_r.n_heads, depth=cfg_base_r.depth, dropout=0.0)
    dit_kwargs_geo = dict(in_dim=2, emb_dim=cfg_geo_r.d_model, d_model=cfg_geo_r.d_model,
                          n_heads=cfg_geo_r.n_heads, depth=cfg_geo_r.depth, dropout=0.0)
    assert dit_kwargs_base == dit_kwargs_geo, "DiT hyperparameters differ between arms"
    torch.manual_seed(123)
    dit_a = DiTCrossAttn1d(**dit_kwargs_base)
    torch.manual_seed(123)
    dit_b = DiTCrossAttn1d(**dit_kwargs_geo)
    sd_base, sd_geo = dit_a.state_dict(), dit_b.state_dict()
    assert sd_base.keys() == sd_geo.keys(), "DiT architecture differs between arms"
    for k in sd_base:
        assert torch.equal(sd_base[k], sd_geo[k]), f"DiT param '{k}' differs between arms"
    # also sanity-check the ACTUAL trained-model DiTs (dit_base/dit_geo, from
    # model_setups above) share the same architecture/shapes, even though
    # their init values legitimately differ per the RNG-stream note above.
    assert dit_base.state_dict().keys() == dit_geo.state_dict().keys()
    for k in dit_base.state_dict():
        assert dit_base.state_dict()[k].shape == dit_geo.state_dict()[k].shape, f"DiT param '{k}' shape differs"

    # Condition net: every sensor PRESENT IN BOTH (wheel, rgb_history, lidar)
    # must be parameter-identical; `goal`/`geometry` only exist in the geo arm.
    shared_sensors = set(cond_base.sensor_names) & set(cond_geo.sensor_names)
    assert shared_sensors == {"wheel", "rgb_history", "lidar"}
    for name in shared_sensors:
        sd_b = cond_base.encoders[name].state_dict()
        sd_g = cond_geo.encoders[name].state_dict()
        assert sd_b.keys() == sd_g.keys(), f"encoder '{name}' architecture differs"
        for k in sd_b:
            assert torch.equal(sd_b[k], sd_g[k]), f"encoder '{name}' param '{k}' differs"
    assert set(cond_geo.sensor_names) - set(cond_base.sensor_names) == {"goal", "geometry"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
