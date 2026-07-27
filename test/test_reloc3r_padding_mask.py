"""Acceptance criterion #5 (docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md):
"Missing sensor history에 대한 padding mask가 attention에 전달돼야 한다."
Verifies the mask isn't just plumbed-but-inert: an episode-start sample (most
of its RGB/wheel history repeat-padded) must produce a real, non-trivial
key_valid_mask, AND actually changes the fusion Transformer's self-attention
output vs. an all-valid mask on the same tokens (i.e. the mask is load-bearing,
not silently ignored).

Run:  python -m pytest test/test_reloc3r_padding_mask.py -v
"""
import os
import sys

import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda key: {"runtime.cwd": REPO}[key])

from utils.modular_dataset import ModularDockingDataset  # noqa: E402
from cleandiffuser.nn_condition.token_sequence_condition import TokenSequenceFusionCondition  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_ds():
    base = OmegaConf.load(os.path.join(REPO, "configs/robot/smr_rgeo.yaml"))
    group = OmegaConf.load(os.path.join(REPO, "configs/robot/sensors_variant/no_goal.yaml"))
    if "defaults" in base:
        del base["defaults"]
    cfg = OmegaConf.merge(base, group)
    cfg.train_data_path = os.path.join(REPO, "dataset/after_0328_train.h5")
    sensors = OmegaConf.to_container(cfg.sensors, resolve=True)
    ds = ModularDockingDataset(
        h5_path=cfg.train_data_path, sensors=sensors, horizon=cfg.horizon,
        obs_horizon=cfg.obs_horizon, train_h5_path=cfg.train_data_path)
    return ds, cfg, sensors


def test_episode_start_sample_has_padded_frames():
    """The very first valid sample of an episode (t == ep_start) must have
    obs_horizon-1 padded (repeated) frames and exactly 1 real frame."""
    ds, cfg, _ = _load_ds()
    first_idx = next(i for i, t in enumerate(ds.index_map) if t == ds.ep_start_map[i])
    sample = ds[first_idx]
    vm = sample["obs"]["wheel_valid_mask"]
    assert vm.shape[0] == cfg.obs_horizon
    assert vm.sum().item() == 1, f"expected exactly 1 valid frame at ep_start, got {vm.sum().item()}"
    assert vm[-1].item() is True or bool(vm[-1]), "the LAST (current) frame must always be valid"
    assert not bool(vm[0]), "the FIRST (oldest) frame at an ep_start sample must be padding"

    # a sample deep into the episode (t = ep_start + obs_horizon) should have
    # a fully valid (no padding) history.
    deep_idx = next(i for i, t in enumerate(ds.index_map)
                    if t == ds.ep_start_map[i] + cfg.obs_horizon)
    sample_deep = ds[deep_idx]
    assert bool(sample_deep["obs"]["wheel_valid_mask"].all())


def test_padding_mask_changes_fusion_attention_output():
    """The mask must actually be USED by TokenSequenceFusionCondition's
    self-attention, not just computed-and-discarded: feeding the padded
    episode-start sample with its real (mostly-invalid) mask must give a
    DIFFERENT fused output than force-marking every token valid."""
    ds, cfg, sensors = _load_ds()
    torch.manual_seed(0)
    cond = TokenSequenceFusionCondition(
        sensors=sensors, d_model=cfg.d_model, nhead=cfg.n_heads,
        num_layers=cfg.get("condition_num_layers", 4), dropout=0.0).to(DEVICE)
    cond.eval()

    first_idx = next(i for i, t in enumerate(ds.index_map) if t == ds.ep_start_map[i])
    sample = ds[first_idx]
    ctx_masked = {k: v.unsqueeze(0).to(DEVICE) for k, v in sample["obs"].items()}
    assert not ctx_masked["wheel_valid_mask"].all(), "fixture must actually contain padding"

    with torch.no_grad():
        out_masked = cond(ctx_masked)

    # now force every valid_mask to True (simulating "mask ignored") and
    # confirm the fused output differs -- proof the mask is load-bearing.
    ctx_unmasked = {k: v.clone() for k, v in ctx_masked.items()}
    for k in list(ctx_unmasked.keys()):
        if k.endswith("_valid_mask"):
            ctx_unmasked[k] = torch.ones_like(ctx_unmasked[k])

    with torch.no_grad():
        out_unmasked = cond(ctx_unmasked)

    assert out_masked.shape == out_unmasked.shape
    assert not torch.allclose(out_masked, out_unmasked, atol=1e-6), (
        "fused condition tokens are IDENTICAL whether padding is masked or not -- "
        "the padding mask is not reaching attention")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
