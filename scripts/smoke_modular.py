"""Smoke test for the modular sensor-fusion framework.

Proves, with synthetic tensors (no h5, CPU): (1) the registry builds a fusion
net from a `sensors` spec; (2) forward returns [B, d_model]; (3) ABLATION works
by simply editing the spec dict -- dropping/adding a sensor yields a valid net
with a different parameter count and still runs. Run: python3 scripts/smoke_modular.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cleandiffuser.nn_condition.modality_encoders import ENCODER_REGISTRY
from cleandiffuser.nn_condition.modular_fusion_condition import ModularSensorFusionCondition

B, D, NHEAD = 4, 64, 4
Tv, Tm, N_PATCH, FEAT, M = 5, 30, 196, 768, 256


def make_batch(spec):
    cond = {}
    for name, s in spec.items():
        enc = s["encoder"]
        if enc in ("motion", "imu"):
            cond[name] = torch.randn(B, s["horizon"], s["dim"])
        elif enc == "dino_image":
            cond[name] = torch.randn(B, s["horizon"], N_PATCH, FEAT)
        elif enc == "pointcloud":
            cond[name] = torch.randn(B, M, 2)
            cond[f"{name}_npoints"] = torch.randint(1, M, (B,))
    return cond


def build_and_run(spec, tag, expect_aux=None):
    net = ModularSensorFusionCondition(spec, d_model=D, nhead=NHEAD, num_layers=2)
    net.train()
    out = net(make_batch(spec))
    n_params = sum(p.numel() for p in net.parameters())
    assert out.shape == (B, D), f"{tag}: expected {(B, D)}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{tag}: non-finite output"
    if expect_aux is True:
        assert net._aux_pred is not None and net._aux_pred.shape == (B, 4), \
            f"{tag}: expected aux pred [B,4], got {None if net._aux_pred is None else tuple(net._aux_pred.shape)}"
        assert torch.isfinite(net._aux_pred).all(), f"{tag}: non-finite aux pred"
    if expect_aux is False:
        assert net._aux_pred is None, f"{tag}: expected NO aux pred"
    aux_tag = "" if expect_aux is None else f", aux={'on' if expect_aux else 'off'}"
    print(f"[OK] {tag}: sensors={list(spec)} -> out {tuple(out.shape)}, params={n_params:,}{aux_tag}")
    return n_params


def main():
    print("Registered encoders:", sorted(ENCODER_REGISTRY))

    full = {
        "velocity": {"encoder": "motion", "dim": 2, "horizon": Tm, "dropout_prob": 0.2},
        "imu": {"encoder": "imu", "dim": 6, "horizon": Tm, "dropout_prob": 0.2},
        "cam_top": {"encoder": "dino_image", "horizon": Tv, "num_latents": 8},
        "cam_bottom": {"encoder": "dino_image", "horizon": Tv, "num_latents": 8},
        "lidar": {"encoder": "pointcloud", "num_latents": 8},
    }
    p_full = build_and_run(full, "full (5 modalities)")

    # Ablation 1: drop imu + lidar by editing the dict.
    abl = {k: full[k] for k in ("velocity", "cam_top", "cam_bottom")}
    p_abl = build_and_run(abl, "ablation (drop imu+lidar)")

    # Ablation 2: single modality.
    solo = {"velocity": full["velocity"]}
    build_and_run(solo, "ablation (velocity only)")

    assert p_abl < p_full, "ablated net should have fewer params than full"

    # Missing-key error path.
    net = ModularSensorFusionCondition(full, d_model=D, nhead=NHEAD, num_layers=2)
    try:
        net({"velocity": torch.randn(B, Tm, 2)})
        raise AssertionError("expected KeyError for missing sensors")
    except KeyError:
        print("[OK] missing-sensor KeyError raised as expected")

    # ---- aux_pose head: ablatable via the `head` field --------------------
    with_head = {**{k: full[k] for k in ("velocity", "lidar")}}
    with_head["lidar"] = {**full["lidar"], "head": "aux_pose"}
    p_head = build_and_run(with_head, "lidar + aux_pose head (cross-attn)", expect_aux=True)

    without_head = {"velocity": full["velocity"], "lidar": full["lidar"]}
    p_nohead = build_and_run(without_head, "lidar, no head (ablated)", expect_aux=False)
    assert p_head > p_nohead, "aux head should add params"

    # Pooled fallback: head on a non-pointcloud sensor.
    pooled = {"velocity": {**full["velocity"], "head": "aux_pose"}}
    build_and_run(pooled, "velocity + aux_pose head (pooled fallback)", expect_aux=True)

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
