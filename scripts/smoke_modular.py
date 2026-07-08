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


def build_and_run(spec, tag):
    net = ModularSensorFusionCondition(spec, d_model=D, nhead=NHEAD, num_layers=2)
    net.train()
    out = net(make_batch(spec))
    n_params = sum(p.numel() for p in net.parameters())
    assert out.shape == (B, D), f"{tag}: expected {(B, D)}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{tag}: non-finite output"
    print(f"[OK] {tag}: sensors={list(spec)} -> out {tuple(out.shape)}, params={n_params:,}")
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

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
