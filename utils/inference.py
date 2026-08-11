"""Rebuild a trained policy from its saved config, and integrate its output.

Both helpers are shared by every offline evaluator (test/eval_run_rgeo.py,
test/eval_align_rgeo.py) and by the calibration scripts. The architecture is
NEVER inferred from the live config: a token-sequence checkpoint cannot
reconstruct its temporal sensor layout from weights alone, so the config
written next to the checkpoint at train time is the only valid source.
"""
import numpy as np
from omegaconf import OmegaConf

from cleandiffuser.nn_condition.token_sequence_condition import TokenSequenceFusionCondition
from cleandiffuser.nn_diffusion import DiTCrossAttn1d
from utils.setups import _select_backbone


def reconstruct_pose_rk4(linear_vels, angular_vels, dt=0.0333, initial_pose=(0.0, 0.0, 0.0)):
    """Integrate a (v, w) command sequence to an SE(2) path via RK4.

    Convention: +x forward, +y left, theta CCW (matches the dataset's body
    frame). Returns [n_steps + 1, 3] poses, starting at `initial_pose`.
    """
    n_steps = len(linear_vels)
    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = initial_pose

    def f(q, v, w):
        return np.array([v * np.cos(q[2]), v * np.sin(q[2]), w])

    curr_q = np.array(initial_pose, dtype=float)
    for i in range(n_steps):
        v, w = linear_vels[i], angular_vels[i]
        k1 = f(curr_q, v, w)
        k2 = f(curr_q + 0.5 * dt * k1, v, w)
        k3 = f(curr_q + 0.5 * dt * k2, v, w)
        k4 = f(curr_q + dt * k3, v, w)
        curr_q += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        curr_q[2] = (curr_q[2] + np.pi) % (2 * np.pi) - np.pi
        trajectory[i + 1] = curr_q
    return trajectory


def build_model_from_cfg(cfg, device):
    """Rebuild the network exactly as model_setups did at TRAIN time, from the
    checkpoint's saved config (no dataset/dataloader)."""
    sensors = OmegaConf.to_container(cfg.sensors, resolve=True)
    nn_condition = TokenSequenceFusionCondition(
        sensors=sensors, d_model=cfg.d_model, nhead=cfg.n_heads,
        num_layers=cfg.get("condition_num_layers", 4), dropout=cfg.dropout,
    ).to(device)
    nn_diffusion_model = DiTCrossAttn1d(
        in_dim=2, emb_dim=cfg.d_model, d_model=cfg.d_model,
        n_heads=cfg.n_heads, depth=cfg.depth, dropout=0.0).to(device)
    Backbone = _select_backbone(cfg)
    nn_diffusion = Backbone(
        nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
        ema_rate=cfg.get("ema_rate", 0.999), device=device)
    return nn_condition, nn_diffusion
