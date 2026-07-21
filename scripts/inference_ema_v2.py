"""Open-loop EMA inference, compatible with CURRENT-generation checkpoints.

Key difference from the old inference_ema.py: the network is built from the
config.yaml SAVED NEXT TO THE CHECKPOINT (logger_setups writes it at train
time), not from the live smr.yaml. So goal/lidar/aux/use_room1/backbone always
match the weights being loaded -- any checkpoint generation loads cleanly.

Runtime knobs (this config / CLI): eval_data_path, episode_idx, max_steps,
num_samples, solver, inference_sampling_steps, w_cfg, traj_ema_alpha,
inference_checkpoint_dir, checkpoint_step.

Run (from repo root):
  WANDB_MODE=disabled HF_HUB_OFFLINE=1 python scripts/inference_ema_v2.py \
    inference_checkpoint_dir=$PWD/outputs/results/<exp>/<ts> \
    checkpoint_step=8470 eval_data_path=$PWD/dataset/after_0328_train.h5
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
from cleandiffuser.nn_condition.modular_fusion_condition import ModularSensorFusionCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.rollout_core import RolloutController
from utils.setups import _select_backbone
from utils.docking_dataset import DockingDataset, denormalize
from utils.modular_dataset import ModularDockingDataset


def reconstruct_pose_rk4(linear_vels, angular_vels, dt=0.0333, initial_pose=(0.0, 0.0, 0.0)):
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
    obs_horizon = cfg.get("obs_horizon", 30)
    if cfg.get("use_modular_fusion", False):
        sensors = OmegaConf.to_container(cfg.sensors, resolve=True)
        nn_condition = ModularSensorFusionCondition(
            sensors=sensors, d_model=cfg.d_model, nhead=cfg.n_heads,
            num_layers=cfg.get("condition_num_layers", 4), dropout=cfg.dropout,
        ).to(device)
    else:
        vision_horizon = len(range(0, obs_horizon, cfg.get("vision_stride", 6)))
        nn_condition = SensorFusionConditionNetwork(
            state_dim=cfg.state_dim,
            obs_horizon=obs_horizon,
            vision_horizon=vision_horizon,
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            num_layers=cfg.get("condition_num_layers", 2),
            dropout=cfg.dropout,
            num_image_latents=cfg.get("num_image_latents", 16),
            velocity_dim=cfg.get("velocity_dim", 2),
            velocity_dropout_prob=cfg.get("velocity_dropout_prob", 0.0),
            use_goal=cfg.get("use_goal", False),
            num_goal_latents=cfg.get("num_goal_latents", 16),
            use_lidar_points=cfg.get("use_lidar_points", False),
            num_lidar_latents=cfg.get("num_lidar_latents", 16),
            use_aux_pose=cfg.get("use_aux_pose", False),
            use_room1=cfg.get("use_room1", True),
            use_goal_lidar=cfg.get("use_goal_lidar", False),
            use_aux_feedback=cfg.get("use_aux_feedback", False),
        ).to(device)

    nn_diffusion_model = DiT1d(
        in_dim=2, emb_dim=cfg.d_model, d_model=cfg.d_model,
        n_heads=cfg.n_heads, depth=cfg.depth, dropout=0.0).to(device)

    Backbone = _select_backbone(cfg)
    nn_diffusion = Backbone(
        nn_diffusion=nn_diffusion_model, nn_condition=nn_condition,
        ema_rate=cfg.get("ema_rate", 0.9999), device=device)
    return nn_condition, nn_diffusion


@hydra.main(config_path="../configs/robot", config_name="smr", version_base=None)
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = HydraConfig.get().runtime.output_dir

    ckpt_dir = args.get("inference_checkpoint_dir")
    if not ckpt_dir:
        raise ValueError("pass inference_checkpoint_dir=<dir with checkpoint + config.yaml>")
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{args.checkpoint_step}.pt")

    # --- Architecture comes from the checkpoint's own config -----------------
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    model_cfg = OmegaConf.load(cfg_path) if os.path.exists(cfg_path) else args
    if not os.path.exists(cfg_path):
        print(f"WARNING: {cfg_path} not found; falling back to the LIVE config "
              f"(architecture mismatch will fail the state_dict load).")
    use_modular = model_cfg.get("use_modular_fusion", False)
    use_goal = model_cfg.get("use_goal", False)
    use_lidar = model_cfg.get("use_lidar_points", False)
    use_room1 = model_cfg.get("use_room1", True)
    obs_h = model_cfg.get("obs_horizon", 30)
    horizon = model_cfg.horizon
    vision_stride = model_cfg.get("vision_stride", 6)
    dt = float(args.get("dt", 0.0333))

    nn_condition, nn_diffusion = build_model_from_cfg(model_cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nn_diffusion.model.load_state_dict(ckpt["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ckpt["ema_state_dict"])
    nn_diffusion.eval()
    act_min, act_scale = ckpt["action_min"], ckpt["action_scale"]
    print(f"loaded {ckpt_path}\n  backbone={model_cfg.get('diffusion_backbone', 'rectified_flow')} "
          f"| modular={use_modular} | goal={use_goal} | lidar={use_lidar} | room1={use_room1}")

    # --- Dataset (open-loop source). Live pixels; flags follow the model. ----
    if use_modular:
        sensors = OmegaConf.to_container(model_cfg.sensors, resolve=True)
        dataset = ModularDockingDataset(
            h5_path=args.eval_data_path, sensors=sensors, horizon=horizon,
            obs_horizon=obs_h, action_key=model_cfg.get("action_key", "encoder"),
            train_h5_path=model_cfg.get("train_data_path", args.eval_data_path))
        dino = None
    else:
        dataset = DockingDataset(
            npz_path=args.eval_data_path, train_npz_path=args.eval_data_path,
            horizon=horizon, obs_horizon=obs_h, dt=dt,
            with_goal=use_goal, goal_mask_prob=1.0,
            with_lidar=use_lidar, use_room1=use_room1)
        from dino.dino_detector import DinoBatchDetector
        dino = DinoBatchDetector(device=device)

    ep_idx = int(args.get("episode_idx", 0))
    ep_start = 0 if ep_idx == 0 else int(dataset.episode_ends[ep_idx - 1])
    ep_end = int(dataset.episode_ends[ep_idx])
    # sample index whose t == ep_start (index_map is grouped by episode)
    base = next(i for i, t in enumerate(dataset.index_map) if t == ep_start)
    ep_steps = ep_end - ep_start - horizon + 1
    max_steps = args.get("max_steps", None)
    if max_steps is not None:
        ep_steps = min(ep_steps, int(max_steps))
    print(f"episode {ep_idx}: frames [{ep_start}, {ep_end}) -> {ep_steps} open-loop steps")

    n_samples = int(args.get("num_samples", 8))
    traj_ema_alpha = float(args.get("traj_ema_alpha", 0.3))

    def encode_hist(img):
        """[T,3,H,W] float in [0,1] -> [1,Tv,196,768] via strided DINO."""
        img = img[::vision_stride].to(device)
        T, C, H, W = img.shape
        with torch.no_grad():
            f, _, _ = dino.get_heatmap(img)
        return f.view(1, T, 196, 768)

    goal_ctx = {}
    if (not use_modular) and use_goal:
        s0 = dataset[base]["obs"]
        with torch.no_grad():
            gf2, _, _ = dino.get_heatmap(s0["goal_image_room2"].unsqueeze(0).to(device))
        goal_ctx["goal_feat2"] = gf2.view(1, 1, 196, 768).repeat(n_samples, 1, 1, 1)
        if use_room1:
            gf1, _, _ = dino.get_heatmap(s0["goal_image_room1"].unsqueeze(0).to(device))
            goal_ctx["goal_feat1"] = gf1.view(1, 1, 196, 768).repeat(n_samples, 1, 1, 1)
        goal_ctx["goal_mask"] = torch.ones(n_samples, device=device)

    dataset._ensure_open()
    gt_actions = np.asarray(
        dataset.root["encoder"][ep_start:ep_start + ep_steps] if use_modular
        else dataset.z_encoder[ep_start:ep_start + ep_steps], dtype=np.float32)

    # Action chunking / temporal-consistency (2026-07-21) -- same knobs and
    # same no-op-by-default contract as test/eval_openloop_metrics.py.rollout().
    # Both now share the sampling/aggregation/EMA/warm-start core via
    # cleandiffuser.rollout_core.RolloutController; this loop only builds the
    # per-frame context (data source differs) and owns the chunk cadence.
    exec_chunk_k = int(os.environ.get("EVAL_CHUNK_K", "1"))
    warm_start = os.environ.get("EVAL_WARM_START", "0") == "1"
    warm_level = float(os.environ.get("EVAL_WARM_LEVEL", "0.3"))
    legacy_rollout = (exec_chunk_k == 1 and not warm_start)

    ctrl = RolloutController(
        nn_diffusion, solver=args.solver, sample_steps=args.inference_sampling_steps,
        use_ema=bool(args.get("use_ema", True)), n_samples=n_samples, horizon=horizon,
        w_cfg=args.w_cfg, agg="mean", traj_ema_alpha=traj_ema_alpha,
        warm_start=warm_start, warm_level=warm_level, device=device)
    selected, v_samp, w_samp = [], [], []
    step = 0
    while step < ep_steps:
        batch = dataset[base + step]
        obs = batch["obs"]

        if use_modular:
            context = {}
            for k, v in obs.items():
                v = v.unsqueeze(0).to(device)
                context[k] = v.repeat(n_samples, *([1] * (v.dim() - 1))) if v.dim() > 0 \
                    else v.repeat(n_samples)
        else:
            context = {
                "dino_feat2": encode_hist(obs["image_room2"]).repeat(n_samples, 1, 1, 1),
                "velocity": obs["velocity"].unsqueeze(0).to(device).repeat(n_samples, 1, 1),
            }
            if use_room1:
                context["dino_feat1"] = encode_hist(obs["image_room1"]).repeat(n_samples, 1, 1, 1)
            if use_lidar:
                context["lidar_points"] = obs["lidar_points"].unsqueeze(0).to(device).repeat(n_samples, 1, 1)
                context["lidar_npoints"] = obs["lidar_npoints"].view(1).to(device).repeat(n_samples)
            context.update(goal_ctx)

        k = 1 if legacy_rollout else min(exec_chunk_k, ep_steps - step, horizon)
        plan = ctrl.plan(context, act_min, act_scale, chunk_shift=k)
        res = plan.samples                        # [N,H,2] denorm, for spread plot

        if legacy_rollout:
            v_samp.append(res[:, 0, 0].copy())
            w_samp.append(res[:, 0, 1].copy())
            selected.append(plan.ema[0, :])
            step += 1
        else:
            # v_samp/w_samp feed the per-step sample-spread plot below, which is
            # indexed 1:1 against gt_actions (always full ep_steps) -- repeat
            # this resample's per-sample column across the k steps it covers so
            # lengths stay aligned when chunking reduces resample frequency.
            for j in range(k):
                v_samp.append(res[:, j, 0].copy())
                w_samp.append(res[:, j, 1].copy())
            selected.extend(list(plan.current[0:k]))
            step += k

        if step % 50 == 0 or step == ep_steps:
            print(f"progress: {step}/{ep_steps}")

    ai_actions = np.array(selected)
    gt_path = reconstruct_pose_rk4(gt_actions[:, 0], gt_actions[:, 1], dt=dt)
    ai_path = reconstruct_pose_rk4(ai_actions[:, 0], ai_actions[:, 1], dt=dt)
    end_err = float(np.hypot(*(gt_path[-1, :2] - ai_path[-1, :2])))
    print(f"endpoint error (GT vs AI, open-loop integration): {end_err*100:.1f} cm")

    # --- plots ----------------------------------------------------------------
    v_np, w_np = np.asarray(v_samp), np.asarray(w_samp)              # [T, N]
    steps = np.arange(len(gt_actions))
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for i, (gt, mean, std, sel, name, col) in enumerate([
            (gt_actions[:, 0], v_np.mean(1), v_np.std(1), ai_actions[:, 0], "linear", "tab:red"),
            (gt_actions[:, 1], w_np.mean(1), w_np.std(1), ai_actions[:, 1], "angular", "tab:blue")]):
        ax[i].plot(steps, gt, 'k--', lw=1.5, label=f'GT {name} vel')
        ax[i].plot(steps, mean, color=col, lw=1.0, label='sample mean')
        ax[i].fill_between(steps, mean - std, mean + std, color=col, alpha=0.2, label='sample std')
        ax[i].plot(steps, sel, color='forestgreen', lw=1.2, label='EMA')
        ax[i].set_ylabel(f"{name} velocity"); ax[i].grid(True, ls='--', alpha=0.5)
        ax[i].legend(loc="upper right", framealpha=0.9)
    ax[1].set_xlabel("Step")
    fig.suptitle(f"Per-step velocity (ep {ep_idx}, EMA a={traj_ema_alpha:.2f}, N={n_samples})")
    plt.tight_layout()
    vel_png = os.path.join(output_dir, f"vel_compare_ep{ep_idx}_step{args.checkpoint_step}.png")
    plt.savefig(vel_png, dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(gt_path[:, 0], gt_path[:, 1], 'k--', lw=1.5, alpha=0.6, label="Ground Truth")
    ax.plot(ai_path[:, 0], ai_path[:, 1], color="forestgreen", lw=1.0, alpha=0.8, label="AI (EMA)")
    ax.scatter(*gt_path[0, :2], c="limegreen", s=100, marker="o", zorder=5, label="start")
    ax.scatter(*gt_path[-1, :2], c="black", s=80, marker="X", zorder=5, label="GT end")
    ax.scatter(*ai_path[-1, :2], c="forestgreen", s=80, marker="X", zorder=5, label="AI end")
    ax.set_title(f"Open-loop rollout ep {ep_idx} | step {args.checkpoint_step} | "
                 f"endpoint err {end_err*100:.1f} cm")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, ls='--', alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    traj_png = os.path.join(output_dir, f"traj_ep{ep_idx}_step{args.checkpoint_step}.png")
    plt.savefig(traj_png, dpi=200); plt.close(fig)
    print(f"saved:\n  {vel_png}\n  {traj_png}")


if __name__ == "__main__":
    main()
