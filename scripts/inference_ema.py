import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def _ensure_hydra_override(key: str, value: str) -> None:
    already_set = any(
        a.split("=", 1)[0].lstrip("+") == key for a in sys.argv[1:] if "=" in a
    )
    if not already_set:
        sys.argv.append(f"{key}={value}")


_ensure_hydra_override("run_kind", "inference")

import zmq
import numpy as np
import matplotlib.pyplot as plt
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork
from cleandiffuser.nn_diffusion import DiT1d

from dataset.docking_dataset import DockingDataset
from utils.inference_ckpt import resolve_inference_checkpoint_path


def denormalize(norm_action, act_scale, act_min):
    return (norm_action + 1.0) / 2.0 * act_scale + act_min


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


# ===== [CHANGED] =====
# RTC helpers removed.
# Trajectory EMA helper added.
def apply_trajectory_ema(current_action, prev_ema_action=None, ema_alpha=0.3):
    """
    current_action: [horizon, 2]
    prev_ema_action: [horizon, 2] or None

    EMA over the whole predicted velocity trajectory.
    """
    if prev_ema_action is None:
        return current_action

    return ema_alpha * current_action + (1.0 - ema_alpha) * prev_ema_action


@hydra.main(config_path="../configs/robot", config_name="smr", version_base=None)
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    original_cwd = get_original_cwd()
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    obs_h = args.get("obs_horizon", 30)
    horizon = args.horizon
    n_samples = args.get("num_samples", 10)

    print("Num samples", n_samples)

    # ===== [CHANGED] =====
    # RTC params removed. EMA param added.
    traj_ema_alpha = float(args.get("traj_ema_alpha", 0.3)) # 1.0 = no ema

    vision_stride = args.get("vision_stride", 6)
    vision_h = len(range(0, obs_h, vision_stride))
    dt = float(args.get("dt", 0.0333))

    vision_backend = args.get("vision_backend", "raw_cnn")

    test_dataset = DockingDataset(
        npz_path=args.eval_data_path,
        train_npz_path=args.train_data_path,
        horizon=args.horizon,
        obs_horizon=obs_h,
        vision_stride=vision_stride,
        dt=dt,
    )

    use_lidar_inf = bool(args.get("use_lidar", False))
    lidar_in_ch = int(args.get("lidar_channels", 2))
    if use_lidar_inf and test_dataset.z_lidar is not None and test_dataset.lidar_meta:
        lidar_in_ch = int(test_dataset.lidar_meta["channels"])
    if use_lidar_inf and test_dataset.z_lidar is None:
        raise ValueError(
            "use_lidar=True but eval HDF5 has no lidar_map. "
            "Build data with preprocessing.py --use_lidar or set use_lidar=false."
        )
    if vision_backend == "dino" and use_lidar_inf:
        raise NotImplementedError("use_lidar with vision_backend=dino is not wired in inference_ema.")

    nn_condition = SensorFusionConditionNetwork(
        state_dim=args.state_dim,
        obs_horizon=obs_h,
        vision_horizon=vision_h,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.get("condition_num_layers", 2),
        dropout=args.dropout,
        num_image_latents=args.get("num_image_latents", 16),
        velocity_dim=args.get("velocity_dim", 2),
        velocity_dropout_prob=args.get("velocity_dropout_prob", 0.0),
        vision_backend=vision_backend,
        use_lidar=use_lidar_inf,
        lidar_in_ch=lidar_in_ch,
        num_lidar_latents=args.get("num_lidar_latents", 16),
        lidar_dropout_prob=0.0,
    ).to(device)

    nn_diffusion_model = DiT1d(
        in_dim=2,
        emb_dim=args.d_model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
    ).to(device)

    nn_diffusion = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion_model,
        nn_condition=nn_condition,
        device=device,
    )

    detector = None
    if vision_backend == "dino":
        from dino.dino_detector import DinoBatchDetector

        detector = DinoBatchDetector(device=device)

    checkpoint_path = resolve_inference_checkpoint_path(args, original_cwd)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    nn_diffusion.model.load_state_dict(ckpt["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ckpt["ema_state_dict"])
    nn_diffusion.eval()

    act_min = ckpt["action_min"]
    act_scale = ckpt["action_scale"]

    ep_end_idx = test_dataset.episode_ends[0]
    ep_steps = ep_end_idx - args.horizon + 1
    gt_actions = test_dataset.z_encoder[0:ep_steps]

    all_selected_actions = []

    # ===== [CHANGED] =====
    # RTC state removed. EMA state added.
    prev_ema_action = None

    # per-step velocity comparison buffers
    all_gt_v = []
    all_gt_w = []
    all_ai_v_samples = []
    all_ai_w_samples = []

    zmq_ctx = zmq.Context()
    socket_viz = zmq_ctx.socket(zmq.PUSH)
    socket_viz.connect("tcp://localhost:5558")
    socket_path = zmq_ctx.socket(zmq.PUSH)
    socket_path.connect("tcp://localhost:5555")

    for step in range(ep_steps):
        batch = test_dataset[step]

        # DockingDataset already returns sparse uint8 image / lidar histories.
        image_room1 = batch["obs"]["image_room1"].unsqueeze(0).to(device).float().div_(255.0)
        image_room2 = batch["obs"]["image_room2"].unsqueeze(0).to(device).float().div_(255.0)
        velocity = batch["obs"]["velocity"].unsqueeze(0).to(device).float()

        # GT first-step velocity for this open-loop step
        gt_first = denormalize(batch["act"].squeeze(0).cpu().numpy(), act_scale, act_min)
        all_gt_v.append(float(gt_first[0, 0]))
        all_gt_w.append(float(gt_first[0, 1]))

        B, T_vis, C, H, W = image_room1.shape

        if vision_backend == "dino":
            image_room1_flat = image_room1.reshape(B * T_vis, C, H, W)
            image_room2_flat = image_room2.reshape(B * T_vis, C, H, W)
            with torch.no_grad():
                dino_feat1, sim_map1, _ = detector.get_heatmap(image_room1_flat)
                dino_feat2, sim_map2, _ = detector.get_heatmap(image_room2_flat)
            dino_feat1 = dino_feat1.view(B, T_vis, 196, 768)
            dino_feat2 = dino_feat2.view(B, T_vis, 196, 768)
            context = {
                "dino_feat1": dino_feat1.repeat(n_samples, 1, 1, 1),
                "dino_feat2": dino_feat2.repeat(n_samples, 1, 1, 1),
                "velocity": velocity.repeat(n_samples, 1, 1),
            }
        else:
            # raw_cnn: condition network encodes RGB inside forward()
            sim_map1 = sim_map2 = None
            context = {
                "raw_image1": image_room1.repeat(n_samples, 1, 1, 1, 1),
                "raw_image2": image_room2.repeat(n_samples, 1, 1, 1, 1),
                "velocity": velocity.repeat(n_samples, 1, 1),
            }

        if use_lidar_inf:
            lidar_map = batch["obs"]["lidar_map"].unsqueeze(0).to(device).float().div_(255.0)
            context["lidar_map"] = lidar_map.repeat(n_samples, 1, 1, 1, 1)

        h, w_ = int(args.image_height), int(args.image_width)
        if sim_map1 is not None and sim_map2 is not None:
            sm1, sm2 = sim_map1[0].cpu().numpy(), sim_map2[0].cpu().numpy()
        else:
            sm1, sm2 = np.zeros((h, w_), dtype=np.float32), np.zeros((h, w_), dtype=np.float32)
        socket_viz.send_pyobj((
            image_room1[0, -1].cpu().numpy(),
            image_room2[0, -1].cpu().numpy(),
            sm1, sm2))

        with torch.no_grad():
            prior = torch.randn(n_samples, horizon, 2, device=device)
            sample_out = nn_diffusion.sample(
                solver=args.solver,
                w_cfg=args.w_cfg,
                prior=prior,
                condition_cfg=context,
                n_samples=n_samples,
                sample_steps=args.inference_sampling_steps,
                use_ema=True,
            )
            out_tensor = sample_out[0] if isinstance(sample_out, tuple) else sample_out
            res = denormalize(out_tensor.cpu().numpy(), act_scale, act_min)

        # keep all sampled first-step velocities for mean/std visualization
        all_ai_v_samples.append(res[:, 0, 0].copy())
        all_ai_w_samples.append(res[:, 0, 1].copy())

        # ===== [CHANGED] =====
        # RTC selection removed.
        # If multiple samples exist, use mean trajectory before EMA.
        if n_samples == 1:
            current_action = res[0]
        else:
            current_action = res.mean(axis=0)

        selected_action = apply_trajectory_ema(
            current_action=current_action,
            prev_ema_action=prev_ema_action,
            ema_alpha=traj_ema_alpha,
        )
        prev_ema_action = selected_action.copy()

        local_predicted_path = reconstruct_pose_rk4(
            selected_action[:, 0],
            selected_action[:, 1],
            dt=dt)
        socket_path.send_pyobj(local_predicted_path)

        all_selected_actions.append(selected_action[0, :])

        if (step + 1) % 10 == 0 or (step + 1) == ep_steps:
            print(
                f"진행: {step + 1} / {ep_steps} steps | "
                f"traj_ema_alpha={traj_ema_alpha:.2f}"
            )

    ai_generated_actions = np.array(all_selected_actions)

    gt_path = reconstruct_pose_rk4(gt_actions[:, 0], gt_actions[:, 1], dt=dt)
    ai_path = reconstruct_pose_rk4(ai_generated_actions[:, 0], ai_generated_actions[:, 1], dt=dt)

    # NEW: per-step velocity comparison plot
    gt_v = np.asarray(all_gt_v, dtype=np.float32)
    gt_w = np.asarray(all_gt_w, dtype=np.float32)
    sel_v = ai_generated_actions[:, 0]
    sel_w = ai_generated_actions[:, 1]

    v_samples_np = np.asarray(all_ai_v_samples, dtype=np.float32)  # [T, N]
    w_samples_np = np.asarray(all_ai_w_samples, dtype=np.float32)  # [T, N]
    v_mean, v_std = v_samples_np.mean(axis=1), v_samples_np.std(axis=1)
    w_mean, w_std = w_samples_np.mean(axis=1), w_samples_np.std(axis=1)

    steps = np.arange(len(gt_v), dtype=np.int32)

    fig_vel, ax_vel = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax_vel[0].plot(steps, gt_v, 'k--', linewidth=1.5, label='GT linear vel')
    ax_vel[0].plot(steps, v_mean, color='tab:red', linewidth=1.0, label='Sample mean linear vel')
    ax_vel[0].fill_between(steps, v_mean - v_std, v_mean + v_std, color='tab:red', alpha=0.2, label='Sample std')
    ax_vel[0].plot(steps, sel_v, color='forestgreen', linewidth=1.2, label='EMA linear vel')
    ax_vel[0].set_ylabel("Linear velocity")
    ax_vel[0].set_title("Per-step Linear Velocity Comparison")
    ax_vel[0].grid(True, linestyle='--', alpha=0.5)
    ax_vel[0].legend(loc="upper right", framealpha=0.9)

    ax_vel[1].plot(steps, gt_w, 'k--', linewidth=1.5, label='GT angular vel')
    ax_vel[1].plot(steps, w_mean, color='tab:blue', linewidth=1.0, label='Sample mean angular vel')
    ax_vel[1].fill_between(steps, w_mean - w_std, w_mean + w_std, color='tab:blue', alpha=0.2, label='Sample std')
    ax_vel[1].plot(steps, sel_w, color='forestgreen', linewidth=1.2, label='EMA angular vel')
    ax_vel[1].set_ylabel("Angular velocity")
    ax_vel[1].set_xlabel("Step")
    ax_vel[1].set_title("Per-step Angular Velocity Comparison")
    ax_vel[1].grid(True, linestyle='--', alpha=0.5)
    ax_vel[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    save_vel_path = os.path.join(output_dir, "velocity_comparison_per_step_80000_test_ema_5_1.png")
    plt.savefig(save_vel_path, dpi=300)
    print(f"velocity comparison image saved to {save_vel_path}")
    plt.close(fig_vel)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(gt_path[:, 0], gt_path[:, 1], color="black", linestyle="--", linewidth=1.5, alpha=0.6, label="Ground Truth")
    ax.plot(ai_path[:, 0], ai_path[:, 1], color="forestgreen", linestyle="-", linewidth=1.0, alpha=0.8,
            label="AI EMA (room1 + room2 + velocity)")

    ax.scatter(gt_path[0, 0], gt_path[0, 1], c="limegreen", s=100, marker="o", zorder=5, label="Start Point")
    ax.scatter(gt_path[-1, 0], gt_path[-1, 1], c="black", s=80, marker="X", zorder=5, label="GT End Point")
    ax.scatter(ai_path[-1, 0], ai_path[-1, 1], c="forestgreen", s=80, marker="X", zorder=5, label="AI EMA End Point")

    # ===== [CHANGED] =====
    # RTC wording removed.
    ax.set_title(
        f"Room1 + Room2 + Velocity + EMA Open-Loop Prediction\n(Vision stride: {vision_stride}, Samples: {n_samples}, EMA alpha: {traj_ema_alpha:.2f})",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Step_80000_test_ema_5_1.png")
    plt.savefig(save_path, dpi=300)
    print(f"trajectory image saved to {save_path}")
    plt.close()

    socket_viz.close()
    socket_path.close()
    zmq_ctx.term()


if __name__ == "__main__":
    main()