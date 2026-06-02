import json
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
from utils.eval_metrics import (
    aggregate_episode_metrics,
    compute_episode_metrics,
    format_metric_textbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def apply_trajectory_ema(current_action, prev_ema_action=None, ema_alpha=0.3):
    if prev_ema_action is None:
        return current_action
    return ema_alpha * current_action + (1.0 - ema_alpha) * prev_ema_action


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------
def _episode_sample_ranges(episode_ends, horizon):
    """
    Yield ``(episode_idx, ep_start_frame, sample_offset, n_samples)`` for every
    episode in dataset order. ``sample_offset`` is the index into
    ``DockingDataset`` corresponding to the first sample of that episode.
    """
    sample_offset = 0
    prev_end = 0
    for ep_idx, ep_end in enumerate(episode_ends):
        ep_end = int(ep_end)
        ep_start = int(prev_end)
        n_samples = max(0, ep_end - ep_start - horizon + 1)
        if n_samples > 0:
            yield ep_idx, ep_start, sample_offset, n_samples
            sample_offset += n_samples
        prev_end = ep_end


def _save_velocity_plot(
    *,
    save_path,
    steps,
    gt_v,
    gt_w,
    sel_v,
    sel_w,
    v_mean,
    v_std,
    w_mean,
    w_std,
    title_prefix,
    metrics,
):
    fig_vel, ax_vel = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax_vel[0].plot(steps, gt_v, "k--", linewidth=1.5, label="GT linear vel")
    ax_vel[0].plot(steps, v_mean, color="tab:red", linewidth=1.0, label="Sample mean linear vel")
    ax_vel[0].fill_between(
        steps, v_mean - v_std, v_mean + v_std, color="tab:red", alpha=0.2, label="Sample std"
    )
    ax_vel[0].plot(steps, sel_v, color="forestgreen", linewidth=1.2, label="EMA linear vel")
    ax_vel[0].set_ylabel("Linear velocity")
    ax_vel[0].set_title(f"{title_prefix} - Per-step Linear Velocity")
    ax_vel[0].grid(True, linestyle="--", alpha=0.5)
    ax_vel[0].legend(loc="upper right", framealpha=0.9)
    ax_vel[0].text(
        0.99,
        0.02,
        format_metric_textbox(metrics, ["v_mae", "v_rmse", "v_jerk", "v_jerk_ratio"]),
        transform=ax_vel[0].transAxes,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=10,
    )

    ax_vel[1].plot(steps, gt_w, "k--", linewidth=1.5, label="GT angular vel")
    ax_vel[1].plot(steps, w_mean, color="tab:blue", linewidth=1.0, label="Sample mean angular vel")
    ax_vel[1].fill_between(
        steps, w_mean - w_std, w_mean + w_std, color="tab:blue", alpha=0.2, label="Sample std"
    )
    ax_vel[1].plot(steps, sel_w, color="forestgreen", linewidth=1.2, label="EMA angular vel")
    ax_vel[1].set_ylabel("Angular velocity")
    ax_vel[1].set_xlabel("Step")
    ax_vel[1].set_title(f"{title_prefix} - Per-step Angular Velocity")
    ax_vel[1].grid(True, linestyle="--", alpha=0.5)
    ax_vel[1].legend(loc="upper right", framealpha=0.9)
    ax_vel[1].text(
        0.99,
        0.02,
        format_metric_textbox(metrics, ["w_mae", "w_rmse", "w_jerk", "w_jerk_ratio"]),
        transform=ax_vel[1].transAxes,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig_vel)


def _save_trajectory_plot(
    *,
    save_path,
    gt_path,
    ai_path,
    title,
    metrics,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        gt_path[:, 0], gt_path[:, 1], color="black", linestyle="--", linewidth=1.5, alpha=0.6,
        label="Ground Truth",
    )
    ax.plot(
        ai_path[:, 0], ai_path[:, 1], color="forestgreen", linestyle="-", linewidth=1.0, alpha=0.85,
        label="AI EMA",
    )

    ax.scatter(gt_path[0, 0], gt_path[0, 1], c="limegreen", s=100, marker="o", zorder=5, label="Start")
    ax.scatter(gt_path[-1, 0], gt_path[-1, 1], c="black", s=80, marker="X", zorder=5, label="GT End")
    ax.scatter(ai_path[-1, 0], ai_path[-1, 1], c="forestgreen", s=80, marker="X", zorder=5, label="AI End")

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)

    all_xy = np.concatenate([gt_path[:, :2], ai_path[:, :2]], axis=0)
    margin = 0.5
    x_min, x_max = float(all_xy[:, 0].min()) - margin, float(all_xy[:, 0].max()) + margin
    y_min, y_max = float(all_xy[:, 1].min()) - margin, float(all_xy[:, 1].max()) + margin
    span = max(x_max - x_min, y_max - y_min)
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    ax.set_xlim([cx - span / 2, cx + span / 2])
    ax.set_ylim([cy - span / 2, cy + span / 2])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.text(
        0.02,
        0.98,
        format_metric_textbox(
            metrics,
            [
                "ade",
                "fde",
                "max_pos_err",
                "heading_mae_deg",
                "heading_final_err_deg",
                "success",
            ],
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="../configs/robot", config_name="smr", version_base=None)
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    original_cwd = get_original_cwd()
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    obs_h = args.get("obs_horizon", 30)
    horizon = args.horizon
    n_samples = args.get("num_samples", 10)
    print("Num samples", n_samples)

    traj_ema_alpha = float(args.get("traj_ema_alpha", 0.3))  # 1.0 = no ema
    vision_stride = args.get("vision_stride", 6)
    vision_h = len(range(0, obs_h, vision_stride))
    dt = float(args.get("dt", 0.0333))

    vision_backend = args.get("vision_backend", "raw_cnn")

    # Episode selection knobs.
    eval_episodes_cfg = args.get("eval_episodes", "all")
    success_threshold_m = float(args.get("success_threshold_m", 0.3))
    heading_threshold_deg = args.get("success_heading_threshold_deg", None)
    if heading_threshold_deg is not None:
        heading_threshold_deg = float(heading_threshold_deg)
    enable_zmq = bool(args.get("enable_zmq", True))

    test_dataset = DockingDataset(
        npz_path=args.eval_data_path,
        train_npz_path=args.train_data_path,
        horizon=args.horizon,
        obs_horizon=obs_h,
        vision_stride=vision_stride,
        dt=dt,
        with_goal=False,  # goal frame is supplied manually per episode below
    )

    use_goal_inf = bool(args.get("use_goal", True))
    goal_mode = str(args.get("goal_mode", "conditioned")).lower()
    if goal_mode not in ("conditioned", "undirected"):
        raise ValueError(f"goal_mode must be 'conditioned' or 'undirected', got {goal_mode!r}")

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
        use_goal=use_goal_inf,
        num_goal_latents=args.get("num_goal_latents", 16),
        goal_mask_prob=args.get("goal_mask_prob", 0.5),
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

    # ZMQ visualization is optional; if no client is listening this still runs
    # (the sockets just buffer / drop), but the user can disable to keep logs tidy.
    socket_viz = socket_path = zmq_ctx = None
    if enable_zmq:
        zmq_ctx = zmq.Context()
        socket_viz = zmq_ctx.socket(zmq.PUSH)
        socket_viz.connect("tcp://localhost:5558")
        socket_path = zmq_ctx.socket(zmq.PUSH)
        socket_path.connect("tcp://localhost:5555")

    episode_plan = list(_episode_sample_ranges(test_dataset.episode_ends, horizon))
    if isinstance(eval_episodes_cfg, str) and eval_episodes_cfg.lower() == "all":
        selected_eps = episode_plan
    else:
        if isinstance(eval_episodes_cfg, int):
            wanted = {int(eval_episodes_cfg)}
        else:
            wanted = {int(e) for e in eval_episodes_cfg}
        selected_eps = [e for e in episode_plan if e[0] in wanted]
    if not selected_eps:
        raise ValueError(
            f"No episodes selected from eval_episodes={eval_episodes_cfg}; "
            f"available episode indices: {[e[0] for e in episode_plan]}"
        )

    print(
        f"Evaluating {len(selected_eps)} episode(s) out of {len(episode_plan)}: "
        f"{[e[0] for e in selected_eps]}"
    )

    per_episode_metrics = []

    for ep_idx, ep_start_frame, sample_offset, n_samples_in_ep in selected_eps:
        print(
            f"\n=== Episode {ep_idx}: frame range "
            f"[{ep_start_frame}, {ep_start_frame + n_samples_in_ep + horizon - 1}], "
            f"{n_samples_in_ep} rollout steps ==="
        )

        gt_actions_ep = test_dataset.z_encoder[
            ep_start_frame: ep_start_frame + n_samples_in_ep
        ][:]

        all_selected_actions = []
        all_gt_v, all_gt_w = [], []
        all_ai_v_samples, all_ai_w_samples = [], []
        prev_ema_action = None

        # NoMaD-style goal: use the episode's final (docked) camera frame as the
        # goal image. Fetched once per episode and reused for every step.
        goal_ctx = {}
        if use_goal_inf:
            goal_frame_idx = int(test_dataset.episode_ends[ep_idx]) - 1
            g_img1 = (
                torch.from_numpy(np.asarray(test_dataset.z_img1[goal_frame_idx]))
                .unsqueeze(0).to(device).float().div_(255.0)
            )  # [1, 3, H, W]
            g_img2 = (
                torch.from_numpy(np.asarray(test_dataset.z_img2[goal_frame_idx]))
                .unsqueeze(0).to(device).float().div_(255.0)
            )
            if vision_backend == "dino":
                with torch.no_grad():
                    g_feat1, _, _ = detector.get_heatmap(g_img1)
                    g_feat2, _, _ = detector.get_heatmap(g_img2)
                goal_ctx["goal_feat1"] = g_feat1.view(1, 196, 768).repeat(n_samples, 1, 1)
                goal_ctx["goal_feat2"] = g_feat2.view(1, 196, 768).repeat(n_samples, 1, 1)
            else:
                goal_ctx["goal_image1"] = g_img1.repeat(n_samples, 1, 1, 1)
                goal_ctx["goal_image2"] = g_img2.repeat(n_samples, 1, 1, 1)
            # goal_mask: 1 = attend (goal-conditioned), 0 = block (undirected).
            goal_active = 1.0 if goal_mode == "conditioned" else 0.0
            goal_ctx["goal_mask"] = torch.full((n_samples,), goal_active, device=device)

        for step_in_ep in range(n_samples_in_ep):
            sample_idx = sample_offset + step_in_ep
            batch = test_dataset[sample_idx]

            image_room1 = (
                batch["obs"]["image_room1"].unsqueeze(0).to(device).float().div_(255.0)
            )
            image_room2 = (
                batch["obs"]["image_room2"].unsqueeze(0).to(device).float().div_(255.0)
            )
            velocity = batch["obs"]["velocity"].unsqueeze(0).to(device).float()

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
                sim_map1 = sim_map2 = None
                context = {
                    "raw_image1": image_room1.repeat(n_samples, 1, 1, 1, 1),
                    "raw_image2": image_room2.repeat(n_samples, 1, 1, 1, 1),
                    "velocity": velocity.repeat(n_samples, 1, 1),
                }

            if use_lidar_inf:
                lidar_map = (
                    batch["obs"]["lidar_map"].unsqueeze(0).to(device).float().div_(255.0)
                )
                context["lidar_map"] = lidar_map.repeat(n_samples, 1, 1, 1, 1)

            # Goal frame + goal mask (same docked goal for every step in episode).
            context.update(goal_ctx)

            if enable_zmq:
                h, w_ = int(args.image_height), int(args.image_width)
                if sim_map1 is not None and sim_map2 is not None:
                    sm1, sm2 = sim_map1[0].cpu().numpy(), sim_map2[0].cpu().numpy()
                else:
                    sm1 = np.zeros((h, w_), dtype=np.float32)
                    sm2 = np.zeros((h, w_), dtype=np.float32)
                socket_viz.send_pyobj(
                    (image_room1[0, -1].cpu().numpy(), image_room2[0, -1].cpu().numpy(), sm1, sm2)
                )

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

            all_ai_v_samples.append(res[:, 0, 0].copy())
            all_ai_w_samples.append(res[:, 0, 1].copy())

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

            if enable_zmq:
                local_predicted_path = reconstruct_pose_rk4(
                    selected_action[:, 0], selected_action[:, 1], dt=dt
                )
                socket_path.send_pyobj(local_predicted_path)

            all_selected_actions.append(selected_action[0, :])

            if (step_in_ep + 1) % 10 == 0 or (step_in_ep + 1) == n_samples_in_ep:
                print(
                    f"  ep{ep_idx} progress: {step_in_ep + 1} / {n_samples_in_ep} steps "
                    f"| traj_ema_alpha={traj_ema_alpha:.2f}"
                )

        # ----------- per-episode aggregation & plots -----------
        ai_actions = np.array(all_selected_actions)
        gt_v = np.asarray(all_gt_v, dtype=np.float32)
        gt_w = np.asarray(all_gt_w, dtype=np.float32)
        sel_v = ai_actions[:, 0]
        sel_w = ai_actions[:, 1]

        v_samples_np = np.asarray(all_ai_v_samples, dtype=np.float32)
        w_samples_np = np.asarray(all_ai_w_samples, dtype=np.float32)
        v_mean, v_std = v_samples_np.mean(axis=1), v_samples_np.std(axis=1)
        w_mean, w_std = w_samples_np.mean(axis=1), w_samples_np.std(axis=1)
        steps_axis = np.arange(len(gt_v), dtype=np.int32)

        gt_path = reconstruct_pose_rk4(gt_actions_ep[:, 0], gt_actions_ep[:, 1], dt=dt)
        ai_path = reconstruct_pose_rk4(ai_actions[:, 0], ai_actions[:, 1], dt=dt)

        ep_metrics = compute_episode_metrics(
            gt_path=gt_path,
            ai_path=ai_path,
            gt_v=gt_v,
            gt_w=gt_w,
            pred_v=sel_v,
            pred_w=sel_w,
            fde_threshold_m=success_threshold_m,
            heading_threshold_deg=heading_threshold_deg,
            sample_v_std=v_std,
            sample_w_std=w_std,
        )
        ep_metrics["episode_idx"] = int(ep_idx)
        ep_metrics["ep_steps"] = int(n_samples_in_ep)
        ep_metrics["n_samples"] = int(n_samples)

        print(
            f"  ep{ep_idx} metrics: ADE={ep_metrics['ade']:.3f} FDE={ep_metrics['fde']:.3f} "
            f"v_mae={ep_metrics['v_mae']:.3f} w_mae={ep_metrics['w_mae']:.3f} "
            f"success={ep_metrics['success']}"
        )

        title_prefix = (
            f"EMA ep{ep_idx} (n_samples={n_samples}, alpha={traj_ema_alpha:.2f})"
        )
        _save_velocity_plot(
            save_path=os.path.join(plot_dir, f"velocity_ep{ep_idx:02d}.png"),
            steps=steps_axis,
            gt_v=gt_v,
            gt_w=gt_w,
            sel_v=sel_v,
            sel_w=sel_w,
            v_mean=v_mean,
            v_std=v_std,
            w_mean=w_mean,
            w_std=w_std,
            title_prefix=title_prefix,
            metrics=ep_metrics,
        )
        _save_trajectory_plot(
            save_path=os.path.join(plot_dir, f"trajectory_ep{ep_idx:02d}.png"),
            gt_path=gt_path,
            ai_path=ai_path,
            title=(
                f"EMA Open-Loop Prediction - episode {ep_idx}\n"
                f"vision_stride={vision_stride}, n_samples={n_samples}, "
                f"alpha={traj_ema_alpha:.2f}"
            ),
            metrics=ep_metrics,
        )

        per_episode_metrics.append(ep_metrics)

    # --------------- aggregate metrics & save JSON ---------------
    aggregated = aggregate_episode_metrics(per_episode_metrics)
    summary = {
        "checkpoint_path": checkpoint_path,
        "eval_data_path": str(args.eval_data_path),
        "train_data_path": str(args.train_data_path),
        "vision_backend": vision_backend,
        "use_lidar": bool(use_lidar_inf),
        "vision_stride": int(vision_stride),
        "obs_horizon": int(obs_h),
        "horizon": int(horizon),
        "n_samples": int(n_samples),
        "traj_ema_alpha": float(traj_ema_alpha),
        "success_threshold_m": float(success_threshold_m),
        "success_heading_threshold_deg": (
            float(heading_threshold_deg) if heading_threshold_deg is not None else None
        ),
        "n_episodes_evaluated": len(per_episode_metrics),
        "aggregate": aggregated,
        "per_episode": per_episode_metrics,
        "inference_kind": "ema",
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    print("\n=== Aggregate ===")
    for k in (
        "ade_mean",
        "fde_mean",
        "v_mae_mean",
        "w_mae_mean",
        "v_jerk_mean",
        "w_jerk_mean",
        "success_rate",
        "n_episodes",
    ):
        if k in aggregated:
            v = aggregated[k]
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:>20s}: {v_str}")

    if enable_zmq:
        socket_viz.close()
        socket_path.close()
        zmq_ctx.term()


if __name__ == "__main__":
    main()
