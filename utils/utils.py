import os
import glob
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import threading
import time
import torch
import uuid
import json
# import wandb
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

def reconstruct_pose_rk4(linear_vels, angular_vels, dt=0.1, initial_pose=(0.0, 0.0, 0.0)):
    n_steps = len(linear_vels)
    # Initialize trajectory: [x, y, theta]
    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = initial_pose

    # State transition function: f(q, u)
    def f(q, v, w):
        return np.array([
            v * np.cos(q[2]),
            v * np.sin(q[2]),
            w
        ])

    curr_q = np.array(initial_pose, dtype=float)

    for i in range(n_steps):
        v = linear_vels[i]
        w = angular_vels[i]

        # RK4 Slopes
        k1 = f(curr_q, v, w)
        k2 = f(curr_q + 0.5 * dt * k1, v, w)
        k3 = f(curr_q + 0.5 * dt * k2, v, w)
        k4 = f(curr_q + dt * k3, v, w)

        # Weighted average update
        curr_q += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Heading normalization to [-pi, pi]
        curr_q[2] = (curr_q[2] + np.pi) % (2 * np.pi) - np.pi

        trajectory[i + 1] = curr_q

    return trajectory

def plot_and_save_trajectories(
    gt_obs_np: np.ndarray,
    generated_traj_np: np.ndarray,
    obs_dim: int,
    ckpt_step: int,
    plot_save_path: str,
):
    """
    Ground Truth와 생성된 궤적을 비교하여 그래프로 그리고 저장합니다.
    """
    save_file = os.path.join(plot_save_path, f'trajectory_plot_step_{ckpt_step}.pdf')

    with PdfPages(save_file) as pdf:
        N_PLOTS = min(10, gt_obs_np.shape[0])
        DIMS_TO_PLOT = range(obs_dim)

        for i in range(N_PLOTS):
            n_dims = len(DIMS_TO_PLOT)
            n_cols = 3
            n_rows = math.ceil(n_dims / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2.5), sharex=True)
            if n_rows == 1: axes = axes.reshape(1, -1) # 행이 1개일 때 shape 맞추기
                
            fig.suptitle(f'Trajectory Comparison (Step: {ckpt_step}, Sample: {i+1})', fontsize=16)

            for j, dim_idx in enumerate(DIMS_TO_PLOT):
                row, col = j // n_cols, j % n_cols
                ax = axes[row, col]
                
                ax.plot(gt_obs_np[i, :, dim_idx], label='Ground Truth', color='royalblue')
                ax.plot(generated_traj_np[i, :, dim_idx], label='Generated', linestyle='--', color='tomato')
                ax.set_title(f'Dimension {dim_idx}')

                if col == 0:
                    ax.set_ylabel('Value')
                
                if j == 0:
                    ax.legend()
            
            for col_idx in range(n_cols):
                if (n_rows - 1) * n_cols + col_idx < n_dims: # 해당 축이 실제로 사용될 때만
                    axes[n_rows - 1, col_idx].set_xlabel('Time Step')

            for j in range(n_dims, n_rows * n_cols):
                row, col = j // n_cols, j % n_cols
                axes[row, col].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"plot saved to {save_file}")


def resume_from_checkpoint(resume_path, diffusions, lr_schedulers):
    start_step = 0
    if resume_path is not None and os.path.exists(resume_path):
        print("======================================================")
        print(f"Resuming training from checkpoint: {resume_path}")

        ckpt_path = os.path.join(resume_path, f'diffusion_ckpt_latest.pt')
        if os.path.exists(ckpt_path):
            diffusions.load(ckpt_path)
            print(f"Loaded diffusion model from {ckpt_path}")

        latest_step = -1
        ckpt_files = glob.glob(os.path.join(resume_path, 'diffusion_ckpt_*.pt'))
        for f in ckpt_files:
            match = re.search(r'_ckpt_(\d+)\.pt$', os.path.basename(f))
            if match:
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
        
        if latest_step != -1:
            start_step = latest_step
            print(f"Resuming from gradient step: {start_step}")
        else:
            print("Warning: Could not determine resume step. Starting from step 0.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for scheduler in lr_schedulers:
                for _ in range(start_step):
                    scheduler.step()
    
    else:
        print("======================================================")
        print("Starting training from scratch...")

    return start_step

def make_dir(dir_path):
    """Create directory if it does not already exist; return Path."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p

class Logger:
    """Primary logger object. Logs in wandb."""
    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / 'models')
        self._cfg = cfg
        #
        # wandb.init(
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     project=getattr(cfg, "project", "cleandiffuser"),
        #     group=getattr(cfg, "group", None),
        #     name=getattr(cfg, "exp_name", None),
        #     id=str(uuid.uuid4()),
        #     mode=getattr(cfg, "wandb_mode", "online"),
        #     dir=str(self._log_dir),
        #     resume="allow",
        # )
        # self._wandb = wandb

    def log(self, d, category):
        """
        d: {'step': int, 'metric1': float, ...}
        category: 'train' or 'inference'
        """
        assert category in ['train', 'inference']
        assert 'step' in d
        step = d['step']
        # 안전한 프린트 포매팅 (숫자만 고정소수점, 나머지는 그대로)
        printable = []
        for k, v in d.items():
            if k == 'step':
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                printable.append(f"{k} {float(v):.4f}")
            else:
                printable.append(f"{k} {v}")
        # print(f"[{step}] " + " | ".join(printable))

        # jsonl 저장
        with (self._log_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                                 for k, v in d.items() if k != 'step'}}) + "\n")

        # wandb 로깅 (카테고리 prefix)
        _d = {}
        for k, v in d.items():
            if k == 'step':
                continue
            _d[f"{category}/{k}"] = v
        # self._wandb.log(_d, step=step)

    def save_agent(self, agent=None, identifier='final'):
        if agent:
            fp = self._model_dir / f'model_{str(identifier)}.pt'
            agent.save(str(fp))
            print(f"model_{str(identifier)} saved")

    def finish(self, agent=None):
        try:
            if agent is not None:
                self.save_agent(agent)
        except Exception as e:
            print(f"Failed to save model: {e}")
        # if self._wandb:
        #     self._wandb.finish()
        print("Finished logging")
