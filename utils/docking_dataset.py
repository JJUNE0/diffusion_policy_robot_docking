from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DockingDataset(Dataset):
    """
    Dataset for the new multimodal setting:
      - motion input  : normalized encoder velocity history [T, 2]
      - vision input  : room1 / room2 image histories [T, 3, 240, 320]
      - target output : normalized future encoder trajectory [H, 2]

    Notes
    -----
    * We keep full image history length = obs_horizon (e.g. 30).
      Vision subsampling such as `::6 -> 5 frames` is intentionally left to
      train / inference code, so raw history remains available for debugging
      and future experiments.
    * Raw encoder history is also returned under key "encoder" for optional
      diagnostics or downstream real-time alignment checks.
    """

    def __init__(
        self,
        npz_path: str,
        train_npz_path: str,
        horizon: int = 16,
        obs_horizon: int = 30,
        dt: float = 0.0333,
        with_goal: bool = False,
        goal_mask_prob: float = 0.5,
        with_lidar: bool = False,
        with_aux: bool = False,
        vision_stride: int = 6,
        sparse_vision_uint8: bool = False,
    ):
        super().__init__()
        # RAM saver (training): return only the sparse vision frames as uint8
        # (~24x less RAM than the full 30-frame float32 history). Default False
        # keeps the old behavior for the inference scripts.
        self.vision_stride = vision_stride
        self.sparse_vision_uint8 = sparse_vision_uint8
        self.h5_path = npz_path
        self.train_h5_path = train_npz_path
        self.horizon = horizon
        self.obs_horizon = obs_horizon
        # Kept for interface compatibility with existing setup / inference code.
        self.dt = dt
        # Goal-feature conditioning (CLAUDE.md §2.3). Goal = the episode's docked
        # (final) frame; goal_mask_prob is the probability the goal is ACTIVE
        # (NoMaD-style: the rest are undirected). with_goal=False = old behavior.
        self.with_goal = with_goal
        self.goal_mask_prob = goal_mask_prob

        self.root = h5py.File(self.h5_path, "r")
        self.train_root = h5py.File(self.train_h5_path, "r")

        self.z_encoder_train = self.train_root["encoder"]
        self.z_encoder = self.root["encoder"]
        self.z_img1 = self.root["image_top"]
        self.z_img2 = self.root["image_bottom"]
        self.episode_ends = self.root["episode_ends"][:]

        # Raw LiDAR points (Option A) + ICP dock-pose aux labels. Auto-detected
        # from the h5 keys (built by utils/preprocessing.py --use_lidar/--with_labels).
        self.with_lidar = with_lidar and ("lidar_points" in self.root)
        self.with_aux = with_aux and ("dock_pose" in self.root)
        self.z_lidar_points = self.root["lidar_points"] if self.with_lidar else None
        self.z_lidar_npoints = self.root["lidar_npoints"] if self.with_lidar else None
        self.z_dock_pose = self.root["dock_pose"] if self.with_aux else None
        self.z_reliable = self.root["reliable"] if self.with_aux else None
        if self.with_aux:
            pose_all = self.z_dock_pose[:]
            rel_all = self.z_reliable[:].astype(bool)
            xy = pose_all[rel_all][:, :2]
            self.dock_xy_mean = xy.mean(axis=0).astype(np.float32)
            self.dock_xy_std = (xy.std(axis=0) + 1e-6).astype(np.float32)

        self.index_map = []
        self.ep_start_map = []
        self.ep_end_map = []

        start_idx = 0
        for end_idx in self.episode_ends:
            for t in range(start_idx, end_idx - self.horizon + 1):
                self.index_map.append(t)
                self.ep_start_map.append(start_idx)
                self.ep_end_map.append(end_idx)
            start_idx = end_idx

        self.action_min = self.z_encoder_train[:].min(axis=0).astype(np.float32)
        self.action_max = self.z_encoder_train[:].max(axis=0).astype(np.float32)
        self.action_scale = np.clip(self.action_max - self.action_min, a_min=1e-5, a_max=None).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index_map)

    def __del__(self):
        try:
            if hasattr(self, "root"):
                self.root.close()
        except Exception:
            pass
        try:
            if hasattr(self, "train_root"):
                self.train_root.close()
        except Exception:
            pass

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return 2.0 * (action - self.action_min) / self.action_scale - 1.0

    def _get_history(self, data, t: int, ep_start: int) -> np.ndarray:
        start_t = t - self.obs_horizon + 1
        if start_t < ep_start:
            valid_data = data[ep_start: t + 1]
            pad_len = self.obs_horizon - len(valid_data)
            padding = np.repeat(valid_data[:1], pad_len, axis=0)
            return np.concatenate([padding, valid_data], axis=0)
        return data[start_t: t + 1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.index_map[idx]
        ep_start = self.ep_start_map[idx]

        # Motion history
        encoder_seq_raw = self._get_history(self.z_encoder, t, ep_start).astype(np.float32)   # [T, 2]
        velocity_seq_norm = self.normalize_action(encoder_seq_raw).astype(np.float32)          # [T, 2]

        # Vision histories. sparse_vision_uint8 (training) returns only the sparse
        # frames as uint8 (~24x less RAM); else full float32 history (inference).
        if self.sparse_vision_uint8:
            image_room1 = np.ascontiguousarray(self._get_history(self.z_img1, t, ep_start)[::self.vision_stride])
            image_room2 = np.ascontiguousarray(self._get_history(self.z_img2, t, ep_start)[::self.vision_stride])
        else:
            image_room1 = self._get_history(self.z_img1, t, ep_start).astype(np.float32) / 255.0  # [T, 3, H, W]
            image_room2 = self._get_history(self.z_img2, t, ep_start).astype(np.float32) / 255.0  # [T, 3, H, W]

        # Future target trajectory in normalized encoder space
        act_traj = self.z_encoder[t: t + self.horizon].astype(np.float32)                      # [H, 2]
        act_traj_norm = self.normalize_action(act_traj).astype(np.float32)

        obs = {
            "encoder": torch.from_numpy(encoder_seq_raw).float(),
            "velocity": torch.from_numpy(velocity_seq_norm).float(),
            "image_room1": torch.from_numpy(image_room1),
            "image_room2": torch.from_numpy(image_room2),
        }

        if self.with_goal:
            # Goal = the episode's docked (final) frame, the ultimate sub-goal.
            gi = int(self.ep_end_map[idx]) - 1
            if self.sparse_vision_uint8:
                goal_room1 = np.ascontiguousarray(self.z_img1[gi])   # uint8 [3, H, W]
                goal_room2 = np.ascontiguousarray(self.z_img2[gi])
            else:
                goal_room1 = self.z_img1[gi].astype(np.float32) / 255.0
                goal_room2 = self.z_img2[gi].astype(np.float32) / 255.0
            goal_active = 1.0 if np.random.rand() < self.goal_mask_prob else 0.0
            obs["goal_image_room1"] = torch.from_numpy(goal_room1)
            obs["goal_image_room2"] = torch.from_numpy(goal_room2)
            obs["goal_mask"] = torch.tensor(goal_active, dtype=torch.float32)

        if self.with_lidar:
            # Current-frame raw points (robot frame), zero-padded to M.
            obs["lidar_points"] = torch.from_numpy(self.z_lidar_points[t].astype(np.float32))
            obs["lidar_npoints"] = torch.tensor(int(self.z_lidar_npoints[t]), dtype=torch.long)

        sample = {"obs": obs, "act": torch.from_numpy(act_traj_norm).float()}

        if self.with_aux:
            # ICP dock-pose label (teacher for the precision aux head). [x_n, y_n, sin, cos].
            p = self.z_dock_pose[t].astype(np.float32)
            valid = not np.any(np.isnan(p))
            rel = float(self.z_reliable[t]) if valid else 0.0
            p = np.nan_to_num(p)
            xy = (p[:2] - self.dock_xy_mean) / self.dock_xy_std
            sample["dock_target"] = torch.tensor(
                [xy[0], xy[1], np.sin(p[2]), np.cos(p[2])], dtype=torch.float32)
            sample["reliable"] = torch.tensor(rel, dtype=torch.float32)

        return sample


def denormalize(norm_action, act_scale, act_min):
    return (norm_action + 1.0) / 2.0 * act_scale + act_min
