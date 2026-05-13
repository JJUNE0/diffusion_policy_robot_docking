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
    ):
        super().__init__()
        self.h5_path = npz_path
        self.train_h5_path = train_npz_path
        self.horizon = horizon
        self.obs_horizon = obs_horizon
        # Kept for interface compatibility with existing setup / inference code.
        self.dt = dt

        self.root = h5py.File(self.h5_path, "r")
        self.train_root = h5py.File(self.train_h5_path, "r")

        self.z_encoder_train = self.train_root["encoder"]
        self.z_encoder = self.root["encoder"]
        self.z_img1 = self.root["image_top"]
        self.z_img2 = self.root["image_bottom"]
        self.episode_ends = self.root["episode_ends"][:]

        # Optional BEV lidar occupancy maps (created by preprocessing.py --use_lidar).
        if "lidar_map" in self.root:
            self.z_lidar = self.root["lidar_map"]
            attrs = self.z_lidar.attrs
            self.lidar_meta = {
                "range_m": float(attrs.get("range_m", 6.0)),
                "resolution": float(attrs.get("resolution", 0.05)),
                "channels": int(attrs.get("channels", self.z_lidar.shape[1])),
                "size": int(attrs.get("size", self.z_lidar.shape[-1])),
            }
        else:
            self.z_lidar = None
            self.lidar_meta = None

        self.index_map = []
        self.ep_start_map = []

        start_idx = 0
        for end_idx in self.episode_ends:
            for t in range(start_idx, end_idx - self.horizon + 1):
                self.index_map.append(t)
                self.ep_start_map.append(start_idx)
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

        # Vision histories (full 30-step history kept here; subsampling is done later)
        image_room1 = self._get_history(self.z_img1, t, ep_start).astype(np.float32) / 255.0  # [T, 3, H, W]
        image_room2 = self._get_history(self.z_img2, t, ep_start).astype(np.float32) / 255.0  # [T, 3, H, W]

        # Future target trajectory in normalized encoder space
        act_traj = self.z_encoder[t: t + self.horizon].astype(np.float32)                      # [H, 2]
        act_traj_norm = self.normalize_action(act_traj).astype(np.float32)

        obs = {
            "encoder": torch.from_numpy(encoder_seq_raw).float(),
            "velocity": torch.from_numpy(velocity_seq_norm).float(),
            "image_room1": torch.from_numpy(image_room1).float(),
            "image_room2": torch.from_numpy(image_room2).float(),
        }

        if self.z_lidar is not None:
            lidar_seq = self._get_history(self.z_lidar, t, ep_start).astype(np.float32) / 255.0  # [T, C, S, S]
            obs["lidar_map"] = torch.from_numpy(lidar_seq).float()

        return {
            "obs": obs,
            "act": torch.from_numpy(act_traj_norm).float(),
        }


def denormalize(norm_action, act_scale, act_min):
    return (norm_action + 1.0) / 2.0 * act_scale + act_min
