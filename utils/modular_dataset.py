"""Config-driven multimodal dataset.

Reads an arbitrary set of modalities from an HDF5 file according to the SAME
`sensors` spec that drives ModularSensorFusionCondition. Each sensor declares:

    source         : HDF5 dataset key (e.g. "encoder", "lidar_points")
    mode           : "history" (length-`horizon` window ending at t)
                   | "current" (the single row at t)
    horizon        : window length for mode=history
    stride         : (optional) subsample the window by ::stride
    channels       : (optional) list of column indices to keep (e.g. IMU planar
                     subset [2, 3, 4] = gyro_z, accl_x, accl_y). Selected before
                     normalization; the sensor's `dim` must match len(channels).
    normalize      : (optional) "action" -> reuse the encoder action normalizer;
                     "zscore" -> per-channel (mean/std over the source) whitening.
    npoints_source : (mode=current, pointcloud) key with the per-row valid count
    head           : (optional) "aux_pose" -> the dataset also emits ICP dock-pose
                     targets ("dock_target" [x_n, y_n, sin, cos] + "reliable"
                     mask) so the aux precision head can be trained/ablated.

Returns {"obs": {name: tensor, ...}, "act": future_traj}. Ablations add/remove
sensors purely by editing the spec; a brand-new modality (e.g. IMU) is: put its
rows under a new h5 key and add one spec entry.

NOTE: `dino_image` sensors expect PRECOMPUTED DINO features stored in the h5
under `source`. Encoding raw pixels with the DINO backbone is a train-loop
concern (kept out of the dataset), same split as the existing pipeline.
"""
from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ModularDockingDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        sensors: Dict[str, dict],
        horizon: int = 60,
        obs_horizon: int = 30,
        action_key: str = "encoder",
        train_h5_path: str = None,
        dock_pose_key: str = "dock_pose",
        reliable_key: str = "reliable",
    ):
        super().__init__()
        self.h5_path = h5_path
        self.sensors = {name: dict(spec) for name, spec in sensors.items()}
        self.horizon = horizon
        self.obs_horizon = obs_horizon
        self.action_key = action_key
        self.train_h5_path = train_h5_path or h5_path
        self.dock_pose_key = dock_pose_key
        self.reliable_key = reliable_key

        # Aux precision head is on iff any sensor declares `head: aux_pose`. Then
        # the dataset must emit dock-pose targets (from the ICP labels in the h5).
        self.aux_sensor = next(
            (n for n, s in self.sensors.items() if s.get("head") == "aux_pose"), None)
        self.with_aux = self.aux_sensor is not None

        # Per-sensor z-score stats (computed once over the source), keyed by name.
        self.norm_stats = {}

        with h5py.File(self.h5_path, "r") as f:
            self.episode_ends = f["episode_ends"][:]
            for name, spec in self.sensors.items():
                if spec["source"] not in f:
                    raise KeyError(
                        f"sensor '{name}': source '{spec['source']}' not in {self.h5_path}. "
                        f"Available: {list(f.keys())}")
                if spec.get("normalize") == "zscore":
                    col = self._select_channels(f[spec["source"]][:], spec).astype(np.float32)
                    mean = col.reshape(-1, col.shape[-1]).mean(axis=0)
                    std = np.clip(col.reshape(-1, col.shape[-1]).std(axis=0), 1e-6, None)
                    self.norm_stats[name] = (mean.astype(np.float32), std.astype(np.float32))
            if self.with_aux:
                for key in (self.dock_pose_key, self.reliable_key):
                    if key not in f:
                        raise KeyError(
                            f"sensor '{self.aux_sensor}' has head=aux_pose but '{key}' "
                            f"is not in {self.h5_path}. Available: {list(f.keys())}")
                pose_all = f[self.dock_pose_key][:]
                rel_all = f[self.reliable_key][:].astype(bool)
                # Guard: only use rows that are BOTH reliable and non-NaN, so a
                # single NaN pose can't poison the whole mean/std (and thus every
                # emitted target) with NaN.
                valid = rel_all & ~np.isnan(pose_all).any(axis=1)
                xy = pose_all[valid][:, :2]
                self.dock_xy_mean = xy.mean(axis=0).astype(np.float32)
                self.dock_xy_std = (xy.std(axis=0) + 1e-6).astype(np.float32)

        with h5py.File(self.train_h5_path, "r") as f:
            act = f[self.action_key][:]
        self.action_min = act.min(axis=0).astype(np.float32)
        self.action_max = act.max(axis=0).astype(np.float32)
        self.action_scale = np.clip(self.action_max - self.action_min, 1e-5, None).astype(np.float32)

        self.index_map, self.ep_start_map = [], []
        start = 0
        for end in self.episode_ends:
            for t in range(start, end - self.horizon + 1):
                self.index_map.append(t)
                self.ep_start_map.append(start)
            start = end
        self.root = None

    def _ensure_open(self):
        if self.root is None:
            self.root = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.index_map)

    def __del__(self):
        try:
            if getattr(self, "root", None) is not None:
                self.root.close()
        except Exception:
            pass

    def normalize_action(self, a):
        return 2.0 * (a - self.action_min) / self.action_scale - 1.0

    @staticmethod
    def _select_channels(arr, spec):
        """Keep only spec['channels'] columns (last axis). No-op if unset."""
        channels = spec.get("channels")
        if channels is None:
            return arr
        return arr[..., list(channels)]

    def _history(self, data, t, ep_start):
        start_t = t - self.obs_horizon + 1
        if start_t < ep_start:
            valid = data[ep_start:t + 1]
            pad = np.repeat(valid[:1], self.obs_horizon - len(valid), axis=0)
            return np.concatenate([pad, valid], axis=0)
        return data[start_t:t + 1]

    def __getitem__(self, idx):
        self._ensure_open()
        t = self.index_map[idx]
        ep_start = self.ep_start_map[idx]
        obs = {}
        for name, spec in self.sensors.items():
            ds = self.root[spec["source"]]
            mode = spec.get("mode", "history")
            if mode == "history":
                arr = self._history(ds, t, ep_start)
                stride = int(spec.get("stride", 1))
                if stride > 1:
                    arr = arr[::stride]
                arr = self._select_channels(arr, spec)
                arr = np.ascontiguousarray(arr).astype(np.float32)
                norm = spec.get("normalize")
                if norm == "action":
                    arr = self.normalize_action(arr).astype(np.float32)
                elif norm == "zscore":
                    mean, std = self.norm_stats[name]
                    arr = ((arr - mean) / std).astype(np.float32)
                obs[name] = torch.from_numpy(arr)
            elif mode == "current":
                arr = self._select_channels(np.ascontiguousarray(ds[t]), spec).astype(np.float32)
                obs[name] = torch.from_numpy(np.ascontiguousarray(arr))
                nsrc = spec.get("npoints_source")
                if nsrc is not None:
                    obs[f"{name}_npoints"] = torch.tensor(int(self.root[nsrc][t]), dtype=torch.long)
            else:
                raise ValueError(f"sensor '{name}': unknown mode '{mode}'.")

        act = self.root[self.action_key][t:t + self.horizon].astype(np.float32)
        act = self.normalize_action(act).astype(np.float32)
        sample = {"obs": obs, "act": torch.from_numpy(act)}

        if self.with_aux:
            # ICP dock-pose label (teacher for the precision aux head).
            p = self.root[self.dock_pose_key][t].astype(np.float32)
            valid = not np.any(np.isnan(p))
            rel = float(self.root[self.reliable_key][t]) if valid else 0.0
            p = np.nan_to_num(p)
            xy = (p[:2] - self.dock_xy_mean) / self.dock_xy_std
            sample["dock_target"] = torch.tensor(
                [xy[0], xy[1], np.sin(p[2]), np.cos(p[2])], dtype=torch.float32)
            sample["reliable"] = torch.tensor(rel, dtype=torch.float32)

        return sample
