"""Config-driven multimodal dataset.

Reads an arbitrary set of modalities from an HDF5 file according to the SAME
`sensors` spec that drives ModularSensorFusionCondition. Each sensor declares:

    source         : HDF5 dataset key (e.g. "encoder", "lidar_points")
    file           : (optional) path to an EXTERNAL h5 holding `source`. Lets a
                     sensor read from a sidecar file, e.g. the legacy DINO cache
                     (after_0328_train_dino_bottom.h5) next to after_0328_train.h5.
                     Row indices must align 1:1 with the main h5.
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

# Percentile used by action_norm="robust" (see _action_stats).
ACTION_ROBUST_PCT = 1.0


def _action_stats(act, action_norm, pct=ACTION_ROBUST_PCT):
    """Affine action normalizer bounds -> (min, max), mapped to [-1, 1].

    "minmax" (legacy) lets the extremes set the scale, and on this dataset the
    extremes are outliers: vx spans +-0.27 m/s while p1/p99 are only
    -0.075/+0.084, so the scale is 0.563 m/s. The docking end-game runs at
    2-15 mm/s, which lands at 0.007-0.053 of a [-1, 1] output range -- under 3%,
    i.e. below the noise a diffusion sampler puts on its own output. That is a
    direct contributor to the terminal stall measured on 2026-07-23
    (test/terminal_metric.py).

    "robust" sets the bounds at p1/p99 instead, shrinking the vx scale ~3.5x and
    the wz scale ~3.4x, so the same physical command occupies several times more
    of the output range. The affine CONTRACT is unchanged --
    `(x + 1) / 2 * action_scale + action_min` still inverts it -- so every
    consumer (inference_ema_v2, rollout_core, the eval scripts, the robot node)
    keeps working with no edit; only the numbers in the checkpoint change.
    The 1% tail beyond the bounds is clipped by normalize_action; those frames
    are faster than anything the docking policy needs to reproduce.
    """
    if action_norm == "robust":
        lo = np.percentile(act, pct, axis=0)
        hi = np.percentile(act, 100.0 - pct, axis=0)
        # keep 0 representable: an asymmetric band would bias "stop"
        m = np.maximum(np.abs(lo), np.abs(hi))
        return -m, m
    if action_norm == "minmax":
        return act.min(axis=0), act.max(axis=0)
    raise ValueError(f"unknown action_norm '{action_norm}' (expected minmax|robust)")


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
        action_norm: str = "minmax",
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
        self.action_norm = action_norm

        # Aux precision head is on iff any sensor declares `head: aux_pose`. Then
        # the dataset must emit dock-pose targets (from the ICP labels in the h5).
        self.aux_sensor = next(
            (n for n, s in self.sensors.items() if s.get("head") == "aux_pose"), None)
        self.with_aux = self.aux_sensor is not None

        # Per-sensor z-score stats (computed once over the source), keyed by name.
        self.norm_stats = {}

        # Sensors may read from a sidecar h5 via `file:` (e.g. a DINO feature
        # cache); everything else (episode_ends, action, aux labels) stays in
        # the main h5.
        self._file_paths = sorted({self.h5_path} | {
            self._sensor_path(spec) for spec in self.sensors.values()})

        files = {p: h5py.File(p, "r") for p in self._file_paths}
        try:
            f = files[self.h5_path]
            self.episode_ends = f["episode_ends"][:]
            n_rows = int(self.episode_ends[-1])
            for name, spec in self.sensors.items():
                sf = files[self._sensor_path(spec)]
                if spec["source"] not in sf:
                    raise KeyError(
                        f"sensor '{name}': source '{spec['source']}' not in {sf.filename}. "
                        f"Available: {list(sf.keys())}")
                if sf[spec["source"]].shape[0] != n_rows:
                    raise ValueError(
                        f"sensor '{name}': '{spec['source']}' in {sf.filename} has "
                        f"{sf[spec['source']].shape[0]} rows but the main h5 has {n_rows}; "
                        f"external files must be row-aligned with the main h5.")
                if spec.get("normalize") == "zscore":
                    col = self._select_channels(sf[spec["source"]][:], spec).astype(np.float32)
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
        finally:
            for h in files.values():
                h.close()

        with h5py.File(self.train_h5_path, "r") as f:
            act = f[self.action_key][:]
        a_min, a_max = _action_stats(act, self.action_norm)
        self.action_min = np.asarray(a_min, np.float32)
        self.action_max = np.asarray(a_max, np.float32)
        self.action_scale = np.clip(self.action_max - self.action_min, 1e-5, None).astype(np.float32)

        self.index_map, self.ep_start_map, self.ep_end_map = [], [], []
        start = 0
        for end in self.episode_ends:
            for t in range(start, end - self.horizon + 1):
                self.index_map.append(t)
                self.ep_start_map.append(start)
                self.ep_end_map.append(int(end))
            start = end
        self.root = None
        self.roots = None

    def _sensor_path(self, spec):
        return spec.get("file") or self.h5_path

    def _ensure_open(self):
        if self.root is None:
            self.roots = {p: h5py.File(p, "r") for p in self._file_paths}
            self.root = self.roots[self.h5_path]

    def __len__(self):
        return len(self.index_map)

    def __del__(self):
        try:
            for h in (getattr(self, "roots", None) or {}).values():
                h.close()
        except Exception:
            pass

    def normalize_action(self, a):
        n = 2.0 * (a - self.action_min) / self.action_scale - 1.0
        # robust bounds sit at p1/p99, so the 1% tail lands outside [-1, 1];
        # clip it rather than let the model spend output range on outliers.
        return np.clip(n, -1.0, 1.0) if self.action_norm == "robust" else n

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

    def _history_valid_mask(self, t, ep_start):
        """bool[obs_horizon], True = real frame, False = repeated episode-start
        padding (same fill rule as _history: pad = repeat(earliest valid, ...),
        prepended). 2026-07-25: acceptance criterion #5 -- padding must reach
        attention, not be silently treated as extra real observations."""
        start_t = t - self.obs_horizon + 1
        valid_len = min(self.obs_horizon, t - ep_start + 1)
        pad_len = self.obs_horizon - valid_len
        mask = np.ones(self.obs_horizon, dtype=bool)
        mask[:pad_len] = False
        return mask

    def __getitem__(self, idx):
        self._ensure_open()
        t = self.index_map[idx]
        ep_start = self.ep_start_map[idx]
        ep_end = self.ep_end_map[idx]
        goal_row = ep_end - 1  # episode's static goal/docked frame (last row),
        # same convention as utils/docking_dataset.py and endgame/se2.py.
        obs = {}
        for name, spec in self.sensors.items():
            sensor_root = self.roots[self._sensor_path(spec)]
            ds = sensor_root[spec["source"]]
            mode = spec.get("mode", "history")
            if mode == "history":
                arr = self._history(ds, t, ep_start)
                valid_mask = self._history_valid_mask(t, ep_start)
                stride = int(spec.get("stride", 1))
                if stride > 1:
                    arr = arr[::stride]
                    valid_mask = valid_mask[::stride]
                arr = self._select_channels(arr, spec)
                arr = np.ascontiguousarray(arr).astype(np.float32)
                norm = spec.get("normalize")
                if norm == "action":
                    arr = self.normalize_action(arr).astype(np.float32)
                elif norm == "zscore":
                    mean, std = self.norm_stats[name]
                    arr = ((arr - mean) / std).astype(np.float32)
                obs[name] = torch.from_numpy(arr)
                obs[f"{name}_valid_mask"] = torch.from_numpy(np.ascontiguousarray(valid_mask))
            elif mode == "current":
                arr = self._select_channels(np.ascontiguousarray(ds[t]), spec).astype(np.float32)
                obs[name] = torch.from_numpy(np.ascontiguousarray(arr))
                nsrc = spec.get("npoints_source")
                if nsrc is not None:
                    obs[f"{name}_npoints"] = torch.tensor(int(sensor_root[nsrc][t]), dtype=torch.long)
            elif mode == "goal":
                # STATIC episode goal frame (docs/0725_reloc3r_test/reloc3r/
                # reloc3r_0725.md: goal_image / geometry sensors read the
                # episode's docked frame, not the current timestep t).
                arr = self._select_channels(np.ascontiguousarray(ds[goal_row]), spec).astype(np.float32)
                obs[name] = torch.from_numpy(np.ascontiguousarray(arr))
                nsrc = spec.get("npoints_source")
                if nsrc is not None:
                    obs[f"{name}_npoints"] = torch.tensor(int(sensor_root[nsrc][goal_row]), dtype=torch.long)
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
