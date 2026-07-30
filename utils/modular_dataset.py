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
                   | "goal_pool_pair" (history ReLoc3R encoder tokens paired
                     with one randomly sampled, camera-synchronized goal from
                     a compiled goal pool)
    horizon        : window length for mode=history
    stride         : (optional) subsample the window by ::stride
    channels       : (optional) list of column indices to keep (e.g. IMU planar
                     subset [2, 3, 4] = gyro_z, accl_x, accl_y). Selected before
                     normalization; the sensor's `dim` must match len(channels).
    normalize      : (optional) "action" -> reuse the encoder action normalizer;
                     "zscore" -> per-channel (mean/std over the source) whitening.
    npoints_source : (mode=current, pointcloud) key with the per-row valid count
    goal_pool_file : (mode=goal_pool_pair) compiled HDF5 snapshot
    goal_source    : ReLoc3R goal encoder-feature key in goal_pool_file
    goal_pool_group: sensors sharing this value sample the SAME goal row
    goal_pool_filter: optional metadata filter, e.g.
                     {split: [train], variant: [original, qr_removed]}
    goal_pool_match: optional episode-aware sampling policy. By default, goals
                     are sampled from the pool rows whose `goal_field` equals
                     the current episode's `episode_field`. A source listed in
                     `cross_map` instead samples the mapped goal group with
                     probability `cross_probability`, e.g. 10% 4F -> 5F.
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
import os
import threading
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
        self._goal_pool_groups = {}

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
        self._file_paths = sorted(
            {self.h5_path}
            | {self._sensor_path(spec) for spec in self.sensors.values()}
            | {
                spec["goal_pool_file"]
                for spec in self.sensors.values()
                if spec.get("mode") == "goal_pool_pair"
            }
        )

        self._ram_cache = {}
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
                if spec.get("mode") == "goal_pool_pair":
                    self._register_goal_pool_group(name, spec, files)
                if spec.get("normalize") == "zscore":
                    col = self._select_channels(sf[spec["source"]][:], spec).astype(np.float32)
                    mean = col.reshape(-1, col.shape[-1]).mean(axis=0)
                    std = np.clip(col.reshape(-1, col.shape[-1]).std(axis=0), 1e-6, None)
                    self.norm_stats[name] = (mean.astype(np.float32), std.astype(np.float32))
                if spec.get("cache_in_ram", False):
                    # One-time BULK sequential read straight into RAM, so every
                    # later __getitem__ indexes a plain numpy array instead of
                    # re-reading this row from the network mount. Random
                    # per-row access over HDF5 (chunks=(1,...) here, see
                    # scripts/precompute_reloc3r_dec_features.py) is
                    # latency-bound on this mount -- measured ~220ms/sample
                    # for this sensor's history window, vs. a full-array
                    # sequential read running at ~180MB/s. Threading the
                    # per-row reads instead (tried first) made things WORSE:
                    # h5py serializes concurrent calls internally, so N
                    # threads just add contention, not throughput.
                    ds = sf[spec["source"]]
                    print(f"[ModularDockingDataset] caching '{name}' "
                          f"({ds.shape}, {ds.dtype}, "
                          f"{ds.nbytes / 1e9:.1f} GB) into RAM ...", flush=True)
                    self._ram_cache[name] = ds[:]
                    print(f"[ModularDockingDataset] '{name}' cached.", flush=True)
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
        # Per-THREAD (not per-instance) file handles: __getitem__ is called
        # concurrently from a thread pool (cleandiffuser.utils.parallel_batch_loader)
        # to hide network-mount read latency (measured ~20x throughput at 4
        # concurrent readers vs. 1, see git history), and h5py.File handles
        # are not safe to share across threads -- each thread lazily opens
        # its OWN handles on first access via `_ensure_open`.
        self._tls = threading.local()

    @staticmethod
    def _string_column(h5, key):
        if key not in h5:
            raise KeyError(
                f"compiled goal pool {h5.filename} is missing metadata '{key}'. "
                f"Available: {list(h5.keys())}"
            )
        return np.asarray(h5[key].asstr()[:], dtype=object)

    def _register_goal_pool_group(self, name, spec, files):
        pool_path = spec.get("goal_pool_file")
        goal_source = spec.get("goal_source")
        if not pool_path or not goal_source:
            raise ValueError(
                f"sensor '{name}' mode=goal_pool_pair requires goal_pool_file and goal_source"
            )
        pool = files[pool_path]
        if goal_source not in pool:
            raise KeyError(
                f"sensor '{name}': goal_source '{goal_source}' not in {pool.filename}. "
                f"Available: {list(pool.keys())}"
            )
        if "goal_id" not in pool:
            raise KeyError(f"{pool.filename}: missing goal_id")
        n_goals = int(pool["goal_id"].shape[0])
        if int(pool[goal_source].shape[0]) != n_goals:
            raise ValueError(
                f"{pool.filename}: {goal_source} has {pool[goal_source].shape[0]} rows, "
                f"goal_id has {n_goals}"
            )
        selected = np.ones(n_goals, dtype=bool)
        filters = spec.get("goal_pool_filter") or {}
        for field, allowed in filters.items():
            allowed = allowed if isinstance(allowed, (list, tuple)) else [allowed]
            values = self._string_column(pool, field)
            selected &= np.isin(values, np.asarray([str(v) for v in allowed], dtype=object))
        rows = np.flatnonzero(selected).astype(np.int64)
        if len(rows) == 0:
            raise ValueError(
                f"sensor '{name}': goal_pool_filter={filters} selected zero goals "
                f"from {pool.filename}"
            )
        sampling = str(spec.get("goal_sampling", "random"))
        variant_weights = {
            str(variant): float(weight)
            for variant, weight in dict(
                spec.get("goal_variant_weights") or {}
            ).items()
        }
        if any(weight < 0 for weight in variant_weights.values()):
            raise ValueError("goal_variant_weights values must be non-negative")
        if variant_weights and not any(
            weight > 0 for weight in variant_weights.values()
        ):
            raise ValueError("goal_variant_weights must contain a positive weight")
        if variant_weights and sampling != "random":
            raise ValueError("goal_variant_weights requires goal_sampling=random")
        goal_variants = self._string_column(pool, "variant")
        selected_variants = set(map(str, goal_variants[rows]))
        unknown_variants = sorted(set(variant_weights) - selected_variants)
        if unknown_variants:
            raise ValueError(
                f"sensor {name!r}: goal_variant_weights references variants "
                f"not present in the filtered pool: {unknown_variants}"
            )
        match = spec.get("goal_pool_match") or {}
        match_info = None
        if match:
            episode_field = str(match.get("episode_field", ""))
            goal_field = str(match.get("goal_field", ""))
            if not episode_field or not goal_field:
                raise ValueError(
                    f"sensor '{name}': goal_pool_match requires episode_field "
                    "and goal_field"
                )
            episode_values = self._string_column(
                files[self.h5_path], episode_field
            )
            if len(episode_values) != len(self.episode_ends):
                raise ValueError(
                    f"sensor '{name}': episode metadata '{episode_field}' has "
                    f"{len(episode_values)} rows but episode_ends has "
                    f"{len(self.episode_ends)}"
                )
            goal_values = self._string_column(pool, goal_field)
            cross_probability = float(match.get("cross_probability", 0.0))
            if not 0.0 <= cross_probability <= 1.0:
                raise ValueError(
                    f"sensor '{name}': goal_pool_match.cross_probability must "
                    f"be in [0, 1], got {cross_probability}"
                )
            cross_map = {
                str(source): str(target)
                for source, target in dict(match.get("cross_map") or {}).items()
            }
            rows_by_value = {
                str(value): rows[goal_values[rows] == value]
                for value in np.unique(goal_values[rows])
            }
            required_values = set(map(str, episode_values))
            episode_value_set = tuple(required_values)
            required_values.update(
                cross_map[source]
                for source in episode_value_set
                if source in cross_map and cross_probability > 0.0
            )
            missing_values = sorted(
                value for value in required_values
                if value not in rows_by_value or len(rows_by_value[value]) == 0
            )
            if missing_values:
                raise ValueError(
                    f"sensor '{name}': goal_pool_match needs goal_field="
                    f"'{goal_field}' values {missing_values}, but the filtered "
                    f"pool has {sorted(rows_by_value)}"
                )
            match_info = {
                "episode_field": episode_field,
                "goal_field": goal_field,
                "episode_values": episode_values,
                "rows_by_value": rows_by_value,
                "cross_probability": cross_probability,
                "cross_map": cross_map,
            }

        group = str(spec.get("goal_pool_group", "default"))
        goal_ids = tuple(self._string_column(pool, "goal_id")[rows].tolist())
        match_signature = None
        if match_info is not None:
            match_signature = (
                match_info["episode_field"],
                match_info["goal_field"],
                match_info["cross_probability"],
                tuple(sorted(match_info["cross_map"].items())),
            )
        signature = (
            os.path.abspath(pool_path),
            tuple(rows.tolist()),
            goal_ids,
            sampling,
            tuple(sorted(variant_weights.items())),
            match_signature,
        )
        previous = self._goal_pool_groups.get(group)
        if previous is not None and previous["signature"] != signature:
            raise ValueError(
                f"goal_pool_group '{group}' must use the same compiled pool/filter "
                f"and sampling policy for every camera; sensor '{name}' does not match"
            )
        self._goal_pool_groups[group] = {
            "signature": signature,
            "rows": rows,
            "sampling": sampling,
            "goal_variants": goal_variants,
            "variant_weights": variant_weights,
            "match": match_info,
        }

    def _sensor_path(self, spec):
        return spec.get("file") or self.h5_path

    def _ensure_open(self):
        """Returns (root, roots) for the CALLING THREAD, opening that
        thread's own h5py.File handles on first use."""
        tls = self._tls
        if getattr(tls, "root", None) is None:
            tls.roots = {p: h5py.File(p, "r") for p in self._file_paths}
            tls.root = tls.roots[self.h5_path]
        return tls.root, tls.roots

    def __len__(self):
        return len(self.index_map)

    def __del__(self):
        # Only closes THIS (the destructor-calling) thread's handles -- any
        # other thread's handles are released by the OS when the process
        # exits. Fine for a training run's lifetime (see _ensure_open).
        try:
            for h in (getattr(self._tls, "roots", None) or {}).values():
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

    def _history(self, data, t, ep_start, stride=1):
        """Read only the temporal rows that survive ``stride``.

        ReLoc3R uses obs_horizon=60 and stride=12, so reading a contiguous
        60-frame HDF5 slice and subsampling it afterwards wastes 11/12 of the
        NFS traffic. Build the padded history indices first and issue one small
        fancy-index read for the rows the model actually consumes. ``h5py``
        requires increasing unique indices, hence the unique/inverse roundtrip
        for repeated episode-start padding.
        """
        stride = int(stride)
        if stride < 1:
            raise ValueError(f"history stride must be >= 1, got {stride}")
        indices = np.arange(
            t - self.obs_horizon + 1, t + 1, dtype=np.int64
        )
        indices = np.maximum(indices, int(ep_start))[::stride]
        unique, inverse = np.unique(indices, return_inverse=True)
        return np.asarray(data[unique])[inverse]

    def _history_valid_mask(self, t, ep_start, stride=1):
        """bool[ceil(obs_horizon/stride)], True=real and False=padded."""
        valid_len = min(self.obs_horizon, t - ep_start + 1)
        pad_len = self.obs_horizon - valid_len
        mask = np.ones(self.obs_horizon, dtype=bool)
        mask[:pad_len] = False
        return mask[::int(stride)]

    def __getitem__(self, idx):
        root, roots = self._ensure_open()
        t = self.index_map[idx]
        ep_start = self.ep_start_map[idx]
        ep_end = self.ep_end_map[idx]
        goal_row = ep_end - 1  # episode's static goal/docked frame (last row),
        # same convention as utils/docking_dataset.py and endgame/se2.py.
        obs = {}
        sampled_goal_rows = {}
        for name, spec in self.sensors.items():
            sensor_root = roots[self._sensor_path(spec)]
            ds = self._ram_cache.get(name)
            if ds is None:
                ds = sensor_root[spec["source"]]
            mode = spec.get("mode", "history")
            if mode == "history":
                stride = int(spec.get("stride", 1))
                arr = self._history(ds, t, ep_start, stride)
                valid_mask = self._history_valid_mask(t, ep_start, stride)
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
            elif mode == "goal_pool_pair":
                stride = int(spec.get("stride", 1))
                history = self._history(ds, t, ep_start, stride)
                valid_mask = self._history_valid_mask(t, ep_start, stride)
                group = str(spec.get("goal_pool_group", "default"))
                if group not in sampled_goal_rows:
                    info = self._goal_pool_groups[group]
                    candidates = info["rows"]
                    match = info["match"]
                    if match is not None:
                        episode_index = int(
                            np.searchsorted(self.episode_ends, t, side="right")
                        )
                        source_value = str(
                            match["episode_values"][episode_index]
                        )
                        target_value = source_value
                        cross_target = match["cross_map"].get(source_value)
                        if (
                            cross_target is not None
                            and torch.rand(1).item()
                            < match["cross_probability"]
                        ):
                            target_value = cross_target
                        candidates = match["rows_by_value"][target_value]
                    if info["sampling"] == "random":
                        variant_weights = info["variant_weights"]
                        if variant_weights:
                            labels = info["goal_variants"][candidates]
                            available = [
                                variant
                                for variant, weight in variant_weights.items()
                                if weight > 0 and np.any(labels == variant)
                            ]
                            if not available:
                                raise RuntimeError(
                                    "goal_variant_weights selected no variant for "
                                    f"the current goal candidate set: {variant_weights}"
                                )
                            weights = torch.tensor(
                                [variant_weights[v] for v in available],
                                dtype=torch.float64,
                            )
                            variant_index = int(
                                torch.multinomial(weights, 1).item()
                            )
                            selected_variant = available[variant_index]
                            candidates = candidates[labels == selected_variant]
                        choice = int(torch.randint(len(candidates), (1,)).item())
                    elif info["sampling"] == "first":
                        choice = 0
                    elif info["sampling"] == "index":
                        choice = int(idx) % len(candidates)
                    else:
                        raise ValueError(
                            f"sensor '{name}': unknown goal_sampling={info['sampling']} "
                            "(expected random|first|index)"
                        )
                    sampled_goal_rows[group] = int(candidates[choice])
                sampled_row = sampled_goal_rows[group]
                goal_root = roots[spec["goal_pool_file"]]
                goal = np.asarray(goal_root[spec["goal_source"]][sampled_row])
                goal_history = np.broadcast_to(goal, history.shape)
                pair = np.stack([history, goal_history], axis=1)
                # Source/history and compiled goal features are float16. Keep
                # them compact across CPU RAM, worker IPC, and H2D; the encoder
                # converts the pair to float32 on GPU before frozen decoding.
                obs[name] = torch.from_numpy(np.ascontiguousarray(pair))
                obs[f"{name}_valid_mask"] = torch.from_numpy(
                    np.ascontiguousarray(valid_mask)
                )
            else:
                raise ValueError(f"sensor '{name}': unknown mode '{mode}'.")

        act = root[self.action_key][t:t + self.horizon].astype(np.float32)
        act = self.normalize_action(act).astype(np.float32)
        sample = {"obs": obs, "act": torch.from_numpy(act)}

        if self.with_aux:
            # ICP dock-pose label (teacher for the precision aux head).
            p = root[self.dock_pose_key][t].astype(np.float32)
            valid = not np.any(np.isnan(p))
            rel = float(root[self.reliable_key][t]) if valid else 0.0
            p = np.nan_to_num(p)
            xy = (p[:2] - self.dock_xy_mean) / self.dock_xy_std
            sample["dock_target"] = torch.tensor(
                [xy[0], xy[1], np.sin(p[2]), np.cos(p[2])], dtype=torch.float32)
            sample["reliable"] = torch.tensor(rel, dtype=torch.float32)

        return sample
