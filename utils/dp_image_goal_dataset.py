"""Raw-image adapter for the official Diffusion Policy image policy.

The adapter deliberately contains no policy implementation.  It translates the
235-episode docking HDF5 and editable goal pool into the observation dictionary
expected by ``DiffusionUnetImagePolicy`` from the official
``real-stanford/diffusion_policy`` repository.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.goal_pool import GoalPool


class DPImageGoalDataset(Dataset):
    """Image history + sampled image goal + wheel history -> raw (v, w) future.

    Each RGB observation key has one explicit official-DP observation step:
    ``[To=1, C, H, W]``. Five named history keys retain temporal identity without
    repeating a static goal five times. ``wheel_history`` is ``[1, 120]`` and
    ``action`` is raw ``[60, 2]`` in physical ``(v, w)`` units. The official
    ``LinearNormalizer`` handles all scaling inside the policy.
    """

    def __init__(
        self,
        h5_path: str,
        goal_pool_db: str,
        *,
        image_key: str = "image_bottom",
        goal_camera: str = "orbbec-0",
        horizon: int = 60,
        obs_horizon: int = 60,
        vision_horizon: int = 5,
        vision_stride: int = 12,
        action_key: str = "encoder",
        goal_splits: Iterable[str] = ("all",),
        goal_variant_weights: Mapping[str, float] | None = None,
        cross_goal_prob: float = 0.1,
        cross_goal_map: Mapping[str, str] | None = None,
        image_cache_mmap: str | None = None,
    ):
        super().__init__()
        self.h5_path = str(Path(h5_path).expanduser().resolve())
        self.goal_pool_db = str(Path(goal_pool_db).expanduser().resolve())
        self.image_key = str(image_key)
        self.goal_camera = str(goal_camera)
        self.horizon = int(horizon)
        self.obs_horizon = int(obs_horizon)
        self.vision_horizon = int(vision_horizon)
        self.vision_stride = int(vision_stride)
        self.action_key = str(action_key)
        self.cross_goal_prob = float(cross_goal_prob)
        self.cross_goal_map = {str(k): str(v) for k, v in (cross_goal_map or {}).items()}
        self.variant_weights = {
            str(k): float(v) for k, v in (
                goal_variant_weights
                or {"original": 0.495, "qr_removed": 0.495, "color_changed": 0.01}
            ).items()
        }
        if self.horizon < 1 or self.obs_horizon < 1:
            raise ValueError("horizon and obs_horizon must be positive")
        if self.vision_stride < 1:
            raise ValueError("vision_stride must be positive")
        expected_vision = len(range(0, self.obs_horizon, self.vision_stride))
        if expected_vision != self.vision_horizon:
            raise ValueError(
                f"vision_horizon={self.vision_horizon}, but obs_horizon="
                f"{self.obs_horizon} and stride={self.vision_stride} produce "
                f"{expected_vision} frames"
            )
        if not 0.0 <= self.cross_goal_prob <= 1.0:
            raise ValueError("cross_goal_prob must be in [0, 1]")
        if not self.variant_weights or any(v < 0 for v in self.variant_weights.values()):
            raise ValueError("goal_variant_weights must be non-empty and non-negative")
        if sum(self.variant_weights.values()) <= 0:
            raise ValueError("goal_variant_weights must contain positive mass")

        with h5py.File(self.h5_path, "r") as h5:
            for key in (self.image_key, self.action_key, "episode_ends"):
                if key not in h5:
                    raise KeyError(f"{self.h5_path} is missing '{key}'")
            self.episode_ends = np.asarray(h5["episode_ends"][:], dtype=np.int64)
            self.actions = np.asarray(h5[self.action_key][:], dtype=np.float32)
            image_ds = h5[self.image_key]
            self.image_shape = tuple(int(x) for x in image_ds.shape[1:])
            self.image_dtype = np.dtype(image_ds.dtype)
            self.image_rows = int(image_ds.shape[0])
            if self.image_shape[0] != 3 or self.image_dtype != np.uint8:
                raise ValueError(
                    f"expected uint8 CHW RGB images, got {image_ds.shape} {image_ds.dtype}"
                )
            if self.image_rows != len(self.actions):
                raise ValueError("image and action row counts differ")
            if "source_dataset" in h5:
                values = h5["source_dataset"].asstr()[:]
                self.episode_sources = np.asarray(values, dtype=object)
            else:
                self.episode_sources = np.asarray(
                    [Path(self.h5_path).stem] * len(self.episode_ends), dtype=object
                )
        if len(self.episode_sources) != len(self.episode_ends):
            raise ValueError("source_dataset must have one value per episode")

        self.action_min = self.actions.min(axis=0).astype(np.float32)
        self.action_max = self.actions.max(axis=0).astype(np.float32)
        self.action_mean = self.actions.mean(axis=0).astype(np.float32)
        self.action_std = self.actions.std(axis=0).astype(np.float32)

        self.index_map: list[int] = []
        self.ep_start_map: list[int] = []
        self.ep_end_map: list[int] = []
        self.ep_id_map: list[int] = []
        start = 0
        for ep_id, end in enumerate(self.episode_ends):
            end = int(end)
            for t in range(start, end - self.horizon + 1):
                self.index_map.append(t)
                self.ep_start_map.append(start)
                self.ep_end_map.append(end)
                self.ep_id_map.append(ep_id)
            start = end

        self._load_goal_images(tuple(str(x) for x in goal_splits))
        self._validate_goal_candidates()

        self.image_cache_mmap = (
            str(Path(image_cache_mmap).expanduser()) if image_cache_mmap else None
        )
        self._image_cache = None
        if self.image_cache_mmap:
            expected_bytes = self.image_rows * int(np.prod(self.image_shape)) * self.image_dtype.itemsize
            actual_bytes = os.path.getsize(self.image_cache_mmap)
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"image cache has {actual_bytes} bytes, expected {expected_bytes} "
                    f"for {(self.image_rows, *self.image_shape)} {self.image_dtype}"
                )
            self._image_cache = np.memmap(
                self.image_cache_mmap,
                mode="r",
                dtype=self.image_dtype,
                shape=(self.image_rows, *self.image_shape),
                order="C",
            )
        self._h5 = None
        self._image_ds = None

    def _load_goal_images(self, splits: tuple[str, ...]) -> None:
        with GoalPool(self.goal_pool_db) as pool:
            records = pool.records(enabled=True, splits=splits)
        images = []
        datasets = []
        variants = []
        ids = []
        for record in records:
            info = record.images.get(self.goal_camera)
            if info is None:
                continue
            with Image.open(info["path"]) as pil:
                rgb_pil = pil.convert("RGB")
                target_hw = self.image_shape[1:]
                if rgb_pil.size != (target_hw[1], target_hw[0]):
                    rgb_pil = rgb_pil.resize(
                        (target_hw[1], target_hw[0]), Image.Resampling.LANCZOS
                    )
                rgb = np.asarray(rgb_pil, dtype=np.uint8)
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
            if chw.shape != self.image_shape:
                raise ValueError(
                    f"goal {record.goal_id} has shape {chw.shape}, expected {self.image_shape}"
                )
            images.append(chw)
            datasets.append(str(record.dataset))
            variants.append(str(record.variant))
            ids.append(str(record.goal_id))
        if not images:
            raise ValueError("goal pool selected no usable goal images")
        self.goal_images = np.stack(images, axis=0)
        self.goal_datasets = np.asarray(datasets, dtype=object)
        self.goal_variants = np.asarray(variants, dtype=object)
        self.goal_ids = tuple(ids)
        grouped: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        for dataset in np.unique(self.goal_datasets):
            dataset_mask = self.goal_datasets == dataset
            for variant in np.unique(self.goal_variants[dataset_mask]):
                rows = np.flatnonzero(dataset_mask & (self.goal_variants == variant))
                grouped[str(dataset)][str(variant)] = rows.astype(np.int64)
        self.goal_rows = dict(grouped)

    def _validate_goal_candidates(self) -> None:
        required = set(map(str, self.episode_sources))
        required.update(
            self.cross_goal_map[src]
            for src in tuple(required)
            if src in self.cross_goal_map and self.cross_goal_prob > 0
        )
        missing = sorted(required - set(self.goal_rows))
        if missing:
            raise ValueError(f"goal pool lacks required datasets: {missing}")
        for dataset in sorted(required):
            available = self.goal_rows[dataset]
            positive = [
                name for name, weight in self.variant_weights.items()
                if weight > 0 and name in available and len(available[name]) > 0
            ]
            if not positive:
                raise ValueError(
                    f"dataset '{dataset}' has none of the weighted variants "
                    f"{sorted(self.variant_weights)}"
                )

    def _ensure_open(self) -> None:
        if self._image_cache is not None or self._h5 is not None:
            return
        self._h5 = h5py.File(self.h5_path, "r")
        self._image_ds = self._h5[self.image_key]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_image_ds"] = None
        return state

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.index_map)

    def history_indices(self, t: int, ep_start: int) -> tuple[np.ndarray, np.ndarray]:
        full = np.arange(t - self.obs_horizon + 1, t + 1, dtype=np.int64)
        full = np.maximum(full, int(ep_start))
        sparse = full[:: self.vision_stride]
        return full, sparse

    def sample_goal_row(self, ep_id: int) -> int:
        source = str(self.episode_sources[int(ep_id)])
        target = source
        cross = self.cross_goal_map.get(source)
        if cross is not None and torch.rand(()).item() < self.cross_goal_prob:
            target = cross
        rows_by_variant = self.goal_rows[target]
        variants = [
            name for name, weight in self.variant_weights.items()
            if weight > 0 and name in rows_by_variant and len(rows_by_variant[name]) > 0
        ]
        weights = torch.tensor(
            [self.variant_weights[name] for name in variants], dtype=torch.float64
        )
        variant = variants[int(torch.multinomial(weights, 1).item())]
        candidates = rows_by_variant[variant]
        return int(candidates[int(torch.randint(len(candidates), (1,)).item())])

    def __getitem__(self, idx: int):
        self._ensure_open()
        t = int(self.index_map[idx])
        ep_start = int(self.ep_start_map[idx])
        ep_id = int(self.ep_id_map[idx])
        full_indices, sparse_indices = self.history_indices(t, ep_start)
        image_source = self._image_cache if self._image_cache is not None else self._image_ds
        # h5py fancy indexing requires increasing unique indices. Episode-start
        # padding repeats indices, so perform the same unique/inverse round-trip
        # used by the RTDP dataset.
        unique, inverse = np.unique(sparse_indices, return_inverse=True)
        history = np.asarray(image_source[unique])[inverse]
        goal_row = self.sample_goal_row(ep_id)

        obs = {
            f"history_{i}": torch.from_numpy(
                np.ascontiguousarray(history[i : i + 1])
            )
            for i in range(self.vision_horizon)
        }
        obs["goal"] = torch.from_numpy(
            np.ascontiguousarray(self.goal_images[goal_row : goal_row + 1])
        )
        wheel = np.ascontiguousarray(self.actions[full_indices].reshape(1, -1))
        obs["wheel_history"] = torch.from_numpy(wheel)
        action = torch.from_numpy(
            np.ascontiguousarray(self.actions[t : t + self.horizon])
        )
        return {"obs": obs, "action": action}

    @property
    def rgb_keys(self) -> tuple[str, ...]:
        return tuple([f"history_{i}" for i in range(self.vision_horizon)] + ["goal"])

    def shape_meta(self) -> dict:
        obs = {
            key: {"shape": list(self.image_shape), "type": "rgb"}
            for key in self.rgb_keys
        }
        obs["wheel_history"] = {
            "shape": [self.obs_horizon * self.actions.shape[1]],
            "type": "low_dim",
        }
        return {"obs": obs, "action": {"shape": [self.actions.shape[1]]}}

    def get_normalizer(self):
        """Build the official DP LinearNormalizer with the RTDP min-max contract."""
        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer,
            SingleFieldLinearNormalizer,
        )

        def manual(lo: np.ndarray, hi: np.ndarray, mean: np.ndarray, std: np.ndarray):
            lo = np.asarray(lo, dtype=np.float32).reshape(-1)
            hi = np.asarray(hi, dtype=np.float32).reshape(-1)
            mean = np.asarray(mean, dtype=np.float32).reshape(-1)
            std = np.asarray(std, dtype=np.float32).reshape(-1)
            span = np.clip(hi - lo, 1e-5, None)
            scale = (2.0 / span).astype(np.float32)
            offset = (-1.0 - scale * lo).astype(np.float32)
            return SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict={"min": lo, "max": hi, "mean": mean, "std": std},
            )

        normalizer = LinearNormalizer()
        normalizer["action"] = manual(
            self.action_min, self.action_max, self.action_mean, self.action_std
        )
        repeats = self.obs_horizon
        normalizer["wheel_history"] = manual(
            np.tile(self.action_min, repeats),
            np.tile(self.action_max, repeats),
            np.tile(self.action_mean, repeats),
            np.tile(self.action_std, repeats),
        )
        image_normalizer = SingleFieldLinearNormalizer.create_manual(
            scale=np.asarray([1.0 / 255.0], dtype=np.float32),
            offset=np.asarray([0.0], dtype=np.float32),
            input_stats_dict={
                "min": np.asarray([0.0], dtype=np.float32),
                "max": np.asarray([255.0], dtype=np.float32),
                "mean": np.asarray([127.5], dtype=np.float32),
                "std": np.asarray([73.9], dtype=np.float32),
            },
        )
        for key in self.rgb_keys:
            normalizer[key] = image_normalizer
        return normalizer
