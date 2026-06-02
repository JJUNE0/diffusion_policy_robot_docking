import os
from typing import Dict, List, Optional

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DockingDataset(Dataset):
    """
    Dataset for the new multimodal setting:
      - motion input  : normalized encoder velocity history [T_obs, 2]   (full obs_horizon)
      - vision input  : room1 / room2 image histories [T_vis, 3, 240, 320]  (sparse)
      - lidar input   : optional BEV occupancy [T_vis, C, S, S]              (sparse)
      - target output : normalized future encoder trajectory [H, 2]

    Memory / IO design notes
    ------------------------
    * Images and lidar are stored on disk as ``uint8`` and are *returned as
      ``torch.uint8`` tensors*. Conversion to float / division by 255 must be
      done on the GPU by the training / inference loop. This keeps CPU-side
      worker buffers ~4x smaller and avoids large float copies through
      ``pin_memory``.
    * Vision and lidar histories are *already subsampled* by ``vision_stride``
      inside ``__getitem__`` (e.g. obs_horizon=30, stride=6 -> 5 frames at
      offsets 0, 6, 12, 18, 24). Earlier code returned the full 30-frame
      history and let the training loop slice ``[:, ::stride]`` afterwards,
      which forced reading and decompressing 6x more HDF5 chunks than needed.
    * HDF5 file handles are opened *lazily* inside the worker process that
      first calls ``__getitem__``. Workers detect a PID change and reopen
      their own handles, because ``h5py`` is not fork-safe and sharing the
      parent's open file across forked DataLoader workers can corrupt reads
      or hang the process.
    """

    def __init__(
        self,
        npz_path: str,
        train_npz_path: str,
        horizon: int = 16,
        obs_horizon: int = 30,
        vision_stride: int = 6,
        dt: float = 0.0333,
        with_goal: bool = True,
        goal_offset_min: Optional[int] = None,
        goal_offset_max: Optional[int] = None,
    ):
        super().__init__()
        self.h5_path = npz_path
        self.train_h5_path = train_npz_path
        self.horizon = horizon
        self.obs_horizon = obs_horizon
        self.vision_stride = max(1, int(vision_stride))
        # Kept for interface compatibility with existing setup / inference code.
        self.dt = dt

        # ------------------------------------------------------------------
        # NoMaD-style goal masking support.
        # When ``with_goal`` is on, __getitem__ also returns a single future
        # camera frame per room ("goal_image1"/"goal_image2") sampled from the
        # *same episode*, somewhere ahead of the current timestep (up to the
        # final, docked frame). The condition network turns this into a goal
        # token that is randomly masked during training.
        # ------------------------------------------------------------------
        self.with_goal = bool(with_goal)
        # Default: sample the goal at least one prediction horizon ahead so it
        # is a genuine future/destination frame, not inside the predicted
        # action window.
        self.goal_offset_min = int(goal_offset_min) if goal_offset_min is not None else int(horizon)
        self.goal_offset_max = int(goal_offset_max) if goal_offset_max is not None else None

        for label, path in (("eval_data", self.h5_path), ("train_norm", self.train_h5_path)):
            p = Path(path).expanduser()
            if not p.is_file():
                if p.is_dir():
                    raise ValueError(
                        f"DockingDataset: {label} must be an HDF5 file path (.h5), not a directory. "
                        f"Got: {path}\n"
                        f"  - eval_data  -> eval_data_path (the episode data to load)\n"
                        f"  - train_norm -> train_data_path (HDF5 used when training, for action normalization)"
                    )
                raise FileNotFoundError(f"DockingDataset: {label} file not found: {path}")

        # ------------------------------------------------------------------
        # Pull all metadata (and the small encoder array used for normalization
        # statistics) up-front so the file handles can stay closed until a
        # worker actually needs them.
        # ------------------------------------------------------------------
        with h5py.File(self.h5_path, "r") as _f:
            self._has_lidar = "lidar_map" in _f
            self.episode_ends = _f["episode_ends"][:]
            if self._has_lidar:
                lidar_dset = _f["lidar_map"]
                attrs = lidar_dset.attrs
                self.lidar_meta = {
                    "range_m": float(attrs.get("range_m", 6.0)),
                    "resolution": float(attrs.get("resolution", 0.05)),
                    "channels": int(attrs.get("channels", lidar_dset.shape[1])),
                    "size": int(attrs.get("size", lidar_dset.shape[-1])),
                }
            else:
                self.lidar_meta = None

        with h5py.File(self.train_h5_path, "r") as _f:
            enc = _f["encoder"][:].astype(np.float32)
        self.action_min = enc.min(axis=0).astype(np.float32)
        self.action_max = enc.max(axis=0).astype(np.float32)
        self.action_scale = np.clip(
            self.action_max - self.action_min, a_min=1e-5, a_max=None
        ).astype(np.float32)

        # Index map: (timestep, ep_start, ep_end) for every valid sample window.
        # ``ep_end`` (exclusive episode boundary) is needed to sample a goal
        # frame from the remaining future of the same episode.
        self.index_map: List[int] = []
        self.ep_start_map: List[int] = []
        self.ep_end_map: List[int] = []
        start_idx = 0
        for end_idx in self.episode_ends:
            for t in range(start_idx, end_idx - self.horizon + 1):
                self.index_map.append(t)
                self.ep_start_map.append(start_idx)
                self.ep_end_map.append(int(end_idx))
            start_idx = end_idx

        # Sparse vision offsets that match the legacy ``[:, ::vision_stride]``
        # slicing, but applied at read time so we never load the full history.
        self._sparse_offsets = list(range(0, self.obs_horizon, self.vision_stride))

        # File handles are opened lazily, per-process.
        self.root = None
        self.train_root = None
        self.z_encoder = None
        self.z_encoder_train = None
        self.z_img1 = None
        self.z_img2 = None
        self.z_lidar = None
        self._opened_pid = None

        # Open in the parent process so direct attribute access (used by the
        # inference scripts, e.g. ``dataset.z_encoder[0:N]``) works without
        # going through ``__getitem__``. Worker processes will reopen their
        # own handles on first ``__getitem__`` call.
        self._ensure_open()

    # ------------------------------------------------------------------
    # File handle management
    # ------------------------------------------------------------------
    def _close_handles(self) -> None:
        for attr in ("root", "train_root"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._opened_pid == pid and self.root is not None and self.train_root is not None:
            return
        # Either first open in this process, or we have been forked from a
        # parent that already had open handles - drop those and reopen.
        self._close_handles()
        self.root = h5py.File(self.h5_path, "r")
        self.train_root = h5py.File(self.train_h5_path, "r")
        self.z_encoder = self.root["encoder"]
        self.z_encoder_train = self.train_root["encoder"]
        self.z_img1 = self.root["image_top"]
        self.z_img2 = self.root["image_bottom"]
        self.z_lidar = self.root["lidar_map"] if self._has_lidar else None
        self._opened_pid = pid

    def __getstate__(self):
        # h5py file objects can't be pickled (DataLoader workers using
        # ``spawn`` / ``forkserver`` would otherwise crash). Strip them and
        # let ``_ensure_open`` reopen on the worker side.
        state = self.__dict__.copy()
        for key in (
            "root",
            "train_root",
            "z_encoder",
            "z_encoder_train",
            "z_img1",
            "z_img2",
            "z_lidar",
        ):
            state[key] = None
        state["_opened_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        self._close_handles()

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index_map)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return 2.0 * (action - self.action_min) / self.action_scale - 1.0

    def _get_history(self, data, t: int, ep_start: int) -> np.ndarray:
        """Dense ``obs_horizon`` history ending at timestep ``t`` (left-padded
        with the first valid frame at episode boundaries)."""
        start_t = t - self.obs_horizon + 1
        if start_t < ep_start:
            valid_data = data[ep_start: t + 1]
            pad_len = self.obs_horizon - len(valid_data)
            padding = np.repeat(valid_data[:1], pad_len, axis=0)
            return np.concatenate([padding, valid_data], axis=0)
        return data[start_t: t + 1]

    def _sparse_indices(self, t: int, ep_start: int) -> np.ndarray:
        """Indices (clamped to ep_start) corresponding to a sparse sub-history
        of length ``len(self._sparse_offsets)`` ending at ``t``. Equivalent to
        ``self._get_history(data, t, ep_start)[::vision_stride]`` but only the
        chosen frames are read from disk."""
        base = t - (self.obs_horizon - 1)
        idx = np.array([max(ep_start, base + off) for off in self._sparse_offsets], dtype=np.int64)
        return idx

    def _read_sparse(self, dset, indices: np.ndarray) -> np.ndarray:
        """h5py needs strictly increasing unique indices for fancy indexing,
        so we de-duplicate, read, then expand back to the requested layout."""
        unique, inverse = np.unique(indices, return_inverse=True)
        chunk = dset[list(unique)]
        return chunk[inverse]

    def _sample_goal_index(self, t: int, ep_end: int) -> int:
        """Pick a goal frame index from the *future* of the current episode.

        ``g`` is sampled uniformly from ``[lo, hi]`` where
        ``lo = min(t + goal_offset_min, ep_end - 1)`` and
        ``hi = ep_end - 1`` (capped by ``goal_offset_max`` if set). Degenerate
        ranges (near the episode end) clamp to the final, docked frame."""
        last = ep_end - 1
        lo = min(t + self.goal_offset_min, last)
        hi = last
        if self.goal_offset_max is not None:
            hi = min(t + self.goal_offset_max, last)
        if lo > hi:
            return last
        return int(np.random.randint(lo, hi + 1))

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()

        t = self.index_map[idx]
        ep_start = self.ep_start_map[idx]
        ep_end = self.ep_end_map[idx]

        # Motion history (cheap, full length).
        encoder_seq_raw = self._get_history(self.z_encoder, t, ep_start).astype(np.float32)   # [T_obs, 2]
        velocity_seq_norm = self.normalize_action(encoder_seq_raw).astype(np.float32)         # [T_obs, 2]

        # Vision histories: read only the sparse frames we'll actually use,
        # keep ``uint8`` so the tensor is 4x smaller. Conversion to float and
        # /255.0 must happen on the GPU in the training / inference loop.
        sparse_idx = self._sparse_indices(t, ep_start)
        image_room1 = self._read_sparse(self.z_img1, sparse_idx)   # [T_vis, 3, H, W] uint8
        image_room2 = self._read_sparse(self.z_img2, sparse_idx)   # [T_vis, 3, H, W] uint8

        # Future target trajectory in normalized encoder space.
        act_traj = self.z_encoder[t: t + self.horizon].astype(np.float32)                     # [H, 2]
        act_traj_norm = self.normalize_action(act_traj).astype(np.float32)

        obs = {
            "encoder": torch.from_numpy(encoder_seq_raw),                  # float32
            "velocity": torch.from_numpy(velocity_seq_norm),               # float32
            "image_room1": torch.from_numpy(image_room1),                  # uint8
            "image_room2": torch.from_numpy(image_room2),                  # uint8
        }

        if self.z_lidar is not None:
            lidar_seq = self._read_sparse(self.z_lidar, sparse_idx)        # [T_vis, C, S, S] uint8
            obs["lidar_map"] = torch.from_numpy(lidar_seq)                 # uint8

        # NoMaD-style goal: a single future camera frame per room (uint8). The
        # condition network masks this token randomly during training.
        if self.with_goal:
            g = self._sample_goal_index(t, ep_end)
            goal_image1 = self.z_img1[g]                                   # [3, H, W] uint8
            goal_image2 = self.z_img2[g]                                   # [3, H, W] uint8
            obs["goal_image1"] = torch.from_numpy(np.asarray(goal_image1))  # uint8
            obs["goal_image2"] = torch.from_numpy(np.asarray(goal_image2))  # uint8

        return {
            "obs": obs,
            "act": torch.from_numpy(act_traj_norm),                        # float32
        }


def denormalize(norm_action, act_scale, act_min):
    return (norm_action + 1.0) / 2.0 * act_scale + act_min
