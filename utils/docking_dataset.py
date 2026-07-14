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
        with_goal_lidar: bool = False,
        aux_relative: bool = False,
        adv_weight: bool = False,
        vision_stride: int = 6,
        sparse_vision_uint8: bool = False,
        dino_cache_path: str = None,
        use_room1: bool = True,
    ):
        super().__init__()
        # DINO feature cache: when set, __getitem__ returns precomputed room2
        # (and optionally room1) DINO features instead of raw image pixels, and
        # the train loop skips the backbone. See scripts/precompute_dino_cache.py.
        self.dino_cache_path = dino_cache_path
        self.use_room1 = use_room1
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

        # IMPORTANT: do NOT keep an open h5 handle in __init__. If we did, every
        # DataLoader worker would fork that one handle, and HDF5 serializes reads
        # across a shared handle -> workers don't parallelize (CPU stays flat)
        # while each still holds read buffers (RAM balloons). Instead read the
        # metadata via short-lived handles here, and reopen ONE handle per worker
        # lazily in __getitem__ (see _ensure_open). This makes num_workers
        # actually parallelize the gzip image decode.
        # Goal-referenced registration conditioning: also return the GOAL frame's
        # LiDAR scan (docked scan) so the condition net can learn current<->goal
        # scan alignment, and (aux_relative) supervise the aux head with the
        # current->goal RELATIVE pose instead of the absolute dock pose.
        self.with_goal_lidar = with_goal_lidar and with_goal and with_lidar
        self.aux_relative = aux_relative

        with h5py.File(self.h5_path, "r") as f:
            self.episode_ends = f["episode_ends"][:]
            self.with_lidar = with_lidar and ("lidar_points" in f)
            self.with_aux = with_aux and ("dock_pose" in f)
            if self.with_aux:
                pose_all = f["dock_pose"][:]
                rel_all = f["reliable"][:].astype(bool)
                # Guard: only use rows that are BOTH reliable and non-NaN, so a
                # single NaN pose can't poison the mean/std (and thus every
                # emitted target) with NaN. (Same guard as ModularDockingDataset.)
                valid = rel_all & ~np.isnan(pose_all).any(axis=1)
                xy = pose_all[valid][:, :2]
                self.dock_xy_mean = xy.mean(axis=0).astype(np.float32)
                self.dock_xy_std = (xy.std(axis=0) + 1e-6).astype(np.float32)
                if self.aux_relative:
                    # Per-episode goal reference pose = the LAST valid-label frame
                    # (the labeler tracks backward from docked, so this is the
                    # docked pose). Episodes without any valid label get None ->
                    # their rows are emitted with reliable=0.
                    self.goal_pose_ep = []
                    s = 0
                    rel_rows = []
                    for e in self.episode_ends:
                        idx = np.where(valid[s:e])[0]
                        g = pose_all[s + idx[-1]] if len(idx) else None
                        self.goal_pose_ep.append(g)
                        if g is not None:
                            rel_rows.append(self._relative_to_goal(pose_all[s:e][valid[s:e]], g))
                        s = e
                    rel_xy = np.concatenate(rel_rows)[:, :2]
                    # separate stats: relative xy is near-zero-centered by design
                    self.rel_xy_mean = rel_xy.mean(axis=0).astype(np.float32)
                    self.rel_xy_std = (rel_xy.std(axis=0) + 1e-6).astype(np.float32)
                # What the emitted dock_target's xy is normalized WITH — the train
                # loop must denormalize with these (not always dock_xy_*).
                self.aux_xy_mean = self.rel_xy_mean if self.aux_relative else self.dock_xy_mean
                self.aux_xy_std = self.rel_xy_std if self.aux_relative else self.dock_xy_std

                # Offline reward-weighted BC (AWR). Emits THREE z-scored advantage
                # components per frame; scripts/train.py mixes them with config
                # weights (adv_mode speed|precision) and a distance gate.
                #
                # Which signals are actually learnable was measured on the 145
                # train demos (spread vs the 5.7mm/0.22deg ICP noise floor):
                #   final x  : 1.9x noise   final y: 1.2x  -> position at docking
                #              is mechanically fixed (robot stops ON the dock,
                #              53.4cm +/-4mm) => NO learnable precision signal
                #   final yaw: 7.1x noise (std 1.57deg, range 11deg) => the ONLY
                #              strong precision signal the demos contain.
                # So docking precision must be rewarded through HEADING ALIGNMENT,
                # not distance. Components:
                #   prog  = dock-approach speed over the horizon        [speed]
                #   align = reduction of |yaw| misalignment over the horizon  [precision]
                #   term  = -|final yaw| of the whole episode (episode-level
                #           credit: "learn more from demos that docked straight") [precision]
                self.adv_weight = adv_weight and with_aux
                if self.adv_weight:
                    # nan_to_num FIRST: unreliable rows hold NaN poses, and a NaN
                    # reaching dock_d survives even where the advantage is 0
                    # (0 * NaN = NaN) -> NaN loss on the very first step.
                    pose_z = np.nan_to_num(pose_all).astype(np.float32)
                    dist_all = np.hypot(pose_z[:, 0], pose_z[:, 1])
                    yaw_err = np.abs(pose_z[:, 2])                        # |yaw| = misalignment
                    n = int(self.episode_ends[-1])
                    prog = np.zeros(n, np.float32)
                    align = np.zeros(n, np.float32)
                    term = np.zeros(n, np.float32)
                    ok = np.zeros(n, bool)
                    dock_d = np.zeros(n, np.float32)
                    s = 0
                    for e in self.episode_ends:
                        idx = np.where(valid[s:e])[0]
                        # episode-level terminal alignment quality (same for all
                        # its frames); episodes with no valid label get 0 = neutral
                        t_align = -float(yaw_err[s + idx[-1]]) if len(idx) else 0.0
                        for t in range(s, e - horizon):
                            # Only trust the distance on labelled rows: 72 unreliable
                            # rows carry blown-up ICP failures (up to 29 km). They are
                            # already neutral (advantage 0), but leaving the garbage in
                            # dock_d would feed the speed gate nonsense.
                            dock_d[t] = dist_all[t] if valid[t] else 0.0
                            if valid[t] and valid[t + horizon]:
                                prog[t] = (dist_all[t] - dist_all[t + horizon]) / (horizon * dt)
                                align[t] = yaw_err[t] - yaw_err[t + horizon]   # >0 = got straighter
                                term[t] = t_align
                                ok[t] = True
                        s = e

                    def _z(v):
                        m = v[ok]
                        mu = float(m.mean()) if len(m) else 0.0
                        sd = float(m.std() + 1e-6) if len(m) else 1.0
                        return ((v - mu) / sd).astype(np.float32)

                    self._adv_prog, self._adv_align, self._adv_term = _z(prog), _z(align), _z(term)
                    self._adv_ok, self._adv_dock_d = ok, dock_d

        # The DINO cache must be row-aligned 1:1 with THIS h5. Catch the silent
        # misalignment of pointing an eval/held-out h5 at the train cache.
        if self.dino_cache_path is not None:
            with h5py.File(self.dino_cache_path, "r") as cf:
                n_cache = cf["dino_bottom"].shape[0]
            n_rows = int(self.episode_ends[-1])
            if n_cache != n_rows:
                raise ValueError(
                    f"DINO cache {self.dino_cache_path} has {n_cache} rows but "
                    f"{self.h5_path} has {n_rows}; the cache must be precomputed "
                    f"from this exact h5 (see scripts/precompute_dino_cache.py).")

        with h5py.File(self.train_h5_path, "r") as f:
            enc = f["encoder"][:]
        self.action_min = enc.min(axis=0).astype(np.float32)
        self.action_max = enc.max(axis=0).astype(np.float32)
        self.action_scale = np.clip(self.action_max - self.action_min, a_min=1e-5, a_max=None).astype(np.float32)

        self.index_map = []
        self.ep_start_map = []
        self.ep_end_map = []
        self.ep_id_map = []
        start_idx = 0
        for ep_id, end_idx in enumerate(self.episode_ends):
            for t in range(start_idx, end_idx - self.horizon + 1):
                self.index_map.append(t)
                self.ep_start_map.append(start_idx)
                self.ep_end_map.append(end_idx)
                self.ep_id_map.append(ep_id)
            start_idx = end_idx

        self.root = None        # opened per worker in _ensure_open()
        self.cache_root = None   # DINO feature cache handle (per worker)

    def _ensure_open(self):
        """Open the h5 handle once per process (lazy). Each DataLoader worker
        opens its OWN handle on first access -> no shared/forked handle, so HDF5
        reads (gzip decode) parallelize across workers."""
        if self.root is not None:
            return
        self.root = h5py.File(self.h5_path, "r")
        self.z_encoder = self.root["encoder"]
        self.z_img1 = self.root["image_top"]
        self.z_img2 = self.root["image_bottom"]
        if self.with_lidar:
            self.z_lidar_points = self.root["lidar_points"]
            self.z_lidar_npoints = self.root["lidar_npoints"]
        if self.with_aux:
            self.z_dock_pose = self.root["dock_pose"]
            self.z_reliable = self.root["reliable"]

        # DINO feature cache (row-aligned 1:1 with the image rows). Opened per
        # worker, same pattern as the image handles above.
        if self.dino_cache_path is not None:
            self.cache_root = h5py.File(self.dino_cache_path, "r")
            self.z_dino2 = self.cache_root["dino_bottom"]   # room2
            if self.use_room1:
                if "dino_top" not in self.cache_root:
                    raise KeyError(
                        "use_room1=True with a DINO cache requires a 'dino_top' dataset "
                        "in the cache; precompute it or set use_room1=False.")
                self.z_dino1 = self.cache_root["dino_top"]

    def __len__(self) -> int:
        return len(self.index_map)

    def __del__(self):
        for attr in ("root", "cache_root"):
            try:
                h = getattr(self, attr, None)
                if h is not None:
                    h.close()
            except Exception:
                pass

    @staticmethod
    def _relative_to_goal(pose: np.ndarray, g: np.ndarray) -> np.ndarray:
        """SE(2) relative pose T_t ∘ inv(T_g) for label rows `pose` [N,3] vs the
        episode's goal reference pose `g` [3]. Its translation is the docked
        (goal) robot pose expressed in the CURRENT robot frame -> exactly the
        "distance + angle to goal" signal (vectorized; frames as in endgame/se2)."""
        pose = np.atleast_2d(pose)
        cg, sg = np.cos(g[2]), np.sin(g[2])
        # inv(T_g) translation and rotation
        tix = -(cg * g[0] + sg * g[1])
        tiy = -(-sg * g[0] + cg * g[1])
        ct, st = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        x = pose[:, 0] + ct * tix - st * tiy
        y = pose[:, 1] + st * tix + ct * tiy
        th = (pose[:, 2] - g[2] + np.pi) % (2 * np.pi) - np.pi
        return np.stack([x, y, th], axis=1)

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
        self._ensure_open()        # per-worker lazy h5 open
        t = self.index_map[idx]
        ep_start = self.ep_start_map[idx]

        # Motion history
        encoder_seq_raw = self._get_history(self.z_encoder, t, ep_start).astype(np.float32)   # [T, 2]
        velocity_seq_norm = self.normalize_action(encoder_seq_raw).astype(np.float32)          # [T, 2]

        # Future target trajectory in normalized encoder space
        act_traj = self.z_encoder[t: t + self.horizon].astype(np.float32)                      # [H, 2]
        act_traj_norm = self.normalize_action(act_traj).astype(np.float32)

        obs = {
            "encoder": torch.from_numpy(encoder_seq_raw).float(),
            "velocity": torch.from_numpy(velocity_seq_norm).float(),
        }

        if self.dino_cache_path is not None:
            # Cached path: return precomputed DINO features (sparse history) instead
            # of raw pixels. Mirror the exact frame selection the live path uses:
            # _get_history(...) -> [::vision_stride]. Features are stored fp16.
            feat2 = self._get_history(self.z_dino2, t, ep_start)[::self.vision_stride]  # [Tv,196,768]
            obs["dino_feat_room2"] = torch.from_numpy(np.ascontiguousarray(feat2))
            if self.use_room1:
                feat1 = self._get_history(self.z_dino1, t, ep_start)[::self.vision_stride]
                obs["dino_feat_room1"] = torch.from_numpy(np.ascontiguousarray(feat1))
        else:
            # Vision histories. sparse_vision_uint8 (training) returns only the sparse
            # frames as uint8 (~24x less RAM); else full float32 history (inference).
            # use_room1=False skips the room1 decode entirely (gzip decode + RAM
            # are wasted otherwise; the single-camera model never reads it).
            if self.sparse_vision_uint8:
                image_room2 = np.ascontiguousarray(self._get_history(self.z_img2, t, ep_start)[::self.vision_stride])
                if self.use_room1:
                    image_room1 = np.ascontiguousarray(self._get_history(self.z_img1, t, ep_start)[::self.vision_stride])
            else:
                image_room2 = self._get_history(self.z_img2, t, ep_start).astype(np.float32) / 255.0  # [T, 3, H, W]
                if self.use_room1:
                    image_room1 = self._get_history(self.z_img1, t, ep_start).astype(np.float32) / 255.0
            obs["image_room2"] = torch.from_numpy(image_room2)
            if self.use_room1:
                obs["image_room1"] = torch.from_numpy(image_room1)

        if self.with_goal:
            # Goal = the episode's docked (final) frame, the ultimate sub-goal.
            gi = int(self.ep_end_map[idx]) - 1
            goal_active = 1.0 if np.random.rand() < self.goal_mask_prob else 0.0
            obs["goal_mask"] = torch.tensor(goal_active, dtype=torch.float32)
            if self.dino_cache_path is not None:
                obs["goal_feat_room2"] = torch.from_numpy(np.ascontiguousarray(self.z_dino2[gi]))  # [196,768]
                if self.use_room1:
                    obs["goal_feat_room1"] = torch.from_numpy(np.ascontiguousarray(self.z_dino1[gi]))
            else:
                if self.sparse_vision_uint8:
                    goal_room2 = np.ascontiguousarray(self.z_img2[gi])   # uint8 [3, H, W]
                else:
                    goal_room2 = self.z_img2[gi].astype(np.float32) / 255.0
                obs["goal_image_room2"] = torch.from_numpy(goal_room2)
                if self.use_room1:
                    if self.sparse_vision_uint8:
                        goal_room1 = np.ascontiguousarray(self.z_img1[gi])
                    else:
                        goal_room1 = self.z_img1[gi].astype(np.float32) / 255.0
                    obs["goal_image_room1"] = torch.from_numpy(goal_room1)

        if self.with_lidar:
            # Current-frame raw points (robot frame), zero-padded to M.
            obs["lidar_points"] = torch.from_numpy(self.z_lidar_points[t].astype(np.float32))
            obs["lidar_npoints"] = torch.tensor(int(self.z_lidar_npoints[t]), dtype=torch.long)

        if self.with_goal_lidar:
            # Goal (docked) frame's scan: the registration reference the condition
            # net aligns the current scan against. Same crop/padding as above.
            gi = int(self.ep_end_map[idx]) - 1
            obs["goal_lidar_points"] = torch.from_numpy(self.z_lidar_points[gi].astype(np.float32))
            obs["goal_lidar_npoints"] = torch.tensor(int(self.z_lidar_npoints[gi]), dtype=torch.long)

        sample = {"obs": obs, "act": torch.from_numpy(act_traj_norm).float()}

        if self.with_aux:
            # ICP dock-pose label (teacher for the precision aux head). [x_n, y_n, sin, cos].
            p = self.z_dock_pose[t].astype(np.float32)
            valid = not np.any(np.isnan(p))
            rel = float(self.z_reliable[t]) if valid else 0.0
            p = np.nan_to_num(p)
            if self.aux_relative:
                # target = current->goal relative pose (distance + angle to goal)
                g = self.goal_pose_ep[self.ep_id_map[idx]]
                if g is None:
                    rel = 0.0
                    p = np.zeros(3, np.float32)
                else:
                    p = self._relative_to_goal(p, g)[0].astype(np.float32)
                xy = (p[:2] - self.rel_xy_mean) / self.rel_xy_std
            else:
                xy = (p[:2] - self.dock_xy_mean) / self.dock_xy_std
            sample["dock_target"] = torch.tensor(
                [xy[0], xy[1], np.sin(p[2]), np.cos(p[2])], dtype=torch.float32)
            sample["reliable"] = torch.tensor(rel, dtype=torch.float32)

        if getattr(self, "adv_weight", False):
            # Raw z-scored advantage components (train loop mixes + gates them).
            # Frames without a valid t/t+H label pair emit zeros -> neutral (w=1).
            ok = bool(self._adv_ok[t])
            sample["adv_prog"] = torch.tensor(self._adv_prog[t] if ok else 0.0, dtype=torch.float32)
            sample["adv_align"] = torch.tensor(self._adv_align[t] if ok else 0.0, dtype=torch.float32)
            sample["adv_term"] = torch.tensor(self._adv_term[t] if ok else 0.0, dtype=torch.float32)
            sample["adv_dock_d"] = torch.tensor(self._adv_dock_d[t], dtype=torch.float32)

        return sample


def denormalize(norm_action, act_scale, act_min):
    return (norm_action + 1.0) / 2.0 * act_scale + act_min
