import os
import json
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
import h5py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Lidar BEV occupancy-map utilities
# ============================================================
def points_to_occupancy_map(
    points_xy: np.ndarray,
    range_m: float = 6.0,
    resolution: float = 0.05,
    n_channels: int = 2,
    density_clip: float = 20.0,
) -> np.ndarray:
    """
    Rasterize a 2D lidar scan (in robot frame) into a BEV occupancy map.

    Convention
    ----------
    * Robot is at the center pixel.
    * +x (forward) -> image up   (decreasing row).
    * +y (left)    -> image left (decreasing col).
    * Output is uint8 with shape (C, H, W); H = W = round(range_m / resolution).

    Channels (when n_channels == 2)
    -------------------------------
    * 0 : binary occupancy mask, 0 or 255.
    * 1 : log-density of hits per cell, scaled to 0..255 via log1p / log1p(density_clip).

    Notes
    -----
    * `range_m` is the full span in meters along one axis (e.g. 6.0 -> robot sees +/- 3.0 m).
    * Out-of-range points are silently dropped.
    * Returning uint8 keeps storage cheap and matches the image branch.
    """
    if n_channels not in (1, 2):
        raise ValueError(f"n_channels must be 1 or 2, got {n_channels}")

    size = int(round(range_m / resolution))
    if size <= 0:
        raise ValueError(f"Invalid range/resolution -> size={size}")

    cy = size // 2
    cx = size // 2

    occ = np.zeros((size, size), dtype=np.float32)

    if points_xy is not None and len(points_xy) > 0:
        x = points_xy[:, 0]
        y = points_xy[:, 1]

        rows = (cy - np.round(x / resolution)).astype(np.int32)
        cols = (cx - np.round(y / resolution)).astype(np.int32)

        valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
        rows = rows[valid]
        cols = cols[valid]

        if len(rows) > 0:
            np.add.at(occ, (rows, cols), 1.0)

    occupied_u8 = (occ > 0).astype(np.uint8) * 255

    if n_channels == 1:
        return occupied_u8[None]

    log_dens = np.log1p(occ) / np.log1p(density_clip)
    log_dens_u8 = np.clip(log_dens * 255.0, 0, 255).astype(np.uint8)

    return np.stack([occupied_u8, log_dens_u8], axis=0)


def _load_lidar_jsonl(lidar_path: str):
    """
    Returns
    -------
    timestamps : np.ndarray of shape (N,) or empty
    scans      : list[np.ndarray]  each (M_i, 2) in float32
    """
    if not os.path.exists(lidar_path):
        return np.array([], dtype=np.float64), []

    timestamps = []
    scans = []
    with open(lidar_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = d.get("ts", None)
            pts = d.get("points", [])
            if ts is None:
                continue

            if len(pts) > 0:
                arr = np.asarray(
                    [(p["x"], p["y"]) for p in pts],
                    dtype=np.float32,
                )
            else:
                arr = np.zeros((0, 2), dtype=np.float32)

            timestamps.append(float(ts))
            scans.append(arr)

    timestamps = np.asarray(timestamps, dtype=np.float64)

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    scans = [scans[i] for i in order]
    return timestamps, scans


# ============================================================
# Episode -> synced metadata
# ============================================================
class StrictSyncRoboticsDataset:
    def __init__(
        self,
        root_dir,
        target_hz=30.0,
        max_time_diff=0.05,
        use_lidar: bool = False,
        lidar_max_time_diff: float = 0.10,
        lidar_range_m: float = 6.0,
        lidar_resolution: float = 0.05,
        lidar_channels: int = 2,
        lidar_format: str = "points",
        lidar_crop_r: float = 0.8,
        lidar_max_points: int = 256,
        use_labels: bool = False,
        labels_dir: str = None,
        max_episodes: int = None,
        episode_start: int = 0,
        use_room1: bool = True,
    ):
        self.root_dir = root_dir
        # Single-camera datasets (only image/room2 recorded) -> skip the room1
        # branch entirely and don't write `image_top`. Matches the training-side
        # `use_room1: false`, and halves the h5.
        self.use_room1 = use_room1
        self.max_episodes = max_episodes
        self.episode_start = episode_start
        self.target_interval = 1.0 / target_hz
        self.max_time_diff = max_time_diff
        self.use_lidar = use_lidar
        self.lidar_max_time_diff = lidar_max_time_diff
        self.lidar_range_m = lidar_range_m
        self.lidar_resolution = lidar_resolution
        self.lidar_channels = lidar_channels
        self.lidar_size = int(round(lidar_range_m / lidar_resolution))
        # LiDAR input format. "points" (default) stores RAW points cropped to the
        # nearest-cluster region (no rasterization -> mm-preserving, efficient,
        # online-consistent). "bev" keeps the occupancy-map path (ablation).
        self.lidar_format = lidar_format
        self.lidar_crop_r = lidar_crop_r
        self.lidar_max_points = lidar_max_points
        # Per-frame ICP dock-pose labels (Option A precision aux head). Indexed by
        # the lidar frame, so use_labels requires use_lidar. Generated offline by
        # scripts/label_subgoals.py -> <labels_dir>/<ep_name>.npz (pose, reliable).
        self.use_labels = use_labels
        self.labels_dir = labels_dir or os.path.join(root_dir, "icp_labels")
        self._label_cache: dict[str, tuple] = {}
        if use_labels and not use_lidar:
            raise ValueError("use_labels requires use_lidar (labels are indexed by lidar frame).")

        self.samples = []
        self.episode_ends = []

        # Cache lidar per episode so get_full_sample() can rasterize on-demand.
        # key = ep_path, value = (timestamps[N], scans[list[(M,2)]])
        self._lidar_cache: dict[str, tuple] = {}

        episode_folders = self._collect_episode_folders()
        if self.episode_start:
            episode_folders = episode_folders[self.episode_start:]
        if self.max_episodes is not None:
            episode_folders = episode_folders[:self.max_episodes]

        current_idx = 0
        for ep_path, ep_name in episode_folders:
            kept_frames = self._process_episode_metadata(ep_path, ep_name)

            if kept_frames > 0:
                current_idx += kept_frames
                self.episode_ends.append(current_idx)

        logger.info(
            f"메타데이터 구축 완료! 총 {len(self.samples)}개의 동기화된 프레임이 식별되었습니다."
            + (f" (use_lidar=True, map={self.lidar_channels}x{self.lidar_size}x{self.lidar_size})"
               if self.use_lidar else "")
        )

    def _lookup_label(self, ep_name, lidar_idx):
        """Return (dock_pose[3] float32, reliable uint8) for one lidar frame.

        Missing label / unreliable / NaN -> NaN pose + reliable=0 (the training
        aux loss masks these out).
        """
        if ep_name not in self._label_cache:
            # ep_name can carry a "dock/" prefix; labels are keyed by leaf name.
            path = os.path.join(self.labels_dir, f"{os.path.basename(ep_name)}.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._label_cache[ep_name] = (z["pose"], z["reliable"])
            else:
                self._label_cache[ep_name] = None
        cache = self._label_cache[ep_name]
        if cache is None or lidar_idx >= len(cache[0]):
            return np.full(3, np.nan, np.float32), np.uint8(0)
        pose = cache[0][lidar_idx].astype(np.float32)
        ok = bool(cache[1][lidar_idx]) and not np.any(np.isnan(pose))
        return pose, np.uint8(1 if ok else 0)

    def _collect_episode_folders(self):
        """
        root_dir 아래의 후보 상위 폴더들을 순회하여 실제 episode 경로 목록 반환.
        """
        candidate_parents = [
            # os.path.join(self.root_dir, "episode"),
            os.path.join(self.root_dir, "dock"),
            # os.path.join(self.root_dir, "valid"),
            # os.path.join(self.root_dir, "valid2"),
            # os.path.join(self.root_dir, "valid3"),
            # os.path.join(self.root_dir, "validation"),
        ]

        collected = []

        for parent in candidate_parents:
            if not os.path.exists(parent):
                logger.warning(f"상위 폴더가 없습니다. 스킵합니다: {parent}")
                continue

            subdirs = sorted([
                d for d in os.listdir(parent)
                if os.path.isdir(os.path.join(parent, d))
            ])

            for d in subdirs:
                ep_path = os.path.join(parent, d)
                ep_name = f"{os.path.basename(parent)}/{d}"
                collected.append((ep_path, ep_name))

        logger.info(f"총 {len(collected)}개의 episode 폴더를 발견했습니다.")
        return collected

    def _get_image_timestamps(self, img_dir):
        if not os.path.exists(img_dir):
            return np.array([]), []

        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        timestamps, valid_files = [], []

        for f in files:
            try:
                ts = float(f.replace('.jpg', '').split('_')[-1])
                timestamps.append(ts)
                valid_files.append(os.path.join(img_dir, f))
            except (ValueError, IndexError):
                continue

        return np.array(timestamps), valid_files

    def _process_episode_metadata(self, ep_path, ep_name):
        """실제 이미지/라이다는 로드하지 않고 경로/타임스탬프만 매칭하여 self.samples에 추가."""
        enc_path = os.path.join(ep_path, 'encoder.csv')
        img_dir_top = os.path.join(ep_path, 'image', 'room1')
        img_dir_bottom = os.path.join(ep_path, 'image', 'room2')
        lidar_path = os.path.join(ep_path, 'lidar.jsonl')

        if not os.path.exists(enc_path):
            logger.warning(f"❌ [{ep_name}] 스킵: encoder.csv 파일이 없습니다. (경로: {enc_path})")
            return 0
        if self.use_room1 and not os.path.exists(img_dir_top):
            logger.warning(f"❌ [{ep_name}] 스킵: image/room1 폴더가 없습니다.")
            return 0
        if not os.path.exists(img_dir_bottom):
            logger.warning(f"❌ [{ep_name}] 스킵: image/room2 폴더가 없습니다.")
            return 0
        if self.use_lidar and not os.path.exists(lidar_path):
            logger.warning(f"❌ [{ep_name}] 스킵: lidar.jsonl 파일이 없습니다.")
            return 0

        df_enc = pd.read_csv(enc_path)

        required_cols = ['ts', 'vx', 'wz']
        for col in required_cols:
            if col not in df_enc.columns:
                logger.warning(f"❌ [{ep_name}] 스킵: encoder.csv에 '{col}' 컬럼이 없습니다.")
                return 0

        ts_bot, files_bot = self._get_image_timestamps(img_dir_bottom)
        if self.use_room1:
            ts_top, files_top = self._get_image_timestamps(img_dir_top)
        else:
            # room2 stands in for room1 in every timestamp/window computation
            # below; nothing is read from it (see get_full_sample).
            ts_top, files_top = ts_bot, files_bot

        if len(ts_top) == 0 or len(ts_bot) == 0:
            logger.warning(f"❌ [{ep_name}] 스킵: room1 또는 room2 이미지가 비어 있습니다.")
            return 0

        if self.use_lidar:
            ts_lidar, scans_lidar = _load_lidar_jsonl(lidar_path)
            if len(ts_lidar) == 0:
                logger.warning(f"❌ [{ep_name}] 스킵: lidar.jsonl이 비어 있거나 파싱 실패.")
                return 0
            self._lidar_cache[ep_path] = (ts_lidar, scans_lidar)
        else:
            ts_lidar = np.array([], dtype=np.float64)

        common_max = [df_enc['ts'].max(), ts_top.max(), ts_bot.max()]
        common_min = [df_enc['ts'].min(), ts_top.min(), ts_bot.min()]
        if self.use_lidar:
            common_max.append(ts_lidar.max())
            common_min.append(ts_lidar.min())

        min_t = max(common_min)
        max_t = min(common_max)

        if min_t >= max_t:
            logger.warning(f"❌ [{ep_name}] 스킵: 유효한 공통 시간 구간이 없습니다.")
            return 0

        target_ts = np.arange(min_t, max_t, self.target_interval)

        interp_enc = interp1d(
            df_enc['ts'],
            df_enc[['vx', 'wz']].values,
            axis=0,
            kind='linear',
            fill_value="extrapolate"
        )

        synced_enc = interp_enc(target_ts)

        count = 0
        for i, t in enumerate(target_ts):
            idx_top = np.abs(ts_top - t).argmin()
            if abs(ts_top[idx_top] - t) > self.max_time_diff:
                continue

            idx_bot = np.abs(ts_bot - t).argmin()
            if abs(ts_bot[idx_bot] - t) > self.max_time_diff:
                continue

            sample = {
                'encoder': synced_enc[i].astype(np.float32),
                'img_top_path': files_top[idx_top],
                'img_bottom_path': files_bot[idx_bot],
            }

            if self.use_lidar:
                idx_lid = int(np.abs(ts_lidar - t).argmin())
                if abs(ts_lidar[idx_lid] - t) > self.lidar_max_time_diff:
                    continue
                sample['lidar_ep_path'] = ep_path
                sample['lidar_idx'] = idx_lid

                if self.use_labels:
                    dock_pose, reliable = self._lookup_label(ep_name, idx_lid)
                    sample['dock_pose'] = dock_pose
                    sample['reliable'] = reliable

            self.samples.append(sample)
            count += 1

        logger.info(f"[{ep_name}] 동기화 완료: {count} 프레임")
        return count

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((3, 240, 320), dtype=np.uint8)

        if img.shape[:2] != (240, 320):
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb.transpose(2, 0, 1)

    def _make_lidar_map(self, ep_path: str, scan_idx: int) -> np.ndarray:
        ts_lidar, scans_lidar = self._lidar_cache[ep_path]
        pts = scans_lidar[scan_idx]
        return points_to_occupancy_map(
            pts,
            range_m=self.lidar_range_m,
            resolution=self.lidar_resolution,
            n_channels=self.lidar_channels,
        )

    def _make_lidar_points(self, ep_path: str, scan_idx: int):
        """Raw points cropped to the nearest-cluster region, padded to M.

        Crop = points within ``lidar_crop_r`` of the nearest scan point (= the
        dock when close). Deterministic, no ICP -> the same crop is reproducible
        online. Returns (points[M,2] float32 zero-padded, n_valid).
        """
        _, scans_lidar = self._lidar_cache[ep_path]
        pts = scans_lidar[scan_idx]
        out = np.zeros((self.lidar_max_points, 2), dtype=np.float32)
        if pts is None or len(pts) == 0:
            return out, 0
        d0 = np.hypot(pts[:, 0], pts[:, 1])
        nearest = pts[d0.argmin()]
        dn = np.hypot(pts[:, 0] - nearest[0], pts[:, 1] - nearest[1])
        keep = pts[dn < self.lidar_crop_r]
        if len(keep) == 0:
            keep = pts[[d0.argmin()]]
        if len(keep) > self.lidar_max_points:           # keep the M closest
            dk = np.hypot(keep[:, 0] - nearest[0], keep[:, 1] - nearest[1])
            keep = keep[np.argsort(dk)[:self.lidar_max_points]]
        n = len(keep)
        out[:n] = keep.astype(np.float32)
        return out, n

    def __len__(self):
        return len(self.samples)

    def get_full_sample(self, idx):
        """저장 시 한 프레임의 모든 데이터를 로드/생성."""
        s = self.samples[idx]
        img_top = self._load_image(s['img_top_path']) if self.use_room1 else None
        img_bot = self._load_image(s['img_bottom_path'])

        if self.use_lidar:
            if self.lidar_format == "points":
                lidar_obj = self._make_lidar_points(s['lidar_ep_path'], s['lidar_idx'])
            else:
                lidar_obj = self._make_lidar_map(s['lidar_ep_path'], s['lidar_idx'])
            return s['encoder'], img_top, img_bot, lidar_obj

        return s['encoder'], img_top, img_bot


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw episode data into HDF5 format")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing dock/ folder with episode data")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output HDF5 file path (e.g. ./dataset.h5)")
    parser.add_argument("--target_hz", type=float, default=30.0)
    parser.add_argument("--max_time_diff", type=float, default=0.05,
                        help="Max time mismatch (s) between target and image timestamp.")

    parser.add_argument("--use_lidar", action="store_true",
                        help="Also rasterize lidar.jsonl into BEV occupancy maps and store under 'lidar_map'.")
    parser.add_argument("--lidar_max_time_diff", type=float, default=0.10,
                        help="Max time mismatch (s) between target and lidar scan timestamp (lidar ~15Hz).")
    parser.add_argument("--lidar_range_m", type=float, default=6.0,
                        help="Full span in meters of the BEV map along one axis (robot at center).")
    parser.add_argument("--lidar_resolution", type=float, default=0.05,
                        help="Meters per pixel for the BEV map.")
    parser.add_argument("--lidar_channels", type=int, default=2, choices=(1, 2),
                        help="(bev format) 1 = occupancy only, 2 = [occupancy, log-density].")
    parser.add_argument("--lidar_format", type=str, default="points", choices=("points", "bev"),
                        help="points = raw points cropped to nearest cluster (default); bev = occupancy map (ablation).")
    parser.add_argument("--lidar_crop_r", type=float, default=0.8,
                        help="(points) keep scan points within this radius of the nearest point.")
    parser.add_argument("--lidar_max_points", type=int, default=256,
                        help="(points) pad/truncate each frame to this many points.")

    parser.add_argument("--with_labels", action="store_true",
                        help="Also store per-frame ICP dock-pose labels (requires --use_lidar). "
                             "Reads <labels_dir>/<ep_name>.npz from scripts/label_subgoals.py.")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Dir of ICP label npz (default: <data_root>/icp_labels).")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Process only N episodes (debug/verification/held-out).")
    parser.add_argument("--episode_start", type=int, default=0,
                        help="Skip the first K episodes (e.g. 30 -> held-out set after training on 0..29).")
    parser.add_argument("--no_room1", action="store_true",
                        help="Single-camera dataset: only image/room2 exists. Skips the room1 branch "
                             "and does not write `image_top` (halves the h5). Pair with use_room1=false.")

    cli_args = parser.parse_args()

    DATA_ROOT_DIR = cli_args.data_root
    SAVE_PATH = cli_args.save_path

    dataset = StrictSyncRoboticsDataset(
        root_dir=DATA_ROOT_DIR,
        target_hz=cli_args.target_hz,
        max_time_diff=cli_args.max_time_diff,
        use_lidar=cli_args.use_lidar,
        lidar_max_time_diff=cli_args.lidar_max_time_diff,
        lidar_range_m=cli_args.lidar_range_m,
        lidar_resolution=cli_args.lidar_resolution,
        lidar_channels=cli_args.lidar_channels,
        lidar_format=cli_args.lidar_format,
        lidar_crop_r=cli_args.lidar_crop_r,
        lidar_max_points=cli_args.lidar_max_points,
        use_labels=cli_args.with_labels,
        labels_dir=cli_args.labels_dir,
        max_episodes=cli_args.max_episodes,
        episode_start=cli_args.episode_start,
        use_room1=not cli_args.no_room1,
    )
    total_samples = len(dataset)

    if total_samples == 0:
        raise RuntimeError("동기화된 프레임이 0개입니다. 입력 경로/옵션을 확인하세요.")

    print(f"\n데이터 저장을 시작합니다: {SAVE_PATH}")
    with h5py.File(SAVE_PATH, 'w') as f:
        dset_enc = f.create_dataset("encoder", (total_samples, 2), dtype='f4')
        dset_img_top = None if cli_args.no_room1 else f.create_dataset(
            "image_top",
            (total_samples, 3, 240, 320),
            dtype='u1',
            compression="gzip",
            chunks=(1, 3, 240, 320)
        )
        dset_img_bot = f.create_dataset(
            "image_bottom",
            (total_samples, 3, 240, 320),
            dtype='u1',
            compression="gzip",
            chunks=(1, 3, 240, 320)
        )
        f.create_dataset("episode_ends", data=np.array(dataset.episode_ends, dtype='i8'))

        points_mode = cli_args.use_lidar and cli_args.lidar_format == "points"
        if points_mode:
            M = cli_args.lidar_max_points
            dset_lpts = f.create_dataset("lidar_points", (total_samples, M, 2), dtype='f4',
                                         compression="gzip", chunks=(1, M, 2))
            dset_lnp = f.create_dataset("lidar_npoints", (total_samples,), dtype='i4')
            dset_lpts.attrs["crop_r"] = cli_args.lidar_crop_r
            dset_lpts.attrs["max_points"] = M
            dset_lpts.attrs["desc"] = "raw scan points (robot frame), cropped to nearest-cluster region, zero-padded"
        elif cli_args.use_lidar:
            C = cli_args.lidar_channels
            S = dataset.lidar_size
            dset_lidar = f.create_dataset(
                "lidar_map",
                (total_samples, C, S, S),
                dtype='u1',
                compression="gzip",
                chunks=(1, C, S, S),
            )
            dset_lidar.attrs["range_m"] = cli_args.lidar_range_m
            dset_lidar.attrs["resolution"] = cli_args.lidar_resolution
            dset_lidar.attrs["channels"] = C
            dset_lidar.attrs["size"] = S
            dset_lidar.attrs["channel_order"] = (
                "occupied" if C == 1 else "occupied,log_density"
            )

        if cli_args.with_labels:
            dset_pose = f.create_dataset("dock_pose", (total_samples, 3), dtype='f4')
            dset_rel = f.create_dataset("reliable", (total_samples,), dtype='u1')
            dset_pose.attrs["desc"] = "ICP dock pose [x,y,theta] (sensor frame); NaN where unreliable"
            dset_rel.attrs["desc"] = "1 = dock_pose is a valid ICP label (precision aux loss mask)"

        for i in tqdm(range(total_samples), desc="Disk Writing"):
            sample = dataset.get_full_sample(i)
            if cli_args.use_lidar:
                enc, top, bot, lidar_obj = sample
                if points_mode:
                    lpts, npts = lidar_obj
                    dset_lpts[i] = lpts
                    dset_lnp[i] = npts
                else:
                    dset_lidar[i] = lidar_obj
            else:
                enc, top, bot = sample

            dset_enc[i] = enc
            if dset_img_top is not None:
                dset_img_top[i] = top
            dset_img_bot[i] = bot
            if cli_args.with_labels:
                dset_pose[i] = dataset.samples[i]["dock_pose"]
                dset_rel[i] = dataset.samples[i]["reliable"]

    print(f"\n모든 데이터가 안전하게 저장되었습니다! (최종 프레임: {total_samples})")
