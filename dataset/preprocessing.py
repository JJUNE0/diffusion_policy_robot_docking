import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
import h5py  # HDF5 포맷 저장을 위해 필요

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class StrictSyncRoboticsDataset:
    def __init__(self, root_dir, target_hz=30.0, max_time_diff=0.05):
        self.root_dir = root_dir
        self.target_interval = 1.0 / target_hz
        self.max_time_diff = max_time_diff
        self.samples = []
        self.episode_ends = []

        # 이번에는 root_dir/episode, root_dir/episode_postech 둘 다 순회
        episode_folders = self._collect_episode_folders()

        current_idx = 0
        for ep_path, ep_name in episode_folders:
            kept_frames = self._process_episode_metadata(ep_path, ep_name)

            if kept_frames > 0:
                current_idx += kept_frames
                self.episode_ends.append(current_idx)

        logger.info(f"메타데이터 구축 완료! 총 {len(self.samples)}개의 동기화된 프레임이 식별되었습니다.")

    def _collect_episode_folders(self):
        """
        root_dir 아래의
          - episode/
          - episode_postech/
        두 폴더를 모두 확인하고,
        그 안의 실제 episode 폴더 경로를 모아 반환
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
                # 이름 충돌 방지를 위해 상위 폴더명까지 포함
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
        """실제 이미지를 로드하지 않고 경로와 센서 데이터만 매칭하여 리스트에 저장"""
        enc_path = os.path.join(ep_path, 'encoder.csv')
        img_dir_top = os.path.join(ep_path, 'image', 'room1')
        img_dir_bottom = os.path.join(ep_path, 'image', 'room2')

        if not os.path.exists(enc_path):
            logger.warning(f"❌ [{ep_name}] 스킵: encoder.csv 파일이 없습니다. (경로: {enc_path})")
            return 0
        if not os.path.exists(img_dir_top):
            logger.warning(f"❌ [{ep_name}] 스킵: image/room1 폴더가 없습니다.")
            return 0
        if not os.path.exists(img_dir_bottom):
            logger.warning(f"❌ [{ep_name}] 스킵: image/room2 폴더가 없습니다.")
            return 0

        df_enc = pd.read_csv(enc_path)

        # 필수 컬럼 체크
        required_cols = ['ts', 'vx', 'wz']
        for col in required_cols:
            if col not in df_enc.columns:
                logger.warning(f"❌ [{ep_name}] 스킵: encoder.csv에 '{col}' 컬럼이 없습니다.")
                return 0

        ts_top, files_top = self._get_image_timestamps(img_dir_top)
        ts_bot, files_bot = self._get_image_timestamps(img_dir_bottom)

        if len(ts_top) == 0 or len(ts_bot) == 0:
            logger.warning(f"❌ [{ep_name}] 스킵: room1 또는 room2 이미지가 비어 있습니다.")
            return 0

        min_t = max(df_enc['ts'].min(), ts_top.min(), ts_bot.min())
        max_t = min(df_enc['ts'].max(), ts_top.max(), ts_bot.max())

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

            self.samples.append({
                'encoder': synced_enc[i].astype(np.float32),
                'img_top_path': files_top[idx_top],
                'img_bottom_path': files_bot[idx_bot]
            })
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

    def __len__(self):
        return len(self.samples)

    def get_full_sample(self, idx):
        """파일 저장을 위해 한 프레임의 모든 데이터를 로드"""
        s = self.samples[idx]
        img_top = self._load_image(s['img_top_path'])
        img_bot = self._load_image(s['img_bottom_path'])
        return s['encoder'], img_top, img_bot


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw episode data into HDF5 format")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing validation/ folder with episode data")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output HDF5 file path (e.g. ./validation_dataset.h5)")
    cli_args = parser.parse_args()

    DATA_ROOT_DIR = cli_args.data_root
    SAVE_PATH = cli_args.save_path

    dataset = StrictSyncRoboticsDataset(root_dir=DATA_ROOT_DIR)
    total_samples = len(dataset)

    print(f"\n데이터 저장을 시작합니다: {SAVE_PATH}")
    with h5py.File(SAVE_PATH, 'w') as f:
        dset_enc = f.create_dataset("encoder", (total_samples, 2), dtype='f4')
        dset_img_top = f.create_dataset(
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

        for i in tqdm(range(total_samples), desc="Disk Writing"):
            enc, top, bot = dataset.get_full_sample(i)
            dset_enc[i] = enc
            dset_img_top[i] = top
            dset_img_bot[i] = bot

    print(f"\n모든 데이터가 안전하게 저장되었습니다! (최종 프레임: {total_samples})")