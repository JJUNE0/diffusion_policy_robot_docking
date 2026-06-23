"""Dataset for the vision precision-docking model (ICP-distillation).

Reads the after_0328 frames + the ICP labels we already generated
(dataset/after_0328/subgoal_labels/<ep>.npz) directly — NO h5 conversion needed.

One sample = (room1 RGB, room2 RGB, LiDAR-BEV image) at a reliable frame  ->  the
ICP dock pose at that frame (the teacher label). Only the ICP-reliable frames are
used, which is exactly the near-dock precision phase.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = "dataset/after_0328/dock"
LABELS = "dataset/after_0328/subgoal_labels"


def _img_table(ep: str, room: str):
    fs = sorted(glob.glob(f"{ep}/image/{room}/*.jpg"))
    ts = np.array([float(os.path.basename(f).rsplit("_", 1)[1][:-4]) for f in fs]) if fs else np.array([])
    return fs, ts


def points_to_bev(points: np.ndarray, range_m: float = 2.0, res: float = 0.02) -> np.ndarray:
    """Rasterize a robot-frame scan into a 1-channel BEV occupancy image (centre = robot)."""
    size = int(round(range_m / res))
    g = np.zeros((size, size), dtype=np.float32)
    if points is not None and len(points):
        c = size // 2
        r = (c - np.round(points[:, 0] / res)).astype(np.int64)
        col = (c - np.round(points[:, 1] / res)).astype(np.int64)
        m = (r >= 0) & (r < size) & (col >= 0) & (col < size)
        g[r[m], col[m]] = 1.0
    return g


class PrecisionDockingFrameDataset(Dataset):
    def __init__(self, img_hw=(96, 128), range_m=2.0, res=0.02,
                 max_episodes=None, crop_dock_m=None):
        self.img_hw = img_hw
        self.range_m = range_m
        self.res = res
        self.crop_dock_m = crop_dock_m         # if set, keep only scan pts within this of dock

        eps = sorted(glob.glob(f"{ROOT}/episode_*_dock"))
        if max_episodes:
            eps = eps[:max_episodes]

        self.index = []                        # (ep, frame_i)
        self.scans, self.imgs, self.labels = {}, {}, {}
        xy = []
        for ep in eps:
            npz = f"{LABELS}/{os.path.basename(ep)}.npz"
            if not os.path.exists(npz):
                continue
            z = np.load(npz)
            pose, rel, ts = z["pose"], z["reliable"], z["ts"]
            scans = [np.array([[p["x"], p["y"]] for p in json.loads(l)["points"]], float)
                     for l in open(f"{ep}/lidar.jsonl").read().splitlines()]
            self.scans[ep] = scans
            self.labels[ep] = (pose, ts)
            self.imgs[ep] = {r: _img_table(ep, r) for r in ("room1", "room2")}
            for i in np.where(rel)[0]:
                if i < len(scans) and not np.any(np.isnan(pose[i])) and len(self.imgs[ep]["room1"][0]):
                    self.index.append((ep, int(i)))
                    xy.append(pose[i, :2])

        xy = np.asarray(xy, dtype=np.float64)
        self.xy_mean = xy.mean(0).astype(np.float32)
        self.xy_std = (xy.std(0) + 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def _load_img(self, fs, ts_arr, t):
        j = int(np.abs(ts_arr - t).argmin())
        im = Image.open(fs[j]).convert("RGB").resize((self.img_hw[1], self.img_hw[0]))
        return torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)

    def denorm_xy(self, xy_norm):
        return xy_norm * self.xy_std + self.xy_mean

    def __getitem__(self, k):
        ep, i = self.index[k]
        pose, ts = self.labels[ep]
        t = float(ts[i])
        fs1, t1 = self.imgs[ep]["room1"]
        fs2, t2 = self.imgs[ep]["room2"]
        r1 = self._load_img(fs1, t1, t)
        r2 = self._load_img(fs2, t2, t)

        scan = self.scans[ep][i]
        if self.crop_dock_m is not None:
            d = np.hypot(scan[:, 0] - pose[i, 0], scan[:, 1] - pose[i, 1])
            scan = scan[d < self.crop_dock_m]
        bev = torch.from_numpy(points_to_bev(scan, self.range_m, self.res))[None]

        x = pose[i]
        xy = (x[:2] - self.xy_mean) / self.xy_std
        target = torch.tensor([xy[0], xy[1], np.sin(x[2]), np.cos(x[2])], dtype=torch.float32)
        return {"room1": r1, "room2": r2, "bev": bev, "target": target}
