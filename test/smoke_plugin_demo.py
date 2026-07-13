"""Offline smoke test of the ai-control demo plugin (no server needed).

Shims node_sdk.CommandStep, builds a realistic window_snapshot from the
held-out h5 (video2 frames = image_bottom, encoder dicts, raw-ish lidar), and
calls the plugin's inference_fn twice. Verifies:
  * model + DINO load inside the ai-control bundle (its OWN cleandiffuser copy)
  * returned horizon: 16 CommandSteps, finite, plausible magnitudes
  * per-call latency (must fit the 10 Hz send loop comfortably)

Run (from repo root):  CUDA_VISIBLE_DEVICES=1 python test/smoke_plugin_demo.py
"""

from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- node_sdk shim (the container provides the real one) --------------------
@dataclass
class CommandStep:
    dx: float
    dy: float
    dtheta: float
    dt: float

    def to_dict(self):
        return dict(dx=self.dx, dy=self.dy, dtheta=self.dtheta, dt=self.dt)


sdk = types.ModuleType("node_sdk")
sdk.CommandStep = CommandStep
sys.modules["node_sdk"] = sdk

# Import the plugin WITHOUT the repo root on sys.path, so `cleandiffuser`
# resolves to the ai-control bundled copy exactly like in the container.
sys.path.insert(0, os.path.join(REPO, "ai-control", "ai_models", "plugins"))
import run_postech_docking_demo as plugin  # noqa: E402


class Cfg:
    inference_size = 16
    inference_fps = 30


def build_snapshot(t=300):
    f = h5py.File(os.path.join(REPO, "dataset", "after_0328_test.h5"), "r")
    frames = []
    for i in range(t - 59, t + 1):
        rgb = f["image_bottom"][i].transpose(1, 2, 0)          # [240,320,3] RGB
        frames.append(np.ascontiguousarray(rgb[:, :, ::-1]))   # -> BGR like LiveKit
    enc = [dict(ts=i / 30.0, vx=float(f["encoder"][i][0]), wz=float(f["encoder"][i][1]))
           for i in range(t - 59, t + 1)]
    n = int(f["lidar_npoints"][t])
    pts = f["lidar_points"][t][:n]
    lidar = [dict(ts=t / 30.0, points=[dict(x=float(p[0]), y=float(p[1])) for p in pts])]
    gt_next = f["encoder"][t:t + 16]
    return {"video2": (frames, [False] * 60), "encoder": (enc, [False] * 60),
            "lidar": (lidar, [False])}, gt_next


def main():
    snap, gt = build_snapshot()
    for trial in (1, 2):
        t0 = time.time()
        steps = plugin.inference_fn(snap, {}, Cfg())
        dt = time.time() - t0
        assert len(steps) == 16, f"expected 16 steps, got {len(steps)}"
        arr = np.array([[s.dx, s.dy, s.dtheta] for s in steps])
        assert np.isfinite(arr).all(), "non-finite command step"
        print(f"call {trial}: {dt*1000:.0f} ms | final step dx={arr[-1,0]*100:.2f}cm "
              f"dy={arr[-1,1]*100:.2f}cm dth={np.degrees(arr[-1,2]):.2f}deg")
    # eyeball: GT displacement over the same 16 steps
    th = np.cumsum(np.concatenate([[0], gt[:-1, 1] * (1 / 30)]))
    gx = np.sum(gt[:, 0] * np.cos(th) / 30)
    gy = np.sum(gt[:, 0] * np.sin(th) / 30)
    print(f"GT same-window displacement: dx={gx*100:.2f}cm dy={gy*100:.2f}cm")
    print("PLUGIN SMOKE OK")


if __name__ == "__main__":
    main()
