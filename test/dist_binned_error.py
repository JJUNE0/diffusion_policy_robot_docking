"""Distance-binned aux dock-pose error analysis (user request 2026-07-10).

The training/eval `aux_pose_mm` metric averages over ALL reliable frames of the
approach (dock pose distance 0.5-1.6 m), so a single "29 mm" number can hide a
much better near-dock precision. This script re-evaluates the aux head of the
trained checkpoints and bins the per-frame error by dock distance.

I/O note: DockingDataset reads the full 30-row DINO history per sample and then
strides (~9 MB/sample) which is far too slow for a 4-variant sweep, so this
script reads only the 5 strided rows (+ goal row) directly from the cache h5
and runs all model variants on the same batch in ONE data pass.

Caveat: only the packed train h5 exists on this machine (no held-out raw
episodes), so the numbers are TRAIN-set errors -> optimistic, but the *shape*
of the error-vs-distance curve is what we are after.

Run (from repo root):  python test/dist_binned_error.py
Outputs: test/out/dist_binned_error.png, test/out/dist_binned_error.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cleandiffuser.nn_condition.sensor_fusion_condition import SensorFusionConditionNetwork  # noqa: E402

# Env-overridable so the same tooling scores held-out data:
#   EVAL_H5 / EVAL_CACHE  -> which data to read
#   EVAL_STATS_H5         -> which data defines the normalization space
# Stats must stay the TRAIN h5 even when EVAL_H5 is the test h5: the model
# learned in train-stat space, so targets must be normalized with train stats
# (same rule as scripts/eval_heldout.py).
H5 = os.environ.get("EVAL_H5", "dataset/after_0328_train.h5")
DINO_CACHE = os.environ.get("EVAL_CACHE", "dataset/after_0328_train_dino_bottom.h5")
STATS_H5 = os.environ.get("EVAL_STATS_H5", "dataset/after_0328_train.h5")
RUNS = {
    "flow": "outputs/train/flow_goal/2026-07-09_12-02-10/checkpoint_step_4230.pt",
    "auxw": "outputs/train/flow_goal_auxw/2026-07-10_15-36-56/checkpoint_step_4230.pt",
}
# Contiguous-block sampling: random single rows force one 301 KB chunk read per
# DINO row (~20 ms each on this disk -> 16+ min); a contiguous slice reads at
# ~18x that rate. 40 blocks x 256 frames still spans many episodes/distances.
N_BLOCKS = 40
BLOCK = 256
OBS_HORIZON, VISION_STRIDE, HORIZON = 30, 6, 60
BIN_EDGES = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.65])
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def build_condition_net() -> SensorFusionConditionNetwork:
    """Mirror utils/setups.py model_setups() for the saved run configs
    (use_room1=False, goal+lidar+aux, d_model=384/nhead=6, num_layers default 2)."""
    return SensorFusionConditionNetwork(
        state_dim=2, obs_horizon=OBS_HORIZON, vision_horizon=5,
        d_model=384, nhead=6, num_layers=2, dropout=0.1,
        num_image_latents=16, velocity_dim=2, velocity_dropout_prob=0.0,
        use_goal=True, num_goal_latents=16,
        use_lidar_points=True, num_lidar_latents=16,
        use_aux_pose=True, use_room1=False,
    )


def load_condition_weights(net, ckpt_path: str, which: str):
    """Extract the `condition.*` sub-tree from the saved joint state dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = {"raw": "model_state_dict", "ema": "ema_state_dict"}[which]
    sd = {k[len("condition."):]: v for k, v in ckpt[key].items() if k.startswith("condition.")}
    net.load_state_dict(sd, strict=True)


class H5Batcher:
    """Lean batch reader mirroring DockingDataset's frame selection/normalization,
    but reading only the strided DINO rows (6x less I/O).

    aux_relative=True mirrors DockingDataset(aux_relative=True): the emitted aux
    target is the current->goal SE(2) relative pose (distance+angle to THIS
    file's own per-episode goal frame), normalized with the STATS file's
    relative-pose stats — not the absolute dock pose. Required for any run
    trained with use_goal_lidar/aux_relative (see utils/docking_dataset.py),
    otherwise pred and target live in different normalized spaces and the mm
    error is meaningless (caught 2026-07-13 evaluating flow_goal_glidar).
    """

    def __init__(self, aux_relative: bool = False):
        f = h5py.File(H5, "r")
        self.f = f
        self.enc = f["encoder"][:].astype(np.float32)           # [N,2] small, preload
        self.episode_ends = f["episode_ends"][:]
        pose = f["dock_pose"][:].astype(np.float32)
        rel = f["reliable"][:].astype(bool)
        self.pose, self.rel_valid = np.nan_to_num(pose), rel & ~np.isnan(pose).any(axis=1)
        self.z_lidar = f["lidar_points"]
        self.z_nlidar = f["lidar_npoints"]
        self.cf = h5py.File(DINO_CACHE, "r")
        self.z_dino = self.cf["dino_bottom"]

        # Normalization space = the STATS file (train h5): dock-pose mean/std and
        # encoder min/max the model learned with — not the eval file's own stats.
        with h5py.File(STATS_H5, "r") as sf:
            s_pose = sf["dock_pose"][:].astype(np.float32)
            s_rel = sf["reliable"][:].astype(bool)
            s_xy = s_pose[s_rel & ~np.isnan(s_pose).any(axis=1)][:, :2]
            self.dock_xy_mean = s_xy.mean(0)
            self.dock_xy_std = s_xy.std(0) + 1e-6
            s_enc = sf["encoder"][:].astype(np.float32)
        self.a_min = s_enc.min(0)
        self.a_scale = np.clip(s_enc.max(0) - self.a_min, 1e-5, None)

        # per-row episode start/end (same index constraint as the dataset)
        n = int(self.episode_ends[-1])
        self.n_rows = n
        self.ep_start = np.zeros(n, np.int64)
        self.ep_end = np.zeros(n, np.int64)
        self.ep_id = np.zeros(n, np.int64)
        s = 0
        self.ok = np.zeros(n, bool)
        for ep_i, e in enumerate(self.episode_ends):
            self.ep_start[s:e] = s
            self.ep_end[s:e] = e
            self.ep_id[s:e] = ep_i
            self.ok[s:e - HORIZON + 1] = True
            s = e

        self.aux_relative = aux_relative
        if aux_relative:
            from utils.docking_dataset import DockingDataset
            # rel-pose normalization stats, computed from STATS_H5 exactly as
            # training does (own dataset instance, aux_relative=True).
            stats_ds = DockingDataset(STATS_H5, STATS_H5, horizon=HORIZON, obs_horizon=OBS_HORIZON,
                                      with_aux=True, aux_relative=True)
            self.rel_xy_mean, self.rel_xy_std = stats_ds.rel_xy_mean, stats_ds.rel_xy_std
            self._relative_to_goal = DockingDataset._relative_to_goal
            del stats_ds
            # THIS file's own per-episode goal reference pose (last valid-label
            # frame) — the actual current->goal target, not the train file's.
            self.goal_pose_ep = []
            s = 0
            for e in self.episode_ends:
                idx = np.where(self.rel_valid[s:e])[0]
                self.goal_pose_ep.append(self.pose[s + idx[-1]] if len(idx) else None)
                s = e

    def norm(self, a):
        return 2.0 * (a - self.a_min) / self.a_scale - 1.0

    def hist_rows(self, t, ep_start, stride=1):
        rows = np.arange(t - OBS_HORIZON + 1, t + 1)
        rows = np.clip(rows, ep_start, None)                    # left-pad with first frame
        return rows[::stride] if stride > 1 else rows

    def batch(self, idxs):
        """idxs must be a sorted block of (near-)contiguous usable frame indices;
        the DINO rows they need are then one contiguous slice (fast seq read)."""
        idxs = np.asarray(idxs)
        B = len(idxs)
        vel = np.stack([self.norm(self.enc[self.hist_rows(t, self.ep_start[t])]) for t in idxs])
        drows = [self.hist_rows(t, self.ep_start[t], VISION_STRIDE) for t in idxs]
        a = int(min(r[0] for r in drows))
        b = int(idxs[-1]) + 1
        block = self.z_dino[a:b]                                 # one sequential read
        dino = np.stack([block[rw - a] for rw in drows])         # [B,5,196,768]
        grows = np.unique(self.ep_end[idxs] - 1)                 # 1-2 goal rows per block
        gfeat = {int(g): self.z_dino[int(g)] for g in grows}
        goal = np.stack([gfeat[int(self.ep_end[t] - 1)] for t in idxs])[:, None]  # [B,1,196,768]
        lid = self.z_lidar[idxs.tolist()].astype(np.float32)
        nlid = self.z_nlidar[idxs.tolist()].astype(np.int64)

        # dock distance for BINNING: always the raw absolute dock pose (physical
        # regime), independent of which target space aux_relative trains on.
        dock_d = np.hypot(self.pose[idxs, 0], self.pose[idxs, 1])

        if self.aux_relative:
            rel_pose = np.zeros((B, 3), np.float32)
            row_valid = np.ones(B, bool)
            for ep_i in np.unique(self.ep_id[idxs]):
                g = self.goal_pose_ep[ep_i]
                m = self.ep_id[idxs] == ep_i
                if g is None:
                    row_valid[m] = False
                    continue
                rel_pose[m] = self._relative_to_goal(self.pose[idxs][m], g)
            xy_n = (rel_pose[:, :2] - self.rel_xy_mean) / self.rel_xy_std
            tgt = np.concatenate([xy_n, np.sin(rel_pose[:, 2:3]), np.cos(rel_pose[:, 2:3])], 1)
            reliable = self.rel_valid[idxs] & row_valid
        else:
            xy_n = (self.pose[idxs, :2] - self.dock_xy_mean) / self.dock_xy_std
            tgt = np.concatenate([xy_n, np.sin(self.pose[idxs, 2:3]), np.cos(self.pose[idxs, 2:3])], 1)
            reliable = self.rel_valid[idxs]

        # Goal-frame scan (this file's own docked-frame lidar per episode). Always
        # included: the condition net only consumes it when use_goal_lidar=True
        # (harmless extra key otherwise), but omitting it for a goal-lidar model
        # would silently drop a token modality it was trained to always receive.
        # h5py fancy-indexing needs strictly increasing unique indices -> read the
        # few unique goal rows once (same pattern as the DINO goal feature above).
        grows = self.ep_end[idxs] - 1
        uniq = np.unique(grows)
        gl_map = {int(r): i for i, r in enumerate(uniq)}
        glid_u = self.z_lidar[uniq.tolist()].astype(np.float32)
        gnlid_u = self.z_nlidar[uniq.tolist()].astype(np.int64)
        pick = np.array([gl_map[int(r)] for r in grows])
        glid, gnlid = glid_u[pick], gnlid_u[pick]

        ctx = {
            "velocity": torch.from_numpy(vel).to(DEVICE),
            "dino_feat2": torch.from_numpy(dino).to(DEVICE).float(),
            "goal_feat2": torch.from_numpy(goal).to(DEVICE).float(),
            "goal_mask": torch.ones(B, device=DEVICE),          # goal-conditioned, as at deployment
            "lidar_points": torch.from_numpy(lid).to(DEVICE),
            "lidar_npoints": torch.from_numpy(nlid).to(DEVICE),
            "goal_lidar_points": torch.from_numpy(glid).to(DEVICE),
            "goal_lidar_npoints": torch.from_numpy(gnlid).to(DEVICE),
        }
        return (ctx, torch.from_numpy(tgt.astype(np.float32)).to(DEVICE), reliable,
                torch.from_numpy(dock_d.astype(np.float32)).to(DEVICE))


def bin_stats(dists, errs):
    med, p90, cnt = [], [], []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (dists >= lo) & (dists < hi)
        cnt.append(int(m.sum()))
        med.append(float(np.median(errs[m])) if m.any() else np.nan)
        p90.append(float(np.percentile(errs[m], 90)) if m.any() else np.nan)
    return np.array(med), np.array(p90), np.array(cnt)


@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = H5Batcher()
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(src.n_rows - BLOCK, size=N_BLOCKS, replace=False))
    blocks = []
    for s in starts:
        blk = np.arange(s, s + BLOCK)
        blk = blk[src.ok[blk]]                                   # drop frames too near an episode end
        if len(blk):
            blocks.append(blk)

    nets = {}
    for run, ckpt in RUNS.items():
        for which in ("raw", "ema"):
            net = build_condition_net().to(DEVICE).eval()
            load_condition_weights(net, ckpt, which)
            nets[f"{run}-{which}"] = net

    std_t = torch.as_tensor(src.dock_xy_std, device=DEVICE)
    acc = {k: dict(d=[], mm=[], deg=[]) for k in nets}
    for bi, idxs in enumerate(blocks):
        ctx, tgt, rel, dock_d = src.batch(idxs)
        relm = torch.from_numpy(rel).to(DEVICE)
        yaw_t = torch.atan2(tgt[:, 2], tgt[:, 3])
        for name, net in nets.items():
            net(dict(ctx))
            pred = net._aux_pred
            mm = torch.hypot(*((pred[:, :2] - tgt[:, :2]) * std_t).T) * 1000.0
            yaw_p = torch.atan2(pred[:, 2], pred[:, 3])
            deg = torch.rad2deg(torch.abs(torch.atan2(torch.sin(yaw_p - yaw_t), torch.cos(yaw_p - yaw_t))))
            acc[name]["d"].append(dock_d[relm].cpu().numpy())
            acc[name]["mm"].append(mm[relm].cpu().numpy())
            acc[name]["deg"].append(deg[relm].cpu().numpy())
        if bi % 5 == 0:
            print(f"block {bi + 1}/{len(blocks)}", flush=True)

    results = {}
    for name, a in acc.items():
        d, e_mm, e_deg = (np.concatenate(a[k]) for k in ("d", "mm", "deg"))
        med, p90, cnt = bin_stats(d, e_mm)
        med_deg, _, _ = bin_stats(d, e_deg)
        results[name] = dict(
            median=med.tolist(), p90=p90.tolist(), count=cnt.tolist(), median_deg=med_deg.tolist(),
            overall_median=float(np.median(e_mm)), overall_mean=float(e_mm.mean()),
            within_1cm=float((e_mm < 10).mean()), within_3cm=float((e_mm < 30).mean()), n=len(e_mm))
        r = results[name]
        print(f"{name}: median {r['overall_median']:.1f} mm | mean {r['overall_mean']:.1f} mm | "
              f"<1cm {r['within_1cm']*100:.0f}% | <3cm {r['within_3cm']*100:.0f}% (n={r['n']})")

    json.dump({"bin_edges": BIN_EDGES.tolist(), "results": results},
              open(os.path.join(OUT_DIR, "dist_binned_error.json"), "w"), indent=1)
    plot(results)


def plot(results):
    centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"flow-raw": "#9ecae1", "flow-ema": "#3182bd", "auxw-raw": "#fdae6b", "auxw-ema": "#e6550d"}
    for name, r in results.items():
        axes[0].plot(centers, r["median"], "-o", color=colors[name], label=f"{name} median")
        axes[0].plot(centers, r["p90"], "--", color=colors[name], alpha=0.5, label=f"{name} p90")
    axes[0].axhline(30, color="0.5", ls=":", label="3 cm")
    axes[0].axhline(10, color="0.7", ls=":", label="1 cm")
    axes[0].set_yscale("log")               # EMA (undertrained, ~15 cm) would flatten linear axes
    axes[0].set_xlabel("dock pose distance [m]")
    axes[0].set_ylabel("aux XY error [mm] (log)")
    axes[0].set_title("aux dock-pose error vs distance (train set)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3, which="both")

    cnt = np.array(results[list(results)[0]]["count"])
    axes[1].bar(centers, cnt, width=np.diff(BIN_EDGES) * 0.9, color="#cccccc")
    axes[1].set_xlabel("dock pose distance [m]")
    axes[1].set_ylabel("reliable frames in sample")
    axes[1].set_title("frame count per bin")
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "dist_binned_error.png")
    plt.savefig(png, dpi=110)
    print(f"saved {png}")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        d = json.load(open(os.path.join(OUT_DIR, "dist_binned_error.json")))
        plot(d["results"])
    else:
        main()
