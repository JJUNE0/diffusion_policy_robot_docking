"""Compare the policy's driven path against the demonstration's, per episode:
metrics (velRMSE / ADE / FDE), a static PNG, and an MP4.

The rollout protocol is IMPORTED from test/eval_run_rgeo.py rather than
re-specified here (same RolloutController knobs, same seed, same
EXEC_CHUNK_K, same MAX_STEPS), so the numbers this prints are the numbers that
evaluator prints -- and `--check-json` asserts exactly that against its saved
result file. The only thing added is recording: every executed action, and
every replanned horizon, kept so they can be drawn.

What "path" means here: both paths are RK4 integrations of a (v, w) command
stream in the robot's body frame (utils/inference.reconstruct_pose_rk4),
anchored at a common origin (0, 0, 0). The GT path integrates the
demonstration's own `encoder` velocities. Neither is an absolute map pose --
this is odometry, which is what the policy actually commands, and it is the
same construction eval_run_rgeo scores ADE/FDE on.

Open-loop, so the divergence is real: the OBSERVATION at every step comes from
the demonstration (the robot is where the demo was), while the ACTIONS being
integrated are the policy's. The two paths therefore separate exactly as much
as the commands differ -- the policy is never corrected back onto the demo.

Run:
  EVAL_H5=dataset/f4hall150_val.h5 \
  EVAL_STATS_H5=dataset/f4hall150_train.h5 \
  EVAL_EPISODES=0,1,2,3 \
  python3.12 test/eval_traj_video_rgeo.py outputs/train/r_relfeat_only_now_f4hall150/<ts>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda key: {"runtime.cwd": REPO}[key])

# Import the evaluator itself so the protocol cannot drift out of sync.
import eval_run_rgeo as ev  # noqa: E402
from utils.inference import build_model_from_cfg, reconstruct_pose_rk4  # noqa: E402
from cleandiffuser.rollout_core import RolloutController  # noqa: E402
from utils.modular_dataset import ModularDockingDataset  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

OUT_DIR = os.path.join(REPO, "test/out/rgeo_traj")

C_GT = "#2c7fb8"        # demonstration
C_AI = "#d95f02"        # policy
C_PLAN = "#7570b3"      # the policy's current replanned horizon


# ---------------------------------------------------------------- rollout ----
@torch.no_grad()
def rollout_recording(nn_diffusion, solver, ds, ep_idx, act_min, act_scale,
                      use_ema, episode_ends, enc_all):
    """ev.rollout(), but keeping the per-step trace needed to draw it.

    Every line that touches the model is a copy of ev.rollout's, including the
    torch.manual_seed(0) placement -- the RNG draw order is what makes the
    metrics reproducible, so it must not move.
    """
    ends = episode_ends.astype(int)
    s = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
    e = int(ends[ep_idx])
    enc = enc_all[s:e]
    horizon = ds.horizon
    ep_steps = min(len(enc) - horizon + 1, ev.MAX_STEPS)
    torch.manual_seed(0)
    ctrl = RolloutController(
        nn_diffusion, solver=solver, sample_steps=ev.SAMPLE_STEPS, use_ema=use_ema,
        n_samples=ev.N_SAMPLES, horizon=horizon, w_cfg=1, agg="mean",
        traj_ema_alpha=ev.TRAJ_EMA_ALPHA, warm_start=False, warm_level=0.3,
        device=ev.DEVICE)

    row_to_didx = {int(t): i for i, t in enumerate(ds.index_map) if ds.ep_start_map[i] == s}
    selected, replans = [], []
    t = 0
    while t < ep_steps:
        di = row_to_didx.get(s + t)
        if di is None:
            break
        obs = ds[di]["obs"]
        context = {k: v.unsqueeze(0).to(ev.DEVICE).repeat(ev.N_SAMPLES, *([1] * v.dim()))
                   for k, v in obs.items()}
        k = min(ev.EXEC_CHUNK_K, ep_steps - t, horizon)
        plan = ctrl.plan(context, act_min, act_scale, chunk_shift=k)
        selected.extend(list(plan.current[0:k]))
        replans.append((t, np.asarray(plan.current, dtype=np.float64)))
        t += k

    ai = np.array(selected)
    gt = enc[:len(ai)]
    gt_path = reconstruct_pose_rk4(gt[:, 0], gt[:, 1], dt=ev.DT)
    ai_path = reconstruct_pose_rk4(ai[:, 0], ai[:, 1], dt=ev.DT)
    disp = np.hypot(*(gt_path[:, :2] - ai_path[:, :2]).T)
    path_len = float(np.sum(np.hypot(*np.diff(gt_path[:, :2], axis=0).T)))

    # Each replan drawn in the world frame, anchored where the policy actually
    # is at that step -- that is what the robot would follow if it stopped
    # replanning. Trim to the steps that remain so the tail plans do not run
    # past the episode.
    plan_paths = []
    for t0, pl in replans:
        n = min(len(pl), len(ai) - t0)
        if n <= 0:
            continue
        plan_paths.append((t0, reconstruct_pose_rk4(pl[:n, 0], pl[:n, 1], dt=ev.DT,
                                                    initial_pose=tuple(ai_path[t0]))))

    return dict(
        ep_idx=ep_idx, src_start=s, steps=int(len(ai)),
        ai=ai, gt=gt, ai_path=ai_path, gt_path=gt_path, disp=disp,
        plan_paths=plan_paths,
        vel_rmse=float(np.sqrt(((ai - gt) ** 2).mean())),
        vx_rmse=float(np.sqrt(((ai[:, 0] - gt[:, 0]) ** 2).mean())),
        wz_rmse=float(np.sqrt(((ai[:, 1] - gt[:, 1]) ** 2).mean())),
        ade_cm=float(disp.mean() * 100), fde_cm=float(disp[-1] * 100),
        path_len_cm=path_len * 100,
        fde_over_path=float(disp[-1] / max(path_len, 1e-9)),
        exec_chunk_k=ev.EXEC_CHUNK_K)


# ---------------------------------------------------------------- drawing ----
def _limits(r, pad=0.15):
    xs = np.concatenate([r["gt_path"][:, 0], r["ai_path"][:, 0]])
    ys = np.concatenate([r["gt_path"][:, 1], r["ai_path"][:, 1]])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span = max(x1 - x0, y1 - y0, 0.5)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = span / 2 * (1 + pad)
    return (cx - h, cx + h), (cy - h, cy + h)


def _heading(ax, pose, color):
    ax.plot([pose[0]], [pose[1]], "o", color=color, ms=7, zorder=6)
    ax.plot([pose[0], pose[0] + 0.12 * np.cos(pose[2])],
            [pose[1], pose[1] + 0.12 * np.sin(pose[2])],
            "-", color=color, lw=2, zorder=6)


def static_figure(r, name, out_png):
    """One page per episode: the two paths, the error over time, and the two
    command channels that produced them."""
    fig = plt.figure(figsize=(15, 5.2), dpi=130)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1, 1], hspace=0.45, wspace=0.28)

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(r["gt_path"][:, 0], r["gt_path"][:, 1], "-", color=C_GT, lw=2.2,
            label="demonstration (GT)")
    ax.plot(r["ai_path"][:, 0], r["ai_path"][:, 1], "-", color=C_AI, lw=2.2,
            label="policy")
    ax.plot([0], [0], "k^", ms=9, label="start")
    _heading(ax, r["gt_path"][-1], C_GT)
    _heading(ax, r["ai_path"][-1], C_AI)
    # the FDE, drawn
    ax.plot([r["gt_path"][-1, 0], r["ai_path"][-1, 0]],
            [r["gt_path"][-1, 1], r["ai_path"][-1, 1]], ":", color="0.35", lw=1.6)
    ax.annotate(f"FDE {r['fde_cm']:.1f} cm",
                ((r["gt_path"][-1, 0] + r["ai_path"][-1, 0]) / 2,
                 (r["gt_path"][-1, 1] + r["ai_path"][-1, 1]) / 2),
                fontsize=8, color="0.25",
                xytext=(6, 6), textcoords="offset points")
    xl, yl = _limits(r)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
    ax.set_xlabel("x forward [m]"); ax.set_ylabel("y left [m]")
    ax.set_title(f"{name} — integrated path", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="best")

    tt = np.arange(len(r["disp"])) * ev.DT
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(tt, r["disp"] * 100, color="0.25", lw=1.6)
    ax.axhline(r["ade_cm"], color=C_AI, ls="--", lw=1.2,
               label=f"ADE {r['ade_cm']:.1f} cm")
    ax.set_ylabel("|error| [cm]"); ax.set_title("path error over time", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    # `disp` is per POSE (T+1 of them); the cumulative distance is per STEP
    # (T of them), so it needs a leading 0 to line the two up.
    travelled = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(r["gt_path"][:, :2], axis=0).T))])
    # Only plot once the demo has actually gone somewhere. Below a few cm the
    # ratio is a millimetre divided by a millimetre -- it rails to hundreds of
    # percent and squashes the part of the curve that means anything.
    ok = travelled >= 0.05
    if ok.any():
        rel = r["disp"][ok] / travelled[ok] * 100
        ax.plot(tt[ok], rel, color="0.25", lw=1.4)
        ax.set_ylim(0, float(rel.max()) * 1.15 + 1)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_ylabel("error / distance [%]"); ax.set_xlabel("t [s]")
    ax.set_title("error relative to distance travelled", fontsize=10)
    ax.grid(alpha=0.25)

    ts = np.arange(r["steps"]) * ev.DT
    for i, (ch, lab, unit) in enumerate(((0, "$v_x$", "m/s"), (1, "$\\omega_z$", "rad/s"))):
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(ts, r["gt"][:, ch], color=C_GT, lw=1.3, label="GT")
        ax.plot(ts, r["ai"][:, ch], color=C_AI, lw=1.3, label="policy")
        rm = r["vx_rmse"] if ch == 0 else r["wz_rmse"]
        ax.set_ylabel(f"{lab} [{unit}]")
        ax.set_title(f"{lab} command  (RMSE {rm:.4f})", fontsize=10)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=4, ncol=2, loc="upper left")
        else:
            ax.set_xlabel("t [s]")

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_video(r, name, images, out_mp4, fps=30, stride=2, dpi=110):
    """Camera view + the two paths growing together, one frame per `stride`
    control steps. Frames go to a temp dir and ffmpeg muxes them -- imageio's
    ffmpeg plugin is not installed in this container, the binary is."""
    n = r["steps"]
    xl, yl = _limits(r)
    plan_at = {t0: p for t0, p in r["plan_paths"]}
    plan_starts = sorted(plan_at)

    fig = plt.figure(figsize=(11.2, 4.6), dpi=dpi)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.15], wspace=0.18,
                  bottom=0.20, top=0.93)
    ax_im = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_im.axis("off")
    im_artist = ax_im.imshow(np.zeros((240, 320, 3), np.uint8))
    im_title = ax_im.set_title("", fontsize=10)

    ax_tr.plot(r["gt_path"][:, 0], r["gt_path"][:, 1], "-", color=C_GT, lw=1.0,
               alpha=0.28)
    (l_gt,) = ax_tr.plot([], [], "-", color=C_GT, lw=2.4, label="demonstration (GT)")
    (l_ai,) = ax_tr.plot([], [], "-", color=C_AI, lw=2.4, label="policy")
    (l_plan,) = ax_tr.plot([], [], "--", color=C_PLAN, lw=1.8, alpha=0.9,
                           label="policy's current 2 s plan")
    (m_gt,) = ax_tr.plot([], [], "o", color=C_GT, ms=7)
    (m_ai,) = ax_tr.plot([], [], "o", color=C_AI, ms=7)
    (l_err,) = ax_tr.plot([], [], ":", color="0.35", lw=1.4)
    ax_tr.plot([0], [0], "k^", ms=9)
    ax_tr.set_xlim(*xl); ax_tr.set_ylim(*yl); ax_tr.set_aspect("equal")
    ax_tr.set_xlabel("x forward [m]"); ax_tr.set_ylabel("y left [m]")
    ax_tr.grid(alpha=0.25)
    ax_tr.legend(fontsize=4, loc="upper left")
    # The readout goes in the dead band under the camera image, not inside the
    # trajectory axes -- a box wide enough for the monospace table collides
    # with both the legend and the paths wherever it is put in there.
    txt = fig.text(0.055, 0.055, "", va="bottom", ha="left",
                   fontsize=8.5, family="monospace",
                   bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=4.0))

    tmp = tempfile.mkdtemp(prefix="rgeo_vid_")
    try:
        frames = list(range(0, n, stride))
        if frames[-1] != n - 1:
            frames.append(n - 1)
        for fi, t in enumerate(frames):
            im_artist.set_data(np.ascontiguousarray(images[t].transpose(1, 2, 0)))
            im_title.set_text(f"{name} — image_bottom (orbbec-0)   t={t * ev.DT:5.1f}s")

            l_gt.set_data(r["gt_path"][:t + 1, 0], r["gt_path"][:t + 1, 1])
            l_ai.set_data(r["ai_path"][:t + 1, 0], r["ai_path"][:t + 1, 1])
            m_gt.set_data([r["gt_path"][t, 0]], [r["gt_path"][t, 1]])
            m_ai.set_data([r["ai_path"][t, 0]], [r["ai_path"][t, 1]])
            l_err.set_data([r["gt_path"][t, 0], r["ai_path"][t, 0]],
                           [r["gt_path"][t, 1], r["ai_path"][t, 1]])

            active = [p for p in plan_starts if p <= t]
            if active:
                pp = plan_at[active[-1]]
                l_plan.set_data(pp[:, 0], pp[:, 1])

            txt.set_text(
                f"step {t:4d}/{n - 1}     v_x[m/s]   w_z[rad/s]\n"
                f"demonstration (GT)   {r['gt'][t, 0]:7.3f}     {r['gt'][t, 1]:7.3f}\n"
                f"policy               {r['ai'][t, 0]:7.3f}     {r['ai'][t, 1]:7.3f}\n"
                f"path error {r['disp'][t] * 100:6.1f} cm      "
                f"episode ADE {r['ade_cm']:.1f} cm")
            fig.savefig(os.path.join(tmp, f"f{fi:06d}.png"))
        plt.close(fig)

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", os.path.join(tmp, "f%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_mp4]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4, len(frames)


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2,
                    help="control steps per video frame (2 -> 15 Hz of a 30 Hz log)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--check-json", default=None,
                    help="a test/out/rgeo/*.json from eval_run_rgeo.py; assert "
                         "this script reproduces its ADE/FDE/velRMSE exactly")
    args = ap.parse_args()
    run_dir = args.run_dir.rstrip("/")

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    exp = str(cfg.experiment_name)
    cks = glob.glob(os.path.join(run_dir, "checkpoint_step_*.pt"))
    ckpt_path = max(cks, key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))

    nn_condition, nn_diffusion = build_model_from_cfg(cfg, ev.DEVICE)
    ck = torch.load(ckpt_path, map_location=ev.DEVICE, weights_only=False)
    nn_diffusion.model.load_state_dict(ck["model_state_dict"])
    nn_diffusion.model_ema.load_state_dict(ck["ema_state_dict"])
    nn_diffusion.eval()
    use_ema = float(cfg.get("ema_rate", 0.999)) <= 0.999
    solver = "euler" if cfg.get("diffusion_backbone") == "rectified_flow" else "ode_dpmsolver++_2M"
    print(f"[{exp}] {os.path.basename(ckpt_path)} | use_ema={use_ema} | solver={solver} | "
          f"EVAL_H5={os.path.basename(ev.EVAL_H5)} | chunk_k={ev.EXEC_CHUNK_K}", flush=True)

    sensors = ev._resolve_sensor_files(OmegaConf.to_container(cfg.sensors, resolve=True),
                                       ev.EVAL_H5)
    ds = ModularDockingDataset(
        h5_path=ev.EVAL_H5, sensors=sensors, horizon=cfg.horizon,
        obs_horizon=cfg.obs_horizon, action_key=cfg.get("action_key", "encoder"),
        train_h5_path=ev.STATS_H5, action_norm=cfg.get("action_norm", "minmax"))
    # Same guard as eval_run_rgeo: a normalizer mismatch produces plausible
    # numbers that mean nothing.
    for nm, got, want in (("action_min", ds.action_min, ck["action_min"]),
                          ("action_scale", ds.action_scale, ck["action_scale"])):
        if not np.allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-4):
            raise SystemExit(f"{nm} mismatch: dataset {np.asarray(got)} vs checkpoint "
                             f"{np.asarray(want)}")

    with h5py.File(ev.EVAL_H5, "r") as f:
        episode_ends = f["episode_ends"][:]
        enc_all = f["encoder"][:].astype(np.float32)
        ep_names = (list(f["source_episode_name"].asstr()[:])
                    if "source_episode_name" in f else
                    [f"episode_{i}" for i in range(len(episode_ends))])

    results, rows = {}, []
    for ep_idx in ev.EVAL_EPISODES:
        name = ep_names[ep_idx]
        r = rollout_recording(nn_diffusion, solver, ds, ep_idx, ck["action_min"],
                              ck["action_scale"], use_ema, episode_ends, enc_all)
        png = static_figure(r, name, os.path.join(OUT_DIR, f"{exp}_{name}_path.png"))
        mp4, nframes = (None, 0)
        if not args.no_video:
            s, e = r["src_start"], r["src_start"] + r["steps"]
            with h5py.File(ev.EVAL_H5, "r") as f:
                imgs = f["image_bottom"][s:e]
            mp4, nframes = render_video(
                r, name, imgs, os.path.join(OUT_DIR, f"{exp}_{name}.mp4"),
                fps=args.fps, stride=args.stride)
            del imgs
        print(f"[{exp}] {name}: velRMSE {r['vel_rmse']:.4f} "
              f"(vx {r['vx_rmse']:.4f} / wz {r['wz_rmse']:.4f}) | "
              f"ADE {r['ade_cm']:.1f} cm | FDE {r['fde_cm']:.1f} cm | "
              f"path {r['path_len_cm']:.0f} cm | {r['steps']} steps"
              + (f" | {os.path.basename(mp4)} ({nframes} frames)" if mp4 else ""),
              flush=True)
        rows.append((name, r))
        results[name] = {k: v for k, v in r.items()
                         if k not in ("ai", "gt", "ai_path", "gt_path", "disp",
                                      "plan_paths")}
        results[name]["png"] = os.path.basename(png)
        if mp4:
            results[name]["mp4"] = os.path.basename(mp4)

    def _agg(key, fn):
        return float(fn([r[key] for _, r in rows]))

    summary = dict(
        n_episodes=len(rows),
        vel_rmse_mean=_agg("vel_rmse", np.mean), vel_rmse_median=_agg("vel_rmse", np.median),
        vx_rmse_mean=_agg("vx_rmse", np.mean), wz_rmse_mean=_agg("wz_rmse", np.mean),
        ade_cm_mean=_agg("ade_cm", np.mean), ade_cm_median=_agg("ade_cm", np.median),
        fde_cm_mean=_agg("fde_cm", np.mean), fde_cm_median=_agg("fde_cm", np.median),
        fde_over_path_mean=_agg("fde_over_path", np.mean))

    print("\n" + "=" * 92)
    print(f"{'episode':<20}{'steps':>7}{'path[cm]':>11}{'velRMSE':>10}"
          f"{'vxRMSE':>9}{'wzRMSE':>9}{'ADE[cm]':>10}{'FDE[cm]':>10}{'FDE/path':>10}")
    print("-" * 92)
    for name, r in rows:
        print(f"{name:<20}{r['steps']:>7}{r['path_len_cm']:>11.0f}{r['vel_rmse']:>10.4f}"
              f"{r['vx_rmse']:>9.4f}{r['wz_rmse']:>9.4f}{r['ade_cm']:>10.1f}"
              f"{r['fde_cm']:>10.1f}{r['fde_over_path'] * 100:>9.1f}%")
    print("-" * 92)
    print(f"{'mean':<20}{'':>7}{'':>11}{summary['vel_rmse_mean']:>10.4f}"
          f"{summary['vx_rmse_mean']:>9.4f}{summary['wz_rmse_mean']:>9.4f}"
          f"{summary['ade_cm_mean']:>10.1f}{summary['fde_cm_mean']:>10.1f}"
          f"{summary['fde_over_path_mean'] * 100:>9.1f}%")
    print(f"{'median':<20}{'':>7}{'':>11}{summary['vel_rmse_median']:>10.4f}"
          f"{'':>9}{'':>9}{summary['ade_cm_median']:>10.1f}"
          f"{summary['fde_cm_median']:>10.1f}")
    print("=" * 92)

    if args.check_json:
        ref = json.load(open(args.check_json))["openloop"]
        bad = []
        for i, (name, r) in zip(ev.EVAL_EPISODES, rows):
            g = ref.get(str(i))
            if g is None:
                bad.append(f"{name}: no episode {i} in {args.check_json}")
                continue
            for k in ("ade_cm", "fde_cm", "vel_rmse"):
                if not np.isclose(r[k], g[k], rtol=1e-9, atol=1e-9):
                    bad.append(f"{name}.{k}: {r[k]!r} vs eval_run_rgeo {g[k]!r}")
        if bad:
            raise SystemExit("PROTOCOL DRIFT vs eval_run_rgeo.py:\n  " + "\n  ".join(bad))
        print(f"cross-check OK: ADE/FDE/velRMSE identical to {os.path.basename(args.check_json)}")

    tag = os.environ.get("EVAL_TAG", "")
    out = os.path.join(OUT_DIR, f"{exp}{'_' + tag if tag else ''}_traj.json")
    json.dump(dict(run_dir=run_dir, ckpt=os.path.basename(ckpt_path),
                   eval_h5=ev.EVAL_H5, stats_h5=ev.STATS_H5,
                   exec_chunk_k=ev.EXEC_CHUNK_K, dt=ev.DT,
                   episodes=results, summary=summary), open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
