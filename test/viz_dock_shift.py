"""Same dock, same hardware — the March demos just never drive all the way in."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, "/home/work/.postech/diffusion_policy_robot_docking")
os.chdir("/home/work/.postech/diffusion_policy_robot_docking")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch
from endgame.se2 import inverse
import h5py

S = ("/tmp/claude-1100/-home-work--postech-diffusion-policy-robot-docking/"
     "a5c26007-c087-4fe0-bb67-75e52d8fc507/scratchpad")
MAR, JUL, MISS = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#1a1a19", "#4a4a46", "#8a8a80"
tracks = json.load(open(f"{S}/dock_track.json"))


def face_y(scan):
    p = scan[(np.abs(scan[:, 0]) < 0.12) & (scan[:, 1] < -0.2) & (scan[:, 1] > -0.9)]
    return -np.median(p[:, 1]) * 1000 if len(p) >= 5 else np.nan


f = h5py.File("dataset/after_0328_train.h5", "r")
dp, ends, rel = f["dock_pose"][:], f["episode_ends"][:], f["reliable"][:]
lp, ln = f["lidar_points"], f["lidar_npoints"]
starts = np.r_[0, ends[:-1]]
mar_depth, mar_face, mar_idx, mar_scans = [], [], [], []
for s, e in zip(starts, ends):
    i = e - 1
    while i > e - 60 and not rel[i]:
        i -= 1
    if not rel[i]:
        continue
    sc = lp[i][:ln[i]].astype(float)
    mar_depth.append(inverse(dp[i])[1] * 1000); mar_face.append(face_y(sc)); mar_idx.append(i)
mar_depth, mar_face = np.array(mar_depth), np.array(mar_face)
pick = np.argsort(np.abs(mar_depth - np.median(mar_depth)))[:12]
mar_scans = [lp[mar_idx[k]][:ln[mar_idx[k]]].astype(float) for k in pick]


def july(rec, path=None):
    ts, sc = [], []
    for line in open(path or f"dataset/demo_0725/{rec}/lidar.jsonl"):
        d = json.loads(line)
        ts.append(d["ts"]); sc.append(np.array([[p["x"], p["y"]] for p in d["points"]]))
    o = np.argsort(ts)
    if rec in tracks:
        j = np.where(np.array(tracks[rec]["ok"], bool))[0][-1]
        return sc[o[j]], np.array(ts)[o][j], inverse(np.array(tracks[rec]["pose"])[j])[1] * 1000
    return sc[o[-1]], np.array(ts)[o][-1], np.nan


SUCC = ["record_00779", "record_00783", "record_00788"]
NEAR = ["record_00777", "record_00786", "record_00787"]
jul_s, jul_n = [july(r) for r in SUCC], [july(r) for r in NEAR]
hum = [july(r, f"{S}/r79x/{r}_lidar.jsonl") for r in ["792", "793", "794", "795", "797"]]

fig = plt.figure(figsize=(15.5, 9.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0], hspace=0.34, wspace=0.24)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d8d8d0")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color="#eeeee8", lw=.7, zorder=0); ax.set_axisbelow(True)


# a — sensor-frame overlay
ax = fig.add_subplot(gs[0, 0]); style(ax)
for sc in mar_scans:
    m = np.hypot(sc[:, 0], sc[:, 1]) < 1.0
    ax.scatter(sc[m, 0], sc[m, 1], s=5, c=MAR, alpha=.35, lw=0, zorder=3)
for sc, _, _ in jul_s:
    m = np.hypot(sc[:, 0], sc[:, 1]) < 1.0
    ax.scatter(sc[m, 0], sc[m, 1], s=5, c=JUL, alpha=.55, lw=0, zorder=4)
ax.scatter([0], [0], marker="s", s=70, c=INK, zorder=6)
ax.annotate("LiDAR (robot)", (0, 0), xytext=(0, -30), textcoords="offset points",
            fontsize=8.5, color=INK, ha="center")
ax.set_xlim(-.75, .75); ax.set_ylim(-.85, .5); ax.set_aspect("equal")
ax.set_xlabel("x  (m, sensor frame)", fontsize=9, color=INK2)
ax.set_ylabel("y  (m)", fontsize=9, color=INK2)
ax.set_title("a.  The dock itself is identical", fontsize=10.5, color=INK, loc="left", pad=8)
ax.scatter([], [], s=22, c=MAR, label="March demos (n=12)")
ax.scatter([], [], s=22, c=JUL, label="July docks (n=3)")
ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)

# b — notch zoom
ax = fig.add_subplot(gs[0, 1]); style(ax)
am = np.concatenate([s[np.hypot(s[:, 0], s[:, 1]) < .75] for s in mar_scans])
aj = np.concatenate([s[np.hypot(s[:, 0], s[:, 1]) < .75] for s, _, _ in jul_s])
ax.scatter(am[:, 0], am[:, 1], s=10, c=MAR, alpha=.30, lw=0, zorder=3)
ax.scatter(aj[:, 0], aj[:, 1], s=10, c=JUL, alpha=.55, lw=0, zorder=4)
ym, yj = -np.nanmedian(mar_face) / 1000, -np.nanmean([face_y(s) for s, _, _ in jul_s]) / 1000
ax.axhline(ym, color=MAR, lw=1.7, ls="--", zorder=5)
ax.axhline(yj, color=JUL, lw=1.7, ls="--", zorder=5)
ax.add_patch(FancyArrowPatch((0.36, ym), (0.36, yj), arrowstyle="<->",
                             mutation_scale=12, color=INK, lw=1.5, zorder=7))
ax.annotate(f"{abs(ym - yj) * 1000:.0f} mm", (0.335, (ym + yj) / 2), fontsize=11.5,
            color=INK, va="center", ha="right", fontweight="bold")
ax.annotate("March demos park here", (-.44, ym), xytext=(0, -14), textcoords="offset points",
            fontsize=9, color=MAR, fontweight="bold", ha="left")
ax.annotate("July docks reach the stop", (-.44, yj), xytext=(0, 7), textcoords="offset points",
            fontsize=9, color=JUL, fontweight="bold", ha="left")
ax.set_xlim(-.45, .45); ax.set_ylim(-.58, -.40); ax.set_aspect("equal")
ax.set_xlabel("x  (m)", fontsize=9, color=INK2); ax.set_ylabel("y  (m)", fontsize=9, color=INK2)
ax.set_title("b.  …but the robot stops 24 mm short", fontsize=10.5, color=INK, loc="left", pad=8)

# c — raw, registration-free measurement
ax = fig.add_subplot(gs[0, 2]); style(ax)
groups = [("March\ndemos", mar_face, MAR, "o"),
          ("July\nnear-miss", np.array([face_y(s) for s, _, _ in jul_n]), MISS, "X"),
          ("July policy\nsuccess", np.array([face_y(s) for s, _, _ in jul_s]), JUL, "o"),
          ("July human\nteleop", np.array([face_y(s) for s, _, _ in hum]), JUL, "v")]
rng = np.random.default_rng(1)
for i, (nm, v, c, mk) in enumerate(groups):
    v = v[~np.isnan(v)]
    ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=26, c=c, marker=mk,
               alpha=.5 if len(v) > 20 else .95, lw=0, zorder=3)
    ax.plot([i - .27, i + .27], [v.mean()] * 2, color=INK, lw=2.2, zorder=5)
    ax.annotate(f"{v.mean():.0f}", (i + .30, v.mean()), fontsize=9.5, color=INK,
                va="center", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=8.5, color=INK2)
ax.set_ylabel("raw range to the dock face  (mm)", fontsize=9, color=INK2)
ax.set_title("c.  No ICP involved — straight off the returns",
             fontsize=10.5, color=INK, loc="left", pad=8)
ax.annotate("near-misses land exactly\non the March distribution", (1.0, 496),
            fontsize=8.5, color=INK2, ha="center")

# d — terminal depth distribution
ax = fig.add_subplot(gs[1, 0]); style(ax)
ax.hist(mar_depth, bins=np.arange(500, 555, 3), color=MAR, alpha=.85, zorder=3,
        label=f"March demos  n={len(mar_depth)}")
for i, (_, _, d) in enumerate(jul_s):
    ax.scatter(d, 4.0, s=54, c=JUL, marker="o", zorder=5,
               label="July policy success" if i == 0 else None)
for i, (_, _, d) in enumerate(jul_n):
    ax.scatter(d, 4.0, s=60, c=MISS, marker="X", zorder=5,
               label="July policy near-miss" if i == 0 else None)
ax.set_xlabel("terminal depth, ICP  (mm)", fontsize=9, color=INK2)
ax.set_ylabel("episodes", fontsize=9, color=INK2)
ax.set_title("d.  1 of 145 March demos gets there", fontsize=10.5, color=INK, loc="left", pad=8)
ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
ax.annotate("43 mm wide — the demos\nnever agree on a stopping point",
            (536, 20), fontsize=8.5, color=INK2, ha="center")

# e,f — camera
def frame_near(rec, ts_ns):
    fs = glob.glob(f"dataset/demo_0725/{rec}/camera_orbbec-0/frames/*.jpg")
    t = np.array([int(os.path.basename(p).split("_")[2].split(".")[0]) for p in fs])
    return fs[int(np.argmin(np.abs(t - ts_ns)))]


for j, (title, img, col, sub) in enumerate([
        ("e.  March demo — its docked frame",
         f["image_bottom"][mar_idx[pick[0]]].transpose(1, 2, 0), MAR,
         f"stops {mar_face[pick[0]]:.0f} mm from the face  ·  camera room2 (orbbec-0)"),
        ("f.  July policy success (788)",
         mpimg.imread(frame_near("record_00788", jul_s[2][1])), JUL,
         f"reaches {face_y(jul_s[2][0]):.0f} mm  ·  same camera, same dock")]):
    ax = fig.add_subplot(gs[1, j + 1])
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(col); sp.set_linewidth(2.4)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel(sub, fontsize=8.5, color=INK2, labelpad=6)

fig.suptitle("Same dock, same hardware — the March demos never drive all the way in",
             fontsize=13.5, color=INK, x=0.008, ha="left", y=0.985, fontweight="bold")
fig.text(0.008, 0.947, "Behaviour cloning can only reproduce the demonstrated stopping "
         "point, and that point is 24 mm short of engagement with 43 mm of scatter.",
         fontsize=9.5, color=MUTED, ha="left")
fig.savefig("outputs/dock_shift_march_vs_july.png", dpi=135, bbox_inches="tight",
            facecolor="#fcfcfb")
print("saved outputs/dock_shift_march_vs_july.png")
