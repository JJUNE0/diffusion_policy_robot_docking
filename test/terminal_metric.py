"""Terminal-band metric — the last few centimetres, which is where docking is
actually won or lost.

Why this exists (2026-07-23 field session, 15 runs / 5 model variants):
every variant drove the approach competently and then STALLED 20-30 mm short.
Scored against the demonstrations at matched remaining distance, the policies
ran at 56-58% of demo speed while far out but collapsed to 10-12% inside
10-50 mm, parking 91% of their frames there against the demos' 31%. Success and
failure separated ~10:1 on that band.

That stretch is <5% of the trajectory and contributes <1% of ADE, which is
exactly why the held-out ADE/FDE table never predicted field results
(docs/ablation_study_2026-07.md section 2.12, docs/offline_metrics.md section 0).
Rank runs by `vx_ratio` and `idle_frac` in the headline band FIRST; keep
ADE/FDE as reporting.

Caveat that outlives the metric: the demos themselves stop ~24 mm short of
engagement and disagree with each other by 43 mm (2026-07-25 audit,
test/viz_dock_shift.py). So vx_ratio ~ 1.0 means "matches the demos", which is
necessary but NOT sufficient to dock. A policy that has to EXCEED the demos --
which, given the audit, every policy does -- wants vx_ratio > 1 here.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from endgame.se2 import inverse  # noqa: E402

# Bands of remaining approach distance to the episode's own docked pose (mm).
BANDS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 10_000)]
HEADLINE = (10, 50)         # the band that separated success from failure 10:1
IDLE_MMS = 5.0              # |vx| below this (mm/s) counts as parked
EXEC_K = int(os.environ.get("TERM_EXEC_K", "8"))   # steps that reach the wheels


def remaining_mm(dock_pose, reliable, episode_ends):
    """Per-frame remaining approach distance to that episode's OWN docked pose.

    `dock_pose` is the dock expressed in the sensor frame; inverting it puts the
    robot in the dock frame, whose y axis is the approach axis (docked sits at
    y ~ +0.53 m). Remaining = y(t) - y(episode's last reliable frame): 0 mm is
    "as deep as this demo ever got", positive is short of it.

    Each episode is referenced to its own terminus on purpose — the demos
    disagree with each other by 43 mm, so a shared threshold would mostly
    measure that disagreement rather than the policy.
    """
    dock_pose = np.asarray(dock_pose, np.float64)
    v = np.asarray(reliable, bool) & ~np.isnan(dock_pose).any(axis=1)
    y = np.full(len(dock_pose), np.nan, np.float64)
    if v.any():
        y[v] = np.array([inverse(p)[1] for p in dock_pose[v]])
    rem = np.full(len(dock_pose), np.nan, np.float64)
    s = 0
    for e in np.asarray(episode_ends, int):
        idx = np.where(v[s:e])[0]
        if len(idx):
            rem[s:e] = (y[s:e] - y[s + idx[-1]]) * 1000.0
        s = e
    return rem


def _band(rem, p_all, d_all, lo, hi):
    m = np.isfinite(rem) & (rem >= lo) & (rem < hi)
    if not m.any():
        return None
    p, d = p_all[m], d_all[m]
    return dict(lo_mm=lo, hi_mm=hi, n=int(m.sum()),
                policy_vx_mms=float(p.mean()), demo_vx_mms=float(d.mean()),
                vx_ratio=float(p.mean() / max(d.mean(), 1e-6)),
                policy_idle_frac=float((p < IDLE_MMS).mean()),
                demo_idle_frac=float((d < IDLE_MMS).mean()))


def band_table(rem, policy_vx_mms, demo_vx_mms):
    """Bin |vx| (mm/s) by remaining distance. Returns (bands, headline) or None.

    All three arrays must be per-frame and index-aligned. The headline band is
    computed over its own span (HEADLINE need not be one of BANDS — the
    breakdown splits 10-50 mm into two rows for diagnosis, but the number runs
    are ranked by pools them).
    """
    rem = np.asarray(rem, float)
    p_all, d_all = np.asarray(policy_vx_mms, float), np.asarray(demo_vx_mms, float)
    bands = [b for b in (_band(rem, p_all, d_all, lo, hi) for lo, hi in BANDS) if b]
    if not bands:
        return None
    headline = _band(rem, p_all, d_all, *HEADLINE)
    if headline is None:                       # nothing in the headline span
        headline = min(bands, key=lambda b: abs(b["lo_mm"] - HEADLINE[0]))
    return bands, headline


def lateral_mm(dock_pose, reliable, episode_ends):
    """Per-frame LATERAL offset from that episode's own docked pose (mm).

    This is the axis that actually decides the dock. The station has a
    protrusion that must enter a hole on the robot, and on 2026-07-23 the
    terminal frames split on lateral, not depth:

        seated (n=3)   |lateral| 5.6 +- 3.4 mm  -> reached 443.5 mm
        jammed (n=4)   |lateral| 16.9 +- 2.8 mm -> stopped at 466.8 mm
        corr(|lateral|, depth reached) = +0.83

    The depth shortfall is a CONSEQUENCE: off-centre, the peg catches the rim
    ("홈 입구에 걸침", record 00778) and cannot go further. Effective clearance
    measured at roughly +-10-12 mm (worst seated +9.0, best jammed +14.5).

    The March demos scatter +-15.7 mm laterally (range -58..+43) — WIDER than
    that clearance — because they stop at the hole entrance by design (pushing
    in draws over-current), so being off-centre there has no consequence and
    the demonstrations never had to be precise about it. 57% of them (83/145)
    still land within +-10 mm of the dock centre, which is the subset worth
    weighting toward.
    """
    dock_pose = np.asarray(dock_pose, np.float64)
    v = np.asarray(reliable, bool) & ~np.isnan(dock_pose).any(axis=1)
    x = np.full(len(dock_pose), np.nan, np.float64)
    if v.any():
        x[v] = np.array([inverse(p)[0] for p in dock_pose[v]])
    lat = np.full(len(dock_pose), np.nan, np.float64)
    s = 0
    for e in np.asarray(episode_ends, int):
        idx = np.where(v[s:e])[0]
        if len(idx):
            lat[s:e] = (x[s:e] - x[s + idx[-1]]) * 1000.0
        s = e
    return lat


def turn_symmetry(policy_wz, demo_wz, thresh=0.02):
    """Left/right balance of the commanded rotation — a steering-bias detector.

    On 2026-07-23 the two variants whose conditioning injects a LEARNED estimate
    of where the goal is (aux-pose feedback; goal-image with no LiDAR branch)
    commanded right turns on 75.1% and 62.1% of their turning frames against a
    50.2% demo baseline, and both went 0/3. Every variant near 50% scored 1/3.
    With thousands of frames the standard error is ~1.4%, so this separates
    cleanly OFFLINE — it would have flagged those two before the field session,
    which no ADE/FDE number did.

    `right_frac` near 0.5 is healthy. Read |right_frac - demo_right_frac| as the
    bias: >0.10 was fatal in every case observed.
    """
    p, d = np.asarray(policy_wz, float), np.asarray(demo_wz, float)
    out = {}
    for tag, v in (("policy", p), ("demo", d)):
        turning = np.abs(v) > thresh
        n = int(turning.sum())
        r = float((v[turning] < 0).mean()) if n else float("nan")
        out[f"{tag}_right_frac"] = r
        out[f"{tag}_n_turning"] = n
        if tag == "policy":
            out["se"] = float(np.sqrt(r * (1 - r) / n)) if n else float("nan")
    out["bias"] = abs(out["policy_right_frac"] - out["demo_right_frac"])
    return out


def report_symmetry(tag, s):
    print(f"[{tag}] TURN SYMMETRY: policy right {s['policy_right_frac']*100:.1f}% "
          f"+-{s['se']*196:.1f} (n={s['policy_n_turning']}) vs demo "
          f"{s['demo_right_frac']*100:.1f}% | bias {s['bias']*100:+.1f} pts"
          f"{'  <-- STEERING BIAS' if s['bias'] > 0.10 else ''}", flush=True)


def summarize(bands, headline):
    return dict(bands=bands, headline_band=[headline["lo_mm"], headline["hi_mm"]],
                term_vx_mms=headline["policy_vx_mms"], term_vx_ratio=headline["vx_ratio"],
                term_idle_frac=headline["policy_idle_frac"],
                demo_vx_mms=headline["demo_vx_mms"],
                demo_idle_frac=headline["demo_idle_frac"], n_frames=headline["n"])


def report(tag, t):
    """Print the headline line plus the per-band breakdown."""
    print(f"[{tag}] TERMINAL {t['headline_band'][0]}-{t['headline_band'][1]}mm: "
          f"policy {t['term_vx_mms']:.1f} mm/s vs demo {t['demo_vx_mms']:.1f} "
          f"(ratio {t['term_vx_ratio']:.2f}) | parked {t['term_idle_frac']*100:.0f}% "
          f"vs demo {t['demo_idle_frac']*100:.0f}% | n={t['n_frames']}", flush=True)
    for b in t["bands"]:
        print(f"[{tag}]   {b['lo_mm']:>4}-{b['hi_mm']:<5} mm | policy {b['policy_vx_mms']:6.1f} "
              f"demo {b['demo_vx_mms']:6.1f} ratio {b['vx_ratio']:5.2f} | parked "
              f"{b['policy_idle_frac']*100:3.0f}%/{b['demo_idle_frac']*100:3.0f}% | "
              f"n={b['n']}", flush=True)
