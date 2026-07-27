"""Live status dashboard for the R-NoGoal/R-Goal/R-Geo training queue
(docs/0725_reloc3r_test/reloc3r/reloc3r_0725.md: "중간 중간 실험 모니터링 기능도
추가하라"). Tails each arm's outputs/train_<name>.log, parses "Step N | Loss:
X" lines, and prints one refreshed table: current step, % done, loss EMA,
step-rate-derived ETA, and last checkpoint written -- so progress across all
three arms is visible at a glance without grepping three log files by hand.

Usage:
  python scripts/monitor_rgeo_training.py [--names r_nogoal,r_goal,r_geo] [--interval 30]
  (Ctrl-C to stop; safe to run alongside test/queue_reloc3r_rgeo.sh, read-only.)
"""
import argparse
import glob
import os
import re
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_RE = re.compile(r"Step (\d+) \| Loss: ([\d.eE+-]+)")
TOTAL_RE = re.compile(r"total_gradient_steps=(\d+)")
DONE_RE = re.compile(r"End Training")


def parse_log(path):
    """Return dict(step, total_steps, loss_ema, done, first_ts, last_ts) by
    scanning the log tail (cheap: only reads the file once per poll)."""
    if not os.path.exists(path):
        return None
    step, total, loss_ema, done = None, None, None, False
    first_step_time, last_step_time = None, None
    mtimes = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = TOTAL_RE.search(line)
            if m:
                total = int(m.group(1))
            m = STEP_RE.search(line)
            if m:
                s, loss = int(m.group(1)), float(m.group(2))
                step = s
                loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss
            if DONE_RE.search(line):
                done = True
    mtime = os.path.getmtime(path)
    return dict(step=step, total=total, loss_ema=loss_ema, done=done, mtime=mtime)


def latest_ckpt(run_glob):
    dirs = sorted(glob.glob(run_glob))
    if not dirs:
        return None, None
    latest_dir = dirs[-1]
    ckpts = glob.glob(os.path.join(latest_dir, "checkpoint_step_*.pt"))
    if not ckpts:
        return latest_dir, None
    best = max(ckpts, key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    return latest_dir, os.path.basename(best)


_last_seen = {}  # name -> (step, wallclock) for rate estimation


def fmt_eta(name, step, total, now):
    if step is None or total is None:
        return "?"
    prev = _last_seen.get(name)
    _last_seen[name] = (step, now)
    if prev is None or prev[0] >= step:
        return "?"
    d_step, d_t = step - prev[0], now - prev[1]
    if d_step <= 0 or d_t <= 0:
        return "?"
    rate = d_step / d_t  # steps/sec
    remain = max(total - step, 0)
    eta_s = remain / rate if rate > 0 else float("inf")
    m, s = divmod(int(eta_s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="r_nogoal,r_goal,r_geo")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    args = ap.parse_args()
    names = args.names.split(",")

    while True:
        now = time.time()
        rows = []
        for name in names:
            log_path = os.path.join(REPO, f"outputs/train_{name}.log")
            info = parse_log(log_path)
            run_dir, ckpt = latest_ckpt(os.path.join(REPO, f"outputs/train/{name}/*"))
            if info is None:
                rows.append((name, "not started", "-", "-", "-", "-", "-"))
                continue
            step, total, loss_ema, done = info["step"], info["total"], info["loss_ema"], info["done"]
            pct = f"{100.0*step/total:.1f}%" if (step and total) else "-"
            status = "DONE" if done else ("running" if now - info["mtime"] < 120 else "STALLED?")
            eta = "-" if done else fmt_eta(name, step, total, now)
            rows.append((
                name, status, f"{step}/{total}" if total else str(step),
                pct, f"{loss_ema:.4f}" if loss_ema is not None else "-",
                eta, ckpt or "-"))

        os.system("clear" if args.once is False else "")
        print(f"=== R-NoGoal/R-Goal/R-Geo training monitor  ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        header = ("arm", "status", "step", "%", "loss(EMA)", "ETA", "last_ckpt")
        widths = [10, 10, 14, 7, 10, 8, 28]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for r in rows:
            print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
