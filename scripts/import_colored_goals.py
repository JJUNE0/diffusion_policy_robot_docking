#!/usr/bin/env python3
"""Import user-provided canonical colored docking goals into GoalPool."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.goal_pool import GoalPool  # noqa: E402


SPECS = {
    "after_0328": {
        "directory": "4f_dock_colored",
        "colors": ("black", "dark_grey", "grey", "white"),
        "orbbec": "4f_orbbec0_{color}.png",
        "usb": "4f_usb0_{color}.jpg",
    },
    "front_dock_5f_2cam": {
        "directory": "5f_colored",
        "colors": ("black", "green", "greenlight", "lime", "white"),
        "orbbec": "orbbec0_{color}_5f.png",
        "usb": "usb0_{color}_5f.jpg",
    },
}


def import_colored(args):
    root = Path(args.source_root).expanduser().resolve()
    with GoalPool(args.db) as pool:
        for dataset, spec in SPECS.items():
            for offset, color in enumerate(spec["colors"], 1):
                paths = {
                    "orbbec-0": root / spec["directory"] / spec["orbbec"].format(color=color),
                    "usb-0": root / spec["directory"] / spec["usb"].format(color=color),
                }
                missing = [str(path) for path in paths.values() if not path.is_file()]
                if missing:
                    raise FileNotFoundError("missing colored goal image(s): " + ", ".join(missing))
                goal_id = f"{dataset}__all__canonical_{color}__color_changed"
                exists = pool.conn.execute(
                    "SELECT 1 FROM goals WHERE goal_id=?", (goal_id,)
                ).fetchone()
                if exists and not args.replace:
                    print(f"skip existing: {goal_id}")
                    continue
                pool.upsert(
                    goal_id=goal_id,
                    dataset=dataset,
                    split="all",
                    episode=-offset,
                    variant="color_changed",
                    images=paths,
                    enabled=True,
                    note=f"user-provided canonical {color} docking-station color",
                    metadata={
                        "source": str(root),
                        "color": color,
                        "canonical_goal": True,
                        "edited_camera": "orbbec-0",
                        "paired_camera": "usb-0",
                    },
                    replace=args.replace,
                )
                print(f"imported: {goal_id}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="dataset/goal_pool/goal_pool.sqlite3")
    p.add_argument("--source-root", default="dataset/goal_pool/images/_colored")
    p.add_argument("--replace", action="store_true")
    return p


if __name__ == "__main__":
    import_colored(parser().parse_args())
