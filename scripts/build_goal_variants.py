#!/usr/bin/env python3
"""Build pixel-aligned QR-removed and station-color goal variants.

The edits deliberately operate only on the orbbec-0 image. The usb-0 camera
never sees the station in this dataset, so its bytes are copied unchanged into
each derived, camera-paired goal. All outputs remain 320x240 JPEG files.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from utils.goal_pool import GoalPool  # noqa: E402

AFTER_DATASET = "after_0328"
F5_DATASET = "front_dock_5f_2cam"
ORIGINAL = "original"
QR_REMOVED = "qr_removed"
COLOR_CHANGED = "color_changed"
BLUE_HUE = 112


def _match(im, template, roi):
    x0, y0, x1, y1 = roi
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(
        gray[y0:y1, x0:x1], template, cv2.TM_CCOEFF_NORMED
    )
    _, score, _, loc = cv2.minMaxLoc(result)
    return x0 + loc[0], y0 + loc[1], template.shape[1], template.shape[0], score


def _inpaint_box(im, box, pad=2, radius=3):
    x, y, w, h, *_ = box
    mask = np.zeros(im.shape[:2], np.uint8)
    cv2.rectangle(
        mask,
        (max(0, x - pad), max(0, y - pad)),
        (min(im.shape[1] - 1, x + w + pad), min(im.shape[0] - 1, y + h + pad)),
        255,
        -1,
    )
    return cv2.inpaint(im, mask, radius, cv2.INPAINT_TELEA), mask


def _recolor(im, mask, hue=BLUE_HUE):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    shifted = hsv.copy()
    shifted[..., 0] = int(hue)
    shifted[..., 1] = np.maximum(shifted[..., 1], 165)
    colored = cv2.cvtColor(shifted, cv2.COLOR_HSV2BGR)
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.7).astype(np.float32)[..., None] / 255.0
    return np.clip(
        im.astype(np.float32) * (1.0 - alpha)
        + colored.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _large_components(mask, min_area=100):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 255
    return out


def _after_variants(im, template):
    box = _match(im, template, (105, 190, 230, 240))
    qr, qr_mask = _inpaint_box(im, box, pad=2)
    x, y, w, h, score = box
    cx = x + w // 2

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([25, 45, 20]), np.array([100, 255, 245]))
    station_region = np.zeros_like(green)
    cv2.fillPoly(
        station_region,
        [np.array([
            [x - 28, y - 64], [x + w + 30, y - 64],
            [x + w + 40, y + 25], [x - 38, y + 25],
        ], np.int32)],
        255,
    )
    color_mask = cv2.bitwise_and(green, station_region)
    cv2.ellipse(color_mask, (cx, y - 42), (39, 27), 0, 0, 360, 0, -1)
    cv2.rectangle(
        color_mask, (x - 3, y - 3), (x + w + 3, min(239, y + h + 3)), 0, -1
    )
    color_mask = _large_components(color_mask, min_area=100)
    if np.count_nonzero(color_mask) < 500:
        raise ValueError(f"after_0328 color mask unexpectedly small: {np.count_nonzero(color_mask)}")
    color = _recolor(im, color_mask)
    return qr, color, qr_mask, color_mask, score


def _f5_variants(im, episode, template):
    if episode == 60:
        box = (181, 29, 29, 22, 1.0)
        qr, qr_mask = _inpaint_box(im, box, pad=1)
        region = np.zeros(im.shape[:2], np.uint8)
        cv2.fillPoly(
            region,
            [
                np.array([[164, 0], [214, 0], [215, 63], [164, 62]], np.int32),
                np.array([[151, 52], [217, 50], [222, 96], [155, 108], [146, 91]], np.int32),
            ],
            255,
        )
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        low_sat = cv2.inRange(hsv, np.array([0, 0, 35]), np.array([179, 120, 235]))
        color_mask = cv2.bitwise_and(region, low_sat)
        cv2.rectangle(color_mask, (177, 0), (196, 11), 0, -1)
        # The color-only variant keeps its QR marker unchanged.
        cv2.rectangle(color_mask, (178, 26), (213, 53), 0, -1)
    else:
        box = _match(im, template, (90, 195, 240, 240))
        qr, qr_mask = _inpaint_box(im, box, pad=1)
        x, y, w, h, _ = box
        top = np.array(
            [[x - 22, y - 43], [x + 76, y - 45], [x + 84, y - 5], [x - 27, y - 2]],
            np.int32,
        )
        front = np.array(
            [[x - 27, y - 2], [x + 84, y - 5], [x + 76, y + 18], [x - 20, y + 18]],
            np.int32,
        )
        region = np.zeros(im.shape[:2], np.uint8)
        cv2.fillPoly(region, [top, front], 255)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        low_sat = cv2.inRange(hsv, np.array([0, 0, 35]), np.array([179, 120, 235]))
        color_mask = cv2.bitwise_and(region, low_sat)
        cv2.ellipse(color_mask, (x + 10, y - 45), (24, 39), 0, 0, 360, 0, -1)
        cv2.rectangle(color_mask, (x - 3, y - 3), (x + w + 3, 239), 0, -1)
    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    if np.count_nonzero(color_mask) < 300:
        raise ValueError(
            f"5F ep{episode:04d} color mask unexpectedly small: "
            f"{np.count_nonzero(color_mask)}"
        )
    color = _recolor(im, color_mask)
    return qr, color, qr_mask, color_mask, box[-1]


def _write_jpeg(path, image, quality):
    if image.shape != (240, 320, 3):
        raise ValueError(f"expected 240x320 BGR image, got {image.shape}")
    if not cv2.imwrite(
        str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ):
        raise IOError(f"failed to write {path}")


def build(args):
    with GoalPool(args.db) as pool:
        originals = pool.records(enabled=True, splits=["all"], variants=[ORIGINAL])
        if len(originals) != args.expected_goals:
            raise ValueError(
                f"expected {args.expected_goals} original paired goals, got {len(originals)}"
            )
        lookup = {(r.dataset, r.episode): r for r in originals}
        after_ref = cv2.imread(lookup[(AFTER_DATASET, 0)].images["orbbec-0"]["path"], 0)
        f5_ref = cv2.imread(lookup[(F5_DATASET, 0)].images["orbbec-0"]["path"], 0)
        after_template = after_ref[207:220, 139:189]
        f5_template = f5_ref[228:240, 135:195]

        qr_scores = []
        for index, rec in enumerate(originals, 1):
            source_orbbec = rec.images["orbbec-0"]["path"]
            source_usb = rec.images["usb-0"]["path"]
            image = cv2.imread(source_orbbec)
            if image is None:
                raise FileNotFoundError(source_orbbec)
            if rec.dataset == AFTER_DATASET:
                qr, color, qr_mask, color_mask, score = _after_variants(
                    image, after_template
                )
            elif rec.dataset == F5_DATASET:
                qr, color, qr_mask, color_mask, score = _f5_variants(
                    image, rec.episode, f5_template
                )
            else:
                raise ValueError(f"unsupported dataset: {rec.dataset}")
            qr_scores.append(float(score))

            with tempfile.TemporaryDirectory(prefix="goal_variant_") as temp_dir:
                temp = Path(temp_dir)
                qr_path = temp / "orbbec-0.jpg"
                color_path = temp / "orbbec-0-color.jpg"
                _write_jpeg(qr_path, qr, args.jpeg_quality)
                _write_jpeg(color_path, color, args.jpeg_quality)
                common = dict(
                    dataset=rec.dataset,
                    split=rec.split,
                    episode=rec.episode,
                    enabled=True,
                    parent_id=rec.goal_id,
                    replace=args.replace,
                )
                pool.upsert(
                    goal_id=pool.default_goal_id(
                        rec.dataset, rec.split, rec.episode, QR_REMOVED
                    ),
                    variant=QR_REMOVED,
                    images={"orbbec-0": qr_path, "usb-0": source_usb},
                    note="pixel-aligned OpenCV QR/fiducial inpainting",
                    metadata={
                        "generator": "scripts/build_goal_variants.py",
                        "algorithm": "cv2.INPAINT_TELEA",
                        "source_goal_id": rec.goal_id,
                        "template_score": float(score),
                        "edited_camera": "orbbec-0",
                        "copied_camera": "usb-0",
                        "mask_pixels": int(np.count_nonzero(qr_mask)),
                    },
                    **common,
                )
                pool.upsert(
                    goal_id=pool.default_goal_id(
                        rec.dataset, rec.split, rec.episode, COLOR_CHANGED
                    ),
                    variant=COLOR_CHANGED,
                    images={"orbbec-0": color_path, "usb-0": source_usb},
                    note="pixel-aligned cobalt-blue station recolor",
                    metadata={
                        "generator": "scripts/build_goal_variants.py",
                        "algorithm": "HSV hue replacement with feathered mask",
                        "source_goal_id": rec.goal_id,
                        "edited_camera": "orbbec-0",
                        "copied_camera": "usb-0",
                        "opencv_hue": BLUE_HUE,
                        "mask_pixels": int(np.count_nonzero(color_mask)),
                    },
                    **common,
                )
            if index % 25 == 0 or index == len(originals):
                print(f"processed {index}/{len(originals)} paired goals")

        print(
            f"Built {len(originals)} {QR_REMOVED} pairs and "
            f"{len(originals)} {COLOR_CHANGED} pairs; "
            f"template score min={min(qr_scores):.3f}, max={max(qr_scores):.3f}"
        )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--expected-goals", type=int, default=236)
    p.add_argument("--jpeg-quality", type=int, default=95, choices=range(1, 101))
    p.add_argument("--replace", action="store_true")
    return p


if __name__ == "__main__":
    build(parser().parse_args())
