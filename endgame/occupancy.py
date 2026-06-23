"""Occupancy raster utilities — AUXILIARY SIGNAL ONLY.

CLAUDE.md §4 / §6 are explicit and non-negotiable:

    "occupancy raster ≠ mm 정밀: grid 양자화로 mm 보장 불가. 종단 pose는
     raw-point known-shape ICP가 책임. occupancy 정합도는 핸드오프 트리거 /
     ICP 수렴검증 / aliasing 방어 보조 신호로만."

So NOTHING in this file ever returns a pose. It returns coarse, grid-quantized
*scores* used only to (a) gate the policy->ICP handoff, (b) cross-check that an
ICP fit is geometrically plausible, and (c) help flag aliasing. The mm pose is
owned exclusively by ``endgame/icp_matcher.py``.

Raster convention matches the repo (robot at centre, +x up, +y left).
"""

from __future__ import annotations

import numpy as np


def points_to_occupancy(
    points_xy: np.ndarray,
    range_m: float = 3.0,
    resolution: float = 0.05,
) -> np.ndarray:
    """Rasterize robot-frame 2D points into a binary occupancy grid (H, W).

    Robot at centre pixel; ``+x`` (forward) -> up, ``+y`` (left) -> left.
    Out-of-range points are dropped. Returns a boolean array.
    """
    size = int(round(range_m / resolution))
    grid = np.zeros((size, size), dtype=bool)
    if points_xy is None or len(points_xy) == 0:
        return grid
    pts = np.asarray(points_xy, dtype=np.float64)
    c = size // 2
    rows = (c - np.round(pts[:, 0] / resolution)).astype(np.int64)
    cols = (c - np.round(pts[:, 1] / resolution)).astype(np.int64)
    valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    grid[rows[valid], cols[valid]] = True
    return grid


def occupancy_iou(grid_a: np.ndarray, grid_b: np.ndarray, dilate: int = 1) -> float:
    """Intersection-over-union of two occupancy grids (coarse overlap score).

    A small ``dilate`` tolerates one-cell quantization jitter — a reminder that
    this number is intentionally fuzzy and must not be read as a metric pose.
    """
    a = _dilate(grid_a, dilate)
    b = _dilate(grid_b, dilate)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _dilate(grid: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return grid
    out = grid.copy()
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            out |= np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
    return out


def occupancy_match_score(
    scan_points: np.ndarray,
    template_points_at_pose: np.ndarray,
    range_m: float = 3.0,
    resolution: float = 0.05,
) -> float:
    """Auxiliary IoU between the observed scan and a posed template.

    Used by the handoff trigger / ICP convergence cross-check. NOT a pose.
    """
    g_scan = points_to_occupancy(scan_points, range_m, resolution)
    g_tmpl = points_to_occupancy(template_points_at_pose, range_m, resolution)
    return occupancy_iou(g_scan, g_tmpl, dilate=1)
