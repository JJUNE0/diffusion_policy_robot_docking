"""Known dock-shape template for raw-point ICP (CLAUDE.md §2.1, §4).

The end-game matcher aligns the live LiDAR scan against a *known* target shape
to recover an mm-level SE(2) pose.  This module holds that shape as a dense 2D
point cloud in the target's own frame, plus the **asymmetry / aliasing metadata**
that CLAUDE.md §4 insists be made explicit ("타겟 형상 가정을 코드 주석/설정에 명시").

The built-in templates intentionally include one *asymmetric* dock (the design
assumption) and one *symmetric* shape used to demonstrate aliasing rejection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def _sample_segment(p0, p1, spacing: float) -> np.ndarray:
    """Evenly sample points along the segment ``p0 -> p1`` at ~``spacing`` (m)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    length = float(np.hypot(*(p1 - p0)))
    n = max(2, int(round(length / max(spacing, 1e-6))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    return p0[None, :] * (1 - ts)[:, None] + p1[None, :] * ts[:, None]


def _polyline(vertices, spacing: float, closed: bool = False) -> np.ndarray:
    """Sample a polyline (list of vertices) into a point cloud."""
    verts = list(vertices)
    if closed:
        verts = verts + [verts[0]]
    chunks = [_sample_segment(verts[i], verts[i + 1], spacing) for i in range(len(verts) - 1)]
    pts = np.concatenate(chunks, axis=0)
    return np.unique(np.round(pts, 6), axis=0)


@dataclass
class TargetTemplate:
    """A known target shape as ``points`` (N, 2) in the target frame."""

    name: str
    points: np.ndarray            # (N, 2) float64, target frame (origin = dock ref)
    is_asymmetric: bool           # design assumption flag (CLAUDE.md §4)
    description: str = ""

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.points)

    def self_similarity_under_rotation(self, yaw: float) -> float:
        """Mean nearest-neighbour distance to itself after rotating by ``yaw``
        about the shape **centroid** (the axis a rigid pose can exploit).

        Near-zero for a yaw that is a rigid symmetry of the shape -> aliasing
        risk. Quantifies the §4 asymmetry assumption rather than just asserting.
        """
        from scipy.spatial import cKDTree

        c, s = np.cos(yaw), np.sin(yaw)
        ctr = self.points.mean(axis=0)
        rot = (self.points - ctr) @ np.array([[c, -s], [s, c]]).T + ctr
        d, _ = cKDTree(self.points).query(rot, k=1)
        return float(np.mean(d))


def make_template(name: str, spacing: float = 0.01) -> TargetTemplate:
    """Build a built-in template at ~``spacing`` (m) point density."""
    if name == "l_notch_dock":
        # Asymmetric V-notch dock: a wide funnel with an off-centre catch.
        # The asymmetry (longer left wing) yields a unique ICP pose.
        verts = [
            (-0.25, 0.30),   # left wing tip (long)
            (-0.05, 0.05),   # notch left shoulder
            (0.00, 0.00),    # notch apex (dock reference origin)
            (0.05, 0.05),    # notch right shoulder
            (0.15, 0.18),    # right wing tip (short -> asymmetry)
        ]
        return TargetTemplate(
            name=name,
            points=_polyline(verts, spacing),
            is_asymmetric=True,
            description="V-notch charging dock with asymmetric wings (design target).",
        )

    if name == "symmetric_rect":
        # CLOSED non-square rectangle: 180-deg rotationally symmetric about its
        # centroid -> two equally-good rigid ICP poses (aliasing). Deliberately
        # exercises the aliasing-rejection path; NOT a valid dock (violates §4).
        verts = [(-0.20, 0.00), (-0.20, 0.20), (0.20, 0.20), (0.20, 0.00)]
        return TargetTemplate(
            name=name,
            points=_polyline(verts, spacing, closed=True),
            is_asymmetric=False,
            description="180-deg symmetric rectangle — aliasing demo, violates §4.",
        )

    if name == "real_dock":
        # Canonical real dock (U-notch + shoulders) aggregated from the
        # after_0328 LiDAR by scripts/build_dock_template.py. Borderline
        # 180-symmetric -> pair with ICPConfig.for_real_dock() (narrow restart
        # band) and accumulate scans at dwell for sub-mm. See memory
        # real-lidar-icp-mm-confirmed / dock-identity-sensor-frame.
        path = os.path.join(os.path.dirname(__file__), "assets", "dock_template_real.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing — run `python scripts/build_dock_template.py` first.")
        return TargetTemplate(
            name=name, points=np.load(path), is_asymmetric=False,
            description="Canonical real after_0328 dock (U-notch + shoulders).")

    raise ValueError(
        f"Unknown template '{name}'. Known: l_notch_dock, symmetric_rect, real_dock.")


def load_template(path: str, name: str = "custom", is_asymmetric: bool = True) -> TargetTemplate:
    """Load a custom ``(N, 2)`` template point cloud from a ``.npy`` file."""
    pts = np.load(path)
    return TargetTemplate(name=name, points=pts, is_asymmetric=is_asymmetric,
                          description=f"Loaded from {path}")


def template_from_config(target_cfg) -> TargetTemplate:
    """Resolve a :class:`endgame.config.TargetShapeConfig` to a template."""
    if target_cfg.template_path:
        return load_template(target_cfg.template_path, is_asymmetric=target_cfg.is_asymmetric)
    return make_template(target_cfg.template_name)
