"""Configuration for the two-regime LiDAR end-game subsystem.

CLAUDE.md design hooks
----------------------
* §2.4 / §4 — *extrinsic calibration sets the mm ceiling*. Every calibration
  number that bounds docking precision is gathered **here, in one place**, so it
  can be audited and version-controlled rather than scattered across modules.
* §4 — *LiDAR geometry aliasing*: the target-shape assumption (the dock has a
  LiDAR-distinguishable, **asymmetric** geometry) is made explicit on
  :class:`TargetShapeConfig` so a violated assumption is visible in config.
* §6 — *occupancy is an auxiliary signal only*: occupancy thresholds live under
  :class:`HandoffConfig`, never on a "produce mm pose" path.

All distances are in **meters**, angles in **radians**, unless noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtrinsicCalibration:
    """LiDAR -> robot-base rigid mounting (SE(2)).

    The end-game module ingests raw scans already expressed in the robot base
    frame.  If your driver delivers points in the LiDAR frame, set these and
    call :meth:`apply` first.

    WARNING (CLAUDE.md §4): any error here is a *hard lower bound* on docking
    precision.  A cm-level extrinsic makes mm docking impossible no matter how
    good the scan matcher is.  Treat these numbers as suspect and re-measure.
    """

    x: float = 0.0          # LiDAR position on robot base, forward (m)
    y: float = 0.0          # LiDAR position on robot base, left (m)
    theta: float = 0.0      # LiDAR yaw w.r.t. robot base (rad)
    # Honest record of how well the above is known. Surfaced in diagnostics so a
    # claimed "mm" result can be sanity-checked against the calibration ceiling.
    est_translation_error_m: float = 0.002
    est_rotation_error_rad: float = 0.005

    def apply(self, points_lidar_frame):
        """Map scan points from the LiDAR frame into the robot base frame."""
        from .se2 import transform_points  # local import to avoid cycle

        return transform_points([self.x, self.y, self.theta], points_lidar_frame)


@dataclass
class TargetShapeConfig:
    """Known dock geometry assumption (CLAUDE.md §4 aliasing guard).

    The whole end-game hinges on the dock presenting a LiDAR-distinguishable
    geometry (a notch / protrusion / asymmetry).  ``is_asymmetric=False`` is a
    red flag: a symmetric target makes the ICP pose ambiguous (aliasing) and the
    matcher will (correctly) refuse to claim a unique mm pose.
    """

    # Named built-in template, see endgame/target_model.py (e.g. "l_notch_dock").
    template_name: str = "l_notch_dock"
    # Path to a custom (N, 2) template point cloud in the target frame (.npy).
    template_path: Optional[str] = None
    # Asserted property: the geometry is non-symmetric -> unique ICP pose.
    is_asymmetric: bool = True
    # Desired pose of the *target*, expressed in the robot frame, when perfectly
    # docked. The controller drives the observed target pose to this.
    dock_target_pose: tuple = (0.40, 0.0, 0.0)  # [x_fwd, y_left, theta]


@dataclass
class ICPConfig:
    """Raw-point known-shape ICP parameters (CLAUDE.md §2.1 mm endgame)."""

    max_iterations: int = 50
    # Correspondence rejection: drop matches farther than this (m). Annealed
    # from *_start to *_end across iterations for coarse-to-fine matching.
    max_correspondence_dist_start: float = 0.30
    max_correspondence_dist_end: float = 0.03
    # Convergence: stop when per-iter pose update falls below these.
    tol_translation_m: float = 1e-4
    tol_rotation_rad: float = 1e-4
    # Acceptance gates for a "trustworthy" fit.
    min_inlier_ratio: float = 0.6
    max_rms_residual_m: float = 0.01
    # Aliasing guard: restart ICP from these yaw offsets (rad) and require a
    # single dominant low-residual basin. Multiple basins => ambiguous => reject.
    restart_yaws: tuple = (0.0, 1.5708, 3.14159, -1.5708)
    # Two restarts count as the "same" pose if within these of each other.
    pose_cluster_translation_m: float = 0.05
    pose_cluster_rotation_rad: float = 0.087  # ~5 deg

    @classmethod
    def for_real_dock(cls) -> "ICPConfig":
        """Settings for the real after_0328 dock (U-notch, borderline 180-symmetric).

        Narrow restart band only: the policy hands off roughly aligned, so a full
        360 deg search is unnecessary and is the sole source of catastrophic 180
        deg flips on this geometry (verified on real data). See memory
        real-lidar-icp-mm-confirmed.
        """
        return cls(restart_yaws=(0.0, 0.087, -0.087, 0.175, -0.175))  # 0, +/-5, +/-10 deg


@dataclass
class HandoffConfig:
    """Explicit, debuggable policy->ICP handoff trigger (CLAUDE.md §2.1, §4).

    Trigger = (LiDAR observes target geometry sufficiently) AND (roughly
    aligned) AND (not aliased).  Occupancy numbers here are *auxiliary only*
    (§6): handoff gating, ICP convergence cross-check, aliasing defense — never
    a mm-pose source.
    """

    # "Geometry observed": coarse ICP must clear these to count as a real lock.
    min_scan_points: int = 30
    min_inlier_ratio: float = 0.55
    max_rms_residual_m: float = 0.02
    # "Roughly aligned": coarse target pose must be within this band of the dock.
    align_translation_tol_m: float = 0.25
    align_rotation_tol_rad: float = 0.35
    # Auxiliary occupancy cross-check (set <=0 to disable): IoU of observed scan
    # occupancy vs template rendered at the estimated pose.
    min_occupancy_iou: float = 0.20
    occupancy_range_m: float = 3.0
    occupancy_resolution_m: float = 0.05
    # Hysteresis: frames the trigger must hold / be lost before switching, to
    # avoid chattering at the handoff boundary.
    engage_consecutive_frames: int = 3
    release_consecutive_frames: int = 5


@dataclass
class EndgameConfig:
    """Top-level container aggregating every end-game knob in one place."""

    extrinsic: ExtrinsicCalibration = field(default_factory=ExtrinsicCalibration)
    target: TargetShapeConfig = field(default_factory=TargetShapeConfig)
    icp: ICPConfig = field(default_factory=ICPConfig)
    handoff: HandoffConfig = field(default_factory=HandoffConfig)
    # Docking is considered DONE when the residual pose error is within these.
    done_translation_tol_m: float = 0.003   # 3 mm (the servo AIMS this tight)
    done_rotation_tol_rad: float = 0.0087   # ~0.5 deg
    # Acceptance / eval gate: a dock attempt counts as successful within this
    # tolerance (user spec: charging contacts tolerate ~1 cm). Looser than the
    # servo target above. Matches the marker<->ICP cross-validation (~1 cm).
    success_translation_tol_m: float = 0.01
