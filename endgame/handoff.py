"""Explicit policy -> LiDAR handoff trigger (CLAUDE.md §2.1, §4).

CLAUDE.md §2.1: "핸드오프 트리거 로직(언제 LiDAR로 넘길지)은 명시적 모듈로
구현하고 디버깅 가능하게 둘 것."  Accordingly this is a *transparent* state
machine: every sub-condition and score that drives a transition is recorded on
the returned :class:`HandoffDecision`, and hysteresis prevents chattering.

Trigger = (LiDAR observes the target geometry sufficiently)
          AND (the robot is roughly aligned with the dock)
          AND (the fit is not aliased)        # §4 aliasing guard

The occupancy IoU cross-check is *auxiliary only* (§6): it can veto a handoff
but never substitutes for the raw-point ICP pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np

from .config import HandoffConfig
from .icp_matcher import ICPMatcher, ICPResult
from .occupancy import occupancy_match_score
from .se2 import pose_distance, transform_points


class Regime(Enum):
    APPROACH = "approach"   # learned diffusion policy drives (wide basin)
    ENGAGED = "engaged"     # LiDAR ICP servo drives (mm end-game)
    DONE = "done"


@dataclass
class HandoffDecision:
    """Transparent record of one handoff evaluation (debug-friendly)."""

    should_engage: bool
    geometry_observed: bool
    roughly_aligned: bool
    not_aliased: bool
    occupancy_ok: bool
    # raw scores behind the booleans
    n_scan_points: int = 0
    inlier_ratio: float = 0.0
    rms_residual_m: float = float("inf")
    translation_err_m: float = float("inf")
    rotation_err_rad: float = float("inf")
    occupancy_iou: float = 0.0
    icp: Optional[ICPResult] = None
    reasons: Dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        flags = "".join("1" if v else "0" for v in
                        (self.geometry_observed, self.roughly_aligned,
                         self.not_aliased, self.occupancy_ok))
        return (f"engage={self.should_engage} [geo/align/uniq/occ={flags}] "
                f"inl={self.inlier_ratio:.2f} rms={self.rms_residual_m*1000:.1f}mm "
                f"pos_err={self.translation_err_m*100:.1f}cm "
                f"yaw_err={np.degrees(self.rotation_err_rad):.1f}deg "
                f"occ_iou={self.occupancy_iou:.2f}")


class HandoffController:
    """Decides when to switch from the learned policy to the LiDAR end-game."""

    def __init__(self, matcher: ICPMatcher, cfg: Optional[HandoffConfig] = None,
                 dock_target_pose=(0.40, 0.0, 0.0)):
        self.matcher = matcher
        self.cfg = cfg or HandoffConfig()
        self.dock_target_pose = np.asarray(dock_target_pose, dtype=np.float64)
        self._engage_streak = 0
        self._release_streak = 0

    def reset(self):
        self._engage_streak = 0
        self._release_streak = 0

    def evaluate(self, scan_pts: np.ndarray, coarse_init_pose) -> HandoffDecision:
        """Run a *coarse* ICP and test all trigger conditions.

        ``coarse_init_pose`` is the policy/odometry guess of the target pose in
        the robot frame (e.g. the configured dock pose) used to seed ICP.
        """
        cfg = self.cfg
        scan_pts = np.asarray(scan_pts, dtype=np.float64).reshape(-1, 2)
        n_pts = len(scan_pts)

        icp = self.matcher.match(scan_pts, coarse_init_pose)

        # 1) geometry observed: enough points + a real lock
        geometry_observed = (
            n_pts >= cfg.min_scan_points
            and icp.inlier_ratio >= cfg.min_inlier_ratio
            and icp.rms_residual_m <= cfg.max_rms_residual_m
        )

        # 2) roughly aligned: coarse pose near the desired dock pose
        trans_err, rot_err = pose_distance(icp.pose, self.dock_target_pose)
        roughly_aligned = (
            trans_err <= cfg.align_translation_tol_m
            and rot_err <= cfg.align_rotation_tol_rad
        )

        # 3) not aliased (§4)
        not_aliased = not icp.ambiguous

        # 4) auxiliary occupancy cross-check (§6) — can veto, never authorises
        if cfg.min_occupancy_iou > 0.0:
            posed_tmpl = transform_points(icp.pose, self.matcher.template.points)
            occ_iou = occupancy_match_score(
                scan_pts, posed_tmpl,
                range_m=cfg.occupancy_range_m, resolution=cfg.occupancy_resolution_m)
            occupancy_ok = occ_iou >= cfg.min_occupancy_iou
        else:
            occ_iou, occupancy_ok = 1.0, True

        should = geometry_observed and roughly_aligned and not_aliased and occupancy_ok
        return HandoffDecision(
            should_engage=should,
            geometry_observed=geometry_observed,
            roughly_aligned=roughly_aligned,
            not_aliased=not_aliased,
            occupancy_ok=occupancy_ok,
            n_scan_points=n_pts,
            inlier_ratio=icp.inlier_ratio,
            rms_residual_m=icp.rms_residual_m,
            translation_err_m=trans_err,
            rotation_err_rad=rot_err,
            occupancy_iou=occ_iou,
            icp=icp,
            reasons={
                "geometry_observed": geometry_observed,
                "roughly_aligned": roughly_aligned,
                "not_aliased": not_aliased,
                "occupancy_ok": occupancy_ok,
            },
        )

    def update(self, decision: HandoffDecision) -> bool:
        """Apply hysteresis; return True once a stable engage is confirmed.

        CLAUDE.md §4 warns about *mis-triggering* when a weak policy never gets
        the robot into the matching basin — so engagement requires a streak and
        is monitored via the streak counters exposed here.
        """
        if decision.should_engage:
            self._engage_streak += 1
            self._release_streak = 0
        else:
            self._release_streak += 1
            self._engage_streak = 0
        return self._engage_streak >= self.cfg.engage_consecutive_frames

    @property
    def engage_streak(self) -> int:
        return self._engage_streak
