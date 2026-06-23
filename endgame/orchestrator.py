"""Two-regime controller: learned policy (approach) + LiDAR ICP (mm end-game).

This is the explicit realisation of CLAUDE.md §2.1's two-regime block:

    [wide/mid range]  learned diffusion policy  -- delegated, NOT reimplemented
            │
        [handoff]     HandoffController (explicit, hysteretic)
            │
    [terminal mm]     LiDAR raw-point ICP servo

The learned policy is injected as an opaque ``policy_fn(obs) -> (v, w)`` so this
module never touches the torch / diffusion training stack — keeping the two
regimes cleanly separated in code as the design demands (work guideline §3).

The end-game servo is a deliberately simple unicycle go-to-pose controller: the
research contribution is the ICP + handoff structure, not the low-level
controller, so it is kept minimal and documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .config import EndgameConfig
from .handoff import HandoffController, Regime
from .icp_matcher import ICPMatcher
from .se2 import compose, inverse, pose_distance, wrap_angle
from .target_model import template_from_config

PolicyFn = Callable[[object], "tuple[float, float]"]


@dataclass
class ServoConfig:
    """Unicycle go-to-pose gains/limits for the end-game (controller tuning)."""

    k_rho: float = 1.2       # forward gain on distance-to-goal
    k_alpha: float = 3.0     # heading gain toward goal bearing
    k_beta: float = -0.8     # final-orientation alignment gain
    v_max: float = 0.15      # m/s
    w_max: float = 0.8       # rad/s
    position_deadband_m: float = 0.004  # below this, only align final heading


class TwoRegimeController:
    """Orchestrates APPROACH -> ENGAGED -> DONE for a single docking run."""

    def __init__(
        self,
        cfg: Optional[EndgameConfig] = None,
        policy_fn: Optional[PolicyFn] = None,
        servo: Optional[ServoConfig] = None,
    ):
        self.cfg = cfg or EndgameConfig()
        self.policy_fn = policy_fn
        self.servo = servo or ServoConfig()

        self.template = template_from_config(self.cfg.target)
        self.matcher = ICPMatcher(self.template, self.cfg.icp)
        self.dock_target_pose = np.asarray(self.cfg.target.dock_target_pose, dtype=np.float64)
        self.handoff = HandoffController(
            self.matcher, self.cfg.handoff, dock_target_pose=self.dock_target_pose)

        self.regime = Regime.APPROACH
        self._release_streak = 0

    def reset(self):
        self.regime = Regime.APPROACH
        self._release_streak = 0
        self.handoff.reset()

    # ------------------------------------------------------------- end-game
    def _servo_to_dock(self, target_pose_robot: np.ndarray):
        """Velocity command that drives the observed target pose to the dock pose.

        If the robot moves by ``g`` (in its own frame), a world-fixed point's
        robot-frame coordinates transform by ``g^{-1}``. We want the target to
        end up at ``dock_target_pose``, so the robot's goal pose is
        ``g = target_pose ∘ dock_target_pose^{-1}`` — the SE(2) move that carries
        the current target observation onto the desired docked observation.
        """
        g = compose(target_pose_robot, inverse(self.dock_target_pose))
        gx, gy, gth = float(g[0]), float(g[1]), float(g[2])
        s = self.servo

        rho = float(np.hypot(gx, gy))
        if rho < s.position_deadband_m:
            # Close enough in position: rotate to the final heading only.
            v = 0.0
            w = float(np.clip(s.k_alpha * wrap_angle(gth), -s.w_max, s.w_max))
            return v, w

        alpha = wrap_angle(np.arctan2(gy, gx))
        beta = wrap_angle(gth - np.arctan2(gy, gx))
        # Slow forward motion while badly misaligned (cos gate keeps it sane).
        v = float(np.clip(s.k_rho * rho * np.cos(alpha), -s.v_max, s.v_max))
        w = float(np.clip(s.k_alpha * alpha + s.k_beta * beta, -s.w_max, s.w_max))
        return v, w

    # ------------------------------------------------------------------ step
    def step(self, scan_points, policy_obs=None) -> Dict:
        """Advance one control tick.

        Parameters
        ----------
        scan_points : (N, 2) array
            Raw 2D LiDAR points in the LiDAR frame. The configured extrinsic is
            applied here (identity by default) to bring them to the robot frame.
        policy_obs : object
            Opaque observation forwarded to ``policy_fn`` during APPROACH.

        Returns a dict with ``v``, ``w``, ``mode``, ``done`` and the relevant
        ICP / handoff diagnostics for logging.
        """
        scan_robot = self.cfg.extrinsic.apply(np.asarray(scan_points, dtype=np.float64).reshape(-1, 2))
        info: Dict = {"mode": self.regime.value, "v": 0.0, "w": 0.0, "done": False}

        if self.regime is Regime.DONE:
            info["done"] = True
            return info

        if self.regime is Regime.APPROACH:
            decision = self.handoff.evaluate(scan_robot, self.dock_target_pose)
            info["handoff"] = decision
            if self.handoff.update(decision):
                self.regime = Regime.ENGAGED
                self._release_streak = 0
                info["mode"] = self.regime.value
                info["event"] = "engaged"
            else:
                v, w = self.policy_fn(policy_obs) if self.policy_fn is not None else (0.0, 0.0)
                info.update(v=float(v), w=float(w))
                return info

        # ENGAGED (also entered on the same tick we just switched)
        result = self.matcher.match(scan_robot, self.dock_target_pose)
        info["icp"] = result
        info["mode"] = self.regime.value

        if not result.is_trustworthy():
            # Lost the geometric lock or became aliased: fall back to the policy
            # rather than command a servo from an untrustworthy pose (§4).
            self._release_streak += 1
            if self._release_streak >= self.cfg.handoff.release_consecutive_frames:
                self.regime = Regime.APPROACH
                self.handoff.reset()
                info["event"] = "released_to_policy"
            v, w = self.policy_fn(policy_obs) if self.policy_fn is not None else (0.0, 0.0)
            info.update(v=float(v), w=float(w), mode=self.regime.value)
            return info

        self._release_streak = 0
        trans_err, rot_err = pose_distance(result.pose, self.dock_target_pose)
        info["translation_err_m"] = trans_err
        info["rotation_err_rad"] = rot_err

        if trans_err <= self.cfg.done_translation_tol_m and rot_err <= self.cfg.done_rotation_tol_rad:
            self.regime = Regime.DONE
            info.update(v=0.0, w=0.0, done=True, mode=self.regime.value, event="docked")
            return info

        v, w = self._servo_to_dock(result.pose)
        info.update(v=v, w=w)
        return info
