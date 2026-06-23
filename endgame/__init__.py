"""Two-regime LiDAR end-game subsystem (CLAUDE.md §2.1).

Terminal mm-precision docking via 2D LiDAR raw-point known-shape ICP, plus the
explicit policy->LiDAR handoff trigger. Kept entirely separate from the learned
diffusion-policy / torch stack (work guideline §3): the learned policy is
injected into :class:`TwoRegimeController` as an opaque ``policy_fn``.

Quick start
-----------
>>> from endgame import EndgameConfig, TwoRegimeController
>>> ctrl = TwoRegimeController(EndgameConfig(), policy_fn=my_policy)
>>> info = ctrl.step(lidar_points_xy, policy_obs)   # -> {'v', 'w', 'mode', ...}
"""

from .config import (
    EndgameConfig,
    ExtrinsicCalibration,
    HandoffConfig,
    ICPConfig,
    TargetShapeConfig,
)
from .handoff import HandoffController, HandoffDecision, Regime
from .icp_matcher import ICPMatcher, ICPResult
from .orchestrator import ServoConfig, TwoRegimeController
from .target_model import TargetTemplate, make_template, template_from_config

__all__ = [
    "EndgameConfig",
    "ExtrinsicCalibration",
    "TargetShapeConfig",
    "ICPConfig",
    "HandoffConfig",
    "ICPMatcher",
    "ICPResult",
    "HandoffController",
    "HandoffDecision",
    "Regime",
    "TwoRegimeController",
    "ServoConfig",
    "TargetTemplate",
    "make_template",
    "template_from_config",
]
