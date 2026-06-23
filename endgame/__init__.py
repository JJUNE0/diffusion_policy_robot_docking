"""LiDAR known-shape ICP — OFFLINE dock-pose labeling & template building.

Option A (single learned model): ICP runs **offline only** — it generates the
dock-pose labels (scripts/label_subgoals.py) and the canonical template
(scripts/build_dock_template.py) that supervise the policy's precision aux head.
ICP does NOT run at inference. (The old runtime two-regime handoff/orchestrator
was removed; see docs/plan/00_overview.md.)
"""

from .config import ExtrinsicCalibration, ICPConfig, TargetShapeConfig
from .icp_matcher import ICPMatcher, ICPResult
from .target_model import TargetTemplate, make_template, template_from_config

__all__ = [
    "ICPConfig",
    "TargetShapeConfig",
    "ExtrinsicCalibration",
    "ICPMatcher",
    "ICPResult",
    "TargetTemplate",
    "make_template",
    "template_from_config",
]
