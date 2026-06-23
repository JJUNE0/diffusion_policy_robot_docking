"""Raw-point known-shape ICP — the mm end-game (CLAUDE.md §2.1, §6).

This is the *only* place a metric SE(2) pose is produced. It aligns the raw
LiDAR scan (no grid quantization) to the known target template via iterative
closest point with a closed-form 2D solve. Occupancy never enters here.

Two CLAUDE.md guards are baked in:

* §2.1 "mm precision" — point-to-point ICP on raw points + coarse-to-fine
  correspondence annealing, with explicit convergence diagnostics (RMS residual,
  inlier ratio) so a bad lock is *detectable* instead of silently trusted.
* §4 "LiDAR geometry aliasing" — ICP is restarted from several yaw seeds; if
  more than one well-separated pose explains the scan comparably well, the
  result is flagged ``ambiguous`` and must NOT be handed mm authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from .config import ICPConfig
from .se2 import compose, pose_distance, transform_points
from .target_model import TargetTemplate


@dataclass
class ICPResult:
    """Outcome of one :meth:`ICPMatcher.match` call (fully introspectable)."""

    pose: np.ndarray                       # target pose in robot frame [x, y, theta]
    converged: bool                        # passed inlier/residual acceptance gates
    rms_residual_m: float                  # RMS nearest-point residual over inliers
    inlier_ratio: float                    # fraction of template points matched
    num_inliers: int
    iterations: int
    ambiguous: bool = False                # §4 aliasing: >1 comparable basin found
    restart_poses: List[np.ndarray] = field(default_factory=list)
    note: str = ""

    def mm_residual(self) -> float:
        return self.rms_residual_m * 1000.0

    def is_trustworthy(self) -> bool:
        """Safe to grant mm authority: converged and not aliased."""
        return self.converged and not self.ambiguous


def _estimate_normals(scan: np.ndarray, tree: cKDTree, k: int = 8) -> np.ndarray:
    """Per-point local line normals via PCA over each point's k neighbours.

    Point-to-line ICP needs the local surface (here, scan-line) direction. The
    normal is the eigenvector of the smallest eigenvalue of the neighbourhood
    covariance. Isolated clutter points get arbitrary normals but are rejected
    by the correspondence-distance gate anyway.
    """
    kk = min(k, len(scan))
    _, idx = tree.query(scan, k=kk)
    idx = np.atleast_2d(idx)
    normals = np.zeros_like(scan)
    for i in range(len(scan)):
        nb = scan[idx[i]]
        nb = nb - nb.mean(axis=0)
        _, vecs = np.linalg.eigh(nb.T @ nb)
        normals[i] = vecs[:, 0]
    return normals


class ICPMatcher:
    """Known-shape ICP matcher for a single dock template."""

    def __init__(self, template: TargetTemplate, cfg: Optional[ICPConfig] = None):
        self.template = template
        self.cfg = cfg or ICPConfig()

    # ------------------------------------------------------------------ core
    def _icp_once(self, scan_pts, tree, normals, init_pose):
        """One point-to-line ICP run (Gauss-Newton increments).

        Point-to-line tolerates the sliding/aperture problem that traps
        point-to-point ICP on sparse LiDAR polylines, giving a much wider
        convergence basin around the handoff seed.
        """
        cfg = self.cfg
        tmpl = self.template.points
        pose = np.asarray(init_pose, dtype=np.float64).copy()
        n_total = len(tmpl)
        iters = 0
        d_start, d_end = cfg.max_correspondence_dist_start, cfg.max_correspondence_dist_end

        for it in range(cfg.max_iterations):
            iters = it + 1
            # Geometric coarse->fine anneal: reach the fine radius quickly, then
            # hold so the convergence test happens at full precision.
            max_d = max(d_end, d_start * (0.7 ** it))
            posed = transform_points(pose, tmpl)
            dists, idx = tree.query(posed, k=1)
            inliers = dists <= max_d
            if int(inliers.sum()) < 3:
                break
            p = posed[inliers]
            q = scan_pts[idx[inliers]]
            n = normals[idx[inliers]]
            # Linearise the rigid update xi=(dx,dy,dtheta) about the robot-frame
            # origin: p' ~ p + [dx,dy] + dtheta*[-py,px]. Residual along normal n.
            A = np.stack([n[:, 0], n[:, 1], n[:, 1] * p[:, 0] - n[:, 0] * p[:, 1]], axis=1)
            b = -np.sum(n * (p - q), axis=1)
            xi, *_ = np.linalg.lstsq(A, b, rcond=None)
            pose = compose(xi, pose)
            if (np.hypot(xi[0], xi[1]) < cfg.tol_translation_m
                    and abs(xi[2]) < cfg.tol_rotation_rad and max_d <= d_end * 1.001):
                break

        # Final diagnostics at the fine correspondence radius (point-to-point).
        posed = transform_points(pose, tmpl)
        dists, _ = tree.query(posed, k=1)
        inliers = dists <= d_end
        n_in = int(inliers.sum())
        rms = float(np.sqrt(np.mean(dists[inliers] ** 2))) if n_in > 0 else np.inf
        inlier_ratio = n_in / n_total if n_total else 0.0
        return pose, rms, inlier_ratio, n_in, iters

    def match(self, scan_pts: np.ndarray, init_pose) -> ICPResult:
        """Fit the template to ``scan_pts`` (robot frame) seeded at ``init_pose``.

        Runs multiple yaw restarts (aliasing guard) and returns the best basin,
        flagging ``ambiguous`` if a second comparable basin exists.
        """
        scan_pts = np.asarray(scan_pts, dtype=np.float64).reshape(-1, 2)
        init_pose = np.asarray(init_pose, dtype=np.float64)
        cfg = self.cfg

        if len(scan_pts) < 3:
            return ICPResult(pose=init_pose.copy(), converged=False, rms_residual_m=np.inf,
                             inlier_ratio=0.0, num_inliers=0, iterations=0,
                             note="too few scan points")

        tree = cKDTree(scan_pts)
        normals = _estimate_normals(scan_pts, tree)
        # Spin restarts about the template centroid (in place) so a symmetry of
        # the shape lands on its alternate fit instead of flying off — the right
        # probe for rotational aliasing (§4).
        ctr = self.template.points.mean(axis=0)

        runs = []
        for dyaw in cfg.restart_yaws:
            c, s = np.cos(dyaw), np.sin(dyaw)
            t_spin = ctr - np.array([[c, -s], [s, c]]) @ ctr
            seed = compose(init_pose, np.array([t_spin[0], t_spin[1], dyaw]))
            pose, rms, inlier_ratio, n_in, iters = self._icp_once(scan_pts, tree, normals, seed)
            accepted = (inlier_ratio >= cfg.min_inlier_ratio and rms <= cfg.max_rms_residual_m)
            runs.append(dict(pose=pose, rms=rms, inlier_ratio=inlier_ratio,
                             num_inliers=n_in, iters=iters, accepted=accepted))

        accepted = [r for r in runs if r["accepted"]]
        # Pick the best basin (lowest residual) among accepted, else best overall.
        pool = accepted if accepted else runs
        best = min(pool, key=lambda r: r["rms"])

        # Aliasing: count distinct accepted basins (pose clusters).
        basins: List[np.ndarray] = []
        for r in accepted:
            if not any(
                (pose_distance(r["pose"], b)[0] < cfg.pose_cluster_translation_m
                 and pose_distance(r["pose"], b)[1] < cfg.pose_cluster_rotation_rad)
                for b in basins
            ):
                basins.append(r["pose"])
        ambiguous = len(basins) >= 2

        note = ""
        if ambiguous:
            note = (f"aliasing: {len(basins)} comparable poses "
                    f"(template asymmetric={self.template.is_asymmetric})")
        elif not accepted:
            note = "no restart met acceptance gates"

        return ICPResult(
            pose=best["pose"],
            converged=best["accepted"],
            rms_residual_m=best["rms"],
            inlier_ratio=best["inlier_ratio"],
            num_inliers=best["num_inliers"],
            iterations=best["iters"],
            ambiguous=ambiguous,
            restart_poses=basins,
            note=note,
        )
