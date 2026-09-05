"""Phase 10: trust the planner only when ensemble disagreement is below a threshold."""

from __future__ import annotations

import numpy as np


class ConfidenceGatedPlanner:
    def __init__(self, planner, policy, threshold: float):
        self.planner = planner
        self.policy = policy
        self.threshold = float(threshold)
        self.last_used_planner = True
        self.planner_frac = 0.0
        self._n = 0
        self._n_plan = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        planned = self.planner.act(obs)
        uncertainty = float(getattr(self.planner, "last_uncertainty", 0.0))
        use_planner = uncertainty < self.threshold
        self.last_used_planner = use_planner
        self._n += 1
        self._n_plan += int(use_planner)
        self.planner_frac = self._n_plan / max(self._n, 1)
        if use_planner:
            return planned
        return self.policy.act(obs, deterministic=True)
