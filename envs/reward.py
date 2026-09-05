"""Known goal score used by the planner (Phase 7, option A)."""

from __future__ import annotations

import numpy as np
import torch

from envs.obs_slices import PICKCUBE_STATE_SLICES

GOAL_THRESH = 0.025


def goal_score(
    tcp_pos: np.ndarray,
    obj_pos: np.ndarray,
    goal_pos: np.ndarray,
    *,
    success: bool = False,
    alpha: float = 1.0,
    beta: float = 0.25,
    gamma: float = 5.0,
) -> float:
    """R = -α||cube-goal|| - β||ee-cube|| + γ 1[success]."""
    cube_err = float(np.linalg.norm(np.asarray(obj_pos) - np.asarray(goal_pos)))
    reach_err = float(np.linalg.norm(np.asarray(tcp_pos) - np.asarray(obj_pos)))
    return -alpha * cube_err - beta * reach_err + (gamma if success else 0.0)


def goal_score_torch(
    obs: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.25,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Batched score from flattened predicted observations. Shape (B,)."""
    tcp = obs[..., PICKCUBE_STATE_SLICES["tcp_pos"]]
    obj = obs[..., PICKCUBE_STATE_SLICES["obj_pos"]]
    goal = obs[..., PICKCUBE_STATE_SLICES["goal_pos"]]
    cube_err = torch.linalg.norm(obj - goal, dim=-1)
    reach_err = torch.linalg.norm(tcp - obj, dim=-1)
    success = (cube_err < GOAL_THRESH).float()
    return -alpha * cube_err - beta * reach_err + gamma * success
