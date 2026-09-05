"""Known slices of flattened PickCube-v1 state observations.

ManiSkill `obs_mode="state"` concatenates `state_dict` in this order for Panda:

    agent.qpos (9) | agent.qvel (9) | is_grasped (1) | tcp_pose (7)
    goal_pos (3) | obj_pose (7) | tcp_to_obj_pos (3) | obj_to_goal_pos (3)

tcp_pose / obj_pose are [x, y, z, qw, qx, qy, qz].
Verify this once with `scripts/inspect_env.py` after installing ManiSkill.
"""

from __future__ import annotations

from typing import Any

import numpy as np


PICKCUBE_STATE_SLICES = {
    "qpos": slice(0, 9),
    "qvel": slice(9, 18),
    "is_grasped": slice(18, 19),
    "tcp_pose": slice(19, 26),
    "tcp_pos": slice(19, 22),
    "tcp_quat": slice(22, 26),
    "goal_pos": slice(26, 29),
    "obj_pose": slice(29, 36),
    "obj_pos": slice(29, 32),
    "obj_quat": slice(32, 36),
    "tcp_to_obj_pos": slice(36, 39),
    "obj_to_goal_pos": slice(39, 42),
}

EXPECTED_OBS_DIM = 42
EXPECTED_ACTION_DIM = 4  # pd_ee_delta_pos: dx, dy, dz, gripper


def observation_loss_weights(obs_dim: int = EXPECTED_OBS_DIM) -> np.ndarray:
    """Higher weight on cube, EE, and relative poses that decide PickCube success."""
    weights = np.ones(obs_dim, dtype=np.float32)
    weights[PICKCUBE_STATE_SLICES["obj_pos"]] = 5.0
    weights[PICKCUBE_STATE_SLICES["obj_to_goal_pos"]] = 5.0
    weights[PICKCUBE_STATE_SLICES["tcp_to_obj_pos"]] = 4.0
    weights[PICKCUBE_STATE_SLICES["tcp_pos"]] = 3.0
    weights[PICKCUBE_STATE_SLICES["is_grasped"]] = 2.0
    weights[PICKCUBE_STATE_SLICES["qpos"]] = 1.5
    return weights


def extract_geometry(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Pull interpretable positions out of a flattened state vector."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    return {
        "tcp_pos": obs[PICKCUBE_STATE_SLICES["tcp_pos"]].copy(),
        "obj_pos": obs[PICKCUBE_STATE_SLICES["obj_pos"]].copy(),
        "goal_pos": obs[PICKCUBE_STATE_SLICES["goal_pos"]].copy(),
        "qpos": obs[PICKCUBE_STATE_SLICES["qpos"]].copy(),
    }


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
