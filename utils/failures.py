"""Episode failure modes for PickCube from a logged trajectory."""

from __future__ import annotations

import numpy as np


REACH_THRESH = 0.05
PLACE_CLOSE = 0.08
LIFT_Z = 0.05
TABLE_Z = 0.02


def label_episode(
    *,
    success: bool,
    is_grasped: np.ndarray,
    tcp_to_obj: np.ndarray,
    obj_to_goal: np.ndarray,
    obj_z: np.ndarray,
) -> str:
    if success:
        return "success"

    grasped = np.asarray(is_grasped, dtype=bool).reshape(-1)
    reach = np.asarray(tcp_to_obj, dtype=np.float64).reshape(-1)
    place = np.asarray(obj_to_goal, dtype=np.float64).reshape(-1)
    z = np.asarray(obj_z, dtype=np.float64).reshape(-1)
    if grasped.size == 0:
        return "timeout"

    ever_grasped = bool(grasped.any())
    if not ever_grasped:
        if float(reach.min()) > REACH_THRESH:
            return "failed_approach"
        return "failed_grasp"

    dropped = False
    for t in range(1, len(grasped)):
        if grasped[t - 1] and (not grasped[t]) and place[t] > PLACE_CLOSE:
            dropped = True
            break
    if dropped:
        return "drop_after_grasp"
    if float(z.max()) < LIFT_Z:
        return "bad_lift"
    if float(place.min()) > PLACE_CLOSE:
        return "bad_transport"
    return "bad_placement"
