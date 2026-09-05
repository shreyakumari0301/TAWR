"""Print observation / action spaces and one reset so the env is actually usable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.obs_slices import EXPECTED_ACTION_DIM, EXPECTED_OBS_DIM, extract_geometry
from envs.pick_place import close_env, get_task_geometry, make_env
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = make_env(args.config)
    try:
        obs, info = env.reset(seed=int(cfg.get("seed", 0)))
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        report = {
            "env_id": cfg.get("env_id"),
            "obs_mode": cfg.get("obs_mode"),
            "control_mode": cfg.get("control_mode"),
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "obs_shape": list(np_shape(obs)),
            "action_shape": list(np_shape(action)),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info_keys": sorted(step_info.keys()),
            "success": bool(step_info.get("success", False)),
            "task_geometry": {k: v.tolist() for k, v in get_task_geometry(env).items()},
            "obs_geometry": {k: v.tolist() for k, v in extract_geometry(next_obs).items()}
            if np_shape(next_obs)[-1] == EXPECTED_OBS_DIM
            else "skipped: unexpected obs dim",
            "expected_obs_dim": EXPECTED_OBS_DIM,
            "expected_action_dim": EXPECTED_ACTION_DIM,
        }
        print(json.dumps(report, indent=2))
        if list(np_shape(obs))[-1] != EXPECTED_OBS_DIM:
            print(
                f"\nWarning: obs dim is {np_shape(obs)[-1]}, expected {EXPECTED_OBS_DIM}. "
                "Update envs/obs_slices.py after inspecting state_dict.",
                file=sys.stderr,
            )
        if list(np_shape(action))[-1] != EXPECTED_ACTION_DIM:
            print(
                f"\nWarning: action dim is {np_shape(action)[-1]}, expected {EXPECTED_ACTION_DIM}.",
                file=sys.stderr,
            )
    finally:
        close_env(env)


def np_shape(x) -> tuple:
    import numpy as np

    return tuple(np.asarray(x).shape)


if __name__ == "__main__":
    main()
