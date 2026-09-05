"""ManiSkill PickCube wrapper used by every script in this project."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from envs.obs_slices import to_numpy
from utils.config import load_config


def _use_pytorch_ik_on_cpu() -> None:
    """Windows/CPU ManiSkill needs Pinocchio for EE controllers.

    Pinocchio does not pip-install cleanly here, but pytorch_kinematics
    already ships with ManiSkill and can run IK on CPU.
    """
    from mani_skill.agents.controllers.utils.kinematics import Kinematics

    if getattr(Kinematics, "_tawr_cpu_pytorch_ik", False):
        return

    def _setup_cpu(self):
        self._setup_gpu()

    Kinematics._setup_cpu = _setup_cpu
    Kinematics._tawr_cpu_pytorch_ik = True


def make_env(
    config_path: str = "configs/env.yaml",
    *,
    overrides: dict[str, Any] | None = None,
    record_metrics: bool = False,
    ignore_terminations: bool = False,
) -> gym.Env:
    """Create a single CPU PickCube env with a standard Gymnasium API.

    Observations are a flat state vector. Actions are 4-D end-effector
    deltas: [dx, dy, dz, gripper]. Rendering is off by default so this
    works on Windows without Vulkan.
    """
    cfg = load_config(config_path)
    if overrides:
        cfg.update(overrides)

    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    _use_pytorch_ik_on_cpu()

    env_kwargs = {
        "obs_mode": cfg.get("obs_mode", "state"),
        "control_mode": cfg.get("control_mode", "pd_ee_delta_pos"),
        "reward_mode": cfg.get("reward_mode", "normalized_dense"),
        "robot_uids": cfg.get("robot_uids", "panda"),
        "num_envs": int(cfg.get("num_envs", 1)),
        "sim_backend": cfg.get("sim_backend", "physx_cpu"),
        "render_backend": cfg.get("render_backend", "none"),
        "reconfiguration_freq": int(cfg.get("reconfiguration_freq", 1)),
    }
    render_mode = cfg.get("render_mode")
    if render_mode:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(cfg.get("env_id", "PickCube-v1"), **env_kwargs)
    env = CPUGymWrapper(
        env,
        ignore_terminations=ignore_terminations,
        record_metrics=record_metrics,
    )
    return env


def close_env(env: gym.Env | None) -> None:
    if env is not None:
        env.close()


def is_success(info: dict[str, Any]) -> bool:
    value = info.get("success", False)
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def get_task_geometry(env: gym.Env) -> dict[str, np.ndarray]:
    """Read cube / EE / goal positions from the simulator, not the vector."""
    base = env.unwrapped
    tcp = _pose_position(getattr(base.agent, "tcp_pose", None) or base.agent.tcp.pose)
    cube = _pose_position(base.cube.pose)
    goal = _pose_position(base.goal_site.pose)
    qpos = to_numpy(base.agent.robot.get_qpos()).reshape(-1).astype(np.float32)
    return {
        "tcp_pos": tcp,
        "obj_pos": cube,
        "goal_pos": goal,
        "qpos": qpos,
    }


def _pose_position(pose: Any) -> np.ndarray:
    pos = pose.p if hasattr(pose, "p") else pose
    return to_numpy(pos).reshape(-1)[:3].astype(np.float32)
