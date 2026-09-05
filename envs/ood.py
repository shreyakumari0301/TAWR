"""Out-of-distribution PickCube shifts that do not require a scene rebuild."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OODConfig:
    name: str
    cube_spawn_half_size: float = 0.1
    cube_spawn_center: tuple[float, float] = (0.0, 0.0)
    max_goal_height: float = 0.3
    robot_init_qpos_noise: float = 0.02
    mass_scale: float = 1.0
    linear_damping: float | None = None


OOD_LEVELS = {
    "id": OODConfig("id"),
    "mild": OODConfig(
        "mild",
        cube_spawn_half_size=0.14,
        max_goal_height=0.38,
        robot_init_qpos_noise=0.06,
        mass_scale=1.5,
        linear_damping=0.15,
    ),
    "hard": OODConfig(
        "hard",
        cube_spawn_half_size=0.20,
        cube_spawn_center=(0.06, 0.05),
        max_goal_height=0.50,
        robot_init_qpos_noise=0.12,
        mass_scale=3.0,
        linear_damping=0.40,
    ),
}


def apply_ood(env, cfg: OODConfig) -> dict:
    """Mutate spawn region, goal height, arm reset noise, and cube mass/damping."""
    base = env.unwrapped
    base.cube_spawn_half_size = float(cfg.cube_spawn_half_size)
    base.cube_spawn_center = tuple(cfg.cube_spawn_center)
    base.max_goal_height = float(cfg.max_goal_height)
    if hasattr(base, "table_scene"):
        base.table_scene.robot_init_qpos_noise = float(cfg.robot_init_qpos_noise)
    if hasattr(base, "robot_init_qpos_noise"):
        base.robot_init_qpos_noise = float(cfg.robot_init_qpos_noise)

    mass_before = None
    try:
        mass = base.cube.get_mass()
        mass_before = float(np.mean(np.asarray(mass, dtype=np.float64)))
        if cfg.mass_scale != 1.0 and mass_before > 0:
            base.cube.set_mass(mass_before * float(cfg.mass_scale))
    except Exception:
        pass
    if cfg.linear_damping is not None:
        try:
            base.cube.set_linear_damping(float(cfg.linear_damping))
        except Exception:
            pass
    return {
        "level": cfg.name,
        "cube_spawn_half_size": base.cube_spawn_half_size,
        "cube_spawn_center": list(base.cube_spawn_center),
        "max_goal_height": base.max_goal_height,
        "robot_init_qpos_noise": float(cfg.robot_init_qpos_noise),
        "mass_scale": cfg.mass_scale,
        "mass_before": mass_before,
        "linear_damping": cfg.linear_damping,
        "note": "Cube geometry is unchanged; OOD is spawn/goal/arm-noise/mass/damping.",
    }
