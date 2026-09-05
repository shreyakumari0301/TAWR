from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from envs.obs_slices import PICKCUBE_STATE_SLICES, extract_geometry
from envs.pick_place import is_success
from utils.failures import label_episode


def evaluate_actor(
    env,
    act_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    seed: int,
    desc: str | None = None,
) -> dict:
    returns = []
    lengths = []
    successes = []
    steps_to_success = []
    t0 = time.perf_counter()
    iterator = range(num_episodes)
    try:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=desc or "eval")
    except ImportError:
        pass

    for ep in iterator:
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_success = bool(is_success(info))
        first_success_step = 1 if ep_success else None

        while not done:
            action = act_fn(np.asarray(obs, dtype=np.float32))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            if is_success(info):
                if first_success_step is None:
                    first_success_step = ep_len
                ep_success = True
            done = bool(terminated or truncated)

        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(float(ep_success))
        if first_success_step is not None:
            steps_to_success.append(first_success_step)

    elapsed = time.perf_counter() - t0
    returns = np.asarray(returns, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    successes = np.asarray(successes, dtype=np.float64)
    return {
        "num_episodes": int(num_episodes),
        "success_rate": float(successes.mean()),
        "success_count": int(successes.sum()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_length": float(lengths.mean()),
        "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else None,
        "steps_per_second": float(lengths.sum() / max(elapsed, 1e-8)),
        "seed": int(seed),
    }


def evaluate_logged(
    env,
    act_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    seed: int,
    *,
    desc: str | None = None,
    planner=None,
    world_model=None,
    keep_timesteps: bool = False,
    gamma: float = 0.8,
) -> dict:
    """Eval with failure labels and optional per-step uncertainty / prediction error."""
    ep_rows = []
    ts_u, ts_err, ts_dev, ts_pred, ts_real = [], [], [], [], []
    ts_amag, ts_state_err, ts_pred_cube, ts_act_cube = [], [], [], []
    latencies = []
    t0 = time.perf_counter()
    iterator = range(num_episodes)
    try:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=desc or "eval")
    except ImportError:
        pass

    for ep in iterator:
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_success = bool(is_success(info))
        first_success_step = 1 if ep_success else None
        grasped, reach, place, z = [], [], [], []
        step_u, step_dev, step_h, step_pred_r, step_r = [], [], [], [], []
        high_u = 0

        while not done:
            obs_np = np.asarray(obs, dtype=np.float32)
            t_act = time.perf_counter()
            action = act_fn(obs_np)
            latencies.append(time.perf_counter() - t_act)
            pred_obj = None
            pred_next = None
            u = 0.0
            dev = 0.0
            amag = 0.0
            H = 0
            pred_ret = 0.0
            if planner is not None and getattr(planner, "last_info", None):
                d = planner.last_info
                u = float(d.get("uncertainty", 0.0))
                dev = float(d.get("action_deviation", 0.0))
                amag = float(d.get("action_magnitude", float(np.linalg.norm(action))))
                H = int(d.get("horizon", 0))
                pred_ret = float(d.get("predicted_return", 0.0))
                pred_obj = d.get("predicted_obj_pos")
                pred_next = d.get("predicted_next")
            elif world_model is not None:
                with torch.no_grad():
                    o = torch.as_tensor(obs_np).unsqueeze(0)
                    a = torch.as_tensor(action).unsqueeze(0)
                    pred = world_model(o, a).squeeze(0).cpu().numpy()
                    pred_next = pred
                    pred_obj = pred[PICKCUBE_STATE_SLICES["obj_pos"]]
                    if hasattr(world_model, "disagreement"):
                        u = float(world_model.disagreement(o, a).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_np = np.asarray(next_obs, dtype=np.float32)
            geom = extract_geometry(next_np)
            cube_err = (
                float(np.linalg.norm(np.asarray(pred_obj) - geom["obj_pos"]))
                if pred_obj is not None
                else float("nan")
            )
            g = extract_geometry(obs_np)
            tcp_to_obj = float(np.linalg.norm(g["tcp_pos"] - g["obj_pos"]))
            obj_to_goal = float(np.linalg.norm(g["obj_pos"] - g["goal_pos"]))
            gval = info.get("is_grasped", False)
            if isinstance(gval, (list, tuple, np.ndarray)):
                gval = bool(np.asarray(gval).reshape(-1)[0])
            grasped.append(bool(gval))
            reach.append(tcp_to_obj)
            place.append(obj_to_goal)
            z.append(float(g["obj_pos"][2]))
            step_u.append(u)
            step_dev.append(dev)
            step_h.append(H)
            step_pred_r.append(pred_ret)
            step_r.append(float(reward))
            if keep_timesteps:
                ts_u.append(u)
                ts_err.append(cube_err)
                ts_dev.append(dev)
                ts_pred.append(pred_ret)
                ts_amag.append(amag if amag else float(np.linalg.norm(action)))
                if pred_next is not None:
                    ts_state_err.append(float(np.linalg.norm(np.asarray(pred_next) - next_np)))
                else:
                    ts_state_err.append(float("nan"))
                ts_pred_cube.append(np.asarray(pred_obj, dtype=np.float64) if pred_obj is not None else np.full(3, np.nan))
                ts_act_cube.append(np.asarray(geom["obj_pos"], dtype=np.float64))
            high_u += int(u >= 0.008)

            ep_return += float(reward)
            ep_len += 1
            if is_success(info):
                if first_success_step is None:
                    first_success_step = ep_len
                ep_success = True
            obs = next_obs
            done = bool(terminated or truncated)

        rewards = np.asarray(step_r, dtype=np.float64)
        remaining = np.zeros_like(rewards)
        acc = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            acc = rewards[t] + gamma * acc
            remaining[t] = acc
        if keep_timesteps:
            ts_real.extend(remaining.tolist())

        mode = label_episode(
            success=ep_success,
            is_grasped=np.asarray(grasped),
            tcp_to_obj=np.asarray(reach),
            obj_to_goal=np.asarray(place),
            obj_z=np.asarray(z),
        )
        ep_rows.append(
            {
                "episode": int(ep),
                "seed": int(seed + ep),
                "success": float(ep_success),
                "return": float(ep_return),
                "length": int(ep_len),
                "failure_mode": mode,
                "mean_uncertainty": float(np.mean(step_u)) if step_u else 0.0,
                "mean_action_deviation": float(np.mean(step_dev)) if step_dev else 0.0,
                "mean_predicted_return": float(np.mean(step_pred_r)) if step_pred_r else 0.0,
                "actual_return": float(ep_return),
                "mean_horizon": float(np.mean(step_h)) if step_h else 0.0,
                "high_uncertainty_frac": float(high_u / max(ep_len, 1)),
                "steps_to_success": int(first_success_step) if first_success_step else None,
            }
        )

    elapsed = time.perf_counter() - t0
    successes = np.asarray([r["success"] for r in ep_rows], dtype=np.float64)
    returns = np.asarray([r["return"] for r in ep_rows], dtype=np.float64)
    lengths = np.asarray([r["length"] for r in ep_rows], dtype=np.float64)
    modes = {}
    for r in ep_rows:
        modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
    out = {
        "num_episodes": int(num_episodes),
        "success_rate": float(successes.mean()) if len(successes) else 0.0,
        "success_count": int(successes.sum()),
        "mean_return": float(returns.mean()) if len(returns) else 0.0,
        "std_return": float(returns.std()) if len(returns) else 0.0,
        "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
        "steps_per_second": float(lengths.sum() / max(elapsed, 1e-8)),
        "seed": int(seed),
        "successes": successes,
        "returns": returns,
        "lengths": lengths,
        "latencies": np.asarray(latencies, dtype=np.float64),
        "failure_counts": modes,
        "episodes": ep_rows,
    }
    if keep_timesteps:
        out["timesteps"] = {
            "uncertainty": np.asarray(ts_u, dtype=np.float64),
            "cube_pred_error_m": np.asarray(ts_err, dtype=np.float64),
            "state_pred_error": np.asarray(ts_state_err, dtype=np.float64),
            "action_deviation": np.asarray(ts_dev, dtype=np.float64),
            "action_magnitude": np.asarray(ts_amag, dtype=np.float64),
            "predicted_return": np.asarray(ts_pred, dtype=np.float64),
            "actual_remaining_return": np.asarray(ts_real, dtype=np.float64),
            "pred_obj_pos": np.asarray(ts_pred_cube, dtype=np.float64),
            "actual_obj_pos": np.asarray(ts_act_cube, dtype=np.float64),
        }
    return out
