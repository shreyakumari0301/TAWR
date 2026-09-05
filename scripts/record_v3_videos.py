"""Record a few PickCube RGB rollouts if the renderer is available."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pick_place import close_env, is_success, make_env
from models.ensemble import EnsembleWorldModel
from models.sac_agent import SACAgent
from planners.mpc import make_planner
from utils.paths import resolve_path


def try_make_rgb_env():
    try:
        return make_env(
            "configs/env.yaml",
            overrides={"render_mode": "rgb_array", "render_backend": "sapien_cpu"},
            record_metrics=True,
        )
    except Exception as exc:
        print(f"RGB env unavailable: {exc}", flush=True)
        return None


def save_gif(frames, path: Path, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
        return
    except Exception:
        pass
    np.savez_compressed(path.with_suffix(".npz"), frames=np.stack(frames))


def rollout(env, act_fn, seed: int):
    frames = []
    obs, info = env.reset(seed=seed)
    done = False
    success = bool(is_success(info))
    try:
        frames.append(env.render())
    except Exception:
        return [], False
    while not done:
        obs, reward, terminated, truncated, info = env.step(act_fn(np.asarray(obs, dtype=np.float32)))
        success = success or bool(is_success(info))
        done = bool(terminated or truncated)
        try:
            frames.append(env.render())
        except Exception:
            break
    frames = [f for f in frames if f is not None]
    return frames, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/sac_best.pt")
    parser.add_argument("--world-model", default="checkpoints/world_model_v2.pt")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--lambda-coef", type=float, default=None)
    args = parser.parse_args()

    env = try_make_rgb_env()
    if env is None:
        note = {"ok": False, "reason": "No RGB renderer on this machine (Windows CPU / render_backend=none)."}
        out = resolve_path("results/figures/v3_videos.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(note, indent=2), encoding="utf-8")
        print(json.dumps(note, indent=2))
        return

    agent = SACAgent.from_checkpoint(str(resolve_path(args.policy)))
    ensemble = EnsembleWorldModel.load(str(resolve_path(args.world_model)))
    lam = 14.72 if args.lambda_coef is None else float(args.lambda_coef)
    planner = make_planner(
        ensemble,
        agent,
        env.action_space.low,
        env.action_space.high,
        horizon=3,
        num_candidates=32,
        discount=0.8,
        policy_guided=True,
        uncertainty_coef=lam,
        value_coef=1.0,
    )
    out_dir = resolve_path("results/figures/videos")
    saved = []
    try:
        for i in range(args.episodes):
            for name, fn in (("sac", lambda o: agent.act(o, deterministic=True)), ("trust_wm", planner.act)):
                frames, success = rollout(env, fn, seed=10_000 + i)
                if not frames:
                    continue
                path = out_dir / f"{name}_ep{i}_{'ok' if success else 'fail'}.gif"
                save_gif(frames, path)
                saved.append({"path": str(path), "method": name, "success": success})
                print(f"wrote {path} success={success}", flush=True)
    finally:
        close_env(env)
    resolve_path("results/figures/v3_videos.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
