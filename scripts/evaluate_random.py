"""Phase 1 milestone: run random episodes and confirm the env loop works."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pick_place import close_env, is_success, make_env
from utils.config import load_config
from utils.paths import resolve_path
from utils.seed import set_seed


def run_random_episodes(env, num_episodes: int, seed: int) -> dict:
    returns = []
    lengths = []
    successes = []
    t0 = time.perf_counter()

    for ep in tqdm(range(num_episodes), desc="random episodes"):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_success = bool(is_success(info))

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            ep_success = ep_success or is_success(info)
            done = bool(terminated or truncated)

        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(float(ep_success))

    elapsed = time.perf_counter() - t0
    returns = np.asarray(returns, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    successes = np.asarray(successes, dtype=np.float64)
    return {
        "method": "random",
        "num_episodes": int(num_episodes),
        "success_rate": float(successes.mean()),
        "success_count": int(successes.sum()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_length": float(lengths.mean()),
        "steps_per_second": float(lengths.sum() / max(elapsed, 1e-8)),
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="results/tables/random_baseline.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    set_seed(seed)

    env = make_env(args.config, record_metrics=True)
    try:
        obs, info = env.reset(seed=seed)
        print("Reset OK")
        print("  obs shape:", np.asarray(obs).shape)
        print("  action space:", env.action_space)
        print("  info keys:", sorted(info.keys()))

        summary = run_random_episodes(env, args.episodes, seed)
    finally:
        close_env(env)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nRandom baseline")
    print(f"  success: {summary['success_count']}/{summary['num_episodes']} "
          f"({100 * summary['success_rate']:.1f}%)")
    print(f"  mean return: {summary['mean_return']:.3f}")
    print(f"  mean length: {summary['mean_length']:.1f}")
    print(f"  throughput: {summary['steps_per_second']:.1f} steps/s")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
