"""Evaluate a trained SAC policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pick_place import close_env, make_env
from models.sac_agent import SACAgent
from utils.evaluate import evaluate_actor
from utils.paths import resolve_path
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sac_best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/tables/sac_baseline.json")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    agent = SACAgent.from_checkpoint(str(resolve_path(args.checkpoint)))
    env = make_env(args.env_config, record_metrics=True)
    try:
        summary = evaluate_actor(
            env,
            lambda obs: agent.act(obs, deterministic=not args.stochastic),
            args.episodes,
            args.seed,
        )
    finally:
        close_env(env)

    summary["method"] = "sac"
    summary["checkpoint"] = str(resolve_path(args.checkpoint))
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
