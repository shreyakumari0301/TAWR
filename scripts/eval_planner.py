"""Evaluate SAC, world-model MPC, and confidence-gated planning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pick_place import close_env, make_env
from models.ensemble import EnsembleWorldModel
from models.sac_agent import SACAgent
from planners.mpc import PlannerConfig, PolicyGuidedMPC
from planners.uncertainty_mpc import ConfidenceGatedPlanner
from utils.config import load_config
from utils.evaluate import evaluate_actor
from utils.paths import resolve_path
from utils.seed import set_seed


def timed(act_fn):
    latencies = []

    def wrapped(obs):
        t0 = time.perf_counter()
        action = act_fn(obs)
        latencies.append(time.perf_counter() - t0)
        return action

    wrapped.latencies = latencies
    return wrapped


def build_actor(method: str, agent, ensemble, cfg, action_low, action_high):
    if method == "sac":
        return lambda obs: agent.act(obs, deterministic=True), None

    planner = PolicyGuidedMPC(
        world_model=ensemble,
        policy=agent,
        config=PlannerConfig(
            horizon=int(cfg["horizon"]),
            num_candidates=int(cfg["num_candidates"]),
            action_noise_std=float(cfg.get("action_noise_std", 0.15)),
            discount=float(cfg.get("discount", 0.8)),
            uncertainty_coef=float(cfg.get("uncertainty_coef", 0.0)),
            value_coef=float(cfg.get("value_coef", 0.0)),
            policy_guided=bool(cfg.get("policy_guided", True)),
            keep_policy_action=bool(cfg.get("keep_policy_action", True)),
        ),
        action_low=action_low,
        action_high=action_high,
    )
    if method == "mpc":
        return planner.act, planner
    if method == "gated":
        gated = ConfidenceGatedPlanner(planner, agent, float(cfg["confidence_threshold"]))
        return gated.act, gated
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--planner-config", default="configs/planner.yaml")
    parser.add_argument("--policy", default="checkpoints/sac_best.pt")
    parser.add_argument("--world-model", default="checkpoints/world_model.pt")
    parser.add_argument("--method", choices=["sac", "mpc", "gated"], default="mpc")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--uncertainty-coef", type=float, default=None)
    parser.add_argument("--policy-guided", type=int, choices=[0, 1], default=None)
    args = parser.parse_args()

    cfg = load_config(args.planner_config)
    if args.horizon is not None:
        cfg["horizon"] = args.horizon
    if args.candidates is not None:
        cfg["num_candidates"] = args.candidates
    if args.value_coef is not None:
        cfg["value_coef"] = args.value_coef
    if args.uncertainty_coef is not None:
        cfg["uncertainty_coef"] = args.uncertainty_coef
    if args.policy_guided is not None:
        cfg["policy_guided"] = bool(args.policy_guided)
    episodes = int(args.episodes if args.episodes is not None else cfg.get("eval_episodes", 40))
    set_seed(args.seed)

    agent = SACAgent.from_checkpoint(str(resolve_path(args.policy)))
    device = torch.device("cpu")
    ensemble = None
    if args.method != "sac":
        ensemble = EnsembleWorldModel.load(str(resolve_path(args.world_model)), device)

    env = make_env(args.env_config, record_metrics=True)
    try:
        act_fn, controller = build_actor(
            args.method,
            agent,
            ensemble,
            cfg,
            env.action_space.low,
            env.action_space.high,
        )
        wrapped = timed(act_fn)
        summary = evaluate_actor(env, wrapped, episodes, args.seed, desc=args.method)
    finally:
        close_env(env)

    lats = np.asarray(wrapped.latencies, dtype=np.float64) if wrapped.latencies else np.zeros(1)
    summary.update(
        {
            "method": args.method,
            "horizon": cfg.get("horizon"),
            "num_candidates": cfg.get("num_candidates"),
            "mean_latency_s": float(lats.mean()),
            "p95_latency_s": float(np.percentile(lats, 95)),
            "planner_frac": float(getattr(controller, "planner_frac", 1.0 if args.method != "sac" else 0.0)),
        }
    )
    out = resolve_path(args.out or f"results/tables/eval_{args.method}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
