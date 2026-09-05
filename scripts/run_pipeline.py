"""Run Phases 3–10 after a SAC checkpoint exists and write the result tables."""

from __future__ import annotations

import argparse
import json
import subprocess
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


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def timed(act_fn):
    latencies = []

    def wrapped(obs):
        t0 = time.perf_counter()
        action = act_fn(obs)
        latencies.append(time.perf_counter() - t0)
        return action

    wrapped.latencies = latencies
    return wrapped


def summarize(name: str, metrics: dict, wrapped, extra: dict | None = None) -> dict:
    lats = np.asarray(wrapped.latencies, dtype=np.float64) if wrapped.latencies else np.zeros(1)
    row = {
        "method": name,
        **metrics,
        "mean_latency_s": float(lats.mean()),
        "p95_latency_s": float(np.percentile(lats, 95)),
    }
    if extra:
        row.update(extra)
    return row


def make_mpc(agent, ensemble, horizon, candidates, uncertainty_coef, action_low, action_high):
    return PolicyGuidedMPC(
        world_model=ensemble,
        policy=agent,
        config=PlannerConfig(
            horizon=horizon,
            num_candidates=candidates,
            action_noise_std=0.15,
            discount=0.99,
            uncertainty_coef=uncertainty_coef,
        ),
        action_low=action_low,
        action_high=action_high,
    )


def choose_threshold(ensemble, data_path: Path) -> float:
    data = np.load(data_path, allow_pickle=True)
    mask = data["split"] == "val"
    obs = torch.as_tensor(data["obs"][mask][:2048], dtype=torch.float32)
    act = torch.as_tensor(data["action"][mask][:2048], dtype=torch.float32)
    with torch.no_grad():
        u = ensemble.disagreement(obs, act).cpu().numpy()
    tau = float(np.quantile(u, 0.6))
    return tau


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/sac_best.pt")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-wm-train", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    policy_path = resolve_path(args.policy)
    if not policy_path.exists():
        policy_path = resolve_path("checkpoints/sac_final.pt")
    if not policy_path.exists():
        raise SystemExit("No SAC checkpoint found. Train Phase 2 first.")

    mixed = resolve_path("data/transitions_mixed.npz")
    wm_path = resolve_path("checkpoints/world_model.pt")
    out_dir = resolve_path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_collect or not mixed.exists():
        run("collect_dataset.py", ["--checkpoint", str(policy_path), "--num-transitions", "12000"])
    if not args.skip_wm_train or not wm_path.exists():
        run("train_world_model.py", ["--data", str(mixed)])
    run("eval_world_model.py", ["--data", str(mixed), "--checkpoint", str(wm_path)])

    agent = SACAgent.from_checkpoint(str(policy_path))
    ensemble = EnsembleWorldModel.load(str(wm_path))
    tau = choose_threshold(ensemble, mixed)
    (out_dir / "confidence_threshold.json").write_text(json.dumps({"tau": tau}, indent=2), encoding="utf-8")

    env = make_env("configs/env.yaml", record_metrics=True)
    rows = []
    try:
        low, high = env.action_space.low, env.action_space.high

        wrapped = timed(lambda o: agent.act(o, deterministic=True))
        rows.append(summarize("SAC", evaluate_actor(env, wrapped, args.episodes, args.seed), wrapped))

        mpc = make_mpc(agent, ensemble, 5, 64, 0.0, low, high)
        wrapped = timed(mpc.act)
        rows.append(summarize("WM-MPC", evaluate_actor(env, wrapped, args.episodes, args.seed + 1), wrapped, {"horizon": 5, "K": 64}))

        mpc_u = make_mpc(agent, ensemble, 5, 64, 0.5, low, high)
        gated = ConfidenceGatedPlanner(mpc_u, agent, tau)
        wrapped = timed(gated.act)
        rows.append(
            summarize(
                "gated-WM",
                evaluate_actor(env, wrapped, args.episodes, args.seed + 2),
                wrapped,
                {"horizon": 5, "K": 64, "tau": tau, "planner_frac": gated.planner_frac},
            )
        )

        horizon_rows = []
        for H in (1, 3, 5, 10):
            planner = make_mpc(agent, ensemble, H, 64, 0.0, low, high)
            wrapped = timed(planner.act)
            horizon_rows.append(
                summarize(
                    f"WM-MPC H={H}",
                    evaluate_actor(env, wrapped, args.episodes, args.seed + 10 + H),
                    wrapped,
                    {"horizon": H, "K": 64},
                )
            )

        candidate_rows = []
        for K in (32, 64, 128):
            planner = make_mpc(agent, ensemble, 5, K, 0.0, low, high)
            wrapped = timed(planner.act)
            candidate_rows.append(
                summarize(
                    f"WM-MPC K={K}",
                    evaluate_actor(env, wrapped, args.episodes, args.seed + 20 + K),
                    wrapped,
                    {"horizon": 5, "K": K},
                )
            )
    finally:
        close_env(env)

    random_path = resolve_path("results/tables/random_baseline.json")
    random_row = json.loads(random_path.read_text(encoding="utf-8")) if random_path.exists() else {}
    wm_eval = json.loads(resolve_path("results/tables/rollout_error.json").read_text(encoding="utf-8"))

    final = {
        "main": [random_row, *rows],
        "horizon_ablation": horizon_rows,
        "candidate_ablation": candidate_rows,
        "world_model": wm_eval,
        "confidence_threshold": tau,
        "policy_checkpoint": str(policy_path),
    }
    out = out_dir / "final_results.json"
    out.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
