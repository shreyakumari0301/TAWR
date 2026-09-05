"""V2 planner matrix: short horizon, policy-guided sampling, SAC terminal value, ensemble uncertainty, DAgger."""

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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from collect_dataset import collect, concat, split_by_episode
from envs.pick_place import close_env, make_env
from models.ensemble import EnsembleWorldModel, SingleMemberWorldModel
from models.sac_agent import SACAgent
from planners.mpc import make_planner
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


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


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
    print(
        f"{name}: success={row['success_rate']:.1%} "
        f"({row['success_count']}/{row['num_episodes']}) "
        f"return={row['mean_return']:.2f}",
        flush=True,
    )
    return row


@torch.no_grad()
def calibrate_uncertainty_coef(ensemble, data_path: Path, target_penalty: float = 0.1, q: float = 0.9) -> dict:
    data = np.load(data_path, allow_pickle=True)
    mask = data["split"] == "val"
    obs = torch.as_tensor(data["obs"][mask][:4096], dtype=torch.float32)
    act = torch.as_tensor(data["action"][mask][:4096], dtype=torch.float32)
    u = ensemble.disagreement(obs, act).cpu().numpy()
    u_q = float(np.quantile(u, q))
    lam = float(target_penalty / max(u_q, 1e-8))
    return {
        "lambda": lam,
        "target_penalty": target_penalty,
        "quantile": q,
        "u_mean": float(u.mean()),
        "u_median": float(np.median(u)),
        "u_q": u_q,
        "u_max": float(u.max()),
        "n": int(u.size),
    }


def eval_row(env, name, act_fn, episodes, seed, extra=None):
    wrapped = timed(act_fn)
    metrics = evaluate_actor(env, wrapped, episodes, seed, desc=name)
    return summarize(name, metrics, wrapped, extra)


def planner_spec(**kwargs) -> dict:
    spec = {
        "horizon": 3,
        "num_candidates": 32,
        "action_noise_std": 0.15,
        "discount": 0.8,
        "uncertainty_coef": 0.0,
        "value_coef": 0.0,
        "policy_guided": True,
        "keep_policy_action": True,
    }
    spec.update(kwargs)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/sac_best.pt")
    parser.add_argument("--data", default="data/transitions_mixed.npz")
    parser.add_argument("--world-model", default="checkpoints/world_model_v2.pt")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--sweep-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dagger-transitions", type=int, default=5000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--skip-dagger", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    policy_path = resolve_path(args.policy)
    data_path = resolve_path(args.data)
    wm_path = resolve_path(args.world_model)
    out_path = resolve_path("results/tables/v2_results.json")
    payload = {
        "policy_checkpoint": str(policy_path),
        "world_model": str(wm_path),
        "episodes": args.episodes,
        "matrix": [],
        "horizon_sweep": [],
        "candidate_sweep": [],
        "dagger": {},
    }
    dump(out_path, payload)

    if not args.skip_train or not wm_path.exists():
        run(
            "train_world_model.py",
            [
                "--data",
                str(data_path),
                "--out",
                str(wm_path),
                "--ensemble-size",
                str(args.ensemble_size),
                "--epochs",
                str(args.epochs),
            ],
        )
        run(
            "eval_world_model.py",
            [
                "--data",
                str(data_path),
                "--checkpoint",
                str(wm_path),
                "--out",
                "results/tables/rollout_error_v2.csv",
            ],
        )

    agent = SACAgent.from_checkpoint(str(policy_path))
    ensemble = EnsembleWorldModel.load(str(wm_path))
    single = SingleMemberWorldModel(ensemble, 0)
    calib = calibrate_uncertainty_coef(ensemble, data_path)
    payload["uncertainty_calibration"] = calib
    wm_eval = resolve_path("results/tables/rollout_error_v2.json")
    if wm_eval.exists():
        payload["world_model_eval"] = json.loads(wm_eval.read_text(encoding="utf-8"))
    dump(out_path, payload)
    lam = calib["lambda"]
    print(json.dumps({"uncertainty_calibration": calib}, indent=2), flush=True)

    env = make_env("configs/env.yaml", record_metrics=True)
    low, high = env.action_space.low, env.action_space.high

    def mpc(spec: dict, model=None):
        return make_planner(model if model is not None else ensemble, agent, low, high, **spec)

    # Rows before Ensemble WM isolate H / sampling / V on one dynamics net.
    # Ensemble WM and later rows use the 5-model mean (and disagreement when λ>0).
    matrix = [
        ("SAC", None, None, {"key_change": "baseline"}),
        (
            "Current best",
            planner_spec(horizon=5, num_candidates=32, discount=0.99, policy_guided=True),
            single,
            {"key_change": "existing H=5 K=32"},
        ),
        (
            "Short WM",
            planner_spec(horizon=3, num_candidates=32, policy_guided=False),
            single,
            {"key_change": "shorter horizon, random shooting"},
        ),
        (
            "Policy-guided WM",
            planner_spec(horizon=3, num_candidates=32, policy_guided=True),
            single,
            {"key_change": "SAC + action noise"},
        ),
        (
            "WM + terminal value",
            planner_spec(horizon=3, num_candidates=32, policy_guided=True, value_coef=1.0),
            single,
            {"key_change": "+V_SAC"},
        ),
        (
            "Ensemble WM",
            planner_spec(horizon=3, num_candidates=32, policy_guided=True),
            ensemble,
            {"key_change": "5 dynamics models"},
        ),
        (
            "Uncertainty MPC",
            planner_spec(horizon=3, num_candidates=32, policy_guided=True, uncertainty_coef=lam),
            ensemble,
            {"key_change": "R - λU"},
        ),
        (
            "Full method",
            planner_spec(
                horizon=3,
                num_candidates=32,
                policy_guided=True,
                value_coef=1.0,
                uncertainty_coef=lam,
            ),
            ensemble,
            {"key_change": "policy-guided + value + uncertainty"},
        ),
    ]

    try:
        for i, (name, spec, model, extra) in enumerate(matrix):
            extra = dict(extra)
            if spec is None:
                row = eval_row(
                    env,
                    name,
                    lambda o: agent.act(o, deterministic=True),
                    args.episodes,
                    args.seed + i,
                    extra,
                )
            else:
                extra.update({"H": spec["horizon"], "K": spec["num_candidates"], **spec})
                extra["uses_ensemble_mean"] = model is ensemble
                planner = mpc(spec, model)
                row = eval_row(env, name, planner.act, args.episodes, args.seed + i, extra)
            payload["matrix"].append(row)
            dump(out_path, payload)

        if not args.skip_sweeps:
            for H in (2, 3, 4, 5):
                spec = planner_spec(horizon=H, num_candidates=32, policy_guided=True)
                planner = mpc(spec, ensemble)
                row = eval_row(
                    env,
                    f"H={H}",
                    planner.act,
                    args.sweep_episodes,
                    args.seed + 100 + H,
                    {"H": H, "K": 32, **spec},
                )
                payload["horizon_sweep"].append(row)
                dump(out_path, payload)

            for K in (8, 16, 24, 32, 48, 64):
                spec = planner_spec(horizon=3, num_candidates=K, policy_guided=True)
                planner = mpc(spec, ensemble)
                row = eval_row(
                    env,
                    f"K={K}",
                    planner.act,
                    args.sweep_episodes,
                    args.seed + 200 + K,
                    {"H": 3, "K": K, **spec},
                )
                payload["candidate_sweep"].append(row)
                dump(out_path, payload)

        if not args.skip_dagger:
            print(">> DAgger: collect planner-visited transitions", flush=True)
            full_spec = planner_spec(
                horizon=3,
                num_candidates=32,
                policy_guided=True,
                value_coef=1.0,
                uncertainty_coef=lam,
            )
            planner = mpc(full_spec, ensemble)

            def planner_act(obs, env_):
                return planner.act(obs)

            planner_data = collect(env, args.dagger_transitions, args.seed + 300, planner_act, "planner")
            mixed_npz = np.load(data_path, allow_pickle=True)
            mixed = {k: mixed_npz[k] for k in mixed_npz.files if k != "split"}
            merged = split_by_episode(concat([mixed, planner_data]), seed=args.seed)
            dagger_path = resolve_path("data/transitions_dagger.npz")
            dagger_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dagger_path, **merged)
            payload["dagger"]["collected"] = {
                "path": str(dagger_path),
                "planner_transitions": int(planner_data["obs"].shape[0]),
                "merged_transitions": int(merged["obs"].shape[0]),
                "sources": {
                    str(s): int(np.sum(merged["source"] == s))
                    for s in np.unique(merged["source"])
                },
            }
            dump(out_path, payload)

            dagger_wm = resolve_path("checkpoints/world_model_v2_dagger.pt")
            print(">> DAgger: retrain world model", flush=True)
            close_env(env)
            env = None
            run(
                "train_world_model.py",
                [
                    "--data",
                    str(dagger_path),
                    "--out",
                    str(dagger_wm),
                    "--ensemble-size",
                    str(args.ensemble_size),
                    "--epochs",
                    str(args.epochs),
                ],
            )
            ensemble_d = EnsembleWorldModel.load(str(dagger_wm))
            calib_d = calibrate_uncertainty_coef(ensemble_d, dagger_path)
            payload["dagger"]["world_model"] = str(dagger_wm)
            payload["dagger"]["uncertainty_calibration"] = calib_d
            dump(out_path, payload)

            env = make_env("configs/env.yaml", record_metrics=True)
            low, high = env.action_space.low, env.action_space.high
            dagger_spec = planner_spec(
                horizon=3,
                num_candidates=32,
                policy_guided=True,
                value_coef=1.0,
                uncertainty_coef=float(calib_d["lambda"]),
            )
            planner_d = make_planner(ensemble_d, agent, low, high, **dagger_spec)
            row = eval_row(
                env,
                "Full method + DAgger",
                planner_d.act,
                args.episodes,
                args.seed + 400,
                {"key_change": "train WM -> plan -> collect -> retrain", **dagger_spec},
            )
            payload["dagger"]["eval"] = row
            dump(out_path, payload)
    finally:
        if env is not None:
            close_env(env)

    dump(out_path, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
