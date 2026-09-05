"""V3 research package: multi-seed confirmation, Adaptive-H, failures, OOD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from envs.ood import OOD_LEVELS, apply_ood
from envs.pick_place import close_env, make_env
from models.ensemble import EnsembleWorldModel, SingleMemberWorldModel
from models.sac_agent import SACAgent
from planners.mpc import make_planner
from plot_v3 import plot_failure_modes, plot_ood, plot_success_ci, plot_uncertainty_vs_error
from utils.evaluate import evaluate_logged
from utils.paths import resolve_path
from utils.seed import set_seed
from utils.stats import mcnemar_exact, summarize_success


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(f"saved {path}", flush=True)


@torch.no_grad()
def calibrate(ensemble, data_path: Path, target_penalty: float = 0.1, q: float = 0.9, adaptive_qs=(0.4, 0.7, 0.9)) -> dict:
    data = np.load(data_path, allow_pickle=True)
    mask = data["split"] == "val"
    obs = torch.as_tensor(data["obs"][mask][:4096], dtype=torch.float32)
    act = torch.as_tensor(data["action"][mask][:4096], dtype=torch.float32)
    u = ensemble.disagreement(obs, act).cpu().numpy()
    u_q = float(np.quantile(u, q))
    return {
        "lambda": float(target_penalty / max(u_q, 1e-8)),
        "u_mean": float(u.mean()),
        "u_median": float(np.median(u)),
        "u_q": u_q,
        "adaptive_thresholds": [float(np.quantile(u, qq)) for qq in adaptive_qs],
        "adaptive_quantiles": list(adaptive_qs),
        "n": int(u.size),
    }


def spec(**kwargs) -> dict:
    cfg = {
        "horizon": 3,
        "num_candidates": 32,
        "action_noise_std": 0.15,
        "discount": 0.8,
        "uncertainty_coef": 0.0,
        "value_coef": 0.0,
        "policy_guided": True,
        "keep_policy_action": True,
        "adaptive_horizon": False,
    }
    cfg.update(kwargs)
    return cfg


def save_run(path: Path, result: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        successes=np.asarray(result["successes"]),
        returns=np.asarray(result["returns"]),
        lengths=np.asarray(result["lengths"]),
        latencies=np.asarray(result["latencies"]),
        extra=json.dumps(jsonable(extra)),
        failure_counts=json.dumps(result.get("failure_counts", {})),
        episodes=json.dumps(jsonable(result.get("episodes", []))),
    )
    if "timesteps" in result:
        ts = result["timesteps"]
        np.savez_compressed(
            path.with_name(path.stem + "_timesteps.npz"),
            **{k: np.asarray(v) for k, v in ts.items()},
        )


def load_run(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    extra = json.loads(str(data["extra"]))
    out = {
        "successes": data["successes"],
        "returns": data["returns"],
        "lengths": data["lengths"],
        "latencies": data["latencies"],
        "failure_counts": json.loads(str(data["failure_counts"])),
        "episodes": json.loads(str(data["episodes"])),
        **extra,
    }
    ts_path = path.with_name(path.stem + "_timesteps.npz")
    if ts_path.exists():
        ts = np.load(ts_path)
        out["timesteps"] = {k: ts[k] for k in ts.files}
    return out


def aggregate(runs: list[dict], method: str) -> dict:
    succ = np.concatenate([r["successes"] for r in runs])
    rets = np.concatenate([r["returns"] for r in runs])
    lens = np.concatenate([r["lengths"] for r in runs])
    lats = np.concatenate([r["latencies"] for r in runs])
    summary = summarize_success(succ, rets, lens, lats)
    seed_rates = [float(np.mean(r["successes"])) for r in runs]
    modes = {}
    for r in runs:
        for k, v in r.get("failure_counts", {}).items():
            modes[k] = modes.get(k, 0) + int(v)
    summary.update(
        {
            "method": method,
            "seed_success_rates": seed_rates,
            "seed_mean": float(np.mean(seed_rates)),
            "seed_std": float(np.std(seed_rates)),
            "failure_counts": modes,
            "n_seeds": len(runs),
        }
    )
    print(
        f"{method}: {summary['success_rate']:.1%} "
        f"[{summary['success_ci95_wilson'][0]:.1%}, {summary['success_ci95_wilson'][1]:.1%}] "
        f"n={summary['n']}",
        flush=True,
    )
    return summary


def build_methods(lam: float, thresholds: list[float]) -> list[tuple[str, dict | None, str]]:
    """(name, planner_spec or None, world_model_kind). kind in {none, single, ensemble}."""
    t = tuple(thresholds)
    return [
        ("SAC", None, "none"),
        ("Single WM-MPC", spec(policy_guided=False, horizon=3), "single"),
        ("Policy-guided WM", spec(policy_guided=True, horizon=3), "single"),
        ("Ensemble WM", spec(policy_guided=True, horizon=3), "ensemble"),
        ("Uncertainty MPC", spec(policy_guided=True, horizon=3, uncertainty_coef=lam), "ensemble"),
        (
            "Full method",
            spec(policy_guided=True, horizon=3, uncertainty_coef=lam, value_coef=1.0),
            "ensemble",
        ),
        (
            "Adaptive-H TRUST-WM",
            spec(
                policy_guided=True,
                uncertainty_coef=lam,
                value_coef=1.0,
                adaptive_horizon=True,
                horizon_thresholds=t,
            ),
            "ensemble",
        ),
    ]


def run_eval(env, agent, ensemble, single, name, spec_d, kind, episodes, reset_seed, keep_ts):
    low, high = env.action_space.low, env.action_space.high
    model = None
    planner = None
    if kind == "single":
        model = single
    elif kind == "ensemble":
        model = ensemble
    if spec_d is None:
        act = lambda o: agent.act(o, deterministic=True)
    else:
        planner = make_planner(model, agent, low, high, **spec_d)
        act = planner.act
    return evaluate_logged(
        env,
        act,
        episodes,
        reset_seed,
        desc=name,
        planner=planner,
        world_model=ensemble if kind != "none" else ensemble,
        keep_timesteps=keep_ts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="checkpoints/sac_best.pt")
    parser.add_argument("--world-model", default="checkpoints/world_model_v2.pt")
    parser.add_argument("--data", default="data/transitions_mixed.npz")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--lambda-episodes", type=int, default=100)
    parser.add_argument("--ood-episodes", type=int, default=200)
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-horizon", action="store_true")
    parser.add_argument("--skip-lambda", action="store_true")
    parser.add_argument("--skip-ood", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    set_seed(0)

    out_dir = resolve_path("results/tables")
    run_dir = out_dir / "v3_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "v3_results.json"
    payload = {
        "episodes": args.episodes,
        "seeds": seeds,
        "matrix": [],
        "horizon": [],
        "lambda_sweep": [],
        "ood": {},
        "tests": {},
        "figures": {},
    }
    if payload_path.exists():
        try:
            payload.update(json.loads(payload_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    agent = SACAgent.from_checkpoint(str(resolve_path(args.policy)))
    ensemble = EnsembleWorldModel.load(str(resolve_path(args.world_model)))
    single = SingleMemberWorldModel(ensemble, 0)
    calib = calibrate(ensemble, resolve_path(args.data))
    payload["calibration"] = calib
    lam = float(calib["lambda"])
    thresholds = calib["adaptive_thresholds"]
    dump(payload_path, payload)
    print(json.dumps({"calibration": calib}, indent=2), flush=True)

    env = make_env("configs/env.yaml", record_metrics=True)

    def eval_named(name, spec_d, kind, seed, episodes, keep_ts=False, tag=None):
        tag = tag or name.replace(" ", "_")
        path = run_dir / f"{tag}_s{seed}.npz"
        reset_seed = seed * 100_000
        extra = {"method": name, "eval_seed": seed, "reset_seed": reset_seed, **(spec_d or {})}
        if path.exists():
            cached = load_run(path)
            if int(np.asarray(cached["successes"]).size) >= episodes:
                print(f"skip {name} seed={seed} (cached)", flush=True)
                return cached
        result = run_eval(env, agent, ensemble, single, f"{name} s{seed}", spec_d, kind, episodes, reset_seed, keep_ts)
        extra["failure_counts"] = result.get("failure_counts", {})
        save_run(path, result, extra)
        return {**result, **extra}

    try:
        methods = build_methods(lam, thresholds)
        if not args.skip_matrix:
            matrix_runs = {name: [] for name, _, _ in methods}
            for seed in seeds:
                for name, spec_d, kind in methods:
                    keep_ts = seed == seeds[0] and name in {"Uncertainty MPC", "Adaptive-H TRUST-WM", "SAC"}
                    row = eval_named(name, spec_d, kind, seed, args.episodes, keep_ts=keep_ts)
                    matrix_runs[name].append(row)
                    payload["matrix"] = [aggregate(matrix_runs[n], n) for n, _, _ in methods if matrix_runs[n]]
                    dump(payload_path, payload)

            names = [n for n, _, _ in methods]
            if "SAC" in matrix_runs and "Uncertainty MPC" in matrix_runs:
                sac = np.concatenate([r["successes"] for r in matrix_runs["SAC"]])
                um = np.concatenate([r["successes"] for r in matrix_runs["Uncertainty MPC"]])
                payload["tests"]["sac_vs_uncertainty"] = mcnemar_exact(sac, um)
            if "SAC" in matrix_runs and "Full method" in matrix_runs:
                sac = np.concatenate([r["successes"] for r in matrix_runs["SAC"]])
                full = np.concatenate([r["successes"] for r in matrix_runs["Full method"]])
                payload["tests"]["sac_vs_full"] = mcnemar_exact(sac, full)
            if "Full method" in matrix_runs and "Adaptive-H TRUST-WM" in matrix_runs:
                full = np.concatenate([r["successes"] for r in matrix_runs["Full method"]])
                adh = np.concatenate([r["successes"] for r in matrix_runs["Adaptive-H TRUST-WM"]])
                payload["tests"]["full_vs_adaptive"] = mcnemar_exact(full, adh)
            dump(payload_path, payload)

        if not args.skip_horizon:
            h_methods = [
                ("H=2", spec(policy_guided=True, horizon=2, uncertainty_coef=lam, value_coef=0.0), "ensemble"),
                ("H=4", spec(policy_guided=True, horizon=4, uncertainty_coef=lam, value_coef=0.0), "ensemble"),
            ]
            grouped = {n: [] for n, _, _ in h_methods}
            h_seeds = seeds[:3]
            for seed in h_seeds:
                for name, spec_d, kind in h_methods:
                    row = eval_named(name, spec_d, kind, seed, args.episodes, tag=f"horizon_{name}")
                    grouped[name].append(row)
                    payload["horizon"] = [aggregate(grouped[n], n) for n, _, _ in h_methods if grouped[n]]
                    dump(payload_path, payload)

        if not args.skip_lambda:
            lam_rows = []
            for lv in (0.0, 5.0, lam, 30.0, 60.0):
                name = f"lambda={lv:.2f}"
                spec_d = spec(policy_guided=True, horizon=3, uncertainty_coef=float(lv), value_coef=0.0)
                row = eval_named(name, spec_d, "ensemble", seeds[0], args.lambda_episodes, tag=f"lambda_{lv:.2f}")
                summary = summarize_success(row["successes"], row["returns"], row["lengths"], row["latencies"])
                summary["method"] = name
                summary["uncertainty_coef"] = float(lv)
                lam_rows.append(summary)
                payload["lambda_sweep"] = lam_rows
                dump(payload_path, payload)

        if not args.skip_ood:
            ood_table = payload.get("ood") or {}
            ood_methods = [
                ("SAC", None, "none"),
                ("Single WM-MPC", spec(policy_guided=False, horizon=3), "single"),
                ("Ensemble WM", spec(policy_guided=True, horizon=3), "ensemble"),
                ("Uncertainty MPC", spec(policy_guided=True, horizon=3, uncertainty_coef=lam), "ensemble"),
            ]
            matrix_by = {r["method"]: r for r in payload.get("matrix", [])}
            if "id" not in ood_table:
                ood_table["id"] = {"applied": {"level": "id", "note": "copied from matrix seed 0 / pooled ID eval"}}
                for name, _, _ in ood_methods:
                    if name in matrix_by:
                        ood_table["id"][name] = {
                            "success_rate": matrix_by[name]["success_rate"],
                            "success_ci95_wilson": matrix_by[name].get("success_ci95_wilson"),
                            "mean_return": matrix_by[name].get("mean_return"),
                            "n": matrix_by[name].get("n"),
                            "copied_from_matrix": True,
                        }
            close_env(env)
            env = None
            for level_name, cfg in OOD_LEVELS.items():
                if level_name == "id":
                    continue
                env = make_env("configs/env.yaml", record_metrics=True)
                applied = apply_ood(env, cfg)
                ood_table.setdefault(level_name, {})["applied"] = applied
                for name, spec_d, kind in ood_methods:
                    tag = f"ood_{level_name}_{name.replace(' ', '_')}"
                    row = eval_named(name, spec_d, kind, seeds[0], args.ood_episodes, tag=tag)
                    summary = summarize_success(row["successes"], row["returns"], row["lengths"], row["latencies"])
                    summary["method"] = name
                    ood_table[level_name][name] = summary
                    payload["ood"] = ood_table
                    dump(payload_path, payload)
                close_env(env)
                env = None
            env = make_env("configs/env.yaml", record_metrics=True)
            payload["ood"] = ood_table
            dump(payload_path, payload)

        by_h = {r["method"]: r for r in payload.get("horizon", [])}
        by_m = {r["method"]: r for r in payload.get("matrix", [])}
        horizon_tbl = []
        if "H=2" in by_h:
            horizon_tbl.append(by_h["H=2"])
        if "Full method" in by_m:
            h3 = dict(by_m["Full method"])
            h3["method"] = "H=3 (Full method)"
            horizon_tbl.append(h3)
        if "H=4" in by_h:
            horizon_tbl.append(by_h["H=4"])
        if "Adaptive-H TRUST-WM" in by_m:
            ad = dict(by_m["Adaptive-H TRUST-WM"])
            ad["method"] = "Adaptive-H"
            horizon_tbl.append(ad)
        if horizon_tbl:
            payload["horizon"] = horizon_tbl
            dump(payload_path, payload)

        figures = {}
        if payload.get("matrix"):
            figures["success_ci"] = plot_success_ci(
                payload["matrix"], resolve_path("results/figures/v3_success_ci.png")
            )
            figures["failure_modes"] = plot_failure_modes(
                {r["method"]: r.get("failure_counts", {}) for r in payload["matrix"]},
                resolve_path("results/figures/v3_failure_modes.png"),
            )
        ts_path = run_dir / "Uncertainty_MPC_s0_timesteps.npz"
        if not ts_path.exists():
            ts_path = run_dir / "Uncertainty_MPC_s0_timesteps.npz"
        # try both naming patterns
        candidates = list(run_dir.glob("*Uncertainty*_timesteps.npz")) + list(run_dir.glob("*TRUST*_timesteps.npz"))
        if candidates:
            ts = np.load(candidates[0])
            u = ts["uncertainty"]
            err = ts["cube_pred_error_m"]
            mask = np.isfinite(u) & np.isfinite(err)
            corr = float(np.corrcoef(u[mask], err[mask])[0, 1]) if mask.sum() > 2 else float("nan")
            figures["uncertainty_vs_error"] = plot_uncertainty_vs_error(
                u, err, resolve_path("results/figures/v3_uncertainty_vs_error.png"), corr
            )
            payload["uncertainty_error_corr"] = corr
        if payload.get("ood"):
            slim = {}
            for level, methods_d in payload["ood"].items():
                slim[level] = {k: v for k, v in methods_d.items() if k != "applied" and isinstance(v, dict) and "success_rate" in v}
            if slim:
                figures["ood"] = plot_ood(slim, resolve_path("results/figures/v3_ood.png"))
        payload["figures"] = figures
        dump(payload_path, payload)
    finally:
        if env is not None:
            close_env(env)

    print(json.dumps(jsonable({k: payload[k] for k in payload if k != "matrix"}), indent=2)[:2000], flush=True)


if __name__ == "__main__":
    main()
