"""Remaining V3 work after seed 0: 5-seed matrix, horizon vs Uncertainty, logged single-WM, OOD.

Does not include Adaptive-H or V_SAC in the final method. Full + V_SAC stays as a
negative ablation in the matrix. Safe to resume: cached 200-episode npz files are skipped.

Do not start this while another PickCube eval is using the simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from envs.ood import OOD_LEVELS, apply_ood
from envs.pick_place import close_env, make_env
from models.ensemble import EnsembleWorldModel, SingleMemberWorldModel
from models.sac_agent import SACAgent
from run_v3_experiments import (
    aggregate,
    calibrate,
    dump,
    jsonable,
    load_run,
    run_eval,
    save_run,
    spec,
)
from utils.paths import resolve_path
from utils.seed import set_seed
from utils.stats import mcnemar_exact, summarize_success


CORE = [
    ("SAC", None, "none"),
    ("Single WM-MPC", spec(policy_guided=False, horizon=3), "single"),
    ("Policy-guided WM", spec(policy_guided=True, horizon=3), "single"),
    ("Ensemble WM", spec(policy_guided=True, horizon=3), "ensemble"),
    ("Uncertainty MPC", spec(policy_guided=True, horizon=3, uncertainty_coef=None), "ensemble"),
    ("Full method", spec(policy_guided=True, horizon=3, uncertainty_coef=None, value_coef=1.0), "ensemble"),
]


def fill_lambda(methods, lam: float):
    out = []
    for name, spec_d, kind in methods:
        if spec_d is None:
            out.append((name, None, kind))
            continue
        cfg = dict(spec_d)
        if cfg.get("uncertainty_coef") is None:
            cfg["uncertainty_coef"] = lam
        out.append((name, cfg, kind))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--ood-episodes", type=int, default=200)
    parser.add_argument("--logged-episodes", type=int, default=200)
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-horizon", action="store_true")
    parser.add_argument("--skip-logged-single", action="store_true")
    parser.add_argument("--skip-ood", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.wait_seconds:
        print(f"waiting {args.wait_seconds}s before touching the env", flush=True)
        time.sleep(args.wait_seconds)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    set_seed(0)
    out_dir = resolve_path("results/tables")
    run_dir = out_dir / "v3_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "v3_validation.json"
    payload = {
        "episodes": args.episodes,
        "seeds": seeds,
        "matrix": [],
        "horizon": [],
        "ood": {},
        "tests": {},
        "architecture": {
            "final": "policy-guided proposals + H=3 + ensemble dynamics + R-λU",
            "dropped": ["V_SAC terminal value", "DAgger", "Adaptive-H"],
            "reason": "seed-0: Uncertainty MPC 74.5% vs Full+V 61.5% vs Adaptive-H 69.5%",
        },
    }
    if payload_path.exists():
        try:
            payload.update(json.loads(payload_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    agent = SACAgent.from_checkpoint(str(resolve_path("checkpoints/sac_best.pt")))
    ensemble = EnsembleWorldModel.load(str(resolve_path("checkpoints/world_model_v2.pt")))
    single = SingleMemberWorldModel(ensemble, 0)
    calib = calibrate(ensemble, resolve_path("data/transitions_mixed.npz"))
    payload["calibration"] = calib
    lam = float(calib["lambda"])
    dump(payload_path, payload)

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
        methods = fill_lambda(CORE, lam)
        if not args.skip_matrix:
            matrix_runs = {name: [] for name, _, _ in methods}
            for seed in seeds:
                for name, spec_d, kind in methods:
                    row = eval_named(name, spec_d, kind, seed, args.episodes)
                    matrix_runs[name].append(row)
                    payload["matrix"] = [aggregate(matrix_runs[n], n) for n, _, _ in methods if matrix_runs[n]]
                    dump(payload_path, payload)
            if matrix_runs["SAC"] and matrix_runs["Uncertainty MPC"]:
                sac = np.concatenate([r["successes"] for r in matrix_runs["SAC"]])
                um = np.concatenate([r["successes"] for r in matrix_runs["Uncertainty MPC"]])
                payload["tests"]["sac_vs_uncertainty"] = mcnemar_exact(sac, um)
            if matrix_runs["Uncertainty MPC"] and matrix_runs["Full method"]:
                um = np.concatenate([r["successes"] for r in matrix_runs["Uncertainty MPC"]])
                full = np.concatenate([r["successes"] for r in matrix_runs["Full method"]])
                payload["tests"]["uncertainty_vs_vsac"] = mcnemar_exact(um, full)
            dump(payload_path, payload)

        if not args.skip_horizon:
            h_rows = []
            for H, tag, name in (
                (2, "horizon_H=2", "H=2 Uncertainty"),
                (3, "Uncertainty_MPC", "H=3 Uncertainty"),
                (4, "horizon_H=4", "H=4 Uncertainty"),
            ):
                spec_d = spec(policy_guided=True, horizon=H, uncertainty_coef=lam, value_coef=0.0)
                row = eval_named(name, spec_d, "ensemble", 0, args.episodes, tag=tag)
                summary = summarize_success(row["successes"], row["returns"], row["lengths"], row["latencies"])
                summary["method"] = name
                summary["horizon"] = H
                h_rows.append(summary)
            ad_path = run_dir / "Adaptive-H_TRUST-WM_s0.npz"
            if ad_path.exists():
                ad = load_run(ad_path)
                summary = summarize_success(ad["successes"], ad["returns"], ad["lengths"], ad["latencies"])
                summary["method"] = "Adaptive-H (seed 0, not in final method)"
                summary["horizon"] = "adaptive"
                h_rows.append(summary)
            payload["horizon"] = h_rows
            dump(payload_path, payload)
            best_fixed = max((r for r in h_rows if isinstance(r.get("horizon"), int)), key=lambda r: r["success_rate"])
            ad = next((r for r in h_rows if r["method"].startswith("Adaptive-H")), None)
            payload["keep_adaptive_h"] = bool(ad and ad["success_rate"] >= best_fixed["success_rate"])
            dump(payload_path, payload)

        if not args.skip_logged_single:
            spec_d = spec(policy_guided=False, horizon=3)
            eval_named(
                "Single WM-MPC logged",
                spec_d,
                "single",
                0,
                args.logged_episodes,
                keep_ts=True,
                tag="Single_WM-MPC_logged",
            )

        if not args.skip_ood:
            ood_methods = [
                ("SAC", None, "none"),
                ("Single WM-MPC", spec(policy_guided=False, horizon=3), "single"),
                ("Ensemble WM", spec(policy_guided=True, horizon=3), "ensemble"),
                ("Uncertainty MPC", spec(policy_guided=True, horizon=3, uncertainty_coef=lam), "ensemble"),
            ]
            ood_table = payload.get("ood") or {}
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
                    row = eval_named(name, spec_d, kind, 0, args.ood_episodes, tag=tag)
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
    finally:
        if env is not None:
            close_env(env)

    print(json.dumps(jsonable(payload), indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
