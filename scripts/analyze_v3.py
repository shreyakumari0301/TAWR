"""V3 analyses that do not need a second simulator: U vs error, exploitation, latency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ensemble import EnsembleSlice, EnsembleWorldModel
from models.sac_agent import SACAgent
from planners.mpc import make_planner
from utils.paths import resolve_path
from utils.stats import mcnemar_exact, summarize_success


def _episodes(npz_path: Path) -> list[dict]:
    data = np.load(npz_path, allow_pickle=True)
    return json.loads(str(data["episodes"]))


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_pred_vs_actual(ts_path: Path, out: Path, title: str) -> dict | None:
    ts = np.load(ts_path)
    if "predicted_return" not in ts.files or "actual_remaining_return" not in ts.files:
        return None
    pred = np.asarray(ts["predicted_return"], dtype=np.float64)
    actual = np.asarray(ts["actual_remaining_return"], dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    if len(pred) < 3:
        return None
    corr = float(np.corrcoef(pred, actual)[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(pred, actual, s=4, alpha=0.18, color="#4C78A8")
    lo = float(min(pred.min(), actual.min()))
    hi = float(max(pred.max(), actual.max()))
    ax.plot([lo, hi], [lo, hi], color="#E45756", lw=1.2, label="y = x")
    ax.set_xlabel("Predicted rollout return")
    ax.set_ylabel("Actual remaining return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return {"figure": _save(fig, out), "corr": corr, "n": int(len(pred))}


def aggregate_available_seeds(run_dir: Path) -> dict:
    methods = [
        "SAC",
        "Single_WM-MPC",
        "Policy-guided_WM",
        "Ensemble_WM",
        "Uncertainty_MPC",
        "Full_method",
        "Adaptive-H_TRUST-WM",
    ]
    labels = {
        "SAC": "SAC",
        "Single_WM-MPC": "Single WM-MPC",
        "Policy-guided_WM": "Policy-guided WM",
        "Ensemble_WM": "Ensemble WM",
        "Uncertainty_MPC": "Uncertainty MPC",
        "Full_method": "Full + V_SAC",
        "Adaptive-H_TRUST-WM": "Adaptive-H",
    }
    rows = []
    paired = {}
    for tag in methods:
        seed_rates = []
        succ_all = []
        ret_all = []
        seeds_found = []
        for seed in range(5):
            path = run_dir / f"{tag}_s{seed}.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            succ = np.asarray(data["successes"], dtype=np.float64)
            if succ.size < 200:
                continue
            seed_rates.append(float(succ.mean()))
            succ_all.append(succ)
            ret_all.append(np.asarray(data["returns"], dtype=np.float64))
            seeds_found.append(seed)
            paired.setdefault(seed, {})[labels[tag]] = succ
        if not seed_rates:
            continue
        succ_cat = np.concatenate(succ_all)
        ret_cat = np.concatenate(ret_all)
        summary = summarize_success(succ_cat, ret_cat, np.ones_like(succ_cat), None)
        summary.update(
            {
                "method": labels[tag],
                "seeds": seeds_found,
                "n_seeds": len(seed_rates),
                "seed_success_rates": seed_rates,
                "seed_mean": float(np.mean(seed_rates)),
                "seed_std": float(np.std(seed_rates, ddof=1) if len(seed_rates) > 1 else 0.0),
            }
        )
        rows.append(summary)
    tests = {}
    for seed, by in paired.items():
        if "SAC" in by and "Uncertainty MPC" in by:
            tests[f"sac_vs_uncertainty_s{seed}"] = mcnemar_exact(by["SAC"], by["Uncertainty MPC"])
        if "Uncertainty MPC" in by and "Full + V_SAC" in by:
            tests[f"uncertainty_vs_vsac_s{seed}"] = mcnemar_exact(by["Uncertainty MPC"], by["Full + V_SAC"])
    return {"methods": rows, "paired_tests": tests}


def plot_u_vs_e(ts_path: Path, out: Path) -> dict:
    ts = np.load(ts_path)
    u = np.asarray(ts["uncertainty"], dtype=np.float64)
    if "state_pred_error" in ts.files and np.isfinite(ts["state_pred_error"]).sum() > 10:
        err = np.asarray(ts["state_pred_error"], dtype=np.float64)
        err_name = "full-state"
    else:
        err = np.asarray(ts["cube_pred_error_m"], dtype=np.float64)
        err_name = "cube"
    mask = np.isfinite(u) & np.isfinite(err)
    u, err = u[mask], err[mask]
    corr = float(np.corrcoef(u, err)[0, 1]) if len(u) > 2 else float("nan")
    # bin reliability: mean error in U quintiles
    qs = np.quantile(u, np.linspace(0, 1, 6))
    bins = []
    for i in range(5):
        m = (u >= qs[i]) & (u <= qs[i + 1] if i == 4 else u < qs[i + 1])
        bins.append(
            {
                "u_lo": float(qs[i]),
                "u_hi": float(qs[i + 1]),
                "mean_u": float(u[m].mean()) if m.any() else 0.0,
                "mean_cube_err_cm": float(100 * err[m].mean()) if m.any() else 0.0,
                "n": int(m.sum()),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].scatter(u, 100 * err, s=4, alpha=0.2, color="#4C78A8")
    axes[0].set_xlabel("Ensemble disagreement U")
    axes[0].set_ylabel("Prediction error (cm)" if err_name == "cube" else "||ŝ − s||")
    axes[0].set_title(f"U vs one-step {err_name} error  (r = {corr:.2f})")
    axes[1].bar(range(5), [b["mean_cube_err_cm"] for b in bins], color="#4C78A8")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels([f"Q{i+1}" for i in range(5)])
    axes[1].set_xlabel("U quintile")
    axes[1].set_ylabel("Mean cube error (cm)")
    axes[1].set_title("Higher U → larger realized error")
    fig.tight_layout()
    path = _save(fig, out)
    pred = np.asarray(ts["predicted_return"], dtype=np.float64)[mask] if "predicted_return" in ts.files else None
    actual = (
        np.asarray(ts["actual_remaining_return"], dtype=np.float64)[mask]
        if "actual_remaining_return" in ts.files
        else None
    )
    pred_corr = None
    if pred is not None and actual is not None:
        pm = np.isfinite(pred) & np.isfinite(actual)
        if pm.sum() > 2:
            pred_corr = float(np.corrcoef(pred[pm], actual[pm])[0, 1])
    return {
        "figure": path,
        "error_kind": err_name,
        "corr_u_vs_error": corr,
        "corr_u_vs_cube_error": corr,
        "n": int(len(u)),
        "quintiles": bins,
        "corr_predicted_vs_actual_return": pred_corr,
    }


def diagnose_single_wm(ts_path: Path, fig_dir: Path) -> dict | None:
    """Logged Single-WM failure: high predicted return ≠ high real return."""
    if not ts_path.exists():
        return None
    ts = np.load(ts_path)
    pred = np.asarray(ts["predicted_return"], dtype=np.float64) if "predicted_return" in ts.files else None
    actual = (
        np.asarray(ts["actual_remaining_return"], dtype=np.float64)
        if "actual_remaining_return" in ts.files
        else None
    )
    amag = np.asarray(ts["action_magnitude"], dtype=np.float64) if "action_magnitude" in ts.files else None
    adev = np.asarray(ts["action_deviation"], dtype=np.float64) if "action_deviation" in ts.files else None
    cube = np.asarray(ts["cube_pred_error_m"], dtype=np.float64) if "cube_pred_error_m" in ts.files else None
    state = np.asarray(ts["state_pred_error"], dtype=np.float64) if "state_pred_error" in ts.files else None
    pred_cube = np.asarray(ts["pred_obj_pos"], dtype=np.float64) if "pred_obj_pos" in ts.files else None
    act_cube = np.asarray(ts["actual_obj_pos"], dtype=np.float64) if "actual_obj_pos" in ts.files else None

    out: dict = {"n_steps": 0}
    if pred is not None and actual is not None:
        mask = np.isfinite(pred) & np.isfinite(actual)
        out["n_steps"] = int(mask.sum())
        if mask.sum() > 2:
            out["corr_pred_vs_actual_return"] = float(np.corrcoef(pred[mask], actual[mask])[0, 1])
            out["mean_predicted_return"] = float(pred[mask].mean())
            out["mean_actual_remaining_return"] = float(actual[mask].mean())
            high = pred[mask] >= np.quantile(pred[mask], 0.9)
            out["top10pct_pred_return_mean"] = float(pred[mask][high].mean())
            out["top10pct_actual_return_mean"] = float(actual[mask][high].mean())
        scatter = plot_pred_vs_actual(
            ts_path, fig_dir / "v3_single_wm_pred_vs_actual.png", "Single WM-MPC · predicted vs actual return"
        )
        if scatter:
            out["pred_vs_actual"] = scatter
    if amag is not None:
        m = np.isfinite(amag)
        out["mean_action_magnitude"] = float(amag[m].mean()) if m.any() else None
    if adev is not None:
        m = np.isfinite(adev)
        out["mean_action_deviation"] = float(adev[m].mean()) if m.any() else None
    if cube is not None:
        m = np.isfinite(cube)
        out["mean_cube_error_cm"] = float(100 * cube[m].mean()) if m.any() else None
        out["p95_cube_error_cm"] = float(100 * np.percentile(cube[m], 95)) if m.any() else None
    if state is not None:
        m = np.isfinite(state)
        out["mean_state_pred_error"] = float(state[m].mean()) if m.any() else None
    if pred_cube is not None and act_cube is not None and pred_cube.ndim == 2 and act_cube.ndim == 2:
        m = np.isfinite(pred_cube).all(axis=1) & np.isfinite(act_cube).all(axis=1)
        if m.any():
            err = np.linalg.norm(pred_cube[m] - act_cube[m], axis=1)
            out["mean_cube_pos_error_cm"] = float(100 * err.mean())
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            ax.scatter(act_cube[m, 0], act_cube[m, 1], s=4, alpha=0.15, label="actual", color="#4C78A8")
            ax.scatter(pred_cube[m, 0], pred_cube[m, 1], s=4, alpha=0.15, label="predicted", color="#E45756")
            ax.set_xlabel("cube x")
            ax.set_ylabel("cube y")
            ax.set_title("Single WM · predicted vs actual cube (xy)")
            ax.legend()
            fig.tight_layout()
            out["cube_xy_figure"] = _save(fig, fig_dir / "v3_single_wm_cube_xy.png")
    return out


def plot_single_wm_exploitation(run_dir: Path, out: Path) -> dict:
    rows = []
    for tag, label in (
        ("Single_WM-MPC_s0", "Single WM-MPC"),
        ("Policy-guided_WM_s0", "Policy-guided WM"),
        ("Ensemble_WM_s0", "Ensemble WM"),
        ("Uncertainty_MPC_s0", "Uncertainty MPC"),
        ("SAC_s0", "SAC"),
    ):
        path = run_dir / f"{tag}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        eps = json.loads(str(data["episodes"]))
        dev = np.asarray([e["mean_action_deviation"] for e in eps], dtype=np.float64)
        ret = np.asarray(data["returns"], dtype=np.float64)
        ok = np.asarray(data["successes"], dtype=np.float64)
        rows.append(
            {
                "method": label,
                "success_rate": float(ok.mean()),
                "mean_return": float(ret.mean()),
                "mean_action_deviation": float(dev.mean()),
                "p95_action_deviation": float(np.percentile(dev, 95)),
            }
        )
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    names = [r["method"] for r in rows]
    ax.bar(range(len(names)), [r["mean_action_deviation"] for r in rows], color="#E45756")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(r"Mean $\|a_{WM}-a_{SAC}\|$")
    ax.set_title("Naïve shooting leaves the SAC action distribution")
    fig.tight_layout()
    path = _save(fig, out)
    return {"figure": path, "rows": rows}


def plot_seed0_matrix(payload_path: Path, out: Path) -> str | None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    rows = payload.get("matrix") or []
    if not rows:
        return None
    names = [r["method"] for r in rows]
    rates = [100 * r["success_rate"] for r in rows]
    lo = [100 * r["success_ci95_wilson"][0] for r in rows]
    hi = [100 * r["success_ci95_wilson"][1] for r in rows]
    yerr = np.vstack([np.asarray(rates) - np.asarray(lo), np.asarray(hi) - np.asarray(rates)])
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(range(len(names)), rates, yerr=yerr, capsize=4, color="#4C78A8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=22, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("V3 seed 0 · 200 matched episodes · 95% Wilson CI")
    fig.tight_layout()
    return _save(fig, out)


def latency_sweep(n_obs: int = 80) -> dict:
    """Time planner.act on stored observations — no second simulator."""
    import time

    data = np.load(resolve_path("data/transitions_mixed.npz"), allow_pickle=True)
    mask = data["split"] == "val"
    obs = np.asarray(data["obs"][mask][:n_obs], dtype=np.float32)
    agent = SACAgent.from_checkpoint(str(resolve_path("checkpoints/sac_best.pt")))
    ensemble = EnsembleWorldModel.load(str(resolve_path("checkpoints/world_model_v2.pt")))
    action_low = np.array([-1, -1, -1, -1], dtype=np.float32)
    action_high = np.array([1, 1, 1, 1], dtype=np.float32)
    rows = []
    for n_models in (1, 3, 5):
        model = EnsembleSlice(ensemble, n_models)
        for K in (16, 32, 64):
            planner = make_planner(
                model,
                agent,
                action_low,
                action_high,
                horizon=3,
                num_candidates=K,
                discount=0.8,
                policy_guided=True,
                uncertainty_coef=14.72 if n_models > 1 else 0.0,
                value_coef=0.0,
            )
            # warmup
            planner.act(obs[0])
            times = []
            for o in obs:
                t0 = time.perf_counter()
                planner.act(o)
                times.append(time.perf_counter() - t0)
            times = np.asarray(times)
            rows.append(
                {
                    "ensemble_size": n_models,
                    "K": K,
                    "mean_latency_ms": float(1000 * times.mean()),
                    "p95_latency_ms": float(1000 * np.percentile(times, 95)),
                }
            )
            print(f"N={n_models} K={K}: {1000*times.mean():.1f} ms", flush=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for n_models in (1, 3, 5):
        xs = [r["K"] for r in rows if r["ensemble_size"] == n_models]
        ys = [r["mean_latency_ms"] for r in rows if r["ensemble_size"] == n_models]
        ax.plot(xs, ys, marker="o", label=f"N={n_models}")
    ax.set_xlabel("Candidates K")
    ax.set_ylabel("Mean planning latency (ms)")
    ax.set_title("Planning cost vs search width and ensemble size")
    ax.legend()
    fig.tight_layout()
    figure = _save(fig, resolve_path("results/figures/v3_latency.png"))
    return {"rows": rows, "figure": figure}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()
    run_dir = resolve_path("results/tables/v3_runs")
    fig_dir = resolve_path("results/figures")
    out = {
        "uncertainty_calibration": None,
        "single_wm_exploitation": None,
        "seed0_matrix": None,
    }
    ts = run_dir / "Uncertainty_MPC_s0_timesteps.npz"
    if ts.exists():
        out["uncertainty_calibration"] = plot_u_vs_e(ts, fig_dir / "v3_uncertainty_vs_error.png")
    out["single_wm_exploitation"] = plot_single_wm_exploitation(
        run_dir, fig_dir / "v3_action_deviation.png"
    )
    payload = resolve_path("results/tables/v3_results.json")
    if payload.exists():
        out["seed0_matrix"] = plot_seed0_matrix(payload, fig_dir / "v3_success_ci.png")
        data = json.loads(payload.read_text(encoding="utf-8"))
        by = {r["method"]: r for r in data.get("matrix", [])}
        if "SAC" in by and "Uncertainty MPC" in by:
            # paired test needs raw successes from npz
            sac = np.load(run_dir / "SAC_s0.npz")["successes"]
            um = np.load(run_dir / "Uncertainty_MPC_s0.npz")["successes"]
            out["sac_vs_uncertainty_seed0"] = mcnemar_exact(sac, um)
            out["sac_seed0"] = summarize_success(
                sac,
                np.load(run_dir / "SAC_s0.npz")["returns"],
                np.load(run_dir / "SAC_s0.npz")["lengths"],
                np.load(run_dir / "SAC_s0.npz")["latencies"],
            )
            out["uncertainty_seed0"] = summarize_success(
                um,
                np.load(run_dir / "Uncertainty_MPC_s0.npz")["returns"],
                np.load(run_dir / "Uncertainty_MPC_s0.npz")["lengths"],
                np.load(run_dir / "Uncertainty_MPC_s0.npz")["latencies"],
            )
    ts_um = run_dir / "Uncertainty_MPC_s0_timesteps.npz"
    if ts_um.exists():
        out["uncertainty_pred_vs_actual"] = plot_pred_vs_actual(
            ts_um, fig_dir / "v3_uncertainty_pred_vs_actual.png", "Uncertainty MPC · predicted vs actual return"
        )
    ts_single = run_dir / "Single_WM-MPC_logged_s0_timesteps.npz"
    out["single_wm_diagnostic"] = diagnose_single_wm(ts_single, fig_dir)
    if ts_single.exists() and out["single_wm_diagnostic"] and "pred_vs_actual" in out["single_wm_diagnostic"]:
        out["single_wm_pred_vs_actual"] = out["single_wm_diagnostic"]["pred_vs_actual"]
    out["seed_aggregate"] = aggregate_available_seeds(run_dir)
    if args.skip_latency:
        out["latency"] = None
    else:
        print("latency sweep...", flush=True)
        out["latency"] = latency_sweep(80)
    dest = resolve_path("results/tables/v3_analysis.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
