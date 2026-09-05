"""Figures for the V3 research package."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_success_ci(rows: list[dict], out: Path) -> str:
    names = [r["method"] for r in rows]
    rates = [100 * r["success_rate"] for r in rows]
    lo = [100 * r["success_ci95_wilson"][0] for r in rows]
    hi = [100 * r["success_ci95_wilson"][1] for r in rows]
    yerr = np.vstack([np.asarray(rates) - np.asarray(lo), np.asarray(hi) - np.asarray(rates)])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(range(len(names)), rates, yerr=yerr, capsize=4, color="#4C78A8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("PickCube success ± 95% Wilson CI")
    ax.axhline(60, color="#54A24B", linestyle="--", linewidth=1, label="V2 SAC point estimate")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def plot_failure_modes(counts_by_method: dict[str, dict[str, int]], out: Path) -> str:
    methods = list(counts_by_method)
    modes = sorted({m for c in counts_by_method.values() for m in c})
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(methods))
    width = 0.8 / max(len(modes), 1)
    for i, mode in enumerate(modes):
        vals = [counts_by_method[m].get(mode, 0) for m in methods]
        ax.bar(x + i * width, vals, width, label=mode)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Episodes")
    ax.set_title("Failure modes by method")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, out)


def plot_uncertainty_vs_error(u: np.ndarray, err: np.ndarray, out: Path, corr: float | None = None) -> str:
    mask = np.isfinite(u) & np.isfinite(err)
    u, err = u[mask], err[mask]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(u, 100 * err, s=6, alpha=0.25, color="#4C78A8")
    ax.set_xlabel("Ensemble disagreement U")
    ax.set_ylabel("Cube prediction error (cm)")
    title = "Uncertainty vs one-step cube error"
    if corr is not None and np.isfinite(corr):
        title += f"  (r = {corr:.2f})"
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def plot_ood(table: dict, out: Path) -> str:
    methods = list(next(iter(table.values())))
    levels = list(table)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(methods))
    width = 0.8 / max(len(levels), 1)
    for i, level in enumerate(levels):
        vals = [100 * table[level][m]["success_rate"] for m in methods]
        ax.bar(x + i * width, vals, width, label=level)
    ax.set_xticks(x + width * (len(levels) - 1) / 2)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("In-distribution vs OOD PickCube")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out)
