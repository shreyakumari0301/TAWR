"""Bootstrap CIs and paired tests for binary success and scalar returns."""

from __future__ import annotations

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=np.mean,
) -> dict:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    point = float(statistic(values))
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = statistic(values[idx], axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "n": int(n)}


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """Paired binary outcomes. b vs a; n01 = a success b fail, n10 = a fail b success."""
    a = np.asarray(a_ok, dtype=bool).reshape(-1)
    b = np.asarray(b_ok, dtype=bool).reshape(-1)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p = float(binomtest(n10, n, 0.5, alternative="two-sided").pvalue)
        except Exception:
            from math import erfc

            z = (abs(n10 - n01) - 1.0) / max(np.sqrt(n), 1e-8)
            p = float(erfc(z / np.sqrt(2.0)))
    return {"n01": n01, "n10": n10, "n11": int(np.sum(a & b)), "n00": int(np.sum(~a & ~b)), "p_value": float(p)}


def summarize_success(successes: np.ndarray, returns: np.ndarray, lengths: np.ndarray, latencies: np.ndarray | None = None) -> dict:
    successes = np.asarray(successes, dtype=np.float64).reshape(-1)
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    k = int(successes.sum())
    n = int(successes.size)
    rate = bootstrap_ci(successes)
    lo, hi = wilson_ci(k, n)
    out = {
        "n": n,
        "success_count": k,
        "success_rate": float(successes.mean()) if n else 0.0,
        "success_ci95_bootstrap": [rate["lo"], rate["hi"]],
        "success_ci95_wilson": [lo, hi],
        "mean_return": float(returns.mean()) if n else 0.0,
        "return_ci95": [bootstrap_ci(returns)["lo"], bootstrap_ci(returns)["hi"]],
        "mean_length": float(lengths.mean()) if n else 0.0,
    }
    if latencies is not None and len(latencies):
        lat = np.asarray(latencies, dtype=np.float64)
        out["mean_latency_s"] = float(lat.mean())
        out["p95_latency_s"] = float(np.percentile(lat, 95))
    return out
