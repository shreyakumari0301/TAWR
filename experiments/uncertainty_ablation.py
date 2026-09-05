"""Uncertainty / gated-planning ablation. Prefer scripts/run_pipeline.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.check_call(
    [
        sys.executable,
        str(ROOT / "scripts" / "eval_planner.py"),
        "--method",
        "gated",
        "--out",
        "results/tables/eval_gated.json",
    ],
    cwd=str(ROOT),
)
