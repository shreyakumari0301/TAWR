"""Horizon ablation wrapper. Prefer scripts/run_pipeline.py for the full matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for H in (1, 3, 5, 10, 20):
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_planner.py"),
            "--method",
            "mpc",
            "--horizon",
            str(H),
            "--out",
            f"results/tables/horizon_H{H}.json",
        ],
        cwd=str(ROOT),
    )
