"""Candidate-count ablation wrapper. Prefer scripts/run_pipeline.py for the full matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for K in (32, 64, 128, 256):
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_planner.py"),
            "--method",
            "mpc",
            "--candidates",
            str(K),
            "--out",
            f"results/tables/candidates_K{K}.json",
        ],
        cwd=str(ROOT),
    )
