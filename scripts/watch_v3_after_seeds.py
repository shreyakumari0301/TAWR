"""After seeds 2–4 land, stop the matrix job and run logged Single-WM + U-vs-E.

Does not start a new method. Horizon and OOD stay skipped until those caches exist.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "tables" / "v3_runs"
CORE = [
    "SAC",
    "Single_WM-MPC",
    "Policy-guided_WM",
    "Ensemble_WM",
    "Uncertainty_MPC",
    "Full_method",
]
LOGGED = RUN_DIR / "Single_WM-MPC_logged_s0.npz"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def core_done(seed: int) -> bool:
    return all((RUN_DIR / f"{tag}_s{seed}.npz").exists() for tag in CORE)


def validation_pids() -> list[int]:
    import os

    here = os.getpid()
    out = []
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        for p in psutil.process_iter(["pid", "cmdline"]):
            if p.pid == here:
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            if "run_v3_validation.py" in cmd and "watch_v3_after_seeds" not in cmd:
                out.append(p.pid)
        return out
    # fallback: Windows CIM
    raw = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -and $_.CommandLine -match 'run_v3_validation.py' "
                "-and $_.CommandLine -notmatch 'watch_v3_after_seeds' } | "
                "Select-Object -ExpandProperty ProcessId"
            ),
        ],
        text=True,
    )
    for line in raw.splitlines():
        line = line.strip()
        if line.isdigit() and int(line) != here:
            out.append(int(line))
    return out


def stop_validation() -> None:
    pids = validation_pids()
    if not pids:
        return
    print(f"stopping validation pids {pids} so logged Single-WM can start", flush=True)
    subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {','.join(map(str, pids))} -Force -ErrorAction SilentlyContinue"])
    time.sleep(8)


def main() -> None:
    print("waiting for seeds 2–4 core caches, then logged Single-WM + U-vs-E", flush=True)
    while not all(core_done(s) for s in (2, 3, 4)):
        done = [s for s in (2, 3, 4) if core_done(s)]
        print(f"seeds complete: {done}", flush=True)
        time.sleep(60)
    print("seeds 2–4 complete", flush=True)
    if not LOGGED.exists():
        stop_validation()
        print("starting logged Single-WM diagnostic", flush=True)
        subprocess.check_call(
            [
                str(PY),
                "-u",
                str(ROOT / "scripts" / "run_v3_validation.py"),
                "--episodes",
                "200",
                "--seeds",
                "0",
                "--skip-matrix",
                "--skip-horizon",
                "--skip-ood",
                "--logged-episodes",
                "200",
            ],
            cwd=str(ROOT),
        )
    print("running U-vs-E + Single-WM analysis", flush=True)
    subprocess.check_call(
        [str(PY), "-u", str(ROOT / "scripts" / "analyze_v3.py"), "--skip-latency"],
        cwd=str(ROOT),
    )
    print("diagnostics done", flush=True)


if __name__ == "__main__":
    main()
