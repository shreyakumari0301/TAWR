"""After seed-1 core methods are cached, stop the original V3 job and start the clean validation.

The original job still evaluates Adaptive-H on every seed and then a mis-specified
horizon / OOD block (V_SAC scoring, Adaptive-H in OOD). Seed 0 already showed
Adaptive-H does not beat Uncertainty MPC, so remaining seeds skip it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "tables" / "v3_runs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CORE = [
    "SAC",
    "Single_WM-MPC",
    "Policy-guided_WM",
    "Ensemble_WM",
    "Uncertainty_MPC",
    "Full_method",
]


def has_run(tag: str, seed: int) -> bool:
    path = RUN / f"{tag}_s{seed}.npz"
    return path.exists() and path.stat().st_size > 1000


def core_done(seed: int) -> bool:
    return all(has_run(tag, seed) for tag in CORE)


def find_v3_pids() -> list[int]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'run_v3_experiments.py' } | "
        "Select-Object -ExpandProperty ProcessId",
    ]
    out = subprocess.check_output(cmd, text=True, cwd=str(ROOT))
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def kill_pids(pids: list[int]) -> None:
    for pid in pids:
        print(f"stopping original V3 job pid={pid}", flush=True)
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)


def remaining_seeds() -> list[int]:
    return [s for s in range(5) if not core_done(s)]


def start_validation(seeds: list[int]) -> None:
    seed_arg = ",".join(str(s) for s in seeds) if seeds else "0"
    cmd = [
        str(PYTHON),
        "-u",
        "scripts/run_v3_validation.py",
        "--episodes",
        "200",
        "--seeds",
        seed_arg,
        "--ood-episodes",
        "200",
        "--logged-episodes",
        "200",
    ]
    if not seeds:
        cmd.append("--skip-matrix")
    print("starting", " ".join(cmd), flush=True)
    log = ROOT / "results" / "tables" / "v3_validation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n# launch seeds={seed_arg}\n")
        subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)


def main() -> None:
    print("watching for seed-1 core completion, then cutting over", flush=True)
    while True:
        done = [s for s in range(5) if core_done(s)]
        print(f"core seeds complete: {done}", flush=True)
        pids = find_v3_pids()
        # Cut over once seed 0 and seed 1 core methods are on disk, before Adaptive-H s1+
        if core_done(0) and core_done(1):
            if pids:
                # Only kill if Adaptive-H s1 is missing or all 5 cores are done
                if (not has_run("Adaptive-H_TRUST-WM", 1)) or all(core_done(s) for s in range(5)):
                    kill_pids(pids)
                    time.sleep(8)
            left = remaining_seeds()
            start_validation(left)
            return
        # Only treat "original job gone" as a cutover if seed 1 core is also done,
        # or the original process is truly absent AND no eval is mid-write.
        if not pids and core_done(0) and core_done(1):
            print("original V3 job already exited; starting remaining work", flush=True)
            start_validation(remaining_seeds())
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
