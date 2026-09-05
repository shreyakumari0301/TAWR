"""One-step and multi-step rollout evaluation of the learned world model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.obs_slices import PICKCUBE_STATE_SLICES
from models.ensemble import EnsembleWorldModel
from utils.paths import resolve_path


def load_ensemble(path: Path, device: torch.device) -> EnsembleWorldModel:
    return EnsembleWorldModel.load(path, device)


def consecutive_windows(episode_id: np.ndarray, horizon: int) -> np.ndarray:
    idx = []
    start = 0
    while start < len(episode_id):
        ep = episode_id[start]
        end = start
        while end < len(episode_id) and episode_id[end] == ep:
            end += 1
        if end - start > horizon:
            idx.extend(range(start, end - horizon))
        start = end
    return np.asarray(idx, dtype=np.int64)


@torch.no_grad()
def rollout_errors(model, obs, actions, nexts, episode_id, horizons, device) -> list[dict]:
    rows = []
    n_take = min(2048, len(obs))
    for H in horizons:
        starts = consecutive_windows(episode_id, H)
        if len(starts) == 0:
            continue
        rng = np.random.default_rng(0)
        starts = rng.choice(starts, size=min(n_take, len(starts)), replace=False)
        state = torch.as_tensor(obs[starts], device=device)
        target = torch.as_tensor(nexts[starts + (H - 1)], device=device)
        for h in range(H):
            act = torch.as_tensor(actions[starts + h], device=device)
            state = model(state, act)
        err = (state - target).cpu().numpy()
        mse = float(np.mean(err**2))
        tcp = np.linalg.norm(err[:, PICKCUBE_STATE_SLICES["tcp_pos"]], axis=1).mean()
        cube = np.linalg.norm(err[:, PICKCUBE_STATE_SLICES["obj_pos"]], axis=1).mean()
        joints = np.mean(np.abs(err[:, PICKCUBE_STATE_SLICES["qpos"]]))
        rows.append(
            {
                "horizon": int(H),
                "mse": mse,
                "tcp_pos_error_m": float(tcp),
                "cube_pos_error_m": float(cube),
                "joint_abs_error": float(joints),
                "n": int(len(starts)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model.yaml")
    parser.add_argument("--data", default="data/transitions_mixed.npz")
    parser.add_argument("--checkpoint", default="checkpoints/world_model.pt")
    parser.add_argument("--out", default="results/tables/rollout_error.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleWorldModel.load(resolve_path(args.checkpoint), device)
    data = np.load(resolve_path(args.data), allow_pickle=True)
    mask = data["split"] == "test"
    obs, act, nxt, epid = data["obs"][mask], data["action"][mask], data["next_obs"][mask], data["episode_id"][mask]

    rows = rollout_errors(model, obs, act, nxt, epid, [1, 2, 3, 5, 10, 20], device)
    out_csv = resolve_path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig_path = resolve_path(f"results/figures/{out_csv.stem}.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    hs = [r["horizon"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(hs, [r["mse"] for r in rows], marker="o", label="overall MSE")
    plt.plot(hs, [r["tcp_pos_error_m"] for r in rows], marker="o", label="TCP error (m)")
    plt.plot(hs, [r["cube_pos_error_m"] for r in rows], marker="o", label="cube error (m)")
    plt.xlabel("rollout horizon (steps)")
    plt.ylabel("prediction error")
    plt.title("World-model rollout error vs horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Uncertainty vs one-step error correlation
    with torch.no_grad():
        o = torch.as_tensor(obs[:2048], device=device)
        a = torch.as_tensor(act[:2048], device=device)
        n = torch.as_tensor(nxt[:2048], device=device)
        pred = model(o, a)
        se = ((pred - n) ** 2).mean(dim=-1).cpu().numpy()
        u = model.disagreement(o, a).cpu().numpy()
    corr = float(np.corrcoef(u, se)[0, 1]) if len(u) > 2 else 0.0
    summary = {"rows": rows, "uncertainty_error_corr": corr, "figure": str(fig_path)}
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
