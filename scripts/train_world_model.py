"""Train an ensemble of delta-prediction world models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.obs_slices import observation_loss_weights
from models.ensemble import EnsembleWorldModel
from utils.config import load_config
from utils.paths import resolve_path
from utils.seed import set_seed


def load_split(path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    mask = data["split"] == split
    return data["obs"][mask], data["action"][mask], data["next_obs"][mask]


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.mean(weights * (pred - target) ** 2)


def train_member(model, loader, optimizer, device, weights) -> float:
    model.train()
    total = 0.0
    n = 0
    for obs, act, nxt in loader:
        obs, act, nxt = obs.to(device), act.to(device), nxt.to(device)
        pred = model(obs, act)
        loss = weighted_mse(pred, nxt, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(obs)
        n += len(obs)
    return total / max(n, 1)


@torch.no_grad()
def eval_mse(model, loader, device, weights) -> float:
    model.eval()
    total = 0.0
    n = 0
    for obs, act, nxt in loader:
        obs, act, nxt = obs.to(device), act.to(device), nxt.to(device)
        pred = model(obs, act)
        total += float(weighted_mse(pred, nxt, weights).item()) * len(obs)
        n += len(obs)
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model.yaml")
    parser.add_argument("--data", default="data/transitions_mixed.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ensemble_size is not None:
        cfg["ensemble_size"] = args.ensemble_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = resolve_path(args.data)
    obs, act, nxt = load_split(data_path, "train")
    val_obs, val_act, val_nxt = load_split(data_path, "val")

    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0).clip(min=1e-6)
    weights = torch.as_tensor(observation_loss_weights(obs.shape[1]), device=device)
    ensemble = EnsembleWorldModel(
        obs_dim=obs.shape[1],
        action_dim=act.shape[1],
        ensemble_size=int(cfg.get("ensemble_size", 3)),
        hidden_sizes=tuple(cfg.get("hidden_sizes", [512, 512, 256])),
        obs_mean=torch.as_tensor(obs_mean),
        obs_std=torch.as_tensor(obs_std),
    ).to(device)

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 256))
    lr = float(cfg.get("lr", 1e-3))
    history = []

    val_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(val_obs),
            torch.as_tensor(val_act),
            torch.as_tensor(val_nxt),
        ),
        batch_size=1024,
        shuffle=False,
    )

    for i, member in enumerate(ensemble.models):
        idx = rng.integers(0, len(obs), size=len(obs))
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(obs[idx]),
                torch.as_tensor(act[idx]),
                torch.as_tensor(nxt[idx]),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        opt = torch.optim.Adam(member.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 1e-5)))
        best_val = 1e9
        best_state = {k: v.detach().cpu().clone() for k, v in member.state_dict().items()}
        for epoch in tqdm(range(1, epochs + 1), desc=f"wm[{i}]"):
            train_loss = train_member(member, loader, opt, device, weights)
            val_loss = eval_mse(member, val_loader, device, weights)
            history.append({"member": i, "epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in member.state_dict().items()}
        member.load_state_dict(best_state)

    out_path = resolve_path(args.out or cfg.get("checkpoint_path", "checkpoints/world_model.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ensemble": ensemble.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "obs_dim": obs.shape[1],
            "action_dim": act.shape[1],
            "cfg": cfg,
            "history": history,
        },
        out_path,
    )
    meta = {
        "path": str(out_path),
        "ensemble_size": len(ensemble.models),
        "weighted_loss": True,
        "final_val_mse": [h["val_mse"] for h in history if h["epoch"] == epochs],
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
