"""Phase 10: ensemble of dynamics models. Disagreement is the uncertainty signal."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.world_model import WorldModel


class EnsembleWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_sizes=(512, 512, 256),
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(
            [
                WorldModel(obs_dim, action_dim, hidden_sizes, obs_mean, obs_std)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(obs, action) for m in self.models], dim=0)
        return preds.mean(dim=0)

    def predict_members(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(obs, action) for m in self.models], dim=0)

    def disagreement(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = self.predict_members(obs, action)
        return preds.var(dim=0).mean(dim=-1)

    @classmethod
    def load(cls, path, device: str | torch.device = "cpu") -> "EnsembleWorldModel":
        device = torch.device(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("cfg", {})
        ensemble_size = int(cfg.get("ensemble_size", 5))
        state = ckpt.get("ensemble", {})
        member_ids = {
            int(key.split(".")[1])
            for key in state
            if key.startswith("models.") and key.split(".")[1].isdigit()
        }
        if member_ids:
            ensemble_size = max(member_ids) + 1
        model = cls(
            obs_dim=int(ckpt["obs_dim"]),
            action_dim=int(ckpt["action_dim"]),
            ensemble_size=ensemble_size,
            hidden_sizes=tuple(cfg.get("hidden_sizes", [512, 512, 256])),
            obs_mean=torch.as_tensor(ckpt["obs_mean"]),
            obs_std=torch.as_tensor(ckpt["obs_std"]),
        ).to(device)
        model.load_state_dict(ckpt["ensemble"])
        model.eval()
        return model


class EnsembleSlice(nn.Module):
    """Use the first N members for ensemble-size latency / robustness sweeps."""

    def __init__(self, ensemble: EnsembleWorldModel, n: int):
        super().__init__()
        n = max(1, min(n, len(ensemble.models)))
        self.models = nn.ModuleList(list(ensemble.models)[:n])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(obs, action) for m in self.models], dim=0)
        return preds.mean(dim=0)

    def disagreement(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if len(self.models) < 2:
            return torch.zeros(obs.shape[0], device=obs.device)
        preds = torch.stack([m(obs, action) for m in self.models], dim=0)
        return preds.var(dim=0).mean(dim=-1)


class SingleMemberWorldModel(nn.Module):
    """Use one ensemble member so the ensemble-mean ablation is meaningful."""

    def __init__(self, ensemble: EnsembleWorldModel, index: int = 0):
        super().__init__()
        self.model = ensemble.models[index]

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model(obs, action)
