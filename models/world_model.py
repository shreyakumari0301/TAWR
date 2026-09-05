"""Phase 4: MLP dynamics that predict Δo, so ô_{t+1} = o_t + fθ(normalize(o_t), a_t)."""

from __future__ import annotations

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes=(512, 512, 256),
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim + action_dim
        for width in hidden_sizes:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        layers.append(nn.Linear(last, obs_dim))
        self.net = nn.Sequential(*layers)
        if obs_mean is None:
            obs_mean = torch.zeros(obs_dim)
        if obs_std is None:
            obs_std = torch.ones(obs_dim)
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32).reshape(-1))
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32).reshape(-1))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / (self.obs_std + 1e-6)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([self.normalize(obs), action], dim=-1))
        return obs + delta
