"""Twin Q-networks for SAC."""

from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = self._build(obs_dim + action_dim, hidden_sizes)
        self.q2 = self._build(obs_dim + action_dim, hidden_sizes)

    @staticmethod
    def _build(in_dim: int, hidden_sizes) -> nn.Sequential:
        layers: list[nn.Module] = []
        last = in_dim
        for width in hidden_sizes:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))
