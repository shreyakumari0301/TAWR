"""Gaussian MLP policy with tanh squashing for SAC."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes=(256, 256),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden_sizes:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, action_dim)
        self.log_std_head = nn.Linear(last, action_dim)

        low = (
            np.full(action_dim, -1.0, dtype=np.float32)
            if action_low is None
            else np.asarray(action_low, dtype=np.float32)
        )
        high = (
            np.ones(action_dim, dtype=np.float32)
            if action_high is None
            else np.asarray(action_high, dtype=np.float32)
        )
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        mean = self.mean_head(x)
        log_std = torch.tanh(self.log_std_head(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        dist = Normal(mean, log_std.exp())
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        det_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, det_action

    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias
