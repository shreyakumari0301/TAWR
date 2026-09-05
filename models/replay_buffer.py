from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.pos = 0
        self.full = False
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.pos] = np.float32(reward)
        self.next_obs[self.pos] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.dones[self.pos] = np.float32(done)
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int) -> ReplayBatch:
        idx = np.random.randint(0, len(self), size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=self.device),
            actions=torch.as_tensor(self.actions[idx], device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device),
            dones=torch.as_tensor(self.dones[idx], device=self.device),
        )
