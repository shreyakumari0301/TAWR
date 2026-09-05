"""Soft Actor-Critic agent used as the model-free baseline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.critic import Critic
from models.policy import Actor
from models.replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes=(256, 256),
        gamma: float = 0.8,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = Actor(obs_dim, action_dim, hidden_sizes, action_low, action_high).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        actions = self.act_batch(np.asarray(obs, dtype=np.float32), deterministic=deterministic)
        if np.asarray(obs).ndim == 1:
            return actions[0]
        return actions

    def act_batch(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_eval_action(obs_t)
            else:
                action, _, _ = self.actor.get_action(obs_t)
        return action.cpu().numpy().astype(np.float32)

    def update(self, replay: ReplayBuffer, batch_size: int) -> dict[str, float]:
        batch = replay.sample(batch_size)
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.get_action(batch.next_obs)
            q1_t, q2_t = self.critic_target(batch.next_obs, next_action)
            min_q_t = torch.min(q1_t, q2_t) - alpha * next_log_prob
            target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * min_q_t

        q1, q2 = self.critic(batch.obs, batch.actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action, log_prob, _ = self.actor.get_action(batch.obs)
        q1_pi, q2_pi = self.critic(batch.obs, action)
        actor_loss = (alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update()
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": self.alpha,
            "q1": float(q1.mean().item()),
        }

    def _soft_update(self) -> None:
        with torch.no_grad():
            for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.mul_(1.0 - self.tau)
                target.data.add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        if "critic" not in state:
            return
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.alpha_opt.load_state_dict(state["alpha_opt"])

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> "SACAgent":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt.get("cfg", {})
        agent = cls(
            obs_dim=int(ckpt["obs_dim"]),
            action_dim=int(ckpt["action_dim"]),
            action_low=np.asarray(ckpt["action_low"], dtype=np.float32),
            action_high=np.asarray(ckpt["action_high"], dtype=np.float32),
            hidden_sizes=tuple(cfg.get("hidden_sizes", [256, 256])),
            gamma=float(cfg.get("gamma", 0.8)),
            tau=float(cfg.get("tau", 0.01)),
            device=map_location,
        )
        agent.load_state_dict(ckpt)
        agent.actor.eval()
        agent.critic.eval()
        agent.critic_target.eval()
        return agent

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s) ≈ min_i Q_i(s, π_det(s)). Shape (B,)."""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        action = self.actor.get_eval_action(obs)
        q1, q2 = self.critic_target(obs, action)
        return torch.min(q1, q2).squeeze(-1)
