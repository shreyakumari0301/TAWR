"""Receding-horizon shooting around a base policy, with optional V, U, and Adaptive-H."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from envs.obs_slices import PICKCUBE_STATE_SLICES
from envs.reward import goal_score_torch


@dataclass
class PlannerConfig:
    horizon: int = 3
    num_candidates: int = 32
    action_noise_std: float = 0.15
    discount: float = 0.8
    uncertainty_coef: float = 0.0
    value_coef: float = 0.0
    policy_guided: bool = True
    keep_policy_action: bool = True
    adaptive_horizon: bool = False
    horizon_thresholds: tuple[float, float, float] = (0.001, 0.003, 0.008)
    horizon_schedule: tuple[int, int, int, int] = (4, 3, 2, 1)


def make_planner(world_model, policy, action_low, action_high, device="cpu", **overrides) -> "PolicyGuidedMPC":
    cfg_fields = {k: v for k, v in overrides.items() if k in PlannerConfig.__dataclass_fields__}
    return PolicyGuidedMPC(
        world_model=world_model,
        policy=policy,
        config=PlannerConfig(**cfg_fields),
        action_low=action_low,
        action_high=action_high,
        device=device,
    )


class PolicyGuidedMPC:
    def __init__(
        self,
        world_model,
        policy,
        config: PlannerConfig,
        action_low,
        action_high,
        device: str | torch.device = "cpu",
    ):
        self.world_model = world_model
        self.policy = policy
        self.config = config
        self.device = torch.device(device)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.last_uncertainty = 0.0
        self.last_horizon = int(config.horizon)
        self.last_info: dict = {}

    def _policy_action(self, state: torch.Tensor) -> torch.Tensor:
        if hasattr(self.policy, "actor"):
            return self.policy.actor.get_eval_action(state)
        return torch.as_tensor(
            self.policy.act_batch(state.cpu().numpy(), deterministic=True),
            device=self.device,
        )

    def _choose_horizon(self, obs_t: torch.Tensor) -> tuple[int, float, torch.Tensor]:
        cfg = self.config
        state0 = obs_t.unsqueeze(0)
        sac = self._policy_action(state0)
        u0 = 0.0
        if hasattr(self.world_model, "disagreement"):
            u0 = float(self.world_model.disagreement(state0, sac).item())
        H = int(cfg.horizon)
        if cfg.adaptive_horizon:
            t1, t2, t3 = cfg.horizon_thresholds
            h4, h3, h2, h1 = cfg.horizon_schedule
            if u0 < t1:
                H = int(h4)
            elif u0 < t2:
                H = int(h3)
            elif u0 < t3:
                H = int(h2)
            else:
                H = int(h1)
        return max(1, H), u0, sac.squeeze(0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.config
        k = cfg.num_candidates
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.world_model.eval()
        with torch.no_grad():
            H, u0, sac_action = self._choose_horizon(obs_t)
            self.last_horizon = H
            state = obs_t.unsqueeze(0).repeat(k, 1)
            scores = torch.zeros(k, device=self.device)
            first_actions = None
            first_u = None
            first_next = None
            gamma = 1.0
            for h in range(H):
                base = self._policy_action(state)
                if cfg.policy_guided:
                    action = (base + torch.randn_like(base) * cfg.action_noise_std).clamp(
                        self.action_low, self.action_high
                    )
                    if cfg.keep_policy_action:
                        action[0] = base[0]
                else:
                    action = torch.rand_like(base) * (self.action_high - self.action_low) + self.action_low

                if hasattr(self.world_model, "disagreement"):
                    penalty = self.world_model.disagreement(state, action)
                else:
                    penalty = torch.zeros(k, device=self.device)

                next_state = self.world_model(state, action)
                reward = goal_score_torch(next_state)
                if h == 0:
                    first_actions = action
                    first_u = penalty
                    first_next = next_state
                scores = scores + gamma * (reward - cfg.uncertainty_coef * penalty)
                state = next_state
                gamma *= cfg.discount

            if cfg.value_coef and hasattr(self.policy, "value"):
                scores = scores + gamma * cfg.value_coef * self.policy.value(state)

            best = int(torch.argmax(scores).item())
            chosen = first_actions[best]
            u_best = float(first_u[best].item()) if first_u is not None else u0
            self.last_uncertainty = u_best
            sac_np = sac_action.detach().cpu().numpy().astype(np.float32)
            wm_np = chosen.detach().cpu().numpy().astype(np.float32)
            pred_next = first_next[best].detach().cpu().numpy().astype(np.float32)
            self.last_info = {
                "uncertainty": u_best,
                "uncertainty_sac": float(u0),
                "horizon": int(H),
                "predicted_return": float(scores[best].item()),
                "action_deviation": float(np.linalg.norm(wm_np - sac_np)),
                "action_magnitude": float(np.linalg.norm(wm_np)),
                "predicted_obj_pos": pred_next[PICKCUBE_STATE_SLICES["obj_pos"]].copy(),
                "predicted_next": pred_next.copy(),
            }
            return wm_np
