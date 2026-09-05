"""Train the SAC model-free baseline on PickCube."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pick_place import close_env, is_success, make_env
from models.replay_buffer import ReplayBuffer
from models.sac_agent import SACAgent
from utils.config import load_config
from utils.evaluate import evaluate_actor
from utils.paths import resolve_path
from utils.seed import set_seed


def save_checkpoint(path: Path, agent: SACAgent, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.state_dict()
    payload.update(extra)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--config", default="configs/sac.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    cfg = load_config(args.config)
    if args.timesteps is not None:
        cfg["total_timesteps"] = args.timesteps
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if args.eval_freq is not None:
        cfg["eval_freq"] = args.eval_freq
    if args.eval_episodes is not None:
        cfg["eval_episodes"] = args.eval_episodes
    seed = int(env_cfg.get("seed", 0) if args.seed is None else args.seed)
    cfg["seed"] = seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = int(cfg["total_timesteps"])
    warmup_steps = int(cfg["warmup_steps"])
    eval_freq = int(cfg["eval_freq"])
    eval_episodes = int(cfg["eval_episodes"])
    log_freq = int(cfg.get("log_freq", 1000))
    updates_per_step = int(cfg.get("updates_per_step", 1))
    update_every = int(cfg.get("update_every", 2))

    train_env = make_env(args.env_config, ignore_terminations=True, record_metrics=True)
    obs, _ = train_env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    obs_dim = int(obs.shape[0])
    action_dim = int(np.prod(train_env.action_space.shape))
    action_low = np.asarray(train_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(train_env.action_space.high, dtype=np.float32)

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=tuple(cfg.get("hidden_sizes", [256, 256])),
        gamma=float(cfg.get("gamma", 0.8)),
        tau=float(cfg.get("tau", 0.01)),
        actor_lr=float(cfg.get("actor_lr", 3e-4)),
        critic_lr=float(cfg.get("critic_lr", 3e-4)),
        alpha_lr=float(cfg.get("alpha_lr", 3e-4)),
        device=device,
    )
    replay = ReplayBuffer(obs_dim, action_dim, int(cfg["buffer_size"]), device)

    log_path = resolve_path(cfg.get("log_path", "results/tables/sac_train_log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = resolve_path(cfg.get("best_path", "checkpoints/sac_best.pt"))
    final_path = resolve_path(cfg.get("final_path", "checkpoints/sac_final.pt"))

    best_success = -1.0
    update_info = {}
    ep_return = 0.0
    ep_len = 0
    ep_success = False
    recent_returns = []
    ckpt_extra = {
        "cfg": cfg,
        "action_low": action_low,
        "action_high": action_high,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }

    print(f"device={device} obs_dim={obs_dim} action_dim={action_dim}")
    print(f"training {total_timesteps} steps, warmup {warmup_steps}")

    t0 = time.perf_counter()
    pbar = tqdm(range(1, total_timesteps + 1), desc="sac")
    for step in pbar:
        if step <= warmup_steps:
            action = train_env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = train_env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = bool(terminated)
        replay.add(obs, action, reward, next_obs, done)

        ep_return += float(reward)
        ep_len += 1
        ep_success = ep_success or is_success(info)
        obs = next_obs

        if terminated or truncated:
            recent_returns.append(ep_return)
            obs, _ = train_env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            ep_return = 0.0
            ep_len = 0
            ep_success = False

        if (
            step >= warmup_steps
            and len(replay) >= int(cfg["batch_size"])
            and step % update_every == 0
        ):
            for _ in range(updates_per_step * update_every):
                update_info = agent.update(replay, int(cfg["batch_size"]))

        if step % log_freq == 0:
            elapsed = time.perf_counter() - t0
            row = {
                "step": step,
                "fps": step / max(elapsed, 1e-8),
                "mean_return_100": float(np.mean(recent_returns[-100:])) if recent_returns else None,
                **update_info,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        if step % 10000 == 0:
            ckpt_extra["global_step"] = step
            save_checkpoint(final_path, agent, ckpt_extra)

        should_eval = step >= warmup_steps and (step % eval_freq == 0 or step == total_timesteps)
        if should_eval:
            eval_env = make_env(args.env_config, record_metrics=True)
            try:
                agent.actor.eval()
                metrics = evaluate_actor(
                    eval_env,
                    lambda o: agent.act(o, deterministic=True),
                    eval_episodes,
                    seed + 10_000,
                )
                agent.actor.train()
            finally:
                close_env(eval_env)
            metrics["step"] = step
            metrics["method"] = "sac"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"eval": metrics}) + "\n")
            pbar.set_postfix(
                success=f"{metrics['success_rate']:.2f}",
                ret=f"{metrics['mean_return']:.2f}",
                alpha=f"{agent.alpha:.3f}",
            )
            ckpt_extra["global_step"] = step
            ckpt_extra["eval"] = metrics
            if metrics["success_rate"] >= best_success:
                best_success = metrics["success_rate"]
                ckpt_extra["best_success_rate"] = best_success
                save_checkpoint(best_path, agent, ckpt_extra)
            save_checkpoint(final_path, agent, ckpt_extra)

    save_checkpoint(final_path, agent, ckpt_extra)
    close_env(train_env)
    print(f"best success rate: {best_success:.3f}")
    print(f"saved best: {best_path}")
    print(f"saved final: {final_path}")


if __name__ == "__main__":
    main()
