"""Collect mixed (random / noisy SAC / SAC) transitions for the world model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pick_place import close_env, is_success, make_env
from models.sac_agent import SACAgent
from utils.config import load_config
from utils.paths import resolve_path
from utils.seed import set_seed


def split_by_episode(data: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    episode_ids = np.unique(data["episode_id"])
    rng.shuffle(episode_ids)
    n = len(episode_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": set(episode_ids[:n_train].tolist()),
        "val": set(episode_ids[n_train : n_train + n_val].tolist()),
        "test": set(episode_ids[n_train + n_val :].tolist()),
    }
    split_name = np.empty(len(data["episode_id"]), dtype=object)
    for name, ids in splits.items():
        split_name[np.isin(data["episode_id"], list(ids))] = name
    data["split"] = split_name.astype(str)
    return data


def collect(env, num_transitions: int, seed: int, act_fn, source: str) -> dict[str, np.ndarray]:
    obs_buf, act_buf, next_obs_buf = [], [], []
    rew_buf, done_buf, success_buf, episode_buf = [], [], [], []
    obs, info = env.reset(seed=seed)
    episode_id = 0
    pbar = tqdm(total=num_transitions, desc=f"collect {source}")
    while len(obs_buf) < num_transitions:
        action = act_fn(np.asarray(obs, dtype=np.float32), env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        obs_buf.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        act_buf.append(np.asarray(action, dtype=np.float32).reshape(-1))
        next_obs_buf.append(np.asarray(next_obs, dtype=np.float32).reshape(-1))
        rew_buf.append(np.float32(reward))
        done_buf.append(np.bool_(done))
        success_buf.append(np.bool_(is_success(info)))
        episode_buf.append(np.int32(episode_id))
        obs = next_obs
        pbar.update(1)
        if done:
            episode_id += 1
            obs, info = env.reset(seed=seed + episode_id)
    pbar.close()
    return {
        "obs": np.stack(obs_buf),
        "action": np.stack(act_buf),
        "next_obs": np.stack(next_obs_buf),
        "reward": np.asarray(rew_buf, dtype=np.float32),
        "done": np.asarray(done_buf, dtype=np.bool_),
        "success": np.asarray(success_buf, dtype=np.bool_),
        "episode_id": np.asarray(episode_buf, dtype=np.int32),
        "source": np.array([source] * len(obs_buf)),
    }


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    offset = 0
    out = {}
    for key in parts[0]:
        if key == "episode_id":
            chunks = []
            for part in parts:
                chunks.append(part[key] + offset)
                offset = int(chunks[-1].max()) + 1 if len(chunks[-1]) else offset
            out[key] = np.concatenate(chunks)
        else:
            out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sac_best.pt")
    parser.add_argument("--num-transitions", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/transitions_mixed.npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed)
    set_seed(seed)
    n = args.num_transitions
    n_random, n_partial, n_sac = int(0.4 * n), int(0.3 * n), n - int(0.4 * n) - int(0.3 * n)

    env = make_env(args.config)
    ckpt = resolve_path(args.checkpoint)
    agent = SACAgent.from_checkpoint(str(ckpt)) if ckpt.exists() else None
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    def random_act(obs, env_):
        return env_.action_space.sample()

    def partial_act(obs, env_):
        if agent is None:
            return env_.action_space.sample()
        return np.clip(agent.act(obs, deterministic=False), action_low, action_high)

    def sac_act(obs, env_):
        if agent is None:
            return env_.action_space.sample()
        return np.clip(agent.act(obs, deterministic=True), action_low, action_high)

    try:
        parts = [
            collect(env, n_random, seed, random_act, "random"),
            collect(env, n_partial, seed + 10_000, partial_act, "partial_sac"),
            collect(env, n_sac, seed + 20_000, sac_act, "sac"),
        ]
    finally:
        close_env(env)

    data = split_by_episode(concat(parts), seed=seed)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    summary = {
        "path": str(out_path),
        "num_transitions": int(data["obs"].shape[0]),
        "num_episodes": int(np.unique(data["episode_id"]).size),
        "sources": {s: int(np.sum(data["source"] == s)) for s in ("random", "partial_sac", "sac")},
        "splits": {name: int(np.sum(data["split"] == name)) for name in ("train", "val", "test")},
        "used_policy": bool(agent is not None),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
