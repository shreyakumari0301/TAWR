# World-model planning for robotic pick-and-place

**Research question:** When should a robot trust short-horizon world-model rollouts over its model-free policy?

This repo learns state-based dynamics for ManiSkill **PickCube-v1**, uses them for receding-horizon planning, and compares that against a SAC baseline. Observations are **state** (not images). The planner uses an **MLP ensemble world model** over a short horizon, then SAC value for the rest.

## Status

V1 numbers are in `results/tables/final_results.json`. V2 (short horizon, policy-guided sampling, SAC terminal value, 5-model ensemble, R−λU, one DAgger pass) is in `results/tables/v2_results.json`.

V2, 30 eval episodes (binomial SE ≈ 7 pp):

| Method | Success | Return | Notes |
| ------ | ------: | -----: | ----- |
| SAC | 60% | 15.2 | baseline |
| H=5 K=32 (single net) | 40% | 17.7 | longer imagination |
| Random shooting H=3 | 3.3% | 2.9 | model exploitation |
| Policy-guided WM | 63% | 14.4 | SAC + action noise |
| WM + V_SAC | 70% | 15.0 | short horizon + terminal value |
| Ensemble WM (5 models) | 77% | 16.1 | mean prediction |
| Uncertainty MPC | **80%** | 12.9 | R − λU, λ≈14.7 |
| Full method | **80%** | 13.7 | policy-guided + V + U |
| Full + 1× DAgger | 70% | 14.5 | 5k planner transitions |

The bottleneck was planner exploitation of model error, not model size. Random shooting failed. Policy-guided H=3, ensemble mean, and an uncertainty penalty are what moved success. One DAgger iteration did not help.

V3 seed 0 (200 matched episodes) **did not reproduce 80%**. The ranking is the result:

| Method | Success | 95% Wilson CI |
| ------ | ------: | ------------- |
| SAC | 64.5% | 58–71% |
| Single WM-MPC (random shooting) | 1.0% | 0.3–4% |
| Policy-guided WM | 56.5% | 50–63% |
| Ensemble WM | 69.5% | 63–76% |
| **Uncertainty MPC (R−λU)** | **74.5%** | **68–80%** |
| Full + V_SAC | 61.5% | 55–68% |
| Adaptive-H | 69.5% | 63–76% |

Frozen architecture if later seeds agree:

**policy-guided proposals + H=3 + ensemble dynamics + uncertainty penalty.**

V_SAC is a negative ablation (74.5% → 61.5%, McNemar p = 0.002). Adaptive-H does not beat fixed-H Uncertainty MPC. Ensemble disagreement vs one-step cube error: r = 0.44 (6237 steps). Five-seed confirmation, logged single-WM failure analysis, and mild/hard OOD are in progress.

```bash
python scripts\run_v3_experiments.py --episodes 200 --seeds 0,1,2,3,4 --skip-videos
python scripts\run_v3_validation.py
python scripts\analyze_v3.py
```

## Setup

Use Python 3.10 or 3.11. ManiSkill is most reliable on Linux with an NVIDIA GPU. On Windows it can still run **CPU simulation** (`physx_cpu`), which is what this repo defaults to.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

A `.venv` is already created in this repo. On this Windows machine, `pd_ee_delta_pos` needs inverse kinematics. Pinocchio does not install via pip here, so `envs/pick_place.py` uses the bundled `pytorch_kinematics` solver on CPU. The Pinocchio warning on startup is expected and can be ignored.

## Phase 1 — already run

```bash
.venv\Scripts\activate
python scripts\inspect_env.py
python scripts\evaluate_random.py --episodes 100
python scripts\collect_dataset.py --num-transitions 5000
```

That produced:

- `results/tables/random_baseline.json`
- `data/transitions_random.npz`

## Results

SAC trained for 80k CPU steps. V1 used 25 episodes per method (`results/tables/final_results.json`). V2 used 30 (`results/tables/v2_results.json`).

V1: the default H=5 / K=64 planner did **not** beat SAC (44% vs 56%). H=3 reached 60% and K=32 reached 64%. Cube error was 1.7 cm at 1 step and diverged by 20 steps.

V2: a 5-model weighted delta WM has 1.0 cm 1-step cube error and 24 cm at H=20. Policy-guided H=3 + ensemble mean + R−λU reached **80%**. Random shooting at the same horizon was 3.3%.

```bash
python scripts\evaluate_policy.py --episodes 100 --checkpoint checkpoints\sac_best.pt
python scripts\run_v2_experiments.py --skip-train
```

## Task

- Env: `PickCube-v1` (Panda)
- Observation: flattened state (`qpos`, `qvel`, TCP pose, cube pose, goal, gripper)
- Action: `pd_ee_delta_pos` → `[dx, dy, dz, gripper]`
- Success: cube within 2.5 cm of the goal and the robot is static

## Later pipeline

```
o_t  →  encoder/identity  →  z_t
(z_t, a_t)  →  world model  →  ẑ_{t+1}

At control time:
  sample K action sequences around SAC
  roll each out for H=3 steps in the world model
  score J = Σ γ^h (R_pred − λU)
  execute the first action of the best candidate
```

## Experiment matrix (after Phase 2)

| Experiment | Question |
| ---------- | -------- |
| Random / SAC | How good is model-free control? |
| SAC + WM-MPC | Does short-horizon planning help? |
| H ∈ {1,3,5,10,20} | Does longer imagination help or hurt? |
| K ∈ {32,64,128,256} | How much search is needed? |
| Ensemble WM | Does disagreement track prediction error? |
| Confidence-gated WM | When should the robot trust the model? |

Primary metrics: **success rate**, **steps to success**, **MSE vs rollout horizon**, **planning latency per action**.

## Layout

```text
configs/     env, SAC, world model, planner
envs/        PickCube wrapper, obs slices, goal score
models/      SAC actor, MLP dynamics, ensemble
planners/    shooting MPC, confidence gating
scripts/     inspect, evaluate, collect, train, eval
results/     tables and figures
checkpoints/ trained weights
data/        transition datasets
```
