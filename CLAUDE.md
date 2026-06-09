# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`roboparty_train` is the Roboparty training workspace for RPO humanoid locomotion policies. It is a thin workspace whose real content lives in two Git submodules:

- `robolab/` — the Roboparty Isaac Lab extension: environments, training/eval scripts, MuJoCo Sim2Sim scripts, robot assets, and motion data.
- `rsl_rl/` — a Roboparty-compatible snapshot of the RSL-RL RL library (PPO, AMP, distillation).

Both are installed editable into the **Isaac Lab Python environment**. Nearly all real work happens inside `robolab/`.

## Environment Requirements

Scripts cannot run without Isaac Sim 5.1.0 / Isaac Lab 2.3.2 and a Python 3.11 env. Always run commands in the Python environment Isaac Lab uses — most `ImportError`s on `isaaclab*` mean the wrong env is active. There is no test suite, lint config, or build step in this repo; "running" means launching a train/play/sim2sim script.

## Setup

```bash
git submodule update --init --recursive   # if robolab/ or rsl_rl/ is empty
pip install -e ./robolab
pip install -e ./rsl_rl
python robolab/scripts/tools/list_envs.py  # verify install; lists exact task ids
```

## Common Commands

Run from the repo root. `--task` takes a registered gym id (see below); always run `list_envs.py` to confirm exact ids.

```bash
# Train (headless, many parallel envs)
python robolab/scripts/rsl_rl/train.py --task=RPO-Flat --headless --logger=tensorboard --num_envs=8192

# Play / evaluate a trained policy
python robolab/scripts/rsl_rl/play.py        --task=RPO-Flat        --num_envs=1
python robolab/scripts/rsl_rl/play_amp.py    --task=RPO-AMP-Play    --num_envs=1
python robolab/scripts/rsl_rl/play_bm.py     --task=RPO-BeyondMimic --num_envs=1
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1

# Export ONNX (requires num_envs=1)
python robolab/scripts/rsl_rl/play_parkour.py --task=RPO-Parkour-Play --num_envs=1 --exportonnx

# Sim2Sim transfer check in MuJoCo (one script per workflow)
python robolab/scripts/mujoco/sim2sim_rpo.py --load_model "/abs/path/to/exported/policy.pt"
```

Training distributes automatically when launched via `torchrun` (`WORLD_SIZE > 1`). Outputs/`logs`/`wandb`/`outputs` are gitignored.

## Architecture

### Environment registration is implicit
`robolab/tasks/__init__.py` calls `isaaclab_tasks.utils.import_packages` to import every sub-package (except `utils`) at import time. Each task's `__init__.py` then calls `gym.register(...)` with two entry points:

- `env_cfg_entry_point` → the environment config class (e.g. `rpo_env_cfg:RPOFlatEnvCfg`)
- `rsl_rl_cfg_entry_point` → the RSL-RL agent/runner config (e.g. `agents/rpo_agent_cfg:RPOFlatAgentCfg`)

A task only becomes visible after `import robolab.tasks` runs (the scripts do this). To add a new env: create a task package under `tasks/`, add `gym.register` in its `__init__.py`, and provide both an env cfg and an agent cfg.

### Two environment styles
Tasks live under one of two trees mirroring Isaac Lab's two paradigms:

- `tasks/direct/` — subclasses `DirectRLEnv`; the env class (e.g. `base/base_env.py:BaseEnv`) owns the observation/reward/reset logic directly. Members: `base` (`RPO-Flat`, `RPO-Rough`), `attn_enc` (`RPO-AttnEnc`), `interrupt` (`RPO-Interrupt`).
- `tasks/manager_based/` — uses Isaac Lab managers; behavior is composed from term configs in `mdp/` (observations, rewards, events, terminations, commands, curriculums) and custom `managers/`. Members: `amp` (`RPO-AMP`, `RPO-AMP-Play`), `beyondmimic` (`RPO-BeyondMimic`, `RPO-Getup-Mimic`), `parkour` (`RPO-Parkour`, `RPO-Parkour-Play`).

Each task package pairs an `*_env_cfg.py` (scene, robot, terrain, mdp terms) with an `agents/` dir holding the RSL-RL config that selects the runner and PPO/AMP hyperparameters.

### Custom RSL-RL fork
Training scripts import runners from the vendored `rsl_rl` submodule, not upstream: `OnPolicyRunner`, `AMPRunner`, `DistillationRunner` (`rsl_rl/runners/`) backed by `algorithms/` (`ppo`, `ppo_amp`, `distillation`). The agent cfg's runner field decides which runner a task uses. Changing PPO/AMP behavior generally means editing `rsl_rl/`, not `robolab/`.

### Robot assets and motion data
- `robolab/robolab/assets/robots/roboparty.py` defines the RPO articulation config consumed by env cfgs.
- `robolab/data/robots/roboparty/rpo/` holds the physical assets: `urdf/`, `mjcf/` (MuJoCo), `meshes/`, `terrain_assets/`.
- `robolab/data/motions/` holds reference motions in three layouts: `rpo_gmr` (raw from GMR), `rpo_lab` (reordered for Isaac Lab), `rpo_bm` (BeyondMimic).

### Motion retargeting pipeline
Datasets from [GMR](https://github.com/Roboparty/GMR) use URDF/XML joint order, which differs from Isaac Lab's joint order. Reorder before training:

```bash
python robolab/scripts/tools/retarget/dataset_retarget.py
```

The joint mapping is a YAML such as `robolab/scripts/tools/retarget/config/rpo.yaml`. AMP and BeyondMimic consume the reordered motions.

### Supporting modules
`robolab/robolab/` also contains shared building blocks reused across tasks: `sensors/` (grouped ray casters, noisy cameras, volume points), `terrains/` (height field, trimesh, virtual obstacles), and `utils/` (buffers, noise, warp helpers).

## Conventions

- All env ids are prefixed `RPO-`; `*-Play` variants are single-/few-env eval configs of a training task.
- ONNX export only works with `--num_envs=1`.
- Source files carry the dual Isaac Lab + RoboLab BSD-3 header; preserve it when adding files.
