# Copyright (c) EC710 Final Project — Humanoid Locomotion via SAC
# SPDX-License-Identifier: BSD-3-Clause
"""Train a SAC policy for bipedal humanoid locomotion in IsaacLab.

Run from the IsaacLab repo root with:

    ./isaaclab.sh -p /path/to/humanoid_sac/train.py \
        --task EC710-Humanoid-SAC-v0 --num_envs 4096 --headless

Add `--video --video_length 200 --video_interval 2000` for periodic video logs.
"""

from __future__ import annotations

import argparse
import sys

# -----------------------------------------------------------------------------
# 1. CLI + AppLauncher  (must run BEFORE any isaaclab/omni.* imports)
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: I001

parser = argparse.ArgumentParser(description="Train SAC humanoid in IsaacLab.")
parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between videos.")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel environments.")
parser.add_argument("--task", type=str, default="EC710-Humanoid-SAC-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override training iterations.")
AppLauncher.add_app_launcher_args(parser)

# Split args so skrl ignores ours and vice-versa
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 2. Imports that depend on the simulator being up
# -----------------------------------------------------------------------------
import os
from datetime import datetime

import gymnasium as gym
import skrl
import torch
import yaml
from packaging import version

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

# Register our task
import humanoid_locomotion  # noqa: F401  (registers the env via __init__.py)

# Sanity check
SKRL_MIN = "1.4.0"
if version.parse(skrl.__version__) < version.parse(SKRL_MIN):
    raise RuntimeError(f"skrl >= {SKRL_MIN} required (have {skrl.__version__}).")


def main():
    # Load task + agent config
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"
    env_cfg.seed = args_cli.seed

    skrl_cfg_path = gym.spec(args_cli.task).kwargs["skrl_cfg_entry_point"]
    with open(skrl_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations

    # Logging directory
    log_root = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    os.makedirs(log_root, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = agent_cfg["agent"]["experiment"]["experiment_name"] or stamp
    log_dir = os.path.join(log_root, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = exp_name

    # Dump configs for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Build the env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda s: s % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    # skrl VecEnv wrapper
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Set seeds
    skrl.config.torch.deterministic = False
    torch.manual_seed(args_cli.seed)

    # Build SAC runner from yaml
        # Build SAC runner from yaml
    print("[DEBUG] policy YAML config:")
    import json
    print(json.dumps(agent_cfg["models"]["policy"], indent=2, default=str))
    print("[DEBUG] critic_1 YAML config:")
    print(json.dumps(agent_cfg["models"]["critic_1"], indent=2, default=str))
    runner = Runner(env, agent_cfg)

    print("[INFO] Logging to:", log_dir)
    print_dict(agent_cfg, nesting=4)

    # Sanity prints — confirms we actually reach training
    print(f"[DEBUG] timesteps        = {agent_cfg['trainer']['timesteps']}")
    print(f"[DEBUG] num_envs         = {env.num_envs}")
    print(f"[DEBUG] device           = {env.device}")
    print(f"[DEBUG] obs_space        = {env.observation_space}")
    print(f"[DEBUG] act_space        = {env.action_space}")
    print("[DEBUG] Calling runner.run('train') now...", flush=True)

    # Train  (skrl 1.4.x requires the explicit 'train' mode)
    runner.run("train")

    print("[DEBUG] runner.run returned cleanly", flush=True)

    # Save final checkpoint
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
        # Bypass the hang in _app_control_on_stop_handle_fn (a known IsaacLab cleanup issue)
        os._exit(0)
