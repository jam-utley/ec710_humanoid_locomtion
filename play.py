from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play SAC humanoid policy.")
parser.add_argument("--task", type=str, default="EC710-Humanoid-SAC-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

print("[BOOT] Launching AppLauncher...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[BOOT] AppLauncher ready", flush=True)

import os

import gymnasium as gym
import torch
import yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

import humanoid_locomotion


def main():
    #Include a ton of comments because, like trianing, this has broken too many times. 
    print("[STEP 1] Building env_cfg", flush=True)
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 4.0  # easier to see during play

    print("[STEP 2] Loading agent yaml", flush=True)
    skrl_cfg_path = gym.spec(args_cli.task).kwargs["skrl_cfg_entry_point"]
    with open(skrl_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    print("[STEP 3] gym.make", flush=True)
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        print("[STEP 3a] Wrapping with RecordVideo", flush=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(
                os.path.dirname(args_cli.checkpoint), "..", "videos", "play"
            ),
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    print("[STEP 4] Wrapping with SkrlVecEnvWrapper", flush=True)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    print(f"[STEP 4] env wrapped. num_envs={env.num_envs} device={env.device} "
          f"obs_space={env.observation_space} act_space={env.action_space}", flush=True)

    print("[STEP 5] Building skrl Runner", flush=True)
    runner = Runner(env, agent_cfg)
    print("[STEP 5] Runner built", flush=True)

    print(f"[STEP 6] Loading checkpoint: {args_cli.checkpoint}", flush=True)
    runner.agent.load(args_cli.checkpoint)
    print("[STEP 6] Checkpoint loaded", flush=True)

    # Put all SAC's networks into eval mode (no-op for layers without train/eval state,
    # disables dropout/batchnorm running stats if you have any)
    for _model in runner.agent.models.values():
        if _model is not None:
            _model.eval()
    print("[STEP 7] Models set to eval mode", flush=True)

    print("[STEP 8] About to call env.reset()", flush=True)
    obs, _ = env.reset()
    print(f"[STEP 8] env.reset() returned. obs shape={obs.shape} "
          f"obs has_nan={torch.isnan(obs).any().item()}", flush=True)

    print("[STEP 9] First agent.act() call (this primes any lazy init)", flush=True)
    with torch.inference_mode():
        first_actions = runner.agent.act(observations=obs, states=obs, timestep=0, timesteps=0)[0]
    print(f"[STEP 9] First actions: shape={first_actions.shape} "
          f"mean={first_actions.mean().item():.4f} "
          f"std={first_actions.std().item():.4f} "
          f"min={first_actions.min().item():.4f} "
          f"max={first_actions.max().item():.4f} "
          f"has_nan={torch.isnan(first_actions).any().item()}", flush=True)

    print("[STEP 10] Entering rollout loop. Watch the IsaacSim viewer.", flush=True)
    step = 0
    try:
        # NOTE: removed `for _ in range(int(1e6))` — replaced with while loop
        # tied to the sim window so closing the window exits cleanly.
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = runner.agent.act(observations=obs, states=obs, timestep=0, timesteps=0)[0]
            obs, _, _, _, _ = env.step(actions)

            if step % 100 == 0:
                print(f"[ROLLOUT] step={step} "
                      f"action mean={actions.mean().item():.3f} "
                      f"std={actions.std().item():.3f} "
                      f"obs mean={obs.mean().item():.3f}", flush=True)
            step += 1
    except Exception as e:
        print(f"[ERROR] Rollout died at step {step}: {type(e).__name__}: {e}", flush=True)
        raise

    print(f"[STEP 11] Rollout exited at step={step}", flush=True)
    env.close()
    print("[STEP 11] env.close() done", flush=True)


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
        #Bypass the IsaacLab Cleanup feature that silently kills code in hangup
        os._exit(0) #Do it! Kill the program! Do it noW!