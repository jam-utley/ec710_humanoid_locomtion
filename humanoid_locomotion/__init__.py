"""
Registers the humanoid locomotion task with gymnasium so train.py and play.py
can reach it via the standard IsaacLab `--task` flag.
"""

import os

import gymnasium as gym

from . import agents
from .humanoid_env import HumanoidLocomotionEnv
from .humanoid_env_cfg import HumanoidLocomotionEnvCfg

##
# Register Gym environments.
##
gym.register(
    id="EC710-Humanoid-SAC-v0",
    entry_point=f"{__name__}.humanoid_env:HumanoidLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HumanoidLocomotionEnvCfg,
        "skrl_cfg_entry_point": os.path.join(
            os.path.dirname(__file__), "agents", "skrl_sac_cfg.yaml"
        ),
    },
)
