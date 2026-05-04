from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

#Robot import
from isaaclab_assets import H1_MINIMAL_CFG


@configclass
class HumanoidLocomotionEnvCfg(DirectRLEnvCfg):

    decimation = 4                  #200 Hz physics, so a 50 Hz policy
    episode_length_s = 20.0         #max ep length in seconds

    action_space = 19               # 19 leg joints on Unitree H1, uses legs-only
    observation_space = 46          #filled in __post_init__
    state_space = 0                 #not used, no asymmetric critic input used here

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,             # 200 Hz physics
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
        ),
    )

    #Terrain: Flat
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", #Change this to make it rough or otehr terrains. Slick may be an option?
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,   #Too many. Adjust in launch
        env_spacing=2.5,
        replicate_physics=True,
    )

    #Robot model
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    #Contact sensor = feet
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_ankle_link",
        history_length=3,
        track_air_time=True,
    )

    action_scale: float = 0.5       #rad scale on the residual
    action_clip: float = 100.0      #safety clip on raw action

    cmd_vel_x_range: tuple = (-1.0, 1.5)
    cmd_vel_y_range: tuple = (-0.5, 0.5)
    cmd_yaw_rate_range: tuple = (-1.0, 1.0)
    cmd_resample_time_s: float = 10.0    #resample command mid-episode

    gait_period_s: float = 0.7

    # Reward weights. These shape the SAC reward r(i, u) used in the
    #Bellman target  y = r + gamma * [min(Q1, Q2) - beta * log mu(u|i)].
    rew_lin_vel_xy: float = 1.5         # tracks vx,vy command
    rew_yaw_rate: float = 0.75          # tracks yaw-rate command
    rew_alive: float = 0.15             # per-step alive bonus
    rew_lin_vel_z: float = -2.0         # penalize vertical motion
    rew_ang_vel_xy: float = -0.05       # penalize roll/pitch rate
    rew_orientation: float = -5.0       # penalize non-upright torso
    rew_base_height: float = -10.0      # encourage standing tall
    rew_action_rate: float = -0.01      # smooth actions
    rew_joint_torque: float = -2.0e-5   # energy penalty
    rew_joint_accel: float = -2.5e-7    # smoothness penalty
    rew_feet_air_time: float = 0.5      # encourage stepping
    rew_undesired_contact: float = -1.0 # any non-foot contact
    rew_termination: float = -200.0     # falling

    base_height_target: float = 0.95    # nominal standing height (m)

    #Termination of simulation threshold
    fall_pitch_roll_threshold: float = 1.0   # rad (~57 deg)
    min_base_height: float = 0.4             # m

    def __post_init__(self):
        n_joints = self.action_space
        obs_dim = (
            n_joints       #joint positions
            + n_joints     #joint velocities
            + 3            #projected gravity (orientation proxy)
            + 3            #base angular velocity
            + 3            #base linear velocity
            + 3            #velocity command
            + n_joints     #last action
            + 2            #gait phase sin/cos
            + 2            #foot contacts (binary)
        )
        self.observation_space = obs_dim
        print(f"[CFG] action_space={n_joints}  observation_space={obs_dim}", flush=True)