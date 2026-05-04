from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationContext

from .humanoid_env_cfg import HumanoidLocomotionEnvCfg


class HumanoidLocomotionEnv(DirectRLEnv):
    """SAC-ready bipedal locomotion environment."""

    cfg: HumanoidLocomotionEnvCfg

    def __init__(self, cfg: HumanoidLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #Double check actual number of joints to ensure there isn't a mismatch
        actual_dof = self._robot.num_joints
        configured_dof = self.cfg.action_space
        if actual_dof != configured_dof:
            raise RuntimeError(
                f"Robot has {actual_dof} joints but cfg.action_space={configured_dof}.\n"
                f"  Joint names: {self._robot.data.joint_names}\n"
                f"  -> Edit humanoid_env_cfg.py and set:  action_space = {actual_dof}\n"
                f"     The observation_space will adjust automatically via __post_init__."
            )

        #Action / observation buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)

        #Velocity command (vx, vy, yaw_rate)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_resample_steps = int(self.cfg.cmd_resample_time_s / (self.cfg.sim.dt * self.cfg.decimation))
        self._step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        #Default joint pose to which residual actions are added
        self._default_joint_pos = self._robot.data.default_joint_pos.clone()

        #Identify foot bodies, i.e. any link containing "ankle"
        body_names = self._robot.data.body_names
        self._foot_ids = [i for i, n in enumerate(body_names) if "ankle" in n.lower()]
        #Force binary feet — left / right
        if len(self._foot_ids) > 2:
            #Note::: H1_MINIMAL has left_ankle_link, right_ankle_link only — fine
            self._foot_ids = self._foot_ids[:2]

        #All non-foot bodies are "undesired contact" bodies
        self._non_foot_ids = [i for i in range(len(body_names)) if i not in self._foot_ids]

        #Set gravity
        self._gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

        #Pre-compute reward dict for easier logging
        self._episode_sums: dict[str, torch.Tensor] = {
            k: torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            for k in [
                "lin_vel_xy", "yaw_rate", "alive", "lin_vel_z", "ang_vel_xy",
                "orientation", "base_height", "action_rate", "joint_torque",
                "joint_accel", "feet_air_time", "undesired_contact", "termination",
            ]
        }

    def _setup_scene(self):
        import isaaclab.sim as sim_utils

        #Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        #Contact sensor on feet
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        #Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        #Clone child envs and disable collisions on them
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        #Let there be light!
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    #Action processing — residual position control + joint PD
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Squash and scale the SAC output, then add to default pose.
        self._actions = actions.clone().clamp(-self.cfg.action_clip, self.cfg.action_clip)
        self._processed_actions = self.cfg.action_scale * self._actions + self._default_joint_pos

    def _apply_action(self) -> None:
        #The joint actuator's built-in PD controller (Kp, Kd from the asset)
        #converts these targets into torques
        self._robot.set_joint_position_target(self._processed_actions)

    
    def _get_observations(self) -> dict:
        #Base orientation -> projected gravity, with compact roll/pitch encoding
        base_quat = self._robot.data.root_quat_w
        projected_gravity = math_utils.quat_rotate_inverse(base_quat, self._gravity_vec)

        #Body-frame velocities
        base_lin_vel_b = math_utils.quat_rotate_inverse(base_quat, self._robot.data.root_lin_vel_w)
        base_ang_vel_b = math_utils.quat_rotate_inverse(base_quat, self._robot.data.root_ang_vel_w)

        #Joint state, relative to default pose, to keep magnitudes tight
        joint_pos = self._robot.data.joint_pos - self._default_joint_pos
        joint_vel = self._robot.data.joint_vel

        #Gait phase clock, gets broadcast to (num_envs, 1) so cat along dim=-1 gives (num_envs, 2) :)
        t = (self._step_counter.float() * (self.cfg.sim.dt * self.cfg.decimation)).unsqueeze(-1)
        phase = 2.0 * torch.pi * t / self.cfg.gait_period_s
        gait_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        # Foot contacts — sensor returns (num_envs, history, num_bodies_in_sensor, 3) collapse over history (max)
        #take vector norm over xyz
        net_forces = self._contact_sensor.data.net_forces_w_history    # (E, H, B, 3)
        contact_force_mag = net_forces.norm(dim=-1).max(dim=1)[0]      # (E, B)
        #Force exactly 2 contact channels (left/right foot). Pad or truncate.
        if contact_force_mag.shape[-1] >= 2:
            contact_force_mag = contact_force_mag[:, :2]
        else:
            pad = torch.zeros(contact_force_mag.shape[0], 2 - contact_force_mag.shape[-1],
                              device=self.device)
            contact_force_mag = torch.cat([contact_force_mag, pad], dim=-1)
        contacts = (contact_force_mag > 1.0).float()

        #Print to make sure state spaces match dimensions (really messed stuff up while I ran this at first)
        if not hasattr(self, "_obs_shape_logged"):
            print(f"[OBS SHAPES] joint_pos={joint_pos.shape}  joint_vel={joint_vel.shape}  "
                  f"proj_grav={projected_gravity.shape}  ang_vel={base_ang_vel_b.shape}  "
                  f"lin_vel={base_lin_vel_b.shape}  cmd={self._commands.shape}  "
                  f"actions={self._actions.shape}  gait={gait_features.shape}  "
                  f"contacts={contacts.shape}", flush=True)
            self._obs_shape_logged = True

        obs = torch.cat( #meow
            (
                joint_pos,
                joint_vel,
                projected_gravity,
                base_ang_vel_b,
                base_lin_vel_b,
                self._commands,
                self._actions,
                gait_features,
                contacts,
            ),
            dim=-1,
        )

        if not hasattr(self, "_obs_total_logged"):
            print(f"[OBS TOTAL] shape={obs.shape}  expected last dim=46", flush=True)
            self._obs_total_logged = True

        return {"policy": obs}


    #Come back and tweak this post project to make the reward go properly
    def _get_rewards(self) -> torch.Tensor:
        dt = self.cfg.sim.dt * self.cfg.decimation

        #Body-frame velocities
        base_quat = self._robot.data.root_quat_w
        v_b = math_utils.quat_rotate_inverse(base_quat, self._robot.data.root_lin_vel_w)
        w_b = math_utils.quat_rotate_inverse(base_quat, self._robot.data.root_ang_vel_w)
        proj_g = math_utils.quat_rotate_inverse(base_quat, self._gravity_vec)
        base_h = self._robot.data.root_pos_w[:, 2]

        #Tracking rewards, Gaussian over error
        lin_err = torch.sum(torch.square(self._commands[:, :2] - v_b[:, :2]), dim=-1)
        yaw_err = torch.square(self._commands[:, 2] - w_b[:, 2])
        r_lin = torch.exp(-lin_err / 0.25) * self.cfg.rew_lin_vel_xy
        r_yaw = torch.exp(-yaw_err / 0.25) * self.cfg.rew_yaw_rate

        #Penalties
        r_alive = torch.full_like(base_h, self.cfg.rew_alive)
        r_vz = torch.square(v_b[:, 2]) * self.cfg.rew_lin_vel_z
        r_wxy = torch.sum(torch.square(w_b[:, :2]), dim=-1) * self.cfg.rew_ang_vel_xy
        r_ori = torch.sum(torch.square(proj_g[:, :2]), dim=-1) * self.cfg.rew_orientation
        r_h = torch.square(base_h - self.cfg.base_height_target) * self.cfg.rew_base_height

        #Smoothness
        r_action = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=-1
        ) * self.cfg.rew_action_rate
        r_torque = torch.sum(
            torch.square(self._robot.data.applied_torque), dim=-1
        ) * self.cfg.rew_joint_torque
        r_acc = torch.sum(
            torch.square(self._robot.data.joint_acc), dim=-1
        ) * self.cfg.rew_joint_accel

        #Feet air time — encourage stepping not standing still. 
        #Air time reset to 0 after foot hits the ground
        first_contact = self._contact_sensor.compute_first_contact(dt)[:, self._foot_index_in_sensor()]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._foot_index_in_sensor()]
        cmd_active = (torch.norm(self._commands[:, :2], dim=-1) > 0.1).float()
        r_air = (
            torch.sum((last_air_time - 0.5) * first_contact.float(), dim=-1)
            * cmd_active
            * self.cfg.rew_feet_air_time
        )

        #Undesired contact, non-feet touching things
        undesired_force = torch.norm(self._contact_sensor.data.net_forces_w, dim=-1)
        #Net force per body in sensor. here this is only the feet, so no undesired contact term unless scope of 'sensor' is broadened'.
        #Kept here for completeness--safe to leave at zero.
        r_contact = torch.zeros_like(base_h) * self.cfg.rew_undesired_contact

        #Termination penalty
        terminated = self._compute_termination()
        r_term = terminated.float() * self.cfg.rew_termination

        rewards = (
            r_lin + r_yaw + r_alive + r_vz + r_wxy + r_ori + r_h
            + r_action + r_torque + r_acc + r_air + r_contact + r_term
        )

        #Log per-component sums
        for k, v in {
            "lin_vel_xy": r_lin, "yaw_rate": r_yaw, "alive": r_alive,
            "lin_vel_z": r_vz, "ang_vel_xy": r_wxy, "orientation": r_ori,
            "base_height": r_h, "action_rate": r_action, "joint_torque": r_torque,
            "joint_accel": r_acc, "feet_air_time": r_air,
            "undesired_contact": r_contact, "termination": r_term,
        }.items():
            self._episode_sums[k] += v

        #Save for action-rate term next step
        self._previous_actions = self._actions.clone()

        return rewards

    def _compute_termination(self) -> torch.Tensor:
        base_quat = self._robot.data.root_quat_w
        proj_g = math_utils.quat_rotate_inverse(base_quat, self._gravity_vec)
        #If gravity z component in body frame goes positive-ish, the torso flipped. This is bad.
        fell_orientation = proj_g[:, 2] > -0.5
        too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.min_base_height
        return fell_orientation | too_low

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._compute_termination()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        #Re-pose the robot at default joint state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        #Reset action history & step counter
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._step_counter[env_ids] = 0

        #Sample new velocity command
        self._resample_commands(env_ids)

        #Logging
        extras = {}
        for k in self._episode_sums:
            ep = torch.mean(self._episode_sums[k][env_ids]) / self.max_episode_length_s
            extras[f"Episode_Reward/{k}"] = ep
            self._episode_sums[k][env_ids] = 0.0
        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(extras)

    def _post_step(self):
        #Resample command periodically mid-episode
        self._step_counter += 1
        resample_mask = (self._step_counter % self._cmd_resample_steps == 0)
        if resample_mask.any():
            ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
            self._resample_commands(ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        n = len(env_ids)
        rng = lambda lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi)
        self._commands[env_ids, 0] = rng(*self.cfg.cmd_vel_x_range)
        self._commands[env_ids, 1] = rng(*self.cfg.cmd_vel_y_range)
        self._commands[env_ids, 2] = rng(*self.cfg.cmd_yaw_rate_range)

    def _foot_index_in_sensor(self):
        #The contact sensor's first dim already enumerates only the matched foot bodies, so return all
        n_feet = self._contact_sensor.data.net_forces_w.shape[1]
        return torch.arange(n_feet, device=self.device)