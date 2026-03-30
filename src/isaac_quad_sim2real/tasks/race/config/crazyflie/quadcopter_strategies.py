# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value
    
    def get_rewards(self) -> torch.Tensor:
        """Reward function for racing through gates with correct traversal direction.
        This includes:
        - gate traversal detection using gate-frame plane crossing
        - waypoint / desired-position updates
        - dense progress reward toward current gate
        - center-alignment reward near the gate opening
        - uprightness reward for stable flight
        - crash detection/penalty using contact sensor data
        - per-timestep reward scaling using train_race.py
        """

        # =========================================================
        # 1) Current geometry relative to active gate
        # =========================================================
        pose_gate = self.env._pose_drone_wrt_gate                      # (N, 3), drone position in current gate frame
        x_gate = pose_gate[:, 0]                                      # along gate normal
        y_gate = pose_gate[:, 1]                                      # lateral offset in gate frame
        z_gate = pose_gate[:, 2]                                      # vertical offset in gate frame

        prev_x_gate = self.env._prev_x_drone_wrt_gate                 # previous x in gate frame

        # Current drone position in world frame
        drone_pos_w = self.env._robot.data.root_link_pos_w

        # =========================================================
        # 2) Dense progress reward toward current desired gate
        #    Use reduction in distance since last step
        # =========================================================
        distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - drone_pos_w, dim=1)
        progress = self.env._last_distance_to_goal - distance_to_goal

        # clamp progress so one weird physics step doesn't dominate learning
        progress = torch.clamp(progress, min=-1.0, max=1.0)

        # =========================================================
        # 3) Gate traversal detection
        #    Correct pass = cross gate plane from +x to -x in gate frame
        #    AND be within the gate opening
        # =========================================================
        gate_half_size = float(self.env._gate_model_cfg_data.gate_side) / 2.0
        inside_gate = (torch.abs(y_gate) < 0.90 * gate_half_size) & (torch.abs(z_gate) < 0.90 * gate_half_size)

        crossed_plane = (prev_x_gate > 0.0) & (x_gate <= 0.0)
        gate_passed = crossed_plane & inside_gate
        ids_gate_passed = torch.where(gate_passed)[0]

        # =========================================================
        # 4) Gate pass bonus and bookkeeping
        # =========================================================
        gate_pass_reward = gate_passed.float()

        if len(ids_gate_passed) > 0:
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1

            # update desired position to next gate
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        # =========================================================
        # 5) Centering reward near gate opening
        #    Encourages passing through the middle instead of clipping edges
        # =========================================================
        radial_offset = torch.sqrt(y_gate**2 + z_gate**2)
        center_reward = torch.exp(-4.0 * radial_offset**2)

        # optionally emphasize it when approaching the gate plane
        near_gate_plane = torch.abs(x_gate) < 1.5
        center_reward = center_reward * near_gate_plane.float()

        # =========================================================
        # 6) Uprightness / stability reward
        #    Helps keep the drone from learning crazy attitudes too early
        # =========================================================
        rot_mats = matrix_from_quat(self.env._robot.data.root_quat_w)   # (N, 3, 3)
        body_z_in_world = rot_mats[:, :, 2]                             # drone body z-axis expressed in world
        upright_reward = torch.clamp(body_z_in_world[:, 2], min=0.0, max=1.0)

        # =========================================================
        # 7) Crash detection using contact forces
        #    Keep the professor's "persistent contact" style accumulation
        # =========================================================
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed_now = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()

        # avoid counting startup ground contact immediately
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed_now * mask

        crash = crashed_now.float()

        # =========================================================
        # 8) Small time penalty
        #    Encourages faster completion / lower lap time
        # =========================================================
        time_penalty = torch.ones(self.num_envs, device=self.device)

        # =========================================================
        # 9) Update state for next step
        # =========================================================
        # Recompute distance-to-goal after possible gate index update
        new_distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - drone_pos_w, dim=1)
        self.env._last_distance_to_goal = new_distance_to_goal

        # store current gate-frame x for next-step crossing detection
        self.env._prev_x_drone_wrt_gate = x_gate.clone()

        # =========================================================
        # 10) Final scaled reward
        # =========================================================
        if self.cfg.is_train:
            rewards = {
                "gate_pass": gate_pass_reward * self.env.rew["gate_pass_reward_scale"],
                "progress": progress * self.env.rew["progress_reward_scale"],
                "center": center_reward * self.env.rew["center_reward_scale"],
                "upright": upright_reward * self.env.rew["upright_reward_scale"],
                "crash": crash * self.env.rew["crash_reward_scale"],
                "time_penalty": time_penalty * self.env.rew["time_penalty_reward_scale"],
            }

            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew["death_cost"],
                reward,
            )

            # logging
            for key, value in rewards.items():
                self._episode_sums[key] += value

        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward
    
    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Observation space for drone racing.

        Includes:
        - body-frame linear velocity
        - body-frame angular velocity
        - gravity direction expressed in body frame
        - current gate relative position in gate frame
        - current gate relative position in body frame
        - previous actions

        This is more task-aligned than raw world position / quaternion.
        """

        # =========================================================
        # 1) Drone state
        # =========================================================
        drone_pos_w = self.env._robot.data.root_link_pos_w                    # (N, 3)
        drone_quat_w = self.env._robot.data.root_quat_w                      # (N, 4)
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b            # (N, 3)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b                # (N, 3)

        # =========================================================
        # 2) Current gate information
        # =========================================================
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]       # (N, 3)

        # Relative position to current gate in gate frame
        gate_pos_gate_frame = self.env._pose_drone_wrt_gate                  # (N, 3)

        # Relative position to current gate in body frame
        gate_pos_b, _ = subtract_frame_transforms(
            drone_pos_w,
            drone_quat_w,
            current_gate_pos_w,
        )                                                                    # (N, 3)

        # =========================================================
        # 3) Attitude cue: gravity in body frame
        #    This is often easier for learning than raw quaternion
        # =========================================================
        rot_mats = matrix_from_quat(drone_quat_w)                            # (N, 3, 3)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        gravity_b = torch.bmm(rot_mats.transpose(1, 2), gravity_w.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # =========================================================
        # 4) Previous action
        # =========================================================
        prev_actions = self.env._previous_actions                            # (N, 4)

        # =========================================================
        # 5) Concatenate final observation
        # =========================================================
        obs = torch.cat(
            [
                drone_lin_vel_b,         # 3
                drone_ang_vel_b,         # 3
                gravity_b,               # 3
                gate_pos_gate_frame,     # 3
                gate_pos_b,              # 3
                prev_actions,            # 4
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # start from the zeroth waypoint (beginning of the race)
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -2.0 * torch.ones(n_reset, device=self.device)
        y_local = torch.zeros(n_reset, device=self.device)
        z_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0