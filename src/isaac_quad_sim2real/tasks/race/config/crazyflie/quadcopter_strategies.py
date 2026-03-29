# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat, quat_apply

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

        ### Custom parameters ###
        ### TODO: Tune these parameters
        self.gate_radius = 0.5

        self.spawn_offset_x = [-3.0, -0.5] # behind the gate
        self.spawn_offset_y = [-1.0,  1.0]
        self.spawn_offset_z = [-0.1,  0.1]

        self.spawn_yaw_noise = [-0.2, 0.2]
        self.spawn_vel_noise = [-0.2, 0.2]

        # Storing last direction to goal for progress reward
        # This prevents the velocity projection (progress_vel) from flipping to a
        # large negative value at the exact frame the drone crosses a gate.
        # Initialize pointing toward waypoint 0 along +X to avoid zero-vector on step 1.
        self._last_dir_to_goal = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_dir_to_goal[:, 0] = 1.0

        # DEBUG counter
        self._debug_step = 0

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_vel_w = self.env._robot.data.root_com_lin_vel_w

        target_pos_w = self.env._waypoints[self.env._idx_wp, :3].clone()
        vec_to_goal = target_pos_w - drone_pos_w
        dist_to_goal_3d = torch.norm(vec_to_goal, dim=1)

        # By projecting current velocity onto the previous frame's target direction, 
        # we eliminate the "reward pollution" (sudden negative dot product) that occurs 
        # when the target waypoint switches immediately after passing a gate.
        progress_vel = torch.sum(drone_vel_w * self._last_dir_to_goal, dim=1)
        progress_vel = torch.clamp(progress_vel, min=0.0, max=15.0)

        # ===== DEBUG =====
        self._debug_step += 1
        if self._debug_step % 2000 == 1:  # print every 2000 steps
            vel_mag = torch.norm(drone_vel_w, dim=1)
            episode_steps = self.env.episode_length_buf

            # stats across all envs
            print(f"\n[DEBUG step={self._debug_step}]")
            print(f"  progress_vel     : mean={progress_vel.mean():.3f}  std={progress_vel.std():.3f}  min={progress_vel.min():.3f}  max={progress_vel.max():.3f}")
            print(f"  drone speed (m/s): mean={vel_mag.mean():.3f}  max={vel_mag.max():.3f}")
            print(f"  _last_dir_to_goal: mean={self._last_dir_to_goal.mean(0).cpu().numpy().round(3)}")
            print(f"  episode_length   : mean={episode_steps.float().mean():.1f}  max={episode_steps.max()}")

            # stats only for envs at episode start (step 1 = first step after reset)
            start_mask = (episode_steps == 1)
            if start_mask.any():
                print(f"  [at episode start, n={start_mask.sum()}]")
                print(f"    progress_vel : mean={progress_vel[start_mask].mean():.3f}")
                print(f"    drone_vel_w  : mean={drone_vel_w[start_mask].mean(0).cpu().numpy().round(3)}")
                print(f"    dir_to_goal  : mean={self._last_dir_to_goal[start_mask].mean(0).cpu().numpy().round(3)}")

            # fraction of envs with negative progress_vel
            neg_frac = (progress_vel < 0).float().mean()
            print(f"  fraction with negative progress_vel: {neg_frac:.2%}")
        # ===== END DEBUG =====

        prev_dist_to_goal = self.env._last_distance_to_goal.clone()
        progress_dist = prev_dist_to_goal - dist_to_goal_3d
        progress_dist = torch.clamp_(progress_dist, min=0.0, max=1.0)

        self.env._last_distance_to_goal[:] = dist_to_goal_3d.detach()

        # drone coordinates in the gate's local frame
        current_gate_pos_w = self.env._waypoints[self.env._idx_wp, :3]
        current_gate_euler = self.env._waypoints[self.env._idx_wp, 3:6]
        current_gate_quat_w = quat_from_euler_xyz(
            current_gate_euler[:, 0], 
            current_gate_euler[:, 1], 
            current_gate_euler[:, 2]
        )

        drone_pos_gate_frame, _ = subtract_frame_transforms(
            current_gate_pos_w,
            current_gate_quat_w,
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w
        )

        current_x = drone_pos_gate_frame[:, 0]
        current_y = drone_pos_gate_frame[:, 1]
        current_z = drone_pos_gate_frame[:, 2]
        
        prev_x = self.env._prev_x_drone_wrt_gate
        
        crossed_plane = (prev_x > 0.0) & (current_x <= 0.0)
        within_bounds = (torch.abs(current_y) < self.gate_radius) & (torch.abs(current_z) < self.gate_radius)
        gate_passed = crossed_plane & within_bounds
        missed_gate = crossed_plane & (~within_bounds)

        self.env._prev_x_drone_wrt_gate = current_x.clone()

        ids_gate_passed = torch.where(gate_passed)[0]

        if len(ids_gate_passed) > 0:
            # Increment waypoint index (with wrap-around for laps)
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            
            # Synchronize desired_pos_w immediately
            new_target_idx = self.env._idx_wp[ids_gate_passed]
            new_target_pos_w = self.env._waypoints[new_target_idx, :3]
            self.env._desired_pos_w[ids_gate_passed] = new_target_pos_w
            self.env._last_distance_to_goal[ids_gate_passed] = torch.norm(
                new_target_pos_w - self.env._robot.data.root_link_pos_w[ids_gate_passed, :3],
                dim=1
            )

            new_gate_euler = self.env._waypoints[new_target_idx, 3:6]
            new_gate_quat_w = quat_from_euler_xyz(new_gate_euler[:, 0], new_gate_euler[:, 1], new_gate_euler[:, 2])
            new_drone_pos_gate_frame, _ = subtract_frame_transforms(
                new_target_pos_w, new_gate_quat_w,
                self.env._robot.data.root_link_pos_w[ids_gate_passed],
                self.env._robot.data.root_quat_w[ids_gate_passed]
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_drone_pos_gate_frame[:, 0]
            
            # Increment total gates passed counter
            self.env._n_gates_passed[ids_gate_passed] += 1

        final_target_pos_w = self.env._waypoints[self.env._idx_wp, :3]
        final_vec_to_goal = final_target_pos_w - self.env._robot.data.root_link_pos_w
        final_dist = torch.norm(final_vec_to_goal, dim=1, keepdim=True)

        self._last_dir_to_goal[:] = (final_vec_to_goal / (final_dist + 1e-6)).detach()

        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        is_contact = (torch.norm(contact_forces, dim=-1) > 0.1).any(dim=-1)
        crashed = is_contact | missed_gate

        # both contact and missed gate accumulate toward termination (grace period of 100 steps)
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + (crashed * mask).int()

        gate_passed_signal = gate_passed.float()

        # Stability
        ## Smoothness penalty (penalize action changes, not magnitude)
        action_l2 = torch.norm(self.env._actions - self.env._previous_actions, dim=1)

        ## Angular velocity penalty
        ang_vel_penalty = torch.norm(self.env._robot.data.root_ang_vel_b, dim=1)

        ## Tilt penalty
        local_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        world_up = quat_apply(self.env._robot.data.root_quat_w, local_up)
        safe_threshold = 0.5 # 60 degree
        tilt_penalty = torch.clamp(safe_threshold - world_up[:, 2], min=0.0)

        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress_vel": progress_vel * self.env.rew['progress_vel_reward_scale'],
                "progress_dist": progress_dist * self.env.rew['progress_dist_reward_scale'],
                "gate_pass": gate_passed_signal * self.env.rew['gate_pass_reward_scale'],
                "crash": crashed.float() * self.env.rew['crash_reward_scale'],
                "action_smoothness": action_l2 * self.env.rew['action_smoothness_reward_scale'],
                "ang_vel_penalty": ang_vel_penalty * self.env.rew['ang_vel_penalty_reward_scale'],
                "tilt_penalty": tilt_penalty * self.env.rew['tilt_penalty_reward_scale'],
                "survival_bonus": torch.ones_like(progress_vel) * self.env.rew['survival_bonus'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward += self.env.reset_terminated.float() * self.env.rew['death_cost']

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        #### Basic drone states, modify for your needs)
        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        drone_quat_inv = self.env._robot.data.root_quat_w.clone()
        drone_quat_inv[:, 1:] = -drone_quat_inv[:, 1:]
        projected_gravity_b = quat_apply(drone_quat_inv, gravity_w)

        ##### Some example observations you may want to explore using
        # Angular velocities (referred to as body rates)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]  # World position of current gate
        current_gate_euler = self.env._waypoints[current_gate_idx, 3:6]

        current_gate_quat_w = quat_from_euler_xyz(
            current_gate_euler[:, 0], 
            current_gate_euler[:, 1], 
            current_gate_euler[:, 2]
        )

        # Relative position to current gate in body frame
        gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            current_gate_pos_w,
            current_gate_quat_w
        )

        local_forward = torch.tensor([-1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        gate_forward_w = quat_apply(current_gate_quat_w, local_forward)
        gate_forward_b = quat_apply(drone_quat_inv, gate_forward_w)

        # Avoid short-sightedness
        num_waypoints = self.env._waypoints.shape[0]
        next_gate_idx = (current_gate_idx + 1) % num_waypoints
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        next_gate_euler = self.env._waypoints[next_gate_idx, 3:6]
        next_gate_quat_w = quat_from_euler_xyz(
            next_gate_euler[:, 0],
            next_gate_euler[:, 1],
            next_gate_euler[:, 2]
        )
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w,
            next_gate_pos_w,
            next_gate_quat_w
        )

        next_gate_forward_w = quat_apply(next_gate_quat_w, local_forward)
        next_gate_forward_b = quat_apply(drone_quat_inv, next_gate_forward_w)

        # Previous actions
        prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed
        # gates_passed = self.env._n_gates_passed.unsqueeze(1).float()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                projected_gravity_b,           # gravity in the body frame (3 dims)
                drone_lin_vel_b,               # velocity in the body frame (3 dims)
                drone_ang_vel_b,               # angular velocity in the body frame (3 dims)
                gate_pos_b / 10.0,             # relative position to current gate in body frame, normalized (3 dims)
                gate_forward_b,                # forward vector of the gate in body frame (3 dims)
                next_gate_pos_b / 10.0,        # relative position to next gate in body frame, normalized (3 dims)
                next_gate_forward_b,           # forward vector of the next gate in body frame (3 dims)
                prev_actions,                  # previous actions (4 dims)
                # drone_pos_gate_frame
            ],
            # TODO ----- END -----
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

        # always spawn behind gate 0 for consistent curriculum learning
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        z_wp  = self.env._waypoints[waypoint_indices][:, 2]
        theta = self.env._waypoints[waypoint_indices][:, -1]

        x_local = torch.empty(n_reset, device=self.device).uniform_(self.spawn_offset_x[0], self.spawn_offset_x[1])
        y_local = torch.empty(n_reset, device=self.device).uniform_(self.spawn_offset_y[0], self.spawn_offset_y[1])
        z_local = torch.empty(n_reset, device=self.device).uniform_(self.spawn_offset_z[0], self.spawn_offset_z[1])

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = torch.clamp(z_local + z_wp, min=0.05) # make sure drone is above ground

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(self.spawn_yaw_noise[0], self.spawn_yaw_noise[1])
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise
        )
        default_root_state[:, 3:7] = quat
        default_root_state[:, 7:10] = torch.empty((n_reset, 3), device=self.device).uniform_(self.spawn_vel_noise[0], self.spawn_vel_noise[1])

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

            default_root_state = self.env._robot.data.default_root_state[env_ids].clone()
            default_root_state[:, 0] = x0.expand(len(env_ids))
            default_root_state[:, 1] = y0.expand(len(env_ids))
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device),
                yaw0.expand(len(env_ids))
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = torch.full((len(env_ids),), self.env._initial_wp, device=self.device, dtype=self.env._idx_wp.dtype)

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :3] - default_root_state[:, :3], dim=1
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
            # self.env._robot.data.root_link_state_w[env_ids, :3] # TODO: is it stale?
            default_root_state[:, :3],
            default_root_state[:, 3:7]
        )

        # self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0]

        self.env._crashed[env_ids] = 0

        vec_to_initial_goal = self.env._waypoints[self.env._idx_wp[env_ids], :3] - default_root_state[:, :3]
        dist_to_initial_goal = torch.norm(vec_to_initial_goal, dim=1, keepdim=True)
        self._last_dir_to_goal[env_ids] = vec_to_initial_goal / (dist_to_initial_goal + 1e-6)