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

        self.randomize = True

        # Domain randomization ranges
        self._twr_min = self.cfg.thrust_to_weight * 0.95
        self._twr_max = self.cfg.thrust_to_weight * 1.05

        self._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
        self._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
        self._k_aero_z_min = self.cfg.k_aero_z * 0.5
        self._k_aero_z_max = self.cfg.k_aero_z * 2.0

        self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
        self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
        self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
        self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
        self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.7
        self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.3

        self._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
        self._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
        self._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
        self._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
        self._kd_omega_y_min = self.cfg.kd_omega_y * 0.7
        self._kd_omega_y_max = self.cfg.kd_omega_y * 1.3

        # Set initial parameters for all envs
        self._randomize_params(torch.arange(self.num_envs, device=self.device))

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
        drone_vel_w = self.env._robot.data.root_com_lin_vel_w

        # =========================================================
        # 2) Dense progress reward toward current desired gate
        #    Use reduction in distance since last step
        # =========================================================

        # Dynamic goal for Gate 3 (Encourage Power Loop)
        dynamic_goal_w = self.env._desired_pos_w.clone()
        is_targeting_gate_3 = (self.env._idx_wp == 3).float().unsqueeze(1)
        powerloop_ghost_vertical_offset = 1.5
        powerloop_ghost_horizontal_offset = 0.5
        dynamic_goal_w[:, 2] += powerloop_ghost_vertical_offset * is_targeting_gate_3[:, 0]
        dynamic_goal_w[:, 1] += powerloop_ghost_horizontal_offset * is_targeting_gate_3[:, 0]

        # Progress in distance
        vec_to_goal_w = dynamic_goal_w - drone_pos_w
        distance_to_goal_3d = torch.linalg.norm(vec_to_goal_w, dim=1)
        distance_to_goal_2d = torch.linalg.norm(vec_to_goal_w[:, :2], dim=1)
        linear_progress_dist = (self.env._last_distance_to_goal - distance_to_goal_3d) * 0.1

        dist_clamped = torch.clamp(distance_to_goal_3d, min=0.1)
        prev_dist_clamped = torch.clamp(self.env._last_distance_to_goal, min=0.1)
        potential_progress_dist = (1./dist_clamped - 1./prev_dist_clamped) * 0.9

        progress_dist = linear_progress_dist + potential_progress_dist
        progress_dist = torch.clamp(progress_dist, min=-1.0, max=1.0)

        # Progress in velocity
        dir_to_goal_w = vec_to_goal_w / (distance_to_goal_3d.unsqueeze(1) + 1e-6)
        progress_vel = torch.sum(drone_vel_w * dir_to_goal_w, dim=1)
        progress_vel = torch.clamp(progress_vel, min=-2.0, max=8.0)

        # Alignment with gate normal (TODO: Do we need this?)
        current_gate_euler = self.env._waypoints[self.env._idx_wp, 3:6]
        current_gate_quat_w = quat_from_euler_xyz(current_gate_euler[:, 0], current_gate_euler[:, 1], current_gate_euler[:, 2])
        local_forward = torch.tensor([-1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        gate_forward_w = quat_apply(current_gate_quat_w, local_forward)
        vel_along_gate_normal = torch.sum(drone_vel_w * gate_forward_w, dim=1)
        near_gate_mask = (distance_to_goal_2d < 1.0).float()
        vel_along_gate_normal = torch.clamp(vel_along_gate_normal, min=-2.0, max=4.0) * near_gate_mask

        speed = torch.linalg.norm(drone_vel_w, dim=1)
        speed_reward = torch.clamp(speed / 8.0, min=0.0, max=1.0)

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

        reverse_cross_plane = (prev_x_gate < 0.0) & (x_gate >= 0.0)
        reverse_cross_gate = reverse_cross_plane & inside_gate
        reverse_gate_cross_penalty = reverse_cross_gate.float()

        # Continuous penalty for approaching the gate from the wrong side
        on_wrong_side = (x_gate <= 0.0) & (~gate_passed) & (inside_gate)
        wrong_side_penalty = on_wrong_side.float() * torch.exp(-0.5 * distance_to_goal_2d)

        # =========================================================
        # 4) Gate pass bonus and bookkeeping
        # =========================================================
        gate_pass_reward = gate_passed.float()

        alignment_reward = torch.zeros(self.num_envs, device=self.device)
        if len(ids_gate_passed) > 0:
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1

            # update desired position to next gate
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

            # Alignment to next gate
            idx_next = self.env._idx_wp[ids_gate_passed]
            pos_next = self.env._waypoints[idx_next, :3].clone()

            is_next_gate_3 = (idx_next == 3).float().unsqueeze(1)
            pos_next[:, 1] += powerloop_ghost_horizontal_offset * is_next_gate_3[:, 0]
            pos_next[:, 2] += powerloop_ghost_vertical_offset * is_next_gate_3[:, 0]
            
            dir_to_next = pos_next - drone_pos_w[ids_gate_passed]
            dir_to_next = dir_to_next / (torch.linalg.norm(dir_to_next, dim=1, keepdim=True) + 1e-6)
            
            vel_dir = drone_vel_w[ids_gate_passed]
            vel_dir = vel_dir / (torch.linalg.norm(vel_dir, dim=1, keepdim=True) + 1e-6)
            
            subset_alignment = torch.sum(vel_dir * dir_to_next, dim=1)
            subset_alignment = torch.clamp(subset_alignment, min=0.0, max=1.0)
            alignment_reward[ids_gate_passed] = subset_alignment * 8.0

        # =========================================================
        # 5) Centering reward near gate opening
        #    Encourages passing through the middle instead of clipping edges
        # =========================================================
        radial_offset = torch.sqrt(y_gate**2 + z_gate**2)
        # center_reward = torch.exp(-1.0 * radial_offset**2)
        tolerance = 0.75 * gate_half_size
        center_reward = torch.clamp(1.0 - torch.relu(radial_offset - tolerance) / (gate_half_size - tolerance + 1e-6), min=0.0, max=1.0)

        # optionally emphasize it when approaching the gate plane
        near_gate_plane = torch.abs(x_gate) < 1.5
        center_reward = center_reward * near_gate_plane.float()

        # =========================================================
        # 6) Uprightness / stability reward
        #    Helps keep the drone from learning crazy attitudes too early
        # =========================================================
        rot_mats = matrix_from_quat(self.env._robot.data.root_quat_w)   # (N, 3, 3)
        body_z_in_world = rot_mats[:, :, 2]                             # drone body z-axis expressed in world
        upright_reward = torch.clamp((body_z_in_world[:, 2] - 0.5) * 2, min=0.0, max=1.0)

        not_powerloop_mask = (self.env._idx_wp != 3).float()
        upright_reward = upright_reward * not_powerloop_mask

        # Encourage aggressive maneuvers (e.g. Powerloop) at gate 3
        is_targeting_gate_3_flat = (self.env._idx_wp == 3).float()
        upside_down_factor = 0.5 * (1.0 - body_z_in_world[:, 2])
        high_enough = (drone_pos_w[:, 2] > 1.0).float()
        inversion_bonus = upside_down_factor * is_targeting_gate_3_flat * high_enough

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
        new_dynamic_goal_w = self.env._desired_pos_w.clone()
        new_is_targeting_gate_3 = (self.env._idx_wp == 3).float().unsqueeze(1)
        new_dynamic_goal_w[:, 2] += powerloop_ghost_vertical_offset * new_is_targeting_gate_3[:, 0]
        new_dynamic_goal_w[:, 1] += powerloop_ghost_horizontal_offset * new_is_targeting_gate_3[:, 0]

        new_distance_to_goal = torch.linalg.norm((new_dynamic_goal_w - drone_pos_w), dim=1)
        self.env._last_distance_to_goal = new_distance_to_goal

        # store current gate-frame x for next-step crossing detection
        self.env._prev_x_drone_wrt_gate = x_gate.clone()

        # =========================================================
        # 10) Action Smoothness + soft stability penalties
        # =========================================================
        action_diff = self.env._actions - self.env._previous_actions
        action_smoothness = torch.norm(action_diff, dim=1)

        # soft penalty: high angular rates (discourage aggressive rotations)
        drone_ang_vel_b = getattr(self.env._robot.data, "root_ang_vel_b", None)
        if drone_ang_vel_b is not None:
            ang_rate = torch.linalg.norm(drone_ang_vel_b, dim=1)
        else:
            ang_rate = torch.zeros(self.num_envs, device=self.device)

        ang_rate_threshold = 6.0
        ang_rate_penalty = torch.clamp((ang_rate - ang_rate_threshold) / (ang_rate_threshold + 1e-6), min=0.0, max=1.0)

        # soft penalty: large tilt (1 - body_z_in_world.z gives tilt magnitude)
        tilt = 1.0 - body_z_in_world[:, 2]
        tilt_penalty = torch.clamp(tilt - 0.4, min=0.0, max=1.0)

        # =========================================================
        # 11) Final scaled reward
        # =========================================================
        if self.cfg.is_train:
            # use safe lookups with sensible defaults so missing keys don't crash
            rew = getattr(self.env, "rew", {})
            rewards = {
                "gate_pass": gate_pass_reward * rew.get("gate_pass_reward_scale", 25.0),
                "reverse_gate_cross_penalty": reverse_gate_cross_penalty * rew.get("reverse_gate_cross_penalty_reward_scale", -30.0),
                "wrong_side_penalty": wrong_side_penalty * rew.get("wrong_side_penalty_reward_scale", -2.0),
                "progress_dist": progress_dist * rew.get("progress_dist_reward_scale", 0.75),
                "progress_vel": progress_vel * rew.get("progress_vel_reward_scale", 0.5),
                "speed": speed_reward * rew.get("speed_reward_scale", 0.3),
                "vel_along_gate_normal": vel_along_gate_normal * rew.get("vel_along_gate_normal_reward_scale", 0.25),
                "exit_alignment": alignment_reward * rew.get("exit_alignment_reward_scale", 0.7),
                "center": center_reward * rew.get("center_reward_scale", 0.0),
                "upright": upright_reward * rew.get("upright_reward_scale", 0.01),
                "inversion_bonus": inversion_bonus * rew.get("inversion_bonus_reward_scale", 0.5),
                "crash": crash * rew.get("crash_reward_scale", -1.5),
                "time_penalty": time_penalty * rew.get("time_penalty_reward_scale", -0.3),
                "action_smoothness": action_smoothness * rew.get("action_smoothness_reward_scale", -0.002),
                # new soft penalties (default to 0.0 so they are opt-in from train script)
                "ang_rate_penalty": ang_rate_penalty * rew.get("ang_rate_penalty_reward_scale", 0.0),
                "tilt_penalty": tilt_penalty * rew.get("tilt_penalty_reward_scale", 0.0),
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
        # 2.1) Current gate information
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

        current_gate_euler = self.env._waypoints[current_gate_idx, 3:6]
        current_gate_quat_w = quat_from_euler_xyz(current_gate_euler[:, 0], current_gate_euler[:, 1], current_gate_euler[:, 2])
        drone_quat_inv = drone_quat_w.clone()
        drone_quat_inv[:, 1:] = -drone_quat_inv[:, 1:]

        local_forward = torch.tensor([-1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        gate_forward_w = quat_apply(current_gate_quat_w, local_forward)
        gate_forward_b = quat_apply(drone_quat_inv, gate_forward_w)          # (N, 3)

        # =========================================================
        # 2.2) Next gate information
        # =========================================================
        next_gate_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]             # (N, 3)

        # Relative position to next gate in body frame
        next_gate_pos_b, _ = subtract_frame_transforms(
            drone_pos_w,
            drone_quat_w,
            next_gate_pos_w,
        )                                                                    # (N, 3)

        next_gate_euler = self.env._waypoints[next_gate_idx, 3:6]
        next_gate_quat_w = quat_from_euler_xyz(next_gate_euler[:, 0], next_gate_euler[:, 1], next_gate_euler[:, 2])
        next_gate_forward_w = quat_apply(next_gate_quat_w, local_forward)
        next_gate_forward_b = quat_apply(drone_quat_inv, next_gate_forward_w)  # (N, 3)

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
                gate_forward_b,          # 3
                next_gate_pos_b,         # 3
                next_gate_forward_b,     # 3
                prev_actions,            # 4
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to randomized racing starts."""

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # =========================================================
        # 1) Logging for training mode
        # =========================================================
        if self.cfg.is_train and hasattr(self, "_episode_sums"):
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

        # =========================================================
        # 2.1) Base robot reset
        # =========================================================
        self.env._robot.reset(env_ids)

        # =========================================================
        # 2.2) Domain Randomization
        # =========================================================
        self._randomize_params(env_ids)

        # =========================================================
        # 3) Initialize model paths if needed
        # =========================================================
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [
                f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)
            ]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = env_ids.shape[0]
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # =========================================================
        # 4) Reset action / controller buffers
        # =========================================================
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()

        # =========================================================
        # 5) Training reset: randomize start around random gates
        # =========================================================
        if self.cfg.is_train:
            num_waypoints = self.env._waypoints.shape[0]

            # sample a random active gate for each reset env
            waypoint_indices = torch.randint(
                low=0, high=num_waypoints, size=(n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )

            # gate world pose
            gate_x = self.env._waypoints[waypoint_indices, 0]
            gate_y = self.env._waypoints[waypoint_indices, 1]
            gate_z = self.env._waypoints[waypoint_indices, 2]
            gate_yaw = self.env._waypoints[waypoint_indices, -1]

            # local spawn around gate:
            # behind gate in gate frame (+x side before crossing)
            x_local = torch.empty(n_reset, device=self.device).uniform_(1.5, 3.0)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-0.6, 0.6)
            z_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)

            cos_yaw = torch.cos(gate_yaw)
            sin_yaw = torch.sin(gate_yaw)

            # local -> world
            x_world_offset = cos_yaw * x_local - sin_yaw * y_local
            y_world_offset = sin_yaw * x_local + cos_yaw * y_local

            initial_x = gate_x + x_world_offset
            initial_y = gate_y + y_world_offset
            initial_z = gate_z + z_local

            # keep altitude in a safe range
            initial_z = torch.clamp(initial_z, min=0.25, max=2.5)

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z

            # point roughly toward gate center with some yaw noise
            desired_yaw = torch.atan2(gate_y - initial_y, gate_x - initial_x)
            yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.25, 0.25)
            initial_yaw = desired_yaw + yaw_noise

            # small roll/pitch noise for robustness
            roll_noise = torch.empty(n_reset, device=self.device).uniform_(-0.08, 0.08)
            pitch_noise = torch.empty(n_reset, device=self.device).uniform_(-0.08, 0.08)

            quat = quat_from_euler_xyz(roll_noise, pitch_noise, initial_yaw)
            default_root_state[:, 3:7] = quat

            # small randomized linear/angular velocity at reset
            default_root_state[:, 7:10] = torch.empty((n_reset, 3), device=self.device).uniform_(-0.2, 0.2)
            default_root_state[:, 10:13] = torch.empty((n_reset, 3), device=self.device).uniform_(-0.1, 0.1)

        # =========================================================
        # 6) Play/eval reset: keep deterministic-ish behavior
        # =========================================================
        else:
            x_local = torch.empty(1, device=self.device).uniform_(1.5, 3.0)
            y_local = torch.empty(1, device=self.device).uniform_(-0.8, 0.8)
            z_local = torch.empty(1, device=self.device).uniform_(-0.1, 0.1)

            gate_x = self.env._waypoints[self.env._initial_wp, 0]
            gate_y = self.env._waypoints[self.env._initial_wp, 1]
            gate_z = self.env._waypoints[self.env._initial_wp, 2]
            gate_yaw = self.env._waypoints[self.env._initial_wp, -1]

            cos_yaw = torch.cos(gate_yaw)
            sin_yaw = torch.sin(gate_yaw)

            x_world_offset = cos_yaw * x_local - sin_yaw * y_local
            y_world_offset = sin_yaw * x_local + cos_yaw * y_local

            x0 = gate_x + x_world_offset
            y0 = gate_y + y_world_offset
            z0 = torch.clamp(gate_z + z_local, min=0.25, max=2.5)

            yaw0 = torch.atan2(gate_y - y0, gate_x - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0).clone()
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0,
            )
            default_root_state[:, 3:7] = quat
            default_root_state[:, 7:13] = 0.0

            waypoint_indices = self.env._initial_wp

        # =========================================================
        # 7) Update env bookkeeping
        # =========================================================
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        # write state first
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._yaw_n_laps[env_ids] = 0
        self.env._n_gates_passed[env_ids] = 0
        self.env._crashed[env_ids] = 0

        # recompute relative pose to gate after reset
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3],
        )

        # initialize gate-crossing memory and last distance
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            (self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids]), dim=1
        )

    def _randomize_params(self, env_ids: torch.Tensor):
        n = len(env_ids)

        if self.randomize:
            self.env._thrust_to_weight[env_ids] = torch.empty(n, device=self.device).uniform_(
                self._twr_min, self._twr_max
            )

            self.env._K_aero[env_ids, :2] = torch.empty(n, 1, device=self.device).uniform_(
                self._k_aero_xy_min, self._k_aero_xy_max
            ).expand(-1, 2)
            self.env._K_aero[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
                self._k_aero_z_min, self._k_aero_z_max
            )

            self.env._kp_omega[env_ids, :2] = torch.empty(n, 1, device=self.device).uniform_(
                self._kp_omega_rp_min, self._kp_omega_rp_max
            ).expand(-1, 2)
            self.env._ki_omega[env_ids, :2] = torch.empty(n, 1, device=self.device).uniform_(
                self._ki_omega_rp_min, self._ki_omega_rp_max
            ).expand(-1, 2)
            self.env._kd_omega[env_ids, :2] = torch.empty(n, 1, device=self.device).uniform_(
                self._kd_omega_rp_min, self._kd_omega_rp_max
            ).expand(-1, 2)

            self.env._kp_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
                self._kp_omega_y_min, self._kp_omega_y_max
            )
            self.env._ki_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
                self._ki_omega_y_min, self._ki_omega_y_max
            )
            self.env._kd_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
                self._kd_omega_y_min, self._kd_omega_y_max
            )

            self.env._tau_m[env_ids] = self.env._tau_m_value

        else:
            self.env._thrust_to_weight[env_ids] = self.env._twr_value
            self.env._K_aero[env_ids, :2] = self.env._k_aero_xy_value
            self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value
            self.env._kp_omega[env_ids, :2] = self.env._kp_omega_rp_value
            self.env._ki_omega[env_ids, :2] = self.env._ki_omega_rp_value
            self.env._kd_omega[env_ids, :2] = self.env._kd_omega_rp_value
            self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value
            self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value
            self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value
            self.env._tau_m[env_ids] = self.env._tau_m_value