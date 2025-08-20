# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
inhand_manipulation_env.py - 在手操作环境实现文件

该文件定义了InHandManipulationEnv类，用于实现手内物体操作的强化学习环境。
支持Allegro Hand和Shadow Hand两种手部模型，通过配置文件进行切换。
环境主要处理手部关节控制、物体位置跟踪、奖励计算等核心功能。
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

if TYPE_CHECKING:
    from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


class InHandManipulationEnv(DirectRLEnv):
    """在手操作环境类，继承自DirectRLEnv
    
    该类实现了手内物体操作的强化学习环境，支持Allegro Hand和Shadow Hand两种手部模型。
    环境主要处理手部关节控制、物体位置跟踪、奖励计算等核心功能。
    """
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化在手操作环境
        
        Args:
            cfg: 环境配置对象，可以是AllegroHandEnvCfg或ShadowHandEnvCfg类型
            render_mode: 渲染模式，可选参数
            **kwargs: 其他关键字参数
        """
        super().__init__(cfg, render_mode, **kwargs)

        # 获取手部关节数量
        self.num_hand_dofs = self.hand.num_joints

        # 位置目标缓冲区，用于存储手部关节的目标位置
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # 驱动关节索引列表，记录哪些关节是被驱动的
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # 手指刚体索引列表，记录手指尖的刚体索引
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # 关节限制，获取手部关节的位置限制
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # 目标重置跟踪，用于跟踪哪些环境需要重置目标
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 用于比较物体位置的参考位置
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        # 默认目标位置和旋转
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        # 初始化目标标记
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # 成功跟踪，记录每个环境的成功次数和连续成功次数
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # 单位张量，用于旋转计算
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        """设置场景，添加手部、物体和目标对象到场景中"""
        # 添加手部、手中物体和目标物体
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # 添加地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 克隆和复制环境（此环境不需要过滤）
        self.scene.clone_environments(copy_from_source=False)
        # 将关节结构添加到场景中 - 必须注册到场景中才能使用EventManager进行随机化
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步骤前的处理，保存动作数据
        
        Args:
            actions: 动作张量
        """
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """应用动作到环境中，控制手部关节位置"""
        # 缩放动作到关节限制范围内
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # 使用移动平均平滑动作
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        # 饱和处理，确保关节位置在限制范围内
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # 更新前一目标位置
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # 设置手部关节位置目标
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        """获取环境观测值
        
        根据配置返回不同类型的观测值，支持简化观测和完整观测两种模式。
        还支持不对称观测，即为critic网络提供更丰富的状态信息。
        
        Returns:
            dict: 包含策略网络和critic网络观测值的字典
        """
        # 如果启用不对称观测，获取指尖力传感器数据
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        # 根据观测类型计算观测值
        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        # 如果启用不对称观测，计算完整状态信息
        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        # 构造观测值字典
        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算奖励值
        
        调用compute_rewards函数计算当前步骤的奖励值，包括距离奖励、旋转奖励、
        动作惩罚等奖励项。同时处理目标重置和成功跟踪逻辑。
        
        Returns:
            torch.Tensor: 每个环境的总奖励值
        """
        # 调用compute_rewards函数计算奖励
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        # 记录连续成功次数到日志中
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # 如果达到了目标，则重置目标
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算环境是否应该结束
        
        检查物体是否掉落或是否达到最大步数，返回两个布尔张量：
        1. 物体是否掉落导致环境结束
        2. 是否达到最大步数导致环境结束
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (物体掉落导致的结束标志, 时间超时导致的结束标志)
        """
        # 计算中间值，包括手部和物体的位置、旋转等信息
        self._compute_intermediate_values()

        # 当立方体掉落时重置环境
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        # 如果配置了最大连续成功次数，处理相关逻辑
        if self.cfg.max_consecutive_success > 0:
            # 如果物体旋转距离小于成功容忍度，则重置进度（episode length buf）
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            # 检查是否达到最大连续成功次数
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        # 检查是否达到最大episode长度
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 如果配置了最大连续成功次数，将时间超时和最大成功次数条件进行或运算
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定环境的索引
        
        重置指定环境中的手部和物体状态，包括位置、旋转、速度等属性。
        同时重置目标姿态和成功计数器。
        
        Args:
            env_ids: 需要重置的环境索引列表，如果为None则重置所有环境
        """
        # 如果未指定环境索引，则重置所有环境
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # 重置关节结构和刚体属性
        super()._reset_idx(env_ids)

        # 重置目标姿态
        self._reset_target_pose(env_ids)

        # 重置物体状态
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # 全局物体位置
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        # 旋转噪声，用于X和Y轴旋转
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # 设置物体速度为零
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # 重置手部状态
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        # 添加位置噪声
        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        # 添加速度噪声
        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        # 更新手部目标位置
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        # 设置手部关节位置和速度目标
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # 重置成功计数器
        self.successes[env_ids] = 0
        # 重新计算中间值
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        """重置目标姿态
        
        为指定环境生成新的随机目标旋转，并更新目标标记。
        
        Args:
            env_ids: 需要重置目标姿态的环境索引列表
        """
        # 重置目标旋转
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # 更新目标姿态和标记
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        # 重置目标缓冲区
        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        """计算中间值
        
        计算手部和物体的各种中间状态值，包括位置、旋转、速度等，
        这些值在奖励计算和其他方法中会被使用。
        """
        # 手部数据
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # 物体数据
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def compute_reduced_observations(self):
        """计算简化观测值
        
        根据论文 https://arxiv.org/pdf/1808.00177.pdf Table 2 中的定义，
        简化观测值包括：
        - 手指尖位置
        - 物体位置（但不包括方向）
        - 相对目标方向
        - 动作值
        
        Returns:
            torch.Tensor: 简化观测值张量
        """
        # 按照论文中的定义计算观测值
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        """计算完整观测值
        
        完整观测值包括手部关节位置和速度、物体位置和旋转、
        物体线速度和角速度、目标位置和旋转、手指尖位置和旋转、
        手指尖速度以及动作值等信息。
        
        Returns:
            torch.Tensor: 完整观测值张量
        """
        obs = torch.cat(
            (
                # 手部信息
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # 物体信息
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # 目标信息
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # 手指尖信息
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # 动作信息
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        """计算完整状态值（用于critic网络）
        
        完整状态值包括手部关节位置和速度、物体位置和旋转、
        物体线速度和角速度、目标位置和旋转、手指尖位置和旋转、
        手指尖速度和力矩传感器数据以及动作值等信息。
        
        Returns:
            torch.Tensor: 完整状态值张量
        """
        states = torch.cat(
            (
                # 手部信息
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # 物体信息
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # 目标信息
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # 手指尖信息
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # 动作信息
                self.actions,
            ),
            dim=-1,
        )
        return states


@torch.jit.script
def scale(x, lower, upper):
    """将输入值从[-1, 1]范围缩放到指定的[lower, upper]范围
    
    Args:
        x: 输入值，范围应在[-1, 1]之间
        lower: 目标范围的下界
        upper: 目标范围的上界
        
    Returns:
        缩放后的值
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """将输入值从指定的[lower, upper]范围反缩放到[-1, 1]范围
    
    Args:
        x: 输入值，范围应在[lower, upper]之间
        lower: 输入范围的下界
        upper: 输入范围的上界
        
    Returns:
        反缩放后的值，范围在[-1, 1]之间
    """
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """生成随机旋转四元数
    
    通过绕X轴和Y轴的随机旋转来生成四元数
    
    Args:
        rand0: X轴旋转的随机因子
        rand1: Y轴旋转的随机因子
        x_unit_tensor: X轴单位向量
        y_unit_tensor: Y轴单位向量
        
    Returns:
        随机旋转四元数
    """
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    """计算两个四元数之间的旋转距离
    
    Args:
        object_rot: 物体当前旋转四元数
        target_rot: 目标旋转四元数
        
    Returns:
        两个四元数之间的旋转距离
    """
    # 手中立方体和目标立方体的方向对齐
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # 改变了四元数约定


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
    """计算奖励值
    
    根据物体位置、旋转、动作等因素计算奖励值，并处理目标重置和成功跟踪逻辑。
    
    Args:
        reset_buf: 环境重置缓冲区
        reset_goal_buf: 目标重置缓冲区
        successes: 成功计数器
        consecutive_successes: 连续成功计数器
        max_episode_length: 最大episode长度
        object_pos: 物体位置
        object_rot: 物体旋转
        target_pos: 目标位置
        target_rot: 目标旋转
        dist_reward_scale: 距离奖励缩放因子
        rot_reward_scale: 旋转奖励缩放因子
        rot_eps: 旋转奖励的epsilon值
        actions: 动作值
        action_penalty_scale: 动作惩罚缩放因子
        success_tolerance: 成功容忍度
        reach_goal_bonus: 达到目标奖励
        fall_dist: 掉落距离阈值
        fall_penalty: 掉落惩罚
        av_factor: 平均因子
        
    Returns:
        tuple: (总奖励值, 目标重置标志, 成功计数器, 连续成功计数器)
    """

    # 计算物体与目标之间的距离
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    # 计算物体与目标之间的旋转距离
    rot_dist = rotation_distance(object_rot, target_rot)

    # 距离奖励：物体与目标距离越近奖励越高
    dist_rew = goal_dist * dist_reward_scale
    # 旋转奖励：物体与目标旋转越接近奖励越高
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # 动作惩罚：动作幅度越大惩罚越重
    action_penalty = torch.sum(actions**2, dim=-1)

    # 总奖励是：位置距离奖励 + 方向对齐奖励 + 动作正则化惩罚 + 成功奖励 + 掉落惩罚
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # 找出哪些环境达到了目标并更新成功计数
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # 成功奖励：当方向在成功容忍度范围内时给予奖励
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # 掉落惩罚：当物体与目标距离超过阈值时给予惩罚
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # 检查环境终止条件，包括最大成功次数
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
