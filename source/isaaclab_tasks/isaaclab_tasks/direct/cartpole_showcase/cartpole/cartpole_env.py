# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg


class CartpoleShowcaseEnv(CartpoleEnv):
    cfg: CartpoleEnvCfg

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 克隆动作张量，避免在后续处理中修改原始数据
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 根据不同的动作空间类型应用相应的动作
        
        # 基本空间类型
        # - Box: 连续动作空间
        if isinstance(self.single_action_space, gym.spaces.Box):
            # 直接将动作缩放后应用到关节
            target = self.cfg.action_scale * self.actions
        # - Discrete: 离散动作空间
        elif isinstance(self.single_action_space, gym.spaces.Discrete):
            # 创建一个零张量作为初始目标力矩
            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            # 根据动作编号设置相应的力矩值
            # 动作 1: 负最大力矩
            target = torch.where(self.actions == 1, -self.cfg.action_scale, target)
            # 动作 2: 正最大力矩
            target = torch.where(self.actions == 2, self.cfg.action_scale, target)
        # - MultiDiscrete: 多维离散动作空间
        elif isinstance(self.single_action_space, gym.spaces.MultiDiscrete):
            # 创建一个零张量作为初始目标力矩
            target = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
            # 根据第一个离散动作设置力矩大小
            # 动作 1: 半最大力矩
            target = torch.where(self.actions[:, [0]] == 1, self.cfg.action_scale / 2.0, target)
            # 动作 2: 最大力矩
            target = torch.where(self.actions[:, [0]] == 2, self.cfg.action_scale, target)
            # 根据第二个离散动作设置力矩方向
            # 动作 0: 负力矩 (一侧)
            target = torch.where(self.actions[:, [1]] == 0, -target, target)
        else:
            raise NotImplementedError(f"动作空间 {type(self.single_action_space)} 尚未实现")

        # 设置目标关节力矩
        self.cartpole.set_joint_effort_target(target, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        # 根据不同的观察空间类型生成相应的观察值

        # 基本空间类型
        # - Box: 连续观察空间
        if isinstance(self.single_observation_space["policy"], gym.spaces.Box):
            # 将杆和小车的关节位置和速度拼接成一个向量
            obs = torch.cat(
                (
                    self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            )
        # - Discrete: 离散观察空间
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Discrete):
            # 将关节位置和速度拼接成一个向量，并判断每个值是否大于等于0
            data = (
                torch.cat(
                    (
                        self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                        self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                        self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                        self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    ),
                    dim=-1,
                )
                >= 0
            )
            # 初始化观察值为0
            obs = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
            # 根据关节位置和速度的符号组合确定离散观察值
            # 这里使用 discretization_indices 函数来判断是否匹配特定的符号组合
            obs = torch.where(discretization_indices(data, [False, False, False, True]), 1, obs)
            obs = torch.where(discretization_indices(data, [False, False, True, False]), 2, obs)
            obs = torch.where(discretization_indices(data, [False, False, True, True]), 3, obs)
            obs = torch.where(discretization_indices(data, [False, True, False, False]), 4, obs)
            obs = torch.where(discretization_indices(data, [False, True, False, True]), 5, obs)
            obs = torch.where(discretization_indices(data, [False, True, True, False]), 6, obs)
            obs = torch.where(discretization_indices(data, [False, True, True, True]), 7, obs)
            obs = torch.where(discretization_indices(data, [True, False, False, False]), 8, obs)
            obs = torch.where(discretization_indices(data, [True, False, False, True]), 9, obs)
            obs = torch.where(discretization_indices(data, [True, False, True, False]), 10, obs)
            obs = torch.where(discretization_indices(data, [True, False, True, True]), 11, obs)
            obs = torch.where(discretization_indices(data, [True, True, False, False]), 12, obs)
            obs = torch.where(discretization_indices(data, [True, True, False, True]), 13, obs)
            obs = torch.where(discretization_indices(data, [True, True, True, False]), 14, obs)
            obs = torch.where(discretization_indices(data, [True, True, True, True]), 15, obs)
        # - MultiDiscrete: 多维离散观察空间
        elif isinstance(self.single_observation_space["policy"], gym.spaces.MultiDiscrete):
            # 创建零和一的张量，用于构建多维离散观察值
            zeros = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
            ones = torch.ones_like(zeros)
            # 分别判断每个关节位置和速度是否大于等于0，并构建多维离散观察值
            obs = torch.cat(
                (
                    torch.where(
                        discretization_indices(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                    torch.where(
                        discretization_indices(self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                    torch.where(
                        discretization_indices(self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                    torch.where(
                        discretization_indices(self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1) >= 0, [True]),
                        ones,
                        zeros,
                    ).unsqueeze(dim=1),
                ),
                dim=-1,
            )
        # 复合空间类型
        # - Tuple: 元组观察空间
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Tuple):
            # 直接返回关节位置和速度的元组
            obs = (self.joint_pos, self.joint_vel)
        # - Dict: 字典观察空间
        elif isinstance(self.single_observation_space["policy"], gym.spaces.Dict):
            # 返回包含关节位置和速度的字典
            obs = {"joint-positions": self.joint_pos, "joint-velocities": self.joint_vel}
        else:
            raise NotImplementedError(
                f"观察空间 {type(self.single_observation_space['policy'])} 尚未实现"
            )

        # 构建最终的观察值字典
        observations = {"policy": obs}
        return observations


def discretization_indices(x: torch.Tensor, condition: list[bool]) -> torch.Tensor:
    # 判断输入张量 x 是否与给定条件匹配
    # 该函数用于确定关节位置和速度的符号组合是否匹配特定的离散观察值
    return torch.prod(x == torch.tensor(condition, device=x.device), axis=-1).to(torch.bool)
