# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg
from .motions import MotionLoader


class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 动作偏移量和缩放因子
        # 获取关节位置的软限制
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        # 计算动作偏移量（关节限制范围的中点）
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        # 计算动作缩放因子（关节限制范围）
        self.action_scale = dof_upper_limits - dof_lower_limits

        # 加载动作数据
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # 获取自由度（DOF）和关键身体部位的索引
        # 定义关键身体部位名称
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        # 获取参考身体部位的索引
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        # 获取关键身体部位的索引列表
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        # 获取机器人关节在动作数据中的索引
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        # 获取参考身体部位在动作数据中的索引
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        # 获取关键身体部位在动作数据中的索引列表
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # 根据观测数量重新配置 AMP 观测空间并创建缓冲区
        # 计算 AMP 观测的总大小
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        # 定义 AMP 观测空间
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        # 创建 AMP 观测缓冲区，用于存储历史观测
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        # 创建机器人实例
        self.robot = Articulation(self.cfg.robot)
        # 添加地面平面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    # 静摩擦系数
                    static_friction=1.0,
                    # 动摩擦系数
                    dynamic_friction=1.0,
                    # 恢复系数
                    restitution=0.0,
                ),
            ),
        )
        # 克隆和复制环境
        self.scene.clone_environments(copy_from_source=False)
        # 对于 CPU 仿真，需要显式过滤碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 将机器人添加到场景中
        self.scene.articulations["robot"] = self.robot
        # 添加光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # 保存动作数据用于后续处理
        self.actions = actions.clone()

    def _apply_action(self):
        # 计算目标关节位置：偏移量 + 缩放因子 * 动作值
        target = self.action_offset + self.action_scale * self.actions
        # 设置机器人关节位置目标
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # 构建任务观测
        obs = compute_obs(
            # 关节位置
            self.robot.data.joint_pos,
            # 关节速度
            self.robot.data.joint_vel,
            # 参考身体部位的世界位置
            self.robot.data.body_pos_w[:, self.ref_body_index],
            # 参考身体部位的世界旋转（四元数）
            self.robot.data.body_quat_w[:, self.ref_body_index],
            # 参考身体部位的线速度
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            # 参考身体部位的角速度
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            # 关键身体部位的世界位置
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # 更新 AMP 观测历史
        # 将历史观测向后移动一位
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # 将当前观测存储在缓冲区的第一位
        self.amp_observation_buffer[:, 0] = obs.clone()
        # 将 AMP 观测添加到额外信息中
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 返回奖励值（这里简单地返回1）
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 时间超时终止条件
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 提前终止条件
        if self.cfg.early_termination:
            # 当参考身体部位（躯干）的高度低于终止高度时终止
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 如果未指定环境ID或环境ID数量等于总环境数，则重置所有环境
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # 重置机器人状态
        self.robot.reset(env_ids)
        # 调用父类的重置方法
        super()._reset_idx(env_ids)

        # 根据重置策略执行不同的重置逻辑
        if self.cfg.reset_strategy == "default":
            # 默认重置策略
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            # 随机重置策略
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"未知的重置策略: {self.cfg.reset_strategy}")

        # 将重置后的状态写入仿真器
        # 设置根链接的姿态
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        # 设置根质心的速度
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        # 设置关节状态
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # 重置策略方法

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 获取默认的根状态
        root_state = self.robot.data.default_root_state[env_ids].clone()
        # 调整根位置以匹配环境原点
        root_state[:, :3] += self.scene.env_origins[env_ids]
        # 获取默认的关节位置和速度
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 采样随机动作时间（如果start为True则为零）
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # 采样随机动作
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # 获取根变换（人形机器人的躯干）
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        # 获取默认的根状态
        root_state = self.robot.data.default_root_state[env_ids].clone()
        # 设置根位置（动作数据中的躯干位置 + 环境原点）
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        # 稍微提升人形机器人以避免与地面碰撞
        root_state[:, 2] += 0.15
        # 设置根旋转
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        # 设置线速度
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        # 设置角速度
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # 获取关节状态
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # 更新 AMP 观测
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # 环境方法

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # 采样随机动作时间（或使用指定的时间）
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        # 计算时间序列：当前时间 - dt * [0, 1, ..., num_amp_observations-1]
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # 获取动作数据
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # 计算 AMP 观测
        amp_observation = compute_obs(
            # 关节位置
            dof_positions[:, self.motion_dof_indexes],
            # 关节速度
            dof_velocities[:, self.motion_dof_indexes],
            # 参考身体部位的位置
            body_positions[:, self.motion_ref_body_index],
            # 参考身体部位的旋转
            body_rotations[:, self.motion_ref_body_index],
            # 参考身体部位的线速度
            body_linear_velocities[:, self.motion_ref_body_index],
            # 参考身体部位的角速度
            body_angular_velocities[:, self.motion_ref_body_index],
            # 关键身体部位的位置
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    # 创建参考切向量和法向量
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    # 设置参考切向量为 [1, 0, 0]
    ref_tangent[..., 0] = 1
    # 设置参考法向量为 [0, 0, 1]
    ref_normal[..., -1] = 1
    # 应用四元数旋转向量
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    # 拼接切向量和法向量作为观测的一部分
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    # 构建观测向量
    obs = torch.cat(
        (
            # 关节位置
            dof_positions,
            # 关节速度
            dof_velocities,
            # 根身体高度（z坐标）
            root_positions[:, 2:3],
            # 根旋转的切向量和法向量
            quaternion_to_tangent_and_normal(root_rotations),
            # 根线速度
            root_linear_velocities,
            # 根角速度
            root_angular_velocities,
            # 关键身体部位相对于根位置的偏移
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
