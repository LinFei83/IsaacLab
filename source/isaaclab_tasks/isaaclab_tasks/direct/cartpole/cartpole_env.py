# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # 环境配置
    decimation = 2  # 控制频率与仿真频率的比率
    episode_length_s = 5.0  # 每个episode的时长（秒）
    action_scale = 100.0  # 动作缩放因子 [N]
    action_space = 1      # 动作空间维度
    observation_space = 4 # 观察空间维度
    state_space = 0       # 状态空间维度

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # 机器人配置
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"  # 小车关节名称
    pole_dof_name = "cart_to_pole"    # 杆关节名称

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # 重置配置
    max_cart_pos = 3.0  # 小车位置超过此值时重置 [m]
    initial_pole_angle_range = [-0.25, 0.25]  # 重置时杆角度的采样范围 [rad]

    # 奖励缩放因子
    rew_scale_alive = 1.0        # 存活奖励
    rew_scale_terminated = -2.0  # 终止奖励
    rew_scale_pole_pos = -1.0    # 杆位置奖励
    rew_scale_cart_vel = -0.01   # 小车速度奖励
    rew_scale_pole_vel = -0.005  # 杆速度奖励


class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        # 创建小车关节
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # 添加地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 克隆和复制环境
        self.scene.clone_environments(copy_from_source=False)
        # CPU仿真需要显式过滤碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # 将关节添加到场景中
        self.scene.articulations["cartpole"] = self.cartpole
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 缩放动作
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # 应用动作到小车关节
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        # 构造观察值：杆角度、杆速度、小车位置、小车速度
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 计算奖励
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 更新关节位置和速度
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        # 判断是否超时
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 判断是否超出边界
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 如果env_ids为None，则重置所有环境
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        # 获取默认关节位置并添加随机扰动
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        # 更新根状态
        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 更新关节位置和速度
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # 将状态写入仿真
        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # 存活奖励：如果未终止，则获得奖励
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # 终止奖励：如果终止，则获得负奖励
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # 杆位置奖励：惩罚杆偏离垂直位置
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # 小车速度奖励：惩罚小车速度过快
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # 杆速度奖励：惩罚杆速度过快
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # 总奖励
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
