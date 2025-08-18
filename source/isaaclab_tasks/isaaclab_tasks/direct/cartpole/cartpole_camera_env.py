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
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class CartpoleRGBCameraEnvCfg(DirectRLEnvCfg):
    # 环境配置
    decimation = 2  # 控制频率与仿真频率的比率
    episode_length_s = 5.0  # 每个episode的时长（秒）
    action_scale = 100.0  # 动作缩放因子 [N]

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # 机器人配置
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"  # 小车关节名称
    pole_dof_name = "cart_to_pole"    # 杆关节名称

    # 相机配置
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],  # 使用RGB图像
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,   # 图像宽度
        height=100,  # 图像高度
    )
    write_image_to_file = False  # 是否将图像写入文件

    # 空间配置
    action_space = 1      # 动作空间维度
    state_space = 0       # 状态空间维度
    observation_space = [tiled_camera.height, tiled_camera.width, 3]  # 观察空间维度

    # 修改查看器设置
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=20.0, replicate_physics=True)

    # 重置配置
    max_cart_pos = 3.0  # 小车位置超过此值时重置 [m]
    initial_pole_angle_range = [-0.125, 0.125]  # 重置时杆角度的采样范围 [rad]

    # 奖励缩放因子
    rew_scale_alive = 1.0        # 存活奖励
    rew_scale_terminated = -2.0  # 终止奖励
    rew_scale_pole_pos = -1.0    # 杆位置奖励
    rew_scale_cart_vel = -0.01   # 小车速度奖励
    rew_scale_pole_vel = -0.005  # 杆速度奖励


@configclass
class CartpoleDepthCameraEnvCfg(CartpoleRGBCameraEnvCfg):
    # 相机配置
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["depth"],  # 使用深度图像
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )

    # 空间配置
    observation_space = [tiled_camera.height, tiled_camera.width, 1]  # 深度图像只有一个通道


class CartpoleCameraEnv(DirectRLEnv):

    cfg: CartpoleRGBCameraEnvCfg | CartpoleDepthCameraEnvCfg

    def __init__(
        self, cfg: CartpoleRGBCameraEnvCfg | CartpoleDepthCameraEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # 获取关节索引
        self._cart_dof_idx, _ = self._cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self._cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        # 关节位置和速度
        self.joint_pos = self._cartpole.data.joint_pos
        self.joint_vel = self._cartpole.data.joint_vel

        # 检查相机数据类型
        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "Cartpole相机环境一次只支持一种图像类型，但提供了以下类型:"
                f" {self.cfg.tiled_camera.data_types}"
            )

    def close(self):
        """清理环境资源。"""
        super().close()

    def _setup_scene(self):
        """设置场景，包括小车和相机。"""
        self._cartpole = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        # 克隆和复制环境
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # CPU仿真需要显式过滤碰撞
            self.scene.filter_collisions(global_prim_paths=[])

        # 将关节和传感器添加到场景中
        self.scene.articulations["cartpole"] = self._cartpole
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 缩放动作
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # 应用动作到小车关节
        self._cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        # 确定图像类型
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        if "rgb" in self.cfg.tiled_camera.data_types:
            camera_data = self._tiled_camera.data.output[data_type] / 255.0
            # 归一化相机数据以获得更好的训练效果
            mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            camera_data -= mean_tensor
        elif "depth" in self.cfg.tiled_camera.data_types:
            camera_data = self._tiled_camera.data.output[data_type]
            # 将无穷大值替换为0
            camera_data[camera_data == float("inf")] = 0
        observations = {"policy": camera_data.clone()}

        # 如果需要，将图像写入文件
        if self.cfg.write_image_to_file:
            save_images_to_file(observations["policy"], f"cartpole_{data_type}.png")

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
        self.joint_pos = self._cartpole.data.joint_pos
        self.joint_vel = self._cartpole.data.joint_vel

        # 判断是否超时
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 判断是否超出边界
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 如果env_ids为None，则重置所有环境
        if env_ids is None:
            env_ids = self._cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        # 获取默认关节位置并添加随机扰动
        joint_pos = self._cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self._cartpole.data.default_joint_vel[env_ids]

        # 更新根状态
        default_root_state = self._cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 更新关节位置和速度
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # 将状态写入仿真
        self._cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


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
