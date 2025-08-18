# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets.robots.ant import ANT_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class AntEnvCfg(DirectRLEnvCfg):
    # 环境配置
    episode_length_s = 15.0  # 每个episode的时长（秒）
    decimation = 2  # 控制频率与仿真频率的比率
    action_scale = 0.5  # 动作缩放因子
    action_space = 8  # 动作空间维度（8个关节）
    observation_space = 36  # 观测空间维度
    state_space = 0  # 状态空间维度

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形的USD路径
        terrain_type="plane",  # 地形类型为平面
        collision_group=-1,  # 碰撞组
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",  # 摩擦系数组合模式
            restitution_combine_mode="average",  # 恢复系数组合模式
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
            restitution=0.0,  # 恢复系数
        ),
        debug_vis=False,  # 是否可视化地形
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # 并行环境数量
        env_spacing=4.0,  # 环境间距
        replicate_physics=True,  # 是否复制物理
        clone_in_fabric=True  # 是否在Fabric中克隆
    )

    # 机器人配置
    robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]  # 关节齿轮比

    heading_weight: float = 0.5  # 朝向奖励权重
    up_weight: float = 0.1  # 直立奖励权重

    energy_cost_scale: float = 0.05  # 能量消耗奖励缩放
    actions_cost_scale: float = 0.005  # 动作奖励缩放
    alive_reward_scale: float = 0.5  # 存活奖励缩放
    dof_vel_scale: float = 0.2  # 关节速度缩放

    death_cost: float = -2.0  # 死亡惩罚
    termination_height: float = 0.31  # 终止高度阈值

    angular_velocity_scale: float = 1.0  # 角速度缩放
    contact_force_scale: float = 0.1  # 接触力缩放


class AntEnv(LocomotionEnv):
    cfg: AntEnvCfg

    def __init__(self, cfg: AntEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
