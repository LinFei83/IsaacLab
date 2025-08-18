# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# 动作文件目录路径
MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """人形机器人 AMP 环境配置 (基类)。"""

    # 环境参数
    # 每个episode的时长（秒）
    episode_length_s = 10.0
    # 降频因子，控制物理模拟与控制更新的比率
    decimation = 2

    # 空间定义
    # 观测空间维度
    observation_space = 81
    # 动作空间维度
    action_space = 28
    # 状态空间维度
    state_space = 0
    # AMP 观测的数量
    num_amp_observations = 2
    # AMP 观测空间维度
    amp_observation_space = 81

    # 终止条件
    # 是否启用提前终止
    early_termination = True
    # 终止高度阈值，当机器人的躯干低于此高度时终止episode
    termination_height = 0.5

    # 动作文件路径
    motion_file: str = MISSING
    # 参考身体部位，用于计算观测和终止条件
    reference_body = "torso"
    # 重置策略：default, random, random-start
    reset_strategy = "random"
    """重置每个环境时（人形机器人的姿态和关节状态）遵循的策略。

    * default: 姿态和关节状态设置为资产的初始状态。
    * random: 姿态和关节状态通过在动作中随机均匀采样时间来设置。
    * random-start: 姿态和关节状态通过在动作开始时（时间为零）采样来设置。
    """

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        # 仿真时间步长
        dt=1 / 60,
        # 渲染间隔
        render_interval=decimation,
        # PhysX 物理引擎配置
        physx=PhysxCfg(
            # GPU 上丢失对的容量
            gpu_found_lost_pairs_capacity=2**23,
            # GPU 上总聚合对的容量
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # 环境数量
        num_envs=4096,
        # 环境间距
        env_spacing=10.0,
        # 是否复制物理
        replicate_physics=True
    )

    # 机器人配置
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                # 匹配所有关节名称
                joint_names_expr=[".*"],
                # 关节刚度（None表示使用默认值）
                stiffness=None,
                # 关节阻尼（None表示使用默认值）
                damping=None,
                # 速度限制
                velocity_limit_sim={
                    ".*": 100.0,
                },
            ),
        },
    )


@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    # 舞蹈动作文件路径
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    # 跑步动作文件路径
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    # 走路动作文件路径
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
