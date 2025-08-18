# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了如何创建一个带有倒立摆的简单环境。它结合了场景、动作、观测和事件管理器的概念来创建环境。

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：创建倒立摆基础环境。")
parser.add_argument("--num_envs", type=int, default=16, help="要生成的环境数量。")

# 添加 AppLauncher cli 参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的所有内容。"""

import math
import torch
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.actions.actions_cfg.JointEffortActionCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
# 动作管理器 (Action Manager) :
# 作用 : 定义智能体（Agent）可以如何与环境交互，即定义动作空间和动作如何施加到机器人上。
class ActionsCfg:
    """环境的动作规范。"""

    # 关节力矩控制，控制小车滑块的移动
    # 动作类型: JointEffortActionCfg - 关节力矩/力控制配置
    # 控制对象: asset_name="robot" - 指向场景中名为"robot"的机器人资产
    # 控制关节: joint_names=["slider_to_cart"] - 控制名为"slider_to_cart"的关节
    # 缩放因子: scale=5.0 - 将输入动作乘以5.0倍
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
# 观测管理器 (Observation Manager) :
# 作用 : 定义智能体能从环境中“看到”什么，即定义观测空间。
class ObservationsCfg:
    """环境的观测规范。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略组的观测。"""

        # 观测项（保持顺序）
        # 相对关节位置：小车和杆子的相对位置
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 相对关节速度：小车和杆子的相对速度
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True # 将joint_pos_rel和joint_vel_rel连接起来

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
# 事件管理器 (Event Manager) :
# 作用 : 调度在仿真特定时刻执行的操作，如重置环境、域随机化（Domain Randomization）等。
class EventCfg:
    """事件配置。"""

    # 启动时事件
    # 随机化杆子的质量，增加环境的多样性
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # 重置时事件
    # 重置小车的位置和速度
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # 重置杆子的位置和速度
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
# 场景管理器 (Scene Manager) :
# 作用 : 定义仿真的世界，包括机器人模型、地面、灯光等所有物理实体。
# 整合为环境总配置 : 创建一个主环境配置类 CartpoleEnvCfg，它继承自 ManagerBasedEnvCfg，并将前面定义的场景、动作、观测、事件等配置作为其属性。
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """倒立摆环境的配置。"""

    # 场景设置
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # 基本设置
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    
    # 自定义仿真参数 : 在主环境配置类中，通过 __post_init__ 方法可以方便地修改仿真参数（如仿真步长 sim.dt、环境步进频率 decimation）和视图参数（如相机位置）。
    def __post_init__(self):
        """后初始化。"""
        # 查看器设置
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # 步长设置
        self.decimation = 4  # 每4个仿真步长执行一次环境步长: 200Hz / 4 = 50Hz
        # 仿真设置
        self.sim.dt = 0.005  # 每5ms执行一次仿真步长: 200Hz


def main():
    """主函数。"""
    # 解析参数
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # 设置基础环境
    env = ManagerBasedEnv(cfg=env_cfg)

    # 模拟物理
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 重置
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: 正在重置环境...")
            # 采样随机动作
            joint_efforts = torch.randn_like(env.action_manager.action)
            # 执行环境步长
            obs, _ = env.step(joint_efforts)
            # obs["policy"] 的形状是 (num_envs, 4)，因为：
            # - joint_pos_rel 贡献 2 个值：[cart_pos_rel, pole_angle_rel]  
            # - joint_vel_rel 贡献 2 个值：[cart_vel_rel, pole_vel_rel]
            # - concatenate_terms=True 将它们连接：[cart_pos_rel, pole_angle_rel, cart_vel_rel, pole_vel_rel]

            # 访问具体数据：
            # env_0_cart_pos_rel = obs["policy"][0][0]      # 环境0的小车相对位置
            # env_0_pole_angle_rel = obs["policy"][0][1]    # 环境0的杆子相对角度  
            # env_0_cart_vel_rel = obs["policy"][0][2]      # 环境0的小车相对速度
            # env_0_pole_vel_rel = obs["policy"][0][3]      # 环境0的杆子相对角速度
            # 打印杆子的当前方向
            print("[Env 0]: 杆子关节: ", obs["policy"][0][1].item())
            # 更新计数器
            count += 1

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()