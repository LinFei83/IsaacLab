# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""该脚本演示了如何生成一个倒立摆并与其交互。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：生成和交互关节物体。")
# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的所有内容。"""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# 预定义配置
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """设计场景。
    
    Returns:
        tuple: 包含场景实体字典和原点坐标的元组
    """
    # 地面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 灯光
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 创建名为 "Origin1"、"Origin2" 的独立组
    # 每个组中将有一个机器人
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # 原点 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # 原点 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # 关节物体
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    # 返回场景信息
    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """运行模拟循环。
    
    Args:
        sim: 模拟上下文对象
        entities: 场景实体字典
        origins: 场景原点坐标张量
    """
    # 提取场景实体
    # 注意：我们在这里这样做只是为了提高可读性。通常，最好直接从字典中访问实体。
    # 在下一个教程中，这个字典将被 InteractiveScene 类替换。
    robot = entities["cartpole"]
    # 定义模拟步进
    sim_dt = sim.get_physics_dt()
    count = 0
    # 模拟循环
    while simulation_app.is_running():
        # 重置
        if count % 500 == 0:
            # 重置计数器
            count = 0
            # 重置场景实体
            # 根状态
            # 我们通过原点偏移根状态，因为状态是在模拟世界坐标系中编写的
            # 如果不这样做，机器人将在模拟世界的 (0, 0, 0) 处生成
            root_state = robot.data.default_root_state.clone()
            # 将每个机器人的初始位置与其对应的 `origins` 相加，确保它们被放置在不同的生成点，而不是都挤在世界原点 (0,0,0)。
            root_state[:, :3] += origins
            # 将计算好的根状态（位姿和速度） **写入** 到物理引擎中，让机器人在仿真世界里“瞬移”到指定状态。
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # 获取默认的关节位置和速度后为初始关节位置增加一点随机噪声
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # 清除内部缓冲区
            robot.reset()
            print("[INFO]: 重置机器人状态...")
        
        # 以下代码在每次循环都会执行

        # 应用随机动作
        # -- 这里生成一个与关节数量相同的随机张量，作为施加到每个关节上的 **力矩（Effort）**。这就是为什么机器人会随机乱动。
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- 将动作应用到机器人
        robot.set_joint_effort_target(efforts)
        # -- 将数据写入模拟器
        robot.write_data_to_sim()
        # 执行步进
        sim.step()
        # 增加计数器
        count += 1
        # 更新缓冲区
        robot.update(sim_dt)


def main():
    """主函数。"""
    # 加载工具助手
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主摄像头
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # 设计场景
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # 启动模拟器
    sim.reset()
    # 现在我们准备好了！
    print("[INFO]: 设置完成...")
    # 运行模拟器
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()