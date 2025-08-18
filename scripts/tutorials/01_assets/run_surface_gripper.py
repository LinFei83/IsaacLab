# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""该脚本演示了如何生成一个配备表面夹爪的抓取和放置机器人并与其交互。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu

当运行此脚本时，请确保 --device 标志设置为 cpu。这是因为表面夹爪目前仅支持在 CPU 上运行。
"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：生成和交互表面夹爪。")
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
from isaaclab.assets import Articulation, SurfaceGripper, SurfaceGripperCfg
from isaaclab.sim import SimulationContext

##
# 预定义配置
##
from isaaclab_assets import PICK_AND_PLACE_CFG  # isort:skip


def design_scene():
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
    origins = [[2.75, 0.0, 0.0], [-2.75, 0.0, 0.0]]
    # 原点 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # 原点 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # 关节物体: 首先定义机器人配置
    pick_and_place_robot_cfg = PICK_AND_PLACE_CFG.copy()
    pick_and_place_robot_cfg.prim_path = "/World/Origin.*/Robot"
    pick_and_place_robot = Articulation(cfg=pick_and_place_robot_cfg)

    # 表面夹爪: 接下来定义表面夹爪配置
    surface_gripper_cfg = SurfaceGripperCfg()
    # 我们需要告诉 View 使用哪个 prim 作为表面夹爪
    surface_gripper_cfg.prim_expr = "/World/Origin.*/Robot/picker_head/SurfaceGripper"
    # 我们可以设置表面夹爪的不同参数，注意如果这些参数没有设置，
    # View 将尝试从 prim 中读取它们。
    surface_gripper_cfg.max_grip_distance = 0.1  # [m] (夹爪能够抓取物体的最大距离)
    surface_gripper_cfg.shear_force_limit = 500.0  # [N] (垂直方向的力限制)
    surface_gripper_cfg.coaxial_force_limit = 500.0  # [N] (夹爪轴向的力限制)
    surface_gripper_cfg.retry_interval = 0.1  # 秒 (夹爪保持抓取状态的时间)
    # 现在我们可以生成表面夹爪
    surface_gripper = SurfaceGripper(cfg=surface_gripper_cfg)

    # 返回场景信息
    scene_entities = {"pick_and_place_robot": pick_and_place_robot, "surface_gripper": surface_gripper}
    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext, entities: dict[str, Articulation | SurfaceGripper], origins: torch.Tensor
):
    """运行模拟循环。
    
    Args:
        sim: 模拟上下文对象
        entities: 场景实体字典
        origins: 场景原点坐标张量
    """
    # 提取场景实体
    robot: Articulation = entities["pick_and_place_robot"]
    surface_gripper: SurfaceGripper = entities["surface_gripper"]

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
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # 设置带有一些噪声的关节位置
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # 清除内部缓冲区
            robot.reset()
            print("[INFO]: 重置机器人状态...")
            # 打开夹爪并确保夹爪处于打开状态
            surface_gripper.reset()
            print("[INFO]: 重置夹爪状态...")

        # 在 -1 和 1 之间采样一个随机命令。
        gripper_commands = torch.rand(surface_gripper.num_instances) * 2.0 - 1.0
        # 夹爪的行为如下：
        # -1 < command < -0.3 --> 夹爪正在打开
        # -0.3 < command < 0.3 --> 夹爪处于空闲状态
        # 0.3 < command < 1 --> 夹爪正在关闭
        print(f"[INFO]: 夹爪命令: {gripper_commands}")
        mapped_commands = [
            "打开" if command < -0.3 else "关闭" if command > 0.3 else "空闲" for command in gripper_commands
        ]
        print(f"[INFO]: 映射命令: {mapped_commands}")
        # 设置夹爪命令
        surface_gripper.set_grippers_command(gripper_commands)
        # 将数据写入模拟器
        surface_gripper.write_data_to_sim()
        # 执行步进
        sim.step()
        # 增加计数器
        count += 1
        # 从模拟器中读取夹爪状态
        surface_gripper.update(sim_dt)
        # 从缓冲区中读取夹爪状态
        surface_gripper_state = surface_gripper.state
        # 夹爪状态是一个整数列表，可以映射为以下内容：
        # -1 --> 打开
        # 0 --> 关闭中
        # 1 --> 关闭
        # 打印夹爪状态
        print(f"[INFO]: 夹爪状态: {surface_gripper_state}")
        mapped_commands = [
            "打开" if state == -1 else "关闭中" if state == 0 else "关闭" for state in surface_gripper_state.tolist()
        ]
        print(f"[INFO]: 映射命令: {mapped_commands}")


def main():
    """主函数。"""
    # 加载工具助手
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主摄像头
    sim.set_camera_view([2.75, 7.5, 10.0], [2.75, 0.0, 0.0])
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