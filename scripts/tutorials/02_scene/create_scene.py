# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""此脚本演示了如何使用交互式场景接口来设置包含多个实体的场景。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="交互式场景接口使用教程。")
parser.add_argument("--num_envs", type=int, default=2, help="要生成的环境数量。")
# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来是其余的所有内容。"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """推车-摆杆场景的配置。"""

    # 地面
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 对于需要在多个环境中复制的物体（如机器人），其 prim_path（路径）应包含 {ENV_REGEX_NS} 占位符。
    # 而没有这个占位符的物体（如灯光）则会被所有环境共享。
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行模拟循环。"""
    # 提取场景实体
    # 注意：我们在这里这样做只是为了提高可读性。
    # 可以通过类似字典的方式，使用在配置类中定义的变量名作为键来访问场景中的特定物体。
    robot = scene["cartpole"]
    # 定义模拟步长
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
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # 设置带有一些噪声的关节位置
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # 清除内部缓冲区
            # 不再对单个机器人调用 reset(), update() 等方法，而是直接对整个 scene 对象进行操作。
            # scene.reset()：重置场景中所有可重置的物体。
            scene.reset()
            print("[INFO]: 重置机器人状态...")
        # 应用随机动作
        # -- 生成随机关节力
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- 将动作应用到机器人
        robot.set_joint_effort_target(efforts)
        # -- 将数据写入模拟器
        scene.write_data_to_sim()
        # 执行步进
        sim.step()
        # 增加计数器
        count += 1
        # 更新缓冲区
        scene.update(sim_dt)


def main():
    """主函数。"""
    # 加载 kit 辅助工具
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主摄像头视角
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # 设计场景
    # 首先创建场景配置类的实例，并传入参数，如环境数量 num_envs 和环境间距 env_spacing
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    # 然后，将这个配置对象传递给 InteractiveScene 的构造函数来创建场景实
    scene = InteractiveScene(scene_cfg)
    # 启动模拟器
    sim.reset()
    # 现在我们准备好了！
    print("[INFO]: 设置完成...")
    # 运行模拟器
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
