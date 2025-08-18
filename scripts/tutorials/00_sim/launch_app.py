# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了如何通过 AppLauncher 运行 IsaacSim

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="教程：通过 AppLauncher 运行 IsaacSim。")
parser.add_argument("--size", type=float, default=1.0, help="立方体的边长")
# SimulationApp 参数 https://docs.omniverse.nvidia.com/py/isaacsim/source/isaacsim.simulation_app/docs/index.html?highlight=simulationapp#isaacsim.simulation_app.SimulationApp
parser.add_argument(
    "--width", type=int, default=1280, help="视口和生成图像的宽度。默认为 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="视口和生成图像的高度。默认为 720"
)

# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下。"""

import isaaclab.sim as sim_utils


def design_scene():
    """通过从 USD 文件中生成地面、灯光、对象和网格来设计场景。"""
    # 地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 生成远处光源
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # 生成一个立方体
    cfg_cuboid = sim_utils.CuboidCfg(
        size=[args_cli.size] * 3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    )
    # 生成立方体，调整 z 轴上的平移以适应其大小
    cfg_cuboid.func("/World/Object", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))


def main():
    """主函数。"""

    # 初始化模拟上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # 通过向场景中添加资产来设计场景
    design_scene()

    # 播放模拟器
    sim.reset()
    # 现在我们准备好了！
    print("[INFO]: 设置完成...")

    # 模拟物理过程
    while simulation_app.is_running():
        # 执行一步模拟
        sim.step()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
