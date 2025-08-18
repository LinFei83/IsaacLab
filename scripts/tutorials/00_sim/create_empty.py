# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""本脚本演示了如何在 Isaac Sim 中创建一个简单的场景。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="教程：创建一个空场景。")
# 添加 AppLauncher 命令行参数 例如无头模式 直播模式等
# app_launcher arguments:
#   AppLauncher 的参数。更多详情，请查阅文档。

#   --headless            始终强制关闭显示。
#   --livestream {0,1,2}  强制启用直播。映射对应于 `LIVESTREAM` 环境变量。
#   --enable_cameras      启用摄像头传感器和相关的扩展依赖。
#   --xr                  为 VR/AR 应用程序启用 XR 模式。
#   --device DEVICE       运行模拟的设备。可以是 "cpu", "cuda", "cuda:N"，其中 N 是设备 ID
#   --verbose             启用 SimulationApp 的详细级别日志输出。
#   --info                启用 SimulationApp 的信息级别日志输出。
#   --experience EXPERIENCE
#                         启动 SimulationApp 时加载的体验文件。如果提供空字符串，体验文件将根据 headless 标志确定。如果
#                         提供相对路径，它将相对于 Isaac Sim 和 Isaac Lab 中的 `apps` 文件夹（按此顺序）解析。
#   --rendering_mode {balanced,performance,quality}
#                         设置渲染模式。预设设置文件可以在 apps/rendering_modes 中找到。可以是 "performance"（性能），"balanced"（平衡），或 "quality"（质量）。
#                         可以使用 RenderCfg 类覆盖单个设置。
#   --kit_args KIT_ARGS   Omniverse Kit 的命令行参数，以空格分隔的字符串形式。示例用法：--kit_args "--ext-folder=/path/to/ext1 --ext-
#                         folder=/path/to/ext2"
#   --anim_recording_enabled
#                         启用从 IsaacLab PhysX 模拟录制时间采样 USD 动画。
#   --anim_recording_start_time ANIM_RECORDING_START_TIME
#                         设置动画录制开始播放的时间。如果未设置，录制将从头开始。
#   --anim_recording_stop_time ANIM_RECORDING_STOP_TIME
#                         设置动画录制停止播放的时间。如果在超过停止时间之前进程关闭，则动画不会被录制。
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""我们已经配置启动仿真应用程序 其余代码如下。"""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """主函数。"""

    # 初始化模拟上下文
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # 设置主相机
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0]) # 设置相机位置和相机应该看向的目标点

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
