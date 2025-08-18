# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了在模拟运行时如何生成日志输出。
它附带了 Docker 使用教程。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse
import os

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="教程：从 Docker 容器内部创建日志。")
# 添加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下。"""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """主函数。"""
    # 指定日志必须存储在 logs/docker_tutorial 目录中
    log_dir_path = os.path.join("logs")
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    # 在容器中，绝对路径将是
    # /workspace/isaaclab/logs/docker_tutorial，因为
    # 所有 Python 执行都是通过 /workspace/isaaclab/isaaclab.sh 完成的
    # 调用进程的路径将是 /workspace/isaaclab
    log_dir_path = os.path.abspath(os.path.join(log_dir_path, "docker_tutorial"))
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    print(f"[INFO] 将实验日志记录到目录: {log_dir_path}")

    # 初始化模拟上下文
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 播放模拟器
    sim.reset()
    # 现在我们准备好了！
    print("[INFO]: 设置完成...")

    # 准备计算模拟时间
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0

    # 打开日志文件
    with open(os.path.join(log_dir_path, "log.txt"), "w") as log_file:
        # 模拟物理过程
        while simulation_app.is_running():
            log_file.write(f"{sim_time}" + "\n")
            # 执行一步模拟
            sim.step()
            sim_time += sim_dt


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
