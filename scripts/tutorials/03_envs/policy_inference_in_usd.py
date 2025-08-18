# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了在预构建的USD环境中的策略推理。

在此示例中，我们使用运动策略来控制H1机器人。该机器人使用Isaac-Velocity-Rough-H1-v0进行训练。机器人被命令以恒定速度向前移动。

.. code-block:: bash

    # 运行脚本
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：在仓库中对H1机器人进行策略推理。")
parser.add_argument("--checkpoint", type=str, help="指向导出为jit的模型检查点的路径。", required=True)

# 添加 AppLauncher cli 参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的所有内容。"""
import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY


def main():
    """主函数。"""
    # 加载训练好的jit策略
    policy_path = os.path.abspath(args_cli.checkpoint)
    # 从指定路径读取策略文件内容
    file_content = omni.client.read_file(policy_path)[2]
    # 将文件内容转换为字节流
    file = io.BytesIO(memoryview(file_content).tobytes())
    # 加载策略模型
    policy = torch.jit.load(file, map_location=args_cli.device)

    # 设置环境
    env_cfg = H1RoughEnvCfg_PLAY()
    # 设置环境数量为1
    env_cfg.scene.num_envs = 1
    # 禁用课程学习
    env_cfg.curriculum = None
    # 设置地形为USD文件
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    # 设置仿真设备
    env_cfg.sim.device = args_cli.device
    # 如果设备是CPU，则禁用fabric
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 使用策略运行推理
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            # 使用策略模型生成动作
            action = policy(obs["policy"])
            # 执行环境步骤
            obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    # 关闭模拟应用
    simulation_app.close()