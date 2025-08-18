# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了如何运行倒立摆平衡任务的RL环境。

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：运行倒立摆RL环境。")
parser.add_argument("--num_envs", type=int, default=16, help="要生成的环境数量。")

# 添加 AppLauncher cli 参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的所有内容。"""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg


def main():
    """主函数。"""
    # 创建环境配置
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # 设置RL环境
    env = ManagerBasedRLEnv(cfg=env_cfg)

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
            # 执行环境步骤
            # 与基础环境不同，env.step() 方法现在会返回一个标准的 RL 元组：(obs, rew, terminated, truncated, info)，包含了奖励和终止信号。
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
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