# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""本脚本演示了如何在场景中生成基本图元。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(
    description="教程：使用给定的渲染模式预设查看仓库场景。"
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """主函数。"""

    # 渲染模式包括 performance（性能）、balanced（平衡）和 quality（质量）
    # 注意，CLI 参数中指定的渲染模式（--rendering_mode）优先级高于此 Render Config 设置
    rendering_mode = "performance"

    # carb 设置字典可以包含任何 rtx carb 设置，这将覆盖原生预设设置
    carb_settings = {"rtx.reflections.enabled": True}

    # 初始化渲染配置
    render_cfg = sim_utils.RenderCfg(
        rendering_mode=rendering_mode,
        carb_settings=carb_settings,
    )

    # 使用渲染配置初始化模拟上下文
    sim_cfg = sim_utils.SimulationCfg(render=render_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 将相机定位在医院大厅区域
    sim.set_camera_view([-11, -0.5, 2], [0, 0, 0.5])

    # 加载医院场景
    hospital_usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"
    cfg = sim_utils.UsdFileCfg(usd_path=hospital_usd_path)
    cfg.func("/Scene", cfg)

    # 播放模拟器
    sim.reset()

    # 现在我们准备好了！
    print("[INFO]: 设置完成...")

    # 运行模拟并查看场景
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
