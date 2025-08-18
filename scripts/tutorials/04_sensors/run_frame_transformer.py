# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了FrameTransformer传感器，通过可视化它创建的帧来展示其功能。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_frame_transformer.py

"""

"""首先启动Isaac Sim模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(
    description="本脚本通过可视化FrameTransformer传感器创建的帧来检查该传感器。"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动omniverse应用
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""其余代码如下。"""

import math
import torch

import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationContext

##
# 预定义配置
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


def define_sensor() -> FrameTransformer:
    """定义要添加到场景中的FrameTransformer传感器。"""
    # 定义偏移量
    # 计算绕Z轴旋转-π/2的四元数
    rot_offset = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor(-math.pi / 2))
    # 应用旋转偏移量到位置向量
    pos_offset = math_utils.quat_apply(rot_offset, torch.tensor([0.08795, 0.01305, -0.33797]))

    # 示例：使用.*获取完整身体+LF_FOOT
    frame_transformer_cfg = FrameTransformerCfg(
        prim_path="/World/Robot/base",  # 传感器的prim路径
        target_frames=[
            # 获取机器人所有部分的帧
            FrameTransformerCfg.FrameCfg(prim_path="/World/Robot/.*"),
            # 特别指定LF_SHANK部分，并设置自定义名称和偏移量
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/Robot/LF_SHANK",
                name="LF_FOOT_USER",
                offset=OffsetCfg(pos=tuple(pos_offset.tolist()), rot=tuple(rot_offset[0].tolist())),
            ),
        ],
        debug_vis=False,  # 不启用调试可视化
    )
    frame_transformer = FrameTransformer(frame_transformer_cfg)

    return frame_transformer


def design_scene() -> dict:
    """设计场景。"""
    # 填充场景
    # -- 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- 灯光
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- 机器人
    robot = Articulation(ANYMAL_C_CFG.replace(prim_path="/World/Robot"))
    # -- 传感器
    frame_transformer = define_sensor()

    # 返回场景信息
    scene_entities = {"robot": robot, "frame_transformer": frame_transformer}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """运行模拟器。"""
    # 定义模拟步进
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 提取实体以简化表示法
    robot: Articulation = scene_entities["robot"]
    frame_transformer: FrameTransformer = scene_entities["frame_transformer"]

    # 我们一次只想要一个可视化。这个可视化器将用于逐步查看每个帧，
    # 以便用户可以验证正确的帧是否与打印到控制台的帧名称相关联
    if not args_cli.headless:
        cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
        cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # 设置帧标记的缩放比例
        transform_visualizer = VisualizationMarkers(cfg)
        # 用于连接帧的线条的调试绘制接口
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    else:
        transform_visualizer = None
        draw_interface = None

    frame_index = 0
    # 模拟物理过程
    while simulation_app.is_running():
        # 以策略控制频率（50 Hz）执行此循环
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        # 执行步进
        sim.step()
        # 更新模拟时间
        sim_time += sim_dt
        count += 1
        # 从模拟器读取数据
        robot.update(sim_dt)
        frame_transformer.update(dt=sim_dt)

        # 更改我们正在可视化的帧，以确保帧名称与帧正确关联
        if not args_cli.headless:
            if count % 50 == 0:
                # 获取帧名称
                frame_names = frame_transformer.data.target_frame_names
                # 增加帧索引
                frame_index += 1
                frame_index = frame_index % len(frame_names)
                print(f"显示帧ID {frame_index}: {frame_names[frame_index]}")

            # 可视化帧
            source_pos = frame_transformer.data.source_pos_w  # 源帧位置
            source_quat = frame_transformer.data.source_quat_w  # 源帧四元数
            target_pos = frame_transformer.data.target_pos_w[:, frame_index]  # 目标帧位置
            target_quat = frame_transformer.data.target_quat_w[:, frame_index]  # 目标帧四元数
            # 绘制帧
            transform_visualizer.visualize(
                torch.cat([source_pos, target_pos], dim=0), torch.cat([source_quat, target_quat], dim=0)
            )
            # 绘制连接帧的线条
            draw_interface.clear_lines()
            # 线条颜色
            lines_colors = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
            line_thicknesses = [5.0] * source_pos.shape[0]
            draw_interface.draw_lines(source_pos.tolist(), target_pos.tolist(), lines_colors, line_thicknesses)


def main():
    """主函数。"""
    # 加载kit助手
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # 设计场景
    scene_entities = design_scene()
    # 运行模拟器
    sim.reset()
    # 现在准备就绪！
    print("[INFO]: 设置完成...")
    # 运行模拟器
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟器
    simulation_app.close()