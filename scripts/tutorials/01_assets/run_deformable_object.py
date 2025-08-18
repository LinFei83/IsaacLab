# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
该脚本演示了如何使用可变形物体并与其交互。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：与可变形物体交互。")
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
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """设计场景。
    
    Returns:
        tuple: 包含场景实体字典和原点坐标的元组
    """
    # 地面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 灯光
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # 创建名为 "Origin1"、"Origin2"、"Origin3" 的独立组
    # 每个组中将有一个机器人
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # 可变形物体
    cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=cfg)

    # 返回场景信息
    scene_entities = {"cube_object": cube_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
    """运行模拟循环。
    
    Args:
        sim: 模拟上下文对象
        entities: 场景实体字典
        origins: 场景原点坐标张量
    """
    # 提取场景实体
    # 注意：我们在这里这样做只是为了提高可读性。通常，最好直接从字典中访问实体。
    # 在下一个教程中，这个字典将被 InteractiveScene 类替换。
    cube_object = entities["cube_object"]
    # 定义模拟步进
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 可变形物体的节点运动学目标
    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone()

    # 模拟物理
    while simulation_app.is_running():
        # 重置
        if count % 250 == 0:
            # 重置计数器
            sim_time = 0.0
            count = 0

            # 重置物体的节点状态
            nodal_state = cube_object.data.default_nodal_state_w.clone()
            # 为物体应用随机姿态
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # 将节点状态写入模拟器
            cube_object.write_nodal_state_to_sim(nodal_state)

            # 将节点状态写入运动学目标并释放所有顶点
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # 重置缓冲区
            cube_object.reset()

            print("----------------------------------------")
            print("[INFO]: 重置物体状态...")

        # 更新索引为 0 和 3 的立方体的运动学目标
        # 我们通过拾取索引为 0 的顶点稍微在 z 方向上移动立方体
        nodal_kinematic_target[[0, 3], 0, 2] += 0.001
        # 将索引为 0 的顶点设置为运动学约束
        # 0: 约束, 1: 自由
        nodal_kinematic_target[[0, 3], 0, 3] = 0.0
        # 将运动学目标写入模拟器
        cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # 将内部数据写入模拟器
        cube_object.write_data_to_sim()
        # 执行步进
        sim.step()
        # 更新模拟时间
        sim_time += sim_dt
        count += 1
        # 更新缓冲区
        cube_object.update(sim_dt)
        # 打印根位置
        if count % 50 == 0:
            print(f"根位置 (在世界坐标系中): {cube_object.data.root_pos_w[:, :3]}")


def main():
    """主函数。"""
    # 加载工具助手
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 设置主摄像头
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])
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