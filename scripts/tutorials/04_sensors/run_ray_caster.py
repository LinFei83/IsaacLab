# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了如何使用光线投射传感器。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster.py

"""

"""首先启动Isaac Sim模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="光线投射测试脚本")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下。"""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def define_sensor() -> RayCaster:
    """定义要添加到场景中的光线投射传感器。"""
    # 创建光线投射传感器
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin.*/ball",  # 传感器的prim路径，使用正则表达式匹配多个球体
        mesh_prim_paths=["/World/ground"],  # 网格prim路径
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),  # 网格模式配置，分辨率为0.1，大小为2.0x2.0
        ray_alignment="yaw",  # 射线对齐方式为偏航角
        debug_vis=not args_cli.headless,  # 根据是否为headless模式决定是否启用调试可视化
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    return ray_caster


def design_scene() -> dict:
    """设计场景。"""
    # 填充场景
    # -- 粗糙地形
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
    # -- 灯光
    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)

    # 创建名为"Origin1", "Origin2", "Origin3"的独立组
    # 每个组中将有一个机器人
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
    # -- 球体
    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/ball",  # 使用正则表达式匹配所有Origin组中的球体
        spawn=sim_utils.SphereCfg(
            radius=0.25,  # 球体半径
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # 刚体属性
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),  # 质量属性，质量为0.5
            collision_props=sim_utils.CollisionPropertiesCfg(),  # 碰撞属性
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # 可视化材质，蓝色
        ),
    )
    balls = RigidObject(cfg)
    # -- 传感器
    ray_caster = define_sensor()

    # 返回场景信息
    scene_entities = {"balls": balls, "ray_caster": ray_caster}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """运行模拟器。"""
    # 提取场景实体以简化表示法
    ray_caster: RayCaster = scene_entities["ray_caster"]
    balls: RigidObject = scene_entities["balls"]

    # 定义传感器的初始位置
    ball_default_state = balls.data.default_root_state.clone()
    ball_default_state[:, :3] = torch.rand_like(ball_default_state[:, :3]) * 10  # 随机设置球体位置

    # 创建一个用于重置场景的计数器
    step_count = 0
    # 模拟物理过程
    while simulation_app.is_running():
        # 重置场景
        if step_count % 250 == 0:
            # 重置球体
            balls.write_root_pose_to_sim(ball_default_state[:, :7])
            balls.write_root_velocity_to_sim(ball_default_state[:, 7:])
            # 重置传感器
            ray_caster.reset()
            # 重置计数器
            step_count = 0
        # 步进模拟
        sim.step()
        # 更新光线投射器
        with Timer(
            f"光线投射器更新，共{4} x {ray_caster.num_rays}条射线，最大高度为"
            f" {torch.max(ray_caster.data.pos_w).item():.2f}"
        ):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        # 更新计数器
        step_count += 1


def main():
    """主函数。"""
    # 加载模拟上下文
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])
    # 设计场景
    scene_entities = design_scene()
    # 运行模拟器
    sim.reset()
    # 现在准备就绪！
    print("[INFO]: 设置完成...")
    # 运行模拟器
    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()