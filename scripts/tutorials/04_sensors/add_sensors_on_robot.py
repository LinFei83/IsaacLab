# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示如何为机器人添加和模拟机载传感器。

我们在四足机器人ANYmal-C（ANYbotics）上添加以下传感器：

* USD-Camera: 这是一个安装在机器人基座上的相机传感器。
* Height Scanner: 这是一个安装在机器人基座上的高度扫描传感器。
* Contact Sensor: 这是一个安装在机器人脚部的接触传感器。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras

"""

"""首先启动Isaac Sim模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="教程：在机器人上添加传感器。")
parser.add_argument("--num_envs", type=int, default=2, help="要生成的环境数量。")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下。"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

##
# 预定义配置
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """设计带有传感器的场景。"""

    # 地面平面
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 机器人
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 传感器配置
    # 相机传感器配置
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,  # 更新周期为0.1秒
        height=480,  # 图像高度
        width=640,  # 图像宽度
        data_types=["rgb", "distance_to_image_plane"],  # 数据类型包括RGB图像和距离图像
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    
    # 高度扫描传感器配置
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,  # 更新周期为0.02秒
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 传感器位置偏移
        ray_alignment="yaw",  # 射线对齐方式
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 扫描网格模式配置
        debug_vis=True,  # 启用调试可视化
        mesh_prim_paths=["/World/defaultGroundPlane"],  # 网格路径
    )
    
    # 接触力传感器配置
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行模拟器。"""
    # 定义模拟步进
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 模拟物理过程
    while simulation_app.is_running():
        # 重置
        if count % 500 == 0:
            # 重置计数器
            count = 0
            # 重置场景实体
            # 根状态
            # 我们通过原点偏移根状态，因为状态是在模拟世界坐标系中编写的
            # 如果不这样做，机器人将在模拟世界的(0, 0, 0)处生成
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # 设置带有一些噪声的关节位置
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # 清除内部缓冲区
            scene.reset()
            print("[INFO]: 重置机器人状态...")
            
        # 对机器人应用默认动作
        # -- 生成动作/命令
        targets = scene["robot"].data.default_joint_pos
        # -- 对机器人应用动作
        scene["robot"].set_joint_position_target(targets)
        # -- 将数据写入模拟器
        scene.write_data_to_sim()
        # 执行步进
        sim.step()
        # 更新模拟时间
        sim_time += sim_dt
        count += 1
        # 更新缓冲区
        scene.update(sim_dt)

        # 打印传感器信息
        print("-------------------------------")
        print(scene["camera"])
        print("接收到的RGB图像形状: ", scene["camera"].data.output["rgb"].shape)
        print("接收到的深度图像形状: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(scene["height_scanner"])
        print("接收到的最大高度值: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")
        print(scene["contact_forces"])
        print("接收到的最大接触力: ", torch.max(scene["contact_forces"].data.net_forces_w).item())


def main():
    """主函数。"""

    # 初始化模拟上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # 设计场景
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # 运行模拟器
    sim.reset()
    # 现在准备就绪！
    print("[INFO]: 设置完成...")
    # 运行模拟器
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()