# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本展示了如何使用Isaac Lab框架中的光线投射相机传感器。

相机传感器基于使用Warp内核对静态网格进行光线投射。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ray_caster_camera.py

"""

"""首先启动Isaac Sim模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="本脚本演示如何使用光线投射相机传感器。")
parser.add_argument("--num_envs", type=int, default=16, help="要生成的环境数量。")
parser.add_argument("--save", action="store_true", default=False, help="将获取的数据保存到磁盘。")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下。"""

import os
import torch

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import project_points, unproject_depth


def define_sensor() -> RayCasterCamera:
    """定义要添加到场景中的光线投射相机传感器。"""
    # 相机基础帧
    # 与USD相机不同，我们将传感器关联到这些位置的prim。
    # 这意味着传感器的父prim是位于此位置的prim。
    prim_utils.create_prim("/World/Origin_00/CameraSensor", "Xform")
    prim_utils.create_prim("/World/Origin_01/CameraSensor", "Xform")

    # 设置相机传感器
    camera_cfg = RayCasterCameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",  # 传感器的prim路径，使用正则表达式匹配多个位置
        mesh_prim_paths=["/World/ground"],  # 网格prim路径
        update_period=0.1,  # 更新周期为0.1秒
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),  # 位置和旋转偏移
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],  # 数据类型
        debug_vis=True,  # 启用调试可视化
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,  # 焦距
            horizontal_aperture=20.955,  # 水平光圈
            height=480,  # 图像高度
            width=640,  # 图像宽度
        ),
    )
    # 创建相机
    camera = RayCasterCamera(cfg=camera_cfg)

    return camera


def design_scene():
    """设计场景。"""
    # 填充场景
    # -- 粗糙地形
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/ground", cfg)
    # -- 灯光
    cfg = sim_utils.DistantLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    # -- 传感器
    camera = define_sensor()

    # 返回场景信息
    scene_entities = {"camera": camera}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """运行模拟器。"""
    # 提取实体以简化表示法
    camera: RayCasterCamera = scene_entities["camera"]

    # 创建复制器写入器
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # 设置姿态：有两种方式可以设置相机的姿态。
    # -- 选项1：使用视图设置姿态
    eyes = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)  # 相机位置
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)  # 目标位置
    camera.set_world_poses_from_view(eyes, targets)
    # -- 选项2：使用ROS设置姿态
    # position = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    # orientation = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]], device=sim.device)
    # camera.set_world_poses(position, orientation, indices=[0], convention="ros")

    # 模拟物理过程
    while simulation_app.is_running():
        # 步进模拟
        sim.step()
        # 更新相机数据
        camera.update(dt=sim.get_physics_dt())

        # 打印相机信息
        print(camera)
        print("接收到的深度图像形状: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # 提取相机数据
        if args_cli.save:
            # 提取相机数据
            camera_index = 0
            # 注意：BasicWriter仅支持以numpy格式保存数据，因此我们需要将数据转换为numpy。
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            # 提取其他信息
            single_cam_info = camera.data.info[camera_index]

            # 将数据重新打包成复制器格式，以便使用其写入器保存
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # 保存图像
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

            # 世界坐标系中的点云
            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
            )

            # 检查方法是否有效
            im_height, im_width = camera.image_shape
            # -- 将点投影到(u, v, d)
            reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices)
            reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            sim_depths = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_close(reproj_depths, sim_depths)


def main():
    """主函数。"""
    # 加载kit助手
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])
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