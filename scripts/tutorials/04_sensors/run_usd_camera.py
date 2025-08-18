# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本展示了如何使用Isaac Lab框架中的相机传感器。

相机传感器通过Omniverse Replicator API创建和接口。然而，我们使用机器人或ROS约定，而不是模拟器或OpenGL约定。

.. code-block:: bash

    # 带GUI的使用方法
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --enable_cameras

    # 无头模式的使用方法
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

"""

"""首先启动Isaac Sim模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="本脚本演示如何使用相机传感器。")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="绘制由``--camera_id``指定索引的相机点云。",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="保存由``--camera_id``指定索引的相机数据。",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "用于显示点或保存相机数据的相机ID。默认为0。"
        " 视口将始终以相机0的视角初始化。"
    ),
)
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下。"""

import numpy as np
import os
import random
import torch

import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import convert_dict_to_backend


def define_sensor() -> Camera:
    """定义要添加到场景中的相机传感器。"""
    # 设置相机传感器
    # 与光线投射相机不同，我们在这些位置生成prim。
    # 这意味着相机传感器将附加到这些prim上。
    prim_utils.create_prim("/World/Origin_00", "Xform")
    prim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",  # 相机传感器的prim路径，使用正则表达式匹配多个位置
        update_period=0,  # 更新周期为0，表示每次仿真步都更新
        height=480,  # 图像高度
        width=640,  # 图像宽度
        data_types=[
            "rgb",  # RGB图像
            "distance_to_image_plane",  # 到图像平面的距离（深度）
            "normals",  # 法线
            "semantic_segmentation",  # 语义分割
            "instance_segmentation_fast",  # 实例分割
            "instance_id_segmentation_fast",  # 实例ID分割
        ],
        colorize_semantic_segmentation=True,  # 彩色化语义分割
        colorize_instance_id_segmentation=True,  # 彩色化实例ID分割
        colorize_instance_segmentation=True,  # 彩色化实例分割
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # 创建相机
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> dict:
    """设计场景。"""
    # 填充场景
    # -- 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- 灯光
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 创建场景实体字典
    scene_entities = {}

    # 用于放置物体的Xform
    prim_utils.create_prim("/World/Objects", "Xform")
    # 随机物体
    for i in range(8):
        # 采样随机位置
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # 采样随机颜色
        color = (random.random(), random.random(), random.random())
        # 选择随机prim类型
        prim_type = random.choice(["Cube", "Cone", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),  # 刚体属性
            "mass_props": sim_utils.MassPropertiesCfg(mass=5.0),  # 质量属性，质量为5.0
            "collision_props": sim_utils.CollisionPropertiesCfg(),  # 碰撞属性
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),  # 可视化材质
            "semantic_tags": [("class", prim_type)],  # 语义标签
        }
        if prim_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common_properties)
        elif prim_type == "Cone":
            shape_cfg = sim_utils.ConeCfg(radius=0.1, height=0.25, **common_properties)
        elif prim_type == "Cylinder":
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **common_properties)
        # 刚体对象
        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",  # 对象的prim路径
            spawn=shape_cfg,  # 生成配置
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),  # 初始状态配置
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)

    # 传感器
    camera = define_sensor()

    # 返回场景信息
    scene_entities["camera"] = camera
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """运行模拟器。"""
    # 提取实体以简化表示法
    camera: Camera = scene_entities["camera"]

    # 创建复制器写入器
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # 相机位置、目标、方向
    camera_positions = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)  # 相机位置
    camera_targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)  # 相机目标
    # 这些方向采用ROS约定，将使相机朝向原点
    camera_orientations = torch.tensor(  # noqa: F841
        [[-0.1759, 0.3399, 0.8205, -0.4247], [-0.4247, 0.8205, -0.3399, 0.1759]], device=sim.device
    )

    # 设置姿态：有两种方式可以设置相机的姿态。
    # -- 选项1：使用视图设置姿态
    camera.set_world_poses_from_view(camera_positions, camera_targets)
    # -- 选项2：使用ROS设置姿态
    # camera.set_world_poses(camera_positions, camera_orientations, convention="ros")

    # 用于可视化和保存的相机索引
    camera_index = args_cli.camera_id

    # 为--draw选项在is_running()循环外创建标记
    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002  # 设置点云点半径
        pc_markers = VisualizationMarkers(cfg)

    # 模拟物理过程
    while simulation_app.is_running():
        # 步进模拟
        sim.step()
        # 更新相机数据
        camera.update(dt=sim.get_physics_dt())

        # 打印相机信息
        print(camera)
        if "rgb" in camera.data.output.keys():
            print("接收到的RGB图像形状        : ", camera.data.output["rgb"].shape)
        if "distance_to_image_plane" in camera.data.output.keys():
            print("接收到的深度图像形状      : ", camera.data.output["distance_to_image_plane"].shape)
        if "normals" in camera.data.output.keys():
            print("接收到的法线形状          : ", camera.data.output["normals"].shape)
        if "semantic_segmentation" in camera.data.output.keys():
            print("接收到的语义分割形状      : ", camera.data.output["semantic_segmentation"].shape)
        if "instance_segmentation_fast" in camera.data.output.keys():
            print("接收到的实例分割形状      : ", camera.data.output["instance_segmentation_fast"].shape)
        if "instance_id_segmentation_fast" in camera.data.output.keys():
            print("接收到的实例ID分割形状    : ", camera.data.output["instance_id_segmentation_fast"].shape)
        print("-------------------------------")

        # 提取相机数据
        if args_cli.save:
            # 保存camera_index指定的相机图像
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
            # 注意：我们需要提供On-time数据以便Replicator保存图像。
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

        # 如果有GUI且传递了--draw参数，则绘制点云
        if sim.has_gui() and args_cli.draw and "distance_to_image_plane" in camera.data.output.keys():
            # 从camera_index指定的相机导出点云
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],  # 内参矩阵
                depth=camera.data.output["distance_to_image_plane"][camera_index],  # 深度数据
                position=camera.data.pos_w[camera_index],  # 相机位置
                orientation=camera.data.quat_w_ros[camera_index],  # 相机方向（ROS约定）
                device=sim.device,  # 设备
            )

            # 在前几步中，物体仍在实例化，Camera.data可能为空。
            # 如果我们尝试可视化一个空的点云，它会使模拟器崩溃，所以我们检查点云不为空。
            if pointcloud.size()[0] > 0:
                pc_markers.visualize(translations=pointcloud)


def main():
    """主函数。"""
    # 加载模拟上下文
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
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
    # 关闭模拟应用
    simulation_app.close()