# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
该脚本演示了如何在模拟器中使用操作空间控制器（OSC）。

OSC控制器可以配置为不同的模式。它使用PhysX计算的动力学量，如雅可比矩阵和
质量矩阵。

.. code-block:: bash

    # 使用方法
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_osc.py

"""

"""首先启动Isaac Sim模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="操作空间控制器使用教程。")
parser.add_argument("--num_envs", type=int, default=128, help="要生成的环境数量。")
# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余代码如下。"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

##
# 预定义配置
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


@configclass
class SceneCfg(InteractiveSceneCfg):
    """带倾斜墙的简单场景配置。"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 倾斜的墙
    tilted_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TiltedWall",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.5, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6 + 0.085, 0.0, 0.3), rot=(0.9238795325, 0.0, -0.3826834324, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/TiltedWall",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.spawn.rigid_props.disable_gravity = True


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行模拟循环。

    Args:
        sim: (SimulationContext) 模拟上下文。
        scene: (InteractiveScene) 交互场景。
    """

    # 提取场景实体以提高可读性。
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    # 获取末端执行器和机械臂关节的索引
    ee_frame_name = "panda_leftfinger"
    arm_joint_names = ["panda_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # 创建OSC
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # 标记器
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # 定义机械臂的目标
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [0.6, 0.15, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.6, -0.3, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.8, 0.0, 0.5, 0.0, 0.92387953, 0.0, 0.38268343],
        ],
        device=sim.device,
    )
    ee_goal_wrench_set_tilted_task = torch.tensor(
        [
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )
    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
        ],
        device=sim.device,
    )
    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task, kp_set_task], dim=-1)

    # 定义模拟步进
    sim_dt = sim.get_physics_dt()

    # 更新现有缓冲区
    # 注意：我们需要在控制器的第一个步骤之前更新缓冲区。
    robot.update(dt=sim_dt)

    # 获取机器人软关节限制的中心
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

    # 获取更新后的状态
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)

    # 跟踪给定的目标命令
    current_goal_idx = 0  # 机械臂的当前目标索引
    command = torch.zeros(
        scene.num_envs, osc.action_dim, device=sim.device
    )  # 通用目标命令，可以是姿态、位置、力等。
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # 体框架中的目标姿态
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # 世界框架中的目标姿态（用于标记器）

    # 将关节努力设置为零
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    count = 0
    # 模拟循环
    while simulation_app.is_running():
        # 每500步重置一次
        if count % 500 == 0:
            # 将关节状态重置为默认值
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)  # 在初始步骤中设置零扭矩
            robot.write_data_to_sim()
            robot.reset()
            # 重置接触传感器
            contact_forces.reset()
            # 重置目标姿态
            robot.update(sim_dt)
            _, _, _, ee_pose_b, _, _, _, _, _, _ = update_states(
                sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces
            )  # 在重置时，雅可比矩阵不会更新到最新状态
            command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                sim, scene, osc, root_pose_w, ee_target_set, current_goal_idx
            )
            # 设置osc命令
            osc.reset()
            command, task_frame_pose_b = convert_to_task_frame(osc, command=command, ee_target_pose_b=ee_target_pose_b)
            osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
        else:
            # 获取更新后的状态
            (
                jacobian_b,
                mass_matrix,
                gravity,
                ee_pose_b,
                ee_vel_b,
                root_pose_w,
                ee_pose_w,
                ee_force_b,
                joint_pos,
                joint_vel,
            ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids, contact_forces)
            # 计算关节命令
            joint_efforts = osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                current_ee_force_b=ee_force_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )
            # 应用动作
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

        # 更新标记器位置
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        # 执行步骤
        sim.step(render=True)
        # 更新机器人缓冲区
        robot.update(sim_dt)
        # 更新缓冲区
        scene.update(sim_dt)
        # 更新模拟时间
        count += 1


# 更新机器人状态
def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    contact_forces,
):
    """更新机器人状态。

    Args:
        sim: (SimulationContext) 模拟上下文。
        scene: (InteractiveScene) 交互场景。
        robot: (Articulation) 机器人关节。
        ee_frame_idx: (int) 末端执行器框架索引。
        arm_joint_ids: (list[int]) 机械臂关节索引。
        contact_forces: (ContactSensor) 接触传感器。

    Returns:
        jacobian_b (torch.tensor): 体框架中的雅可比矩阵。
        mass_matrix (torch.tensor): 质量矩阵。
        gravity (torch.tensor): 重力向量。
        ee_pose_b (torch.tensor): 体框架中的末端执行器姿态。
        ee_vel_b (torch.tensor): 体框架中的末端执行器速度。
        root_pose_w (torch.tensor): 世界框架中的根姿态。
        ee_pose_w (torch.tensor): 世界框架中的末端执行器姿态。
        ee_force_b (torch.tensor): 体框架中的末端执行器力。
        joint_pos (torch.tensor): 关节位置。
        joint_vel (torch.tensor): 关节速度。

    Raises:
        ValueError: 未定义的目标类型。
    """
    # 从模拟中获取与动力学相关的量
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    # 将雅可比矩阵从世界框架转换到根框架
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # 计算末端执行器的当前姿态
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # 计算末端执行器的当前速度
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # 提取世界框架中的末端执行器速度
    root_vel_w = robot.data.root_vel_w  # 提取世界框架中的根速度
    relative_vel_w = ee_vel_w - root_vel_w  # 计算世界框架中的相对速度
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # 从世界框架到根框架
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # 计算接触力
    ee_force_w = torch.zeros(scene.num_envs, 3, device=sim.device)
    sim_dt = sim.get_physics_dt()
    contact_forces.update(sim_dt)  # 更新接触传感器
    # 通过在最后四个时间步上取平均值（即平滑）来计算接触力，并
    # 取三个表面的最大值，因为只有一个应该是感兴趣的接触
    ee_force_w, _ = torch.max(torch.mean(contact_forces.data.net_forces_w_history, dim=1), dim=1)

    # 这是一个简化，仅用于测试。
    ee_force_b = ee_force_w

    # 获取关节位置和速度
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    )


# 更新目标命令
def update_target(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    ee_target_set: torch.tensor,
    current_goal_idx: int,
):
    """更新操作空间控制器的目标。

    Args:
        sim: (SimulationContext) 模拟上下文。
        scene: (InteractiveScene) 交互场景。
        osc: (OperationalSpaceController) 操作空间控制器。
        root_pose_w: (torch.tensor) 世界框架中的根姿态。
        ee_target_set: (torch.tensor) 末端执行器目标集。
        current_goal_idx: (int) 当前目标索引。

    Returns:
        command (torch.tensor): 更新后的目标命令。
        ee_target_pose_b (torch.tensor): 体框架中更新后的目标姿态。
        ee_target_pose_w (torch.tensor): 世界框架中更新后的目标姿态。
        next_goal_idx (int): 下一个目标索引。

    Raises:
        ValueError: 未定义的目标类型。
    """

    # 更新末端执行器的期望命令
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    # 更新末端执行器的期望姿态
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass  # 对于力控制，ee_target_pose_b可以保持在根框架，重要的是ee_target_b
        else:
            raise ValueError("在update_target()中未定义的目标类型。")

    # 更新世界框架中的目标期望姿态（用于标记器）
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


# 将目标命令转换到任务框架
def convert_to_task_frame(osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor):
    """将目标命令转换到任务框架。

    Args:
        osc: OperationalSpaceController对象。
        command: 要转换的命令。
        ee_target_pose_b: 体框架中的目标姿态。

    Returns:
        command (torch.tensor): 任务框架中的目标命令。
        task_frame_pose_b (torch.tensor): 任务框架中的目标姿态。

    Raises:
        ValueError: 未定义的目标类型。
    """
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        elif target_type == "wrench_abs":
            # 这些已经在目标框架中定义了ee_goal_wrench_set_tilted_task（因为它
            # 更容易），所以不进行转换
            cmd_idx += 6
        else:
            raise ValueError("在_convert_to_task_frame()中未定义的目标类型。")

    return command, task_frame_pose_b


def main():
    """主函数。"""
    # 加载kit助手
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主摄像头
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # 设计场景
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # 启动模拟器
    sim.reset()
    # 现在我们准备好了！
    print("[INFO]: 设置完成...")
    # 运行模拟器
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()
