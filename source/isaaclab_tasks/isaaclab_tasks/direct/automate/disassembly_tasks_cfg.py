# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/AutoMate"

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@configclass
class FixedAssetCfg:
    # USD文件路径
    usd_path: str = ""
    # 直径
    diameter: float = 0.0
    # 高度
    height: float = 0.0
    # 底座高度，用于计算持有资产的质心
    base_height: float = 0.0
    # 摩擦系数
    friction: float = 0.75
    # 质量
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    # USD文件路径
    usd_path: str = ""
    # 直径，用于计算夹爪宽度
    diameter: float = 0.0
    # 高度
    height: float = 0.0
    # 摩擦系数
    friction: float = 0.75
    # 质量
    mass: float = 0.05


@configclass
class RobotCfg:
    # 机器人USD文件路径
    robot_usd: str = ""
    # Franka手指垫长度
    franka_fingerpad_length: float = 0.017608
    # 摩擦系数
    friction: float = 0.75


@configclass
class DisassemblyTask:
    # 机器人配置
    robot_cfg: RobotCfg = RobotCfg()
    # 任务名称
    name: str = ""
    # 任务持续时间（秒）
    duration_s = 5.0

    # 固定资产配置
    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    # 持有资产配置
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    # 资产尺寸
    asset_size: float = 0.0

    # 手掌到手指中心的距离
    # palm_to_finger_dist: float = 0.1034
    palm_to_finger_dist: float = 0.1134

    # 机器人相关配置
    # 手部初始位置，相对于固定资产顶端
    hand_init_pos: list = [0.0, 0.0, 0.015]
    # 手部初始位置噪声
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    # 手部初始方向（欧拉角）
    hand_init_orn: list = [3.1416, 0, 2.356]
    # 手部初始方向噪声
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # 动作相关配置
    # 是否启用单向旋转
    unidirectional_rot: bool = False

    # 固定资产相关配置（适用于所有任务）
    # 固定资产初始位置噪声
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    # 固定资产初始方向角度（度）
    fixed_asset_init_orn_deg: float = 0.0
    # 固定资产初始方向角度范围（度）
    fixed_asset_init_orn_range_deg: float = 10.0

    # 机器人轨迹路点数量
    num_point_robot_traj: int = 10


@configclass
class Peg8mm(HeldAssetCfg):
    # 插头USD文件路径
    usd_path = "plug.usd"
    # 插头OBJ文件路径
    obj_path = "plug.obj"
    # 插头直径
    diameter = 0.007986
    # 插头高度
    height = 0.050
    # 插头质量
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    # 插座USD文件路径
    usd_path = "socket.usd"
    # 插座OBJ文件路径
    obj_path = "socket.obj"
    # 插座直径
    diameter = 0.0081
    # 插座高度
    height = 0.050896
    # 插座底座高度
    base_height = 0.0


@configclass
class Extraction(DisassemblyTask):
    # 任务名称
    name = "extraction"

    # 装配件ID
    assembly_id = "00015"
    # 装配件目录
    assembly_dir = f"{ASSET_DIR}/{assembly_id}/"
    # 拆卸数据保存目录
    disassembly_dir = "disassembly_dir"
    # 记录轨迹数量
    num_log_traj = 1000

    # 固定资产配置
    fixed_asset_cfg = Hole8mm()
    # 持有资产配置
    held_asset_cfg = Peg8mm()
    # 资产尺寸
    asset_size = 8.0
    # 任务持续时间
    duration_s = 10.0

    # 插头抓取点JSON文件路径
    plug_grasp_json = f"{ASSET_DIR}/plug_grasps.json"
    # 拆卸距离JSON文件路径
    disassembly_dist_json = f"{ASSET_DIR}/disassembly_dist.json"

    # 移动夹爪的仿真步数
    move_gripper_sim_steps = 64
    # 拆卸过程的仿真步数
    disassemble_sim_steps = 64

    # 机器人相关配置
    # 手部初始位置，相对于固定资产顶端
    hand_init_pos: list = [0.0, 0.0, 0.047]
    # 手部初始位置噪声
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    # 手部初始方向（欧拉角）
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    # 手部初始方向噪声
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    # 夹爪最大开口宽度
    hand_width_max: float = 0.080

    # 固定资产相关配置（适用于所有任务）
    # 固定资产初始位置噪声
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    # 固定资产初始方向角度（度）
    fixed_asset_init_orn_deg: float = 0.0
    # 固定资产初始方向角度范围（度）
    fixed_asset_init_orn_range_deg: float = 10.0
    # 固定资产Z轴偏移量
    fixed_asset_z_offset: float = 0.1435

    # 指尖中点初始位置（相对于桌面）
    fingertip_centered_pos_initial: list = [
        0.0,
        0.0,
        0.2,
    ]
    # 指尖初始旋转（欧拉角）
    fingertip_centered_rot_initial: list = [3.141593, 0.0, 0.0]
    # 夹爪随机位置噪声
    gripper_rand_pos_noise: list = [0.05, 0.05, 0.05]
    # 夹爪随机旋转噪声（roll/pitch/yaw各±10度）
    gripper_rand_rot_noise: list = [0.174533, 0.174533, 0.174533]
    # 夹爪随机Z轴偏移量
    gripper_rand_z_offset: float = 0.05

    # 固定资产配置
    fixed_asset: ArticulationCfg = ArticulationCfg(
        # fixed_asset: RigidObjectCfg = RigidObjectCfg(
        # 固定资产在场景中的路径
        prim_path="/World/envs/env_.*/FixedAsset",
        # 固定资产模型加载配置
        spawn=sim_utils.UsdFileCfg(
            # USD文件路径
            usd_path=f"{assembly_dir}{fixed_asset_cfg.usd_path}",
            # 激活接触传感器
            activate_contact_sensors=True,
            # 刚体属性配置
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 启用重力
                disable_gravity=False,
                # 最大去穿透速度
                max_depenetration_velocity=5.0,
                # 线性阻尼
                linear_damping=0.0,
                # 角阻尼
                angular_damping=0.0,
                # 最大线速度
                max_linear_velocity=1000.0,
                # 最大角速度
                max_angular_velocity=3666.0,
                # 启用陀螺力
                enable_gyroscopic_forces=True,
                # 位置求解器迭代次数
                solver_position_iteration_count=192,
                # 速度求解器迭代次数
                solver_velocity_iteration_count=1,
                # 最大接触冲量
                max_contact_impulse=1e32,
            ),
            # 关节属性配置
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # 启用自碰撞
                enabled_self_collisions=True,
                # 固定根链接，使固定资产具有固定基座
                fix_root_link=True,
            ),
            # 质量属性配置
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            # 碰撞属性配置
            collision_props=sim_utils.CollisionPropertiesCfg(
                # 接触偏移
                contact_offset=0.005,
                # 静止偏移
                rest_offset=0.0
            ),
        ),
        # 初始状态配置
        init_state=ArticulationCfg.InitialStateCfg(
            # init_state=RigidObjectCfg.InitialStateCfg(
            # 初始位置
            pos=(0.6, 0.0, 0.05),
            # 初始旋转（四元数）
            rot=(1.0, 0.0, 0.0, 0.0),
            # 关节初始位置
            joint_pos={},
            # 关节初始速度
            joint_vel={},
        ),
        # 执行器配置
        actuators={},
    )
    # held_asset: ArticulationCfg = ArticulationCfg(
    # 持有资产配置
    held_asset: RigidObjectCfg = RigidObjectCfg(
        # 持有资产在场景中的路径
        prim_path="/World/envs/env_.*/HeldAsset",
        # 持有资产模型加载配置
        spawn=sim_utils.UsdFileCfg(
            # USD文件路径
            usd_path=f"{assembly_dir}{held_asset_cfg.usd_path}",
            # 激活接触传感器
            activate_contact_sensors=True,
            # 刚体属性配置
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 禁用重力
                disable_gravity=True,
                # 最大去穿透速度
                max_depenetration_velocity=5.0,
                # 线性阻尼
                linear_damping=0.0,
                # 角阻尼
                angular_damping=0.0,
                # 最大线速度
                max_linear_velocity=1000.0,
                # 最大角速度
                max_angular_velocity=3666.0,
                # 启用陀螺力
                enable_gyroscopic_forces=True,
                # 位置求解器迭代次数
                solver_position_iteration_count=192,
                # 速度求解器迭代次数
                solver_velocity_iteration_count=1,
                # 最大接触冲量
                max_contact_impulse=1e32,
            ),
            # 质量属性配置
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            # 碰撞属性配置
            collision_props=sim_utils.CollisionPropertiesCfg(
                # 接触偏移
                contact_offset=0.005,
                # 静止偏移
                rest_offset=0.0
            ),
        ),
        # init_state=ArticulationCfg.InitialStateCfg(
        # 初始状态配置
        init_state=RigidObjectCfg.InitialStateCfg(
            # 初始位置
            pos=(0.0, 0.4, 0.1),
            # 初始旋转（四元数）
            rot=(1.0, 0.0, 0.0, 0.0),
            # joint_pos={},
            # joint_vel={}
        ),
        # actuators={}
    )
