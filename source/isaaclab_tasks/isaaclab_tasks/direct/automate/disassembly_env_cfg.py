# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .disassembly_tasks_cfg import ASSET_DIR, Extraction

OBS_DIM_CFG = {
    "joint_pos": 7,
    "fingertip_pos": 3,
    "fingertip_quat": 4,
    "fingertip_goal_pos": 3,
    "fingertip_goal_quat": 4,
    "delta_pos": 3,
}

STATE_DIM_CFG = {
    "joint_pos": 7,
    "joint_vel": 7,
    "fingertip_pos": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "fingertip_goal_pos": 3,
    "fingertip_goal_quat": 4,
    "held_pos": 3,
    "held_quat": 4,
    "delta_pos": 3,
}


@configclass
class ObsRandCfg:
    # 固定资产位置的噪声配置
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    # 指数移动平均因子
    ema_factor = 0.2

    # 位置动作边界
    pos_action_bounds = [0.1, 0.1, 0.1]
    # 旋转动作边界
    rot_action_bounds = [0.01, 0.01, 0.01]

    # 位置动作阈值
    pos_action_threshold = [0.01, 0.01, 0.01]
    # 旋转动作阈值
    rot_action_threshold = [0.01, 0.01, 0.01]

    # 重置时的关节位置
    reset_joints = [0.0, 0.0, 0.0, -1.870, 0.0, 1.8675, 0.785398]
    # 重置时的任务比例增益
    reset_task_prop_gains = [1000, 1000, 1000, 50, 50, 50]
    # reset_rot_deriv_scale = 1.0
    # default_task_prop_gains = [1000, 1000, 1000, 50, 50, 50]
    # reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    # 重置时的旋转导数缩放因子
    reset_rot_deriv_scale = 10.0
    # 默认任务比例增益
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # 零空间参数
    # 默认自由度位置张量
    default_dof_pos_tensor = [0.0, 0.0, 0.0, -1.870, 0.0, 1.8675, 0.785398]
    # 零空间比例增益
    kp_null = 10.0
    # 零空间微分增益
    kd_null = 6.3246


@configclass
class DisassemblyEnvCfg(DirectRLEnvCfg):
    # 控制决策频率与物理仿真的比率
    decimation = 8
    # 动作空间维度
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    # 观察空间维度
    observation_space = 24
    # 状态空间维度
    state_space = 44
    # 观察顺序配置
    obs_order: list = [
        "joint_pos",          # 关节位置
        "fingertip_pos",      # 指尖位置
        "fingertip_quat",     # 指尖四元数
        "fingertip_goal_pos", # 指尖目标位置
        "fingertip_goal_quat",# 指尖目标四元数
        "delta_pos",          # 位置差值
    ]
    # 状态顺序配置
    state_order: list = [
        "joint_pos",          # 关节位置
        "joint_vel",          # 关节速度
        "fingertip_pos",      # 指尖位置
        "fingertip_quat",     # 指尖四元数
        "ee_linvel",          # 末端线速度
        "ee_angvel",          # 末端角速度
        "fingertip_goal_pos", # 指尖目标位置
        "fingertip_goal_quat",# 指尖目标四元数
        "held_pos",           # 持有物体位置
        "held_quat",          # 持有物体四元数
        "delta_pos",          # 位置差值
    ]

    # 任务名称：extraction(提取), peg_insertion(插销), gear_meshing(齿轮啮合), nut_threading(螺母螺纹)
    task_name: str = "extraction"
    # 任务配置字典
    tasks: dict = {"extraction": Extraction()}
    # 观察噪声配置
    obs_rand: ObsRandCfg = ObsRandCfg()
    # 控制配置
    ctrl: CtrlCfg = CtrlCfg()

    # episode_length_s = 10.0  # Probably need to override.
    # 回合长度(秒)
    episode_length_s = 5.0
    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        # 使用的设备
        device="cuda:0",
        # 仿真时间步长
        dt=1 / 120,
        # 重力设置
        gravity=(0.0, 0.0, -9.81),
        # PhysX物理引擎配置
        physx=PhysxCfg(
            # 求解器类型
            solver_type=1,
            # 最大位置迭代次数，重要参数以避免穿透
            max_position_iteration_count=192,
            # 最大速度迭代次数
            max_velocity_iteration_count=1,
            # 弹跳阈值速度
            bounce_threshold_velocity=0.2,
            # 摩擦偏移阈值
            friction_offset_threshold=0.01,
            # 摩擦相关距离
            friction_correlation_distance=0.00625,
            # GPU最大刚体接触数量
            gpu_max_rigid_contact_count=2**23,
            # GPU最大刚体补丁数量
            gpu_max_rigid_patch_count=2**23,
            # GPU最大分区数，对稳定仿真很重要
            gpu_max_num_partitions=1,
        ),
        # 物理材质配置
        physics_material=RigidBodyMaterialCfg(
            # 静摩擦系数
            static_friction=1.0,
            # 动摩擦系数
            dynamic_friction=1.0,
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # 环境数量
        num_envs=128,
        # 环境间距
        env_spacing=2.0
    )

    # 机器人配置
    robot = ArticulationCfg(
        # 机器人在场景中的路径
        prim_path="/World/envs/env_.*/Robot",
        # 机器人模型加载配置
        spawn=sim_utils.UsdFileCfg(
            # 机器人USD文件路径
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
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
            # 关节属性配置
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # 启用自碰撞
                enabled_self_collisions=False,
                # 位置求解器迭代次数
                solver_position_iteration_count=192,
                # 速度求解器迭代次数
                solver_velocity_iteration_count=1,
            ),
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
            # 关节初始位置
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            # 位置初始化
            pos=(0.0, 0.0, 0.0),
            # 旋转初始化(四元数)
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        # 执行器配置
        actuators={
            # 机械臂前4个关节执行器配置
            "panda_arm1": ImplicitActuatorCfg(
                # 关节名称表达式
                joint_names_expr=["panda_joint[1-4]"],
                # 刚度
                stiffness=0.0,
                # 阻尼
                damping=0.0,
                # 摩擦
                friction=0.0,
                # 电枢
                armature=0.0,
                # 力限制
                effort_limit=87,
                # 速度限制
                velocity_limit=124.6,
            ),
            # 机械臂后3个关节执行器配置
            "panda_arm2": ImplicitActuatorCfg(
                # 关节名称表达式
                joint_names_expr=["panda_joint[5-7]"],
                # 刚度
                stiffness=0.0,
                # 阻尼
                damping=0.0,
                # 摩擦
                friction=0.0,
                # 电枢
                armature=0.0,
                # 力限制
                effort_limit=12,
                # 速度限制
                velocity_limit=149.5,
            ),
            # 机械手执行器配置
            "panda_hand": ImplicitActuatorCfg(
                # 关节名称表达式
                joint_names_expr=["panda_finger_joint[1-2]"],
                # 力限制
                effort_limit=40.0,
                # 速度限制
                velocity_limit=0.04,
                # 刚度
                stiffness=7500.0,
                # 阻尼
                damping=173.0,
                # 摩擦
                friction=0.1,
                # 电枢
                armature=0.0,
            ),
        },
    )
