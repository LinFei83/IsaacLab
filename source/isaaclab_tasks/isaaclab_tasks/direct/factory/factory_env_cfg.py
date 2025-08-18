# 版权所有 (c) 2022-2025, The Isaac Lab Project 开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利.
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

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert

# 观测维度配置
OBS_DIM_CFG = {
    "fingertip_pos": 3,          # 指尖位置
    "fingertip_pos_rel_fixed": 3,  # 指尖相对于固定物体的位置
    "fingertip_quat": 4,       # 指尖四元数
    "ee_linvel": 3,            # 末端执行器线速度
    "ee_angvel": 3,            # 末端执行器角速度
}

# 状态维度配置
STATE_DIM_CFG = {
    "fingertip_pos": 3,          # 指尖位置
    "fingertip_pos_rel_fixed": 3,  # 指尖相对于固定物体的位置
    "fingertip_quat": 4,       # 指尖四元数
    "ee_linvel": 3,            # 末端执行器线速度
    "ee_angvel": 3,            # 末端执行器角速度
    "joint_pos": 7,            # 关节位置
    "held_pos": 3,             # 持有物体的位置
    "held_pos_rel_fixed": 3,   # 持有物体相对于固定物体的位置
    "held_quat": 4,            # 持有物体的四元数
    "fixed_pos": 3,            # 固定物体的位置
    "fixed_quat": 4,           # 固定物体的四元数
    "task_prop_gains": 6,      # 任务比例增益
    "ema_factor": 1,           # 指数移动平均因子
    "pos_threshold": 3,        # 位置阈值
    "rot_threshold": 3,        # 旋转阈值
}


@configclass
class ObsRandCfg:
    # 固定物体位置的观测噪声
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    # 指数移动平均因子
    ema_factor = 0.2

    # 动作边界
    pos_action_bounds = [0.05, 0.05, 0.05]  # 位置动作边界
    rot_action_bounds = [1.0, 1.0, 1.0]     # 旋转动作边界

    # 动作阈值
    pos_action_threshold = [0.02, 0.02, 0.02]  # 位置动作阈值
    rot_action_threshold = [0.097, 0.097, 0.097]  # 旋转动作阈值

    # 重置时的关节位置和任务增益
    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]  # 重置时的任务比例增益
    reset_rot_deriv_scale = 10.0  # 重置时的旋转导数缩放
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]  # 默认任务比例增益

    # 零空间参数
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]  # 默认自由度位置张量
    kp_null = 10.0  # 零空间位置增益
    kd_null = 6.3246  # 零空间速度增益


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    # 环境配置
    decimation = 8  # 控制频率与仿真频率的比率
    action_space = 6  # 动作空间维度
    # num_*: 将被覆盖以对应 obs_order, state_order.
    observation_space = 21  # 观测空间维度
    state_space = 72  # 状态空间维度
    
    # 观测和状态的顺序
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    # 任务名称和配置
    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()  # 观测噪声配置
    ctrl: CtrlCfg = CtrlCfg()  # 控制配置

    # 回合长度
    episode_length_s = 10.0  # 回合长度（秒），可能需要覆盖.
    
    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",  # 使用的设备
        dt=1 / 120,  # 仿真时间步长
        gravity=(0.0, 0.0, -9.81),  # 重力
        physx=PhysxCfg(
            solver_type=1,  # 求解器类型
            max_position_iteration_count=192,  # 最大位置迭代次数，重要以避免穿透.
            max_velocity_iteration_count=1,  # 最大速度迭代次数
            bounce_threshold_velocity=0.2,  # 弹跳阈值速度
            friction_offset_threshold=0.01,  # 摩擦偏移阈值
            friction_correlation_distance=0.00625,  # 摩擦相关距离
            gpu_max_rigid_contact_count=2**23,  # GPU最大刚体接触数
            gpu_max_rigid_patch_count=2**23,  # GPU最大刚体补丁数
            gpu_collision_stack_size=2**28,  # GPU碰撞堆栈大小
            gpu_max_num_partitions=1,  # GPU最大分区数，对稳定仿真很重要.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=True)

    # 机器人配置
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",  # 机器人在场景中的路径
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",  # USD文件路径
            activate_contact_sensors=True,  # 激活接触传感器
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # 禁用重力
                max_depenetration_velocity=5.0,  # 最大去穿透速度
                linear_damping=0.0,  # 线性阻尼
                angular_damping=0.0,  # 角阻尼
                max_linear_velocity=1000.0,  # 最大线速度
                max_angular_velocity=3666.0,  # 最大角速度
                enable_gyroscopic_forces=True,  # 启用陀螺力
                solver_position_iteration_count=192,  # 求解器位置迭代次数
                solver_velocity_iteration_count=1,  # 求解器速度迭代次数
                max_contact_impulse=1e32,  # 最大接触冲量
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # 禁用自碰撞
                solver_position_iteration_count=192,  # 求解器位置迭代次数
                solver_velocity_iteration_count=1,  # 求解器速度迭代次数
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),  # 碰撞属性
        ),
        init_state=ArticulationCfg.InitialStateCfg(
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
            pos=(0.0, 0.0, 0.0),  # 初始位置
            rot=(1.0, 0.0, 0.0, 0.0),  # 初始旋转
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],  # 关节名称表达式
                stiffness=0.0,  # 刚度
                damping=0.0,  # 阻尼
                friction=0.0,  # 摩擦
                armature=0.0,  # 电枢
                effort_limit_sim=87,  # 仿真中的力限制
                velocity_limit_sim=124.6,  # 仿真中的速度限制
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],  # 关节名称表达式
                stiffness=0.0,  # 刚度
                damping=0.0,  # 阻尼
                friction=0.0,  # 摩擦
                armature=0.0,  # 电枢
                effort_limit_sim=12,  # 仿真中的力限制
                velocity_limit_sim=149.5,  # 仿真中的速度限制
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],  # 关节名称表达式
                effort_limit_sim=40.0,  # 仿真中的力限制
                velocity_limit_sim=0.04,  # 仿真中的速度限制
                stiffness=7500.0,  # 刚度
                damping=173.0,  # 阻尼
                friction=0.1,  # 摩擦
                armature=0.0,  # 电枢
            ),
        },
    )


@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"  # 任务名称
    task = PegInsert()  # 任务配置
    episode_length_s = 10.0  # 回合长度（秒）


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"  # 任务名称
    task = GearMesh()  # 任务配置
    episode_length_s = 20.0  # 回合长度（秒）


@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"  # 任务名称
    task = NutThread()  # 任务配置
    episode_length_s = 30.0  # 回合长度（秒）
