# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand环境配置模块。
该模块定义了Shadow Hand环境的配置类，包括基础环境配置和OpenAI风格环境配置。
"""

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg


@configclass
class EventCfg:
    """随机化配置类。
    
    该类定义了环境中各种随机化事件的配置，包括机器人、物体和场景的物理属性随机化。
    """

    # -- 机器人随机化
    # 机器人物理材质随机化
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),  # 静摩擦系数范围
            "dynamic_friction_range": (1.0, 1.0),  # 动摩擦系数范围
            "restitution_range": (1.0, 1.0),  # 恢复系数范围
            "num_buckets": 250,  # 桶数量
        },
    )
    
    # 机器人关节刚度和阻尼随机化
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 所有关节
            "stiffness_distribution_params": (0.75, 1.5),  # 刚度分布参数
            "damping_distribution_params": (0.3, 3.0),  # 阻尼分布参数
            "operation": "scale",  # 操作类型：缩放
            "distribution": "log_uniform",  # 分布类型：对数均匀分布
        },
    )
    
    # 机器人关节位置限制随机化
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 所有关节
            "lower_limit_distribution_params": (0.00, 0.01),  # 下限分布参数
            "upper_limit_distribution_params": (0.00, 0.01),  # 上限分布参数
            "operation": "add",  # 操作类型：加法
            "distribution": "gaussian",  # 分布类型：高斯分布
        },
    )
    
    # 机器人肌腱属性随机化
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),  # 所有肌腱
            "stiffness_distribution_params": (0.75, 1.5),  # 刚度分布参数
            "damping_distribution_params": (0.3, 3.0),  # 阻尼分布参数
            "operation": "scale",  # 操作类型：缩放
            "distribution": "log_uniform",  # 分布类型：对数均匀分布
        },
    )

    # -- 物体随机化
    # 物体物理材质随机化
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),  # 物体
            "static_friction_range": (0.7, 1.3),  # 静摩擦系数范围
            "dynamic_friction_range": (1.0, 1.0),  # 动摩擦系数范围
            "restitution_range": (1.0, 1.0),  # 恢复系数范围
            "num_buckets": 250,  # 桶数量
        },
    )
    
    # 物体质量缩放随机化
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),  # 物体
            "mass_distribution_params": (0.5, 1.5),  # 质量分布参数
            "operation": "scale",  # 操作类型：缩放
            "distribution": "uniform",  # 分布类型：均匀分布
        },
    )

    # -- 场景随机化
    # 重力随机化
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",  # 间隔模式
        is_global_time=True,  # 使用全局时间
        interval_range_s=(36.0, 36.0),  # 间隔时间（秒）= 步数 * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),  # 重力分布参数
            "operation": "add",  # 操作类型：加法
            "distribution": "gaussian",  # 分布类型：高斯分布
        },
    )


@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    """Shadow Hand基础环境配置类。"""
    
    # 环境参数
    decimation = 2  # 控制频率降采样
    episode_length_s = 10.0  # 回合长度（秒）
    action_space = 20  # 动作空间维度
    observation_space = 157  # 观测空间维度（完整观测）
    state_space = 0  # 状态空间维度
    asymmetric_obs = False  # 是否使用非对称观测
    obs_type = "full"  # 观测类型

    # 仿真参数
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 仿真时间步长
        render_interval=decimation,  # 渲染间隔
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,  # 弹跳阈值速度
        ),
    )
    
    # 机器人配置
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # 初始位置
            rot=(1.0, 0.0, 0.0, 0.0),  # 初始旋转（四元数）
            joint_pos={".*": 0.0},  # 所有关节初始位置
        )
    )
    
    # 驱动关节名称列表
    actuated_joint_names = [
        "robot0_WRJ1",  # 腕关节1
        "robot0_WRJ0",  # 腕关节0
        "robot0_FFJ3",  # 食指关节3
        "robot0_FFJ2",  # 食指关节2
        "robot0_FFJ1",  # 食指关节1
        "robot0_MFJ3",  # 中指关节3
        "robot0_MFJ2",  # 中指关节2
        "robot0_MFJ1",  # 中指关节1
        "robot0_RFJ3",  # 无名指关节3
        "robot0_RFJ2",  # 无名指关节2
        "robot0_RFJ1",  # 无名指关节1
        "robot0_LFJ4",  # 小指关节4
        "robot0_LFJ3",  # 小指关节3
        "robot0_LFJ2",  # 小指关节2
        "robot0_LFJ1",  # 小指关节1
        "robot0_THJ4",  # 拇指关节4
        "robot0_THJ3",  # 拇指关节3
        "robot0_THJ2",  # 拇指关节2
        "robot0_THJ1",  # 拇指关节1
        "robot0_THJ0",  # 拇指关节0
    ]
    
    # 指尖刚体名称列表
    fingertip_body_names = [
        "robot0_ffdistal",  # 食指末端
        "robot0_mfdistal",  # 中指末端
        "robot0_rfdistal",  # 无名指末端
        "robot0_lfdistal",  # 小指末端
        "robot0_thdistal",  # 拇指末端
    ]

    # 手中物体配置
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",  # 物体路径
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # USD文件路径
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # 启用运动学
                disable_gravity=False,  # 禁用重力
                enable_gyroscopic_forces=True,  # 启用陀螺力
                solver_position_iteration_count=8,  # 位置求解器迭代次数
                solver_velocity_iteration_count=0,  # 速度求解器迭代次数
                sleep_threshold=0.005,  # 休眠阈值
                stabilization_threshold=0.0025,  # 稳定化阈值
                max_depenetration_velocity=1000.0,  # 最大去穿透速度
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),  # 质量属性（密度）
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),  # 初始状态
    )
    
    # 目标物体配置（可视化标记）
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",  # 标记路径
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  # USD文件路径
                scale=(1.0, 1.0, 1.0),  # 缩放
            )
        },
    )
    
    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,  # 环境数量
        env_spacing=0.75,  # 环境间距
        replicate_physics=True,  # 复制物理
        clone_in_fabric=True  # 在Fabric中克隆
    )

    # 重置参数
    reset_position_noise = 0.01  # 重置时位置噪声范围
    reset_dof_pos_noise = 0.2  # 重置时自由度位置噪声范围
    reset_dof_vel_noise = 0.0  # 重置时自由度速度噪声范围
    
    # 奖励缩放因子
    dist_reward_scale = -10.0  # 距离奖励缩放
    rot_reward_scale = 1.0  # 旋转奖励缩放
    rot_eps = 0.1  # 旋转奖励的epsilon值
    action_penalty_scale = -0.0002  # 动作惩罚缩放
    reach_goal_bonus = 250  # 达到目标奖励
    fall_penalty = 0  # 掉落惩罚
    fall_dist = 0.24  # 掉落距离阈值
    vel_obs_scale = 0.2  # 速度观测缩放
    success_tolerance = 0.1  # 成功容忍度
    max_consecutive_success = 0  # 最大连续成功次数
    av_factor = 0.1  # 平均值因子
    act_moving_average = 1.0  # 动作移动平均
    force_torque_obs_scale = 10.0  # 力/力矩观测缩放


@configclass
class ShadowHandOpenAIEnvCfg(ShadowHandEnvCfg):
    """Shadow Hand OpenAI风格环境配置类。"""
    
    # 环境参数
    decimation = 3  # 控制频率降采样
    episode_length_s = 8.0  # 回合长度（秒）
    action_space = 20  # 动作空间维度
    observation_space = 42  # 观测空间维度
    state_space = 187  # 状态空间维度
    asymmetric_obs = True  # 是否使用非对称观测
    obs_type = "openai"  # 观测类型
    
    # 仿真参数
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,  # 仿真时间步长
        render_interval=decimation,  # 渲染间隔
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,  # 弹跳阈值速度
            gpu_max_rigid_contact_count=2**23,  # GPU最大刚体接触数量
            gpu_max_rigid_patch_count=2**23,  # GPU最大刚体补丁数量
        ),
    )
    
    # 重置参数
    reset_position_noise = 0.01  # 重置时位置噪声范围
    reset_dof_pos_noise = 0.2  # 重置时自由度位置噪声范围
    reset_dof_vel_noise = 0.0  # 重置时自由度速度噪声范围
    
    # 奖励缩放因子
    dist_reward_scale = -10.0  # 距离奖励缩放
    rot_reward_scale = 1.0  # 旋转奖励缩放
    rot_eps = 0.1  # 旋转奖励的epsilon值
    action_penalty_scale = -0.0002  # 动作惩罚缩放
    reach_goal_bonus = 250  # 达到目标奖励
    fall_penalty = -50  # 掉落惩罚
    fall_dist = 0.24  # 掉落距离阈值
    vel_obs_scale = 0.2  # 速度观测缩放
    success_tolerance = 0.4  # 成功容忍度
    max_consecutive_success = 50  # 最大连续成功次数
    av_factor = 0.1  # 平均值因子
    act_moving_average = 0.3  # 动作移动平均
    force_torque_obs_scale = 10.0  # 力/力矩观测缩放
    
    # 域随机化配置
    events: EventCfg = EventCfg()
    
    # 动作噪声模型：在每个时间步添加高斯噪声+偏置，偏置是在重置时采样的高斯噪声
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),  # 噪声配置
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),  # 偏置噪声配置
    )
    
    # 观测噪声模型：在每个时间步添加高斯噪声+偏置，偏置是在重置时采样的高斯噪声
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),  # 噪声配置
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),  # 偏置噪声配置
    )
