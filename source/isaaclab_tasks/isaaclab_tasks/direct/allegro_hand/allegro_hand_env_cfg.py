# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# 导入Allegro手部的机器人配置
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

# 导入仿真工具和相关配置类
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class AllegroHandEnvCfg(DirectRLEnvCfg):
    # 环境参数
    # 降频因子，控制动作执行频率与仿真频率的比例
    decimation = 4
    # 每个训练回合的时长（秒）
    episode_length_s = 10.0
    # 动作空间维度（16个关节）
    action_space = 16
    # 观测空间维度（完整观测）
    observation_space = 124  # (full)
    # 状态空间维度
    state_space = 0
    # 是否使用非对称观测
    asymmetric_obs = False
    # 观测类型
    obs_type = "full"
    
    # 仿真参数配置
    sim: SimulationCfg = SimulationCfg(
        # 仿真时间步长（1/120秒）
        dt=1 / 120,
        # 渲染间隔，与降频因子保持一致
        render_interval=decimation,
        # 物理材质配置，设置静摩擦和动摩擦系数
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # PhysX物理引擎配置
        physx=PhysxCfg(
            # 弹跳阈值速度
            bounce_threshold_velocity=0.2,
        ),
    )
    
    # 机器人配置，使用预定义的Allegro手部配置并替换场景路径
    robot_cfg: ArticulationCfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # 驱动关节名称列表（16个关节）
    actuated_joint_names = [
        "index_joint_0",    # 食指关节0
        "middle_joint_0",   # 中指关节0
        "ring_joint_0",     # 无名指关节0
        "thumb_joint_0",    # 拇指关节0
        "index_joint_1",    # 食指关节1
        "index_joint_2",    # 食指关节2
        "index_joint_3",    # 食指关节3
        "middle_joint_1",   # 中指关节1
        "middle_joint_2",   # 中指关节2
        "middle_joint_3",   # 中指关节3
        "ring_joint_1",     # 无名指关节1
        "ring_joint_2",     # 无名指关节2
        "ring_joint_3",     # 无名指关节3
        "thumb_joint_1",    # 拇指关节1
        "thumb_joint_2",    # 拇指关节2
        "thumb_joint_3",    # 拇指关节3
    ]
    
    # 手指尖端刚体名称列表
    fingertip_body_names = [
        "index_link_3",     # 食指尖端
        "middle_link_3",    # 中指尖端
        "ring_link_3",      # 无名指尖端
        "thumb_link_3",     # 拇指尖端
    ]

    # 手中物体配置
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        # 物体在场景中的路径
        prim_path="/World/envs/env_.*/object",
        # 物体的USD文件配置
        spawn=sim_utils.UsdFileCfg(
            # 使用Isaac Nucleus中的立方体模型
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # 刚体属性配置
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 是否启用运动学模式（false表示使用动力学）
                kinematic_enabled=False,
                # 是否禁用重力（false表示启用重力）
                disable_gravity=False,
                # 是否启用陀螺力
                enable_gyroscopic_forces=True,
                # 位置求解器迭代次数
                solver_position_iteration_count=8,
                # 速度求解器迭代次数
                solver_velocity_iteration_count=0,
                # 睡眠阈值
                sleep_threshold=0.005,
                # 稳定化阈值
                stabilization_threshold=0.0025,
                # 最大去穿透速度
                max_depenetration_velocity=1000.0,
            ),
            # 质量属性配置，设置密度
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            # 物体缩放比例
            scale=(1.2, 1.2, 1.2),
        ),
        # 初始状态配置
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    # 目标物体可视化配置
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        # 标记在场景中的路径
        prim_path="/Visuals/goal_marker",
        # 标记配置字典
        markers={
            "goal": sim_utils.UsdFileCfg(
                # 使用与手中物体相同的USD文件
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                # 缩放比例
                scale=(1.2, 1.2, 1.2),
            )
        },
    )
    
    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # 并行环境数量（8192个）
        num_envs=8192,
        # 环境间距
        env_spacing=0.75,
        # 是否复制物理属性
        replicate_physics=True,
        # 是否在Fabric中克隆
        clone_in_fabric=True
    )
    
    # 重置参数
    # 重置时位置噪声范围
    reset_position_noise = 0.01  # range of position at reset
    # 重置时关节位置噪声范围
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    # 重置时关节速度噪声范围
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    
    # 奖励缩放因子
    # 距离奖励缩放因子
    dist_reward_scale = -10.0
    # 旋转奖励缩放因子
    rot_reward_scale = 1.0
    # 旋转奖励的epsilon值
    rot_eps = 0.1
    # 动作惩罚缩放因子
    action_penalty_scale = -0.0002
    # 达到目标奖励
    reach_goal_bonus = 250
    # 掉落惩罚
    fall_penalty = 0
    # 掉落距离阈值
    fall_dist = 0.24
    # 速度观测缩放因子
    vel_obs_scale = 0.2
    # 成功容忍度
    success_tolerance = 0.2
    # 最大连续成功次数
    max_consecutive_success = 0
    # 平均值因子
    av_factor = 0.1
    # 动作移动平均值
    act_moving_average = 1.0
    # 力/力矩观测缩放因子
    force_torque_obs_scale = 10.0
