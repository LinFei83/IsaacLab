# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback机器人的配置文件
该文件定义了Leatherback机器人在Isaac Lab仿真环境中的完整配置，包括物理属性、关节设置和执行器配置
"""

import os
import isaaclab.sim as sim_utils  # Isaac Lab仿真工具模块
from isaaclab.actuators import ImplicitActuatorCfg  # 隐式执行器配置类
from isaaclab.assets import ArticulationCfg  # 关节机器人配置类

# 获取工作空间根目录的绝对路径
# 通过当前文件位置向上查找4级目录来定位工作空间根目录
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))

# USD文件路径，确保跨平台兼容性
# USD (Universal Scene Description) 是Pixar开发的3D场景描述格式，用于定义机器人的3D模型
USD_PATH = os.path.join(WORKSPACE_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "direct", "leatherback", "custom_assets", "leatherback_simple_better.usd")

# Leatherback机器人的主配置对象
# 定义了机器人在仿真环境中的所有物理和行为特性
LEATHERBACK_CFG = ArticulationCfg(
    # 生成配置：定义如何在仿真中创建机器人实例
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,  # 3D模型文件路径
        # 刚体物理属性配置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,  # 启用刚体物理
            max_linear_velocity=1000.0,  # 最大线性速度 (m/s)
            max_angular_velocity=1000.0,  # 最大角速度 (rad/s)
            max_depenetration_velocity=100.0,  # 最大穿透恢复速度
            enable_gyroscopic_forces=True,  # 启用陀螺仪力效应
        ),
        # 关节机器人根部属性配置
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 禁用自碰撞检测
            solver_position_iteration_count=4,  # 位置求解器迭代次数
            solver_velocity_iteration_count=0,  # 速度求解器迭代次数
            sleep_threshold=0.005,  # 休眠阈值，低于此速度时物体进入休眠状态
            stabilization_threshold=0.001,  # 稳定化阈值
        ),
    ),
    # 初始状态配置：定义机器人在仿真开始时的位置和关节角度
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # 初始位置 (x, y, z)，z=0.05表示离地面5cm
        # 各关节的初始角度设置（单位：弧度）
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,    # 前左轮与转向节连接关节
            "Wheel__Knuckle__Front_Right": 0.0,   # 前右轮与转向节连接关节
            "Wheel__Upright__Rear_Right": 0.0,    # 后右轮与立柱连接关节
            "Wheel__Upright__Rear_Left": 0.0,     # 后左轮与立柱连接关节
            "Knuckle__Upright__Front_Right": 0.0, # 前右转向节与立柱连接关节（转向关节）
            "Knuckle__Upright__Front_Left": 0.0,  # 前左转向节与立柱连接关节（转向关节）
        },
    ),
    # 执行器配置：定义机器人的驱动系统
    actuators={
        # 油门执行器：控制所有车轮的驱动
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],  # 匹配所有以"Wheel"开头的关节
            effort_limit=40000.0,  # 最大输出力矩 (N·m)
            velocity_limit=100.0,  # 最大角速度 (rad/s)
            stiffness=0.0,         # 刚度系数，0表示纯速度控制
            damping=100000.0,      # 阻尼系数，用于稳定控制
        ),
        # 转向执行器：控制前轮转向
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],  # 匹配前轮转向关节
            effort_limit=40000.0,  # 最大输出力矩 (N·m)
            velocity_limit=100.0,  # 最大角速度 (rad/s)
            stiffness=1000.0,      # 刚度系数，用于位置控制
            damping=0.0,           # 阻尼系数，0表示纯位置控制
        ),
    },
)
