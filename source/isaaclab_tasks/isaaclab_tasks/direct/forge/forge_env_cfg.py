# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)。
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# 从工厂环境配置导入观察和状态维度配置、控制配置、工厂环境配置基类、观察随机化配置
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

# 导入自定义的死区随机化函数
from .forge_events import randomize_dead_zone
# 导入 Forge 任务配置类
from .forge_tasks_cfg import ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask

# 更新观察维度配置，添加力阈值和力传感器力数据
OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

# 更新状态维度配置，添加力阈值和力传感器力数据
STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    """Forge 环境的控制配置类"""
    # 指数移动平均因子范围，用于平滑控制信号
    ema_factor_range = [0.025, 0.1]
    # 默认任务比例增益，用于控制算法
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    # 任务比例增益噪声水平，用于增加控制的鲁棒性
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    # 位置阈值噪声水平
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    # 旋转阈值噪声水平
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    # 默认死区设置，用于减少微小控制信号的影响
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    """Forge 环境的观察随机化配置类"""
    # 指尖位置噪声标准差
    fingertip_pos = 0.00025
    # 指尖旋转噪声标准差（度）
    fingertip_rot_deg = 0.1
    # 力传感器力噪声标准差
    ft_force = 1.0


@configclass
class EventCfg:
    """Forge 环境的事件配置类，定义了各种随机化事件"""
    # 物体质量随机化事件，在每次重置时执行
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # 持有资产的物理材质随机化事件，在启动时执行
    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # 固定资产的物理材质随机化事件，在启动时执行
    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: 根据资产类型设置这些值。
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    # 机器人物理材质随机化事件，在启动时执行
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # 死区阈值随机化事件，以固定时间间隔执行
    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone, mode="interval", interval_range_s=(2.0, 2.0)  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    """Forge 环境配置类，继承自工厂环境配置"""
    # 动作空间维度
    action_space: int = 7
    # 观察随机化配置
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    # 控制配置
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    # 任务配置
    task: ForgeTask = ForgeTask()
    # 事件配置
    events: EventCfg = EventCfg()

    # 力传感器数据平滑因子
    ft_smoothing_factor: float = 0.25

    # 观察数据的顺序定义
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    # 状态数据的顺序定义
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
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    """插销插入任务配置类"""
    # 任务名称
    task_name = "peg_insert"
    # 任务配置实例
    task = ForgePegInsert()
    # 回合长度（秒）
    episode_length_s = 10.0


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    """齿轮啮合任务配置类"""
    # 任务名称
    task_name = "gear_mesh"
    # 任务配置实例
    task = ForgeGearMesh()
    # 回合长度（秒）
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    """螺母螺纹任务配置类"""
    # 任务名称
    task_name = "nut_thread"
    # 任务配置实例
    task = ForgeNutThread()
    # 回合长度（秒）
    episode_length_s = 30.0
