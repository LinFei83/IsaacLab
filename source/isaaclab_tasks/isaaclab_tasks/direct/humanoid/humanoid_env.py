# 版权所有 (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利.
#
# SPDX-License-Identifier: BSD-3-Clause

# 启用Python 3.7+的类型注解功能
from __future__ import annotations

# 导入人形机器人的默认配置
from isaaclab_assets import HUMANOID_CFG

# 导入仿真工具和相关配置类
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# 导入基础的运动环境类
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # 环境配置参数
    # 每个episode的持续时间（秒）
    episode_length_s = 15.0
    # 控制决策频率与仿真频率的降采样因子
    decimation = 2
    # 动作缩放因子
    action_scale = 1.0
    # 动作空间维度（关节数量）
    action_space = 21
    # 观测空间维度
    observation_space = 75
    # 状态空间维度（0表示不使用状态空间）
    state_space = 0

    # 仿真配置
    # 仿真器配置，dt为仿真时间步长，render_interval为渲染间隔
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # 地形配置
    terrain = TerrainImporterCfg(
        # 地形的USD路径
        prim_path="/World/ground",
        # 地形类型为平面
        terrain_type="plane",
        # 碰撞组设置为-1（默认组）
        collision_group=-1,
        # 物理材质配置
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # 摩擦系数组合模式为平均值
            friction_combine_mode="average",
            # 恢复系数组合模式为平均值
            restitution_combine_mode="average",
            # 静摩擦系数
            static_friction=1.0,
            # 动摩擦系数
            dynamic_friction=1.0,
            # 恢复系数（弹性）
            restitution=0.0,
        ),
        # 是否启用调试可视化
        debug_vis=False,
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # 并行环境数量
        num_envs=4096,
        # 环境间距
        env_spacing=4.0,
        # 是否复制物理属性以提高性能
        replicate_physics=True,
        # 是否在Fabric中克隆以提高性能
        clone_in_fabric=True
    )

    # 机器人配置
    # 使用HUMANOID_CFG作为基础配置，并替换USD路径
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # 关节齿轮比配置，用于计算关节力矩
    joint_gears: list = [
        67.5000,  # lower_waist 腰部下关节
        67.5000,  # lower_waist 腰部下关节
        67.5000,  # right_upper_arm 右上臂
        67.5000,  # right_upper_arm 右上臂
        67.5000,  # left_upper_arm 左上臂
        67.5000,  # left_upper_arm 左上臂
        67.5000,  # pelvis 骨盆
        45.0000,  # right_lower_arm 右下臂
        45.0000,  # left_lower_arm 左下臂
        45.0000,  # right_thigh: x 右大腿x轴
        135.0000,  # right_thigh: y 右大腿y轴
        45.0000,  # right_thigh: z 右大腿z轴
        45.0000,  # left_thigh: x 左大腿x轴
        135.0000,  # left_thigh: y 左大腿y轴
        45.0000,  # left_thigh: z 左大腿z轴
        90.0000,  # right_knee 右膝盖
        90.0000,  # left_knee 左膝盖
        22.5,  # right_foot 右脚
        22.5,  # right_foot 右脚
        22.5,  # left_foot 左脚
        22.5,  # left_foot 左脚
    ]

    # 奖励权重配置
    # 朝向奖励权重
    heading_weight: float = 0.5
    # 直立奖励权重
    up_weight: float = 0.1

    # 成本缩放因子
    # 能量消耗成本缩放因子
    energy_cost_scale: float = 0.05
    # 动作成本缩放因子
    actions_cost_scale: float = 0.01
    # 存活奖励缩放因子
    alive_reward_scale: float = 2.0
    # 关节速度缩放因子
    dof_vel_scale: float = 0.1

    # 惩罚配置
    # 死亡惩罚值
    death_cost: float = -1.0
    # 终止高度阈值（低于此高度认为机器人死亡）
    termination_height: float = 0.8

    # 其他缩放因子
    # 角速度缩放因子
    angular_velocity_scale: float = 0.25
    # 接触力缩放因子
    contact_force_scale: float = 0.01


class HumanoidEnv(LocomotionEnv):
    # 环境配置
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        # 调用父类构造函数初始化人形机器人运动环境
        # cfg: 环境配置对象
        # render_mode: 渲染模式（如"human"表示可视化渲染）
        # **kwargs: 其他传递给父类的参数
        super().__init__(cfg, render_mode, **kwargs)
