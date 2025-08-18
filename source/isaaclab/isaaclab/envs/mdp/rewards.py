# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用于启用奖励函数的通用函数集合。

这些函数可以传递给 :class:`isaaclab.managers.RewardTermCfg` 对象，
以包含由该函数引入的奖励项。

本模块提供了多种类型的奖励和惩罚函数：
- 一般奖励：存活奖励、终止惩罚等
- 根部（基座）惩罚：线性/角速度、姿态、高度等
- 关节惩罚：关节力矩、速度、加速度、位置限制等  
- 动作惩罚：动作变化率、动作幅度等
- 接触传感器：期望/非期望接触检测
- 速度跟踪奖励：线性和角速度命令跟踪
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
通用奖励函数。
这些函数提供基本的奖励机制，如存活奖励和终止惩罚。
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活奖励函数。
    
    为每个未终止的环境实例提供奖励，鼓励智能体保持存活状态。
    这是强化学习中常用的基础奖励机制。
    
    Args:
        env: 基于管理器的强化学习环境实例
        
    Returns:
        torch.Tensor: 存活奖励张量，未终止为1.0，已终止为0.0
    """
    # 返回未终止状态的浮点值 (True->1.0, False->0.0)
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """对非超时终止的回合进行惩罚。
    
    当环境因为失败条件（而非时间限制）终止时给予惩罚，
    用于训练智能体避免不良行为导致的提前终止。
    
    Args:
        env: 基于管理器的强化学习环境实例
        
    Returns:
        torch.Tensor: 终止惩罚张量，已终止为1.0，未终止为0.0
    """
    # 返回终止状态的浮点值，用作惩罚信号
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """对特定的非超时终止条件进行惩罚的管理器类。

    此类用于精确控制哪些终止条件应该被惩罚，而不是简单地惩罚所有终止。
    这对于复杂环境中的细粒度奖励设计非常有用。

    参数说明：

    * attr:`term_keys`: 需要惩罚的终止条件。可以是字符串、字符串列表
      或正则表达式。默认为 ".*"，表示惩罚所有终止条件。

    奖励计算方式：
    奖励值等于非超时终止条件的数量总和。
    这意味着：
    - 如果因超时而终止，奖励为 0
    - 如果有两个终止条件同时激活，奖励为 2（即惩罚值为 2）
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # 初始化基类
        super().__init__(cfg, env)
        # 查找并存储终止条件
        term_keys = cfg.params.get("term_keys", ".*")  # 获取配置中的终止条件键名
        self._term_names = env.termination_manager.find_terms(term_keys)  # 从终止管理器中查找匹配的条件名称

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        """计算终止惩罚奖励。
        
        Args:
            env: 环境实例
            term_keys: 终止条件键名（仅为接口兼容，实际使用初始化时的配置）
            
        Returns:
            torch.Tensor: 终止惩罚值，仅对非超时终止进行惩罚
        """
        # 返回终止条件的未加权奖励（实际上是惩罚）
        reset_buf = torch.zeros(env.num_envs, device=env.device)  # 初始化重置缓冲区
        for term in self._term_names:
            # 累加终止条件值，以处理同一步中多个终止条件的情况
            reset_buf += env.termination_manager.get_term(term)

        # 仅对非超时终止进行惩罚（排除正常的时间限制终止）
        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
根部（基座）惩罚函数。
这些函数用于约束机器人或智能体的基座运动，包括线速度、角速度、姿态和高度等。
通过这些惩罚可以使智能体保持稳定和适当的运动模式。
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2平方核惩罚z轴基座线速度。
    
    此函数用于惩罚机器人在垂直方向（z轴）上的运动，通常用于约束机器人保持在地面上，
    避免不必要的跳跃或垂直运动。L2平方核提供平滑的惩罚梯度。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: z轴线速度的L2平方惩罚值
    """
    # 提取所需的数量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算z轴线速度的平方，作为惩罚项
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2平方核惩罚xy轴基座角速度。
    
    此函数用于惩罚机器人在roll和pitch方向上的旋转运动，帮助机器人保持稳定的姿态。
    通常用于四足机器人或双足机器人的平衡控制。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: xy轴角速度的L2平方惩罚值之和
    """
    # 提取所需的数量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算x和y轴角速度平方的总和，惩罚roll和pitch旋转
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2平方核惩罚非水平基座姿态。

    通过惩罚投影重力向量的xy分量来计算此惩罚。当机器人保持水平姿态时，
    重力向量在机器人坐标系中应该指向-z方向，xy分量应该接近零。
    这是一种有效的姿态稳定方法，广泛用于移动机器人控制。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 非水平姿态的L2平方惩罚值
    """
    # 提取所需的数量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 通过惩罚投影重力向量的xy分量来约束机器人保持水平姿态
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """使用L2平方核惩罚资产高度与目标高度的偏差。

    此函数用于约束机器人在特定高度运行，常用于飞行器或四足机器人的高度控制。
    支持平坡和崎岖地形两种情况。

    注意:
        对于平坡地形，目标高度在世界坐标系中。对于崎岖地形，
        传感器读数可以调整目标高度以适应地形变化。
        
    Args:
        env: 基于管理器的强化学习环境实例
        target_height: 目标高度值
        asset_cfg: 场景实体配置，默认为"robot"
        sensor_cfg: 传感器配置，用于地形高度检测（可选）
        
    Returns:
        torch.Tensor: 高度偏差的L2平方惩罚值
    """
    # 提取所需的数量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # 使用传感器数据调整目标高度（用于崎岖地形）
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # 对于平坡地形，直接使用提供的目标高度
        adjusted_target_height = target_height
    # 计算L2平方惩罚（实际高度与目标高度的差值平方）
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2核惩罚身体的线性加速度。
    
    此函数用于约束机器人各个身体部件的加速度，促进平滑运动，减少震动。
    通常用于多关节机器人的运动控制优化。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 所有指定身体部件线性加速度的L2范数之和
    """
    # 获取关节化资产（多关节机器人）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算指定身体部件的线性加速度范数，并求和作为惩罚项
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
关节惩罚函数。
这些函数用于约束机器人关节的运动，包括关节力矩、速度、加速度和位置限制等。
通过这些惩罚可以使机器人的关节运动更加平滑、节能和安全。
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2平方核惩罚关节体上的关节力矩。

    此函数用于鼓励机器人使用更小的关节力矩，从而提高能效并减少电机负荷。
    这对于节能和设备保护非常重要。

    注意: 仅有在 :attr:`asset_cfg.joint_ids` 中配置的关节才会对该项贡献关节力矩。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 指定关节力矩的L2平方惩罚值之和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算指定关节的力矩平方和，作为惩罚项鼓励低力矩运动
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """使用L1核惩罚关节体上的关节速度。
    
    L1核相比L2核对小速度的惩罚更轻，对大速度的惩罚相对更重，
    适用于需要稀疏性约束的情况。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，必须指定 joint_ids
        
    Returns:
        torch.Tensor: 指定关节速度的L1惩罚值之和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算指定关节速度的绝对值和，使用L1范数作为惩罚
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2平方核惩罚关节体上的关节速度。

    L2平方核提供平滑的惩罚梯度，适用于鼓励平滑的关节运动，
    减少高速关节运动带来的震动和磨损。

    注意: 仅有在 :attr:`asset_cfg.joint_ids` 中配置的关节才会对该项贡献关节速度。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 指定关节速度的L2平方惩罚值之和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算指定关节速度的平方和，作为平滑运动的惩罚项
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用L2平方核惩罚关节体上的关节加速度。

    此函数约束关节加速度，鼓励平滑的加速度变化，减少机器人运动中的冲击和震动。
    这对于提高运动质量和设备寿命非常重要。

    注意: 仅有在 :attr:`asset_cfg.joint_ids` 中配置的关节才会对该项贡献关节加速度。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 指定关节加速度的L2平方惩罚值之和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算指定关节加速度的平方和，作为平滑加速度的惩罚项
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚偏离默认位置的关节位置。
    
    此函数鼓励机器人保持接近默认姿态，常用于人形机器人或四足机器人的姿态控制。
    默认位置通常是机器人的休息或中性姿态。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 关节位置偏差的L1惩罚值之和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算关节位置与默认位置的偏差
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # 返回偏差的绝对值之和，作为惩罚项
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """当关节位置超过软限制时进行惩罚。

    此函数通过计算关节位置与软限制之间差值的绝对值之和来计算惩罚。
    软限制相比硬限制更加安全，可以在接近限制时提供渐进式惩罚。
    
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 关节位置限制超出量的总和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算超出限制的约束
    # 计算低于下限的部分（负值被截断为0）
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    # 计算高于上限的部分（负值被截断为0）
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    # 返回所有超限量的总和
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当关节速度超过软限制时进行惩罚。

    此函数通过计算关节速度与软限制之间差值的绝对值之和来计算惩罚。
    软限制可以防止关节运动过快，保护机器人的安全性和稳定性。

    Args:
        env: 基于管理器的强化学习环境实例
        soft_ratio: 使用的软限制比例（通常小于1.0，提供安全缓冲）
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 关节速度限制超出量的总和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算超出限制的约束
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])  # 关节速度的绝对值
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio  # 软限制乘以比例系数
    )
    # 限制最大误差为每个关节 1 rad/s，以避免巨大的惩罚
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
动作惩罚函数。
这些函数用于约束智能体的动作输出，包括动作幅度、变化率和力矩限制等。
通过这些惩罚可以使智能体的动作更加平滑、节能和安全。
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """当应用力矩超过限制时进行惩罚。

    此函数通过计算应用功矩与限制之间差值的绝对值之和来计算惩罚。
    这可以防止电机过载，保护机器人的硬件安全。

    .. caution::
        目前仅适用于显式执行器，因为我们手动计算应用功矩。
        对于隐式执行器，我们目前无法从物理引擎中获取应用功矩。
        
    Args:
        env: 基于管理器的强化学习环境实例
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: 应用功矩超限量的总和
    """
    # 提取所需的数量（用于类型提示）
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算超出限制的约束
    # TODO: 我们需要修复这个问题以支持隐式关节。
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """使用L2平方核惩罚动作的变化率。
    
    此函数鼓励平滑的动作变化，减少动作的突变和震动。
    这对于提高机器人运动的平滑性和稳定性非常重要。
    
    Args:
        env: 基于管理器的强化学习环境实例
        
    Returns:
        torch.Tensor: 动作变化率的L2平方惩罚值
    """
    # 计算当前动作与上一步动作之间的差值平方和
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """使用L2平方核惩罚动作幅度。
    
    此函数鼓励使用较小的动作幅度，从而提高能效并减少能耗。
    常用于鼓励节能的控制策略。
    
    Args:
        env: 基于管理器的强化学习环境实例
        
    Returns:
        torch.Tensor: 动作幅度的L2平方惩罚值
    """
    # 计算当前动作的平方和，作为动作幅度的惩罚
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
接触传感器相关函数。
这些函数用于处理接触传感器数据，包括期望接触和非期望接触的检测。
广泛用于步行机器人、操作机器人等需要接触反馈的应用。
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚非期望接触，计算超过阈值的违规数量。
    
    此函数用于惩罚不应该发生的接触，如机器人身体与障碍物的碰撞。
    常用于避障和安全控制。
    
    Args:
        env: 基于管理器的强化学习环境实例
        threshold: 接触力阈值，超过此值将被认为是不期望的接触
        sensor_cfg: 接触传感器配置
        
    Returns:
        torch.Tensor: 每个环境中非期望接触的数量
    """
    # 提取所需的数量（用于类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检查接触力是否超过阈值
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 对指定身体部件的接触力进行范数计算，并判断是否超过阈值
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # 对每个环境的接触进行求和
    return torch.sum(is_contact, dim=1)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """当没有期望的接触存在时进行惩罚。
    
    此函数用于鼓励维持必要的接触，如步行机器人的脚部与地面的接触。
    当所有指定的身体部件都没有接触时，将给出惩罚。
    
    Args:
        env: 基于管理器的强化学习环境实例
        sensor_cfg: 接触传感器配置
        threshold: 接触力阈值，默认为1.0
        
    Returns:
        torch.Tensor: 缺少期望接触的惩罚值（没有接触时1.0，有接触时0.0）
    """
    # 获取接触传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检查指定身体部件是否有超过阈值的接触力
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    # 检查是否所有指定部件都没有接触（全为False）
    zero_contact = (~contacts).all(dim=1)
    # 返回惩罚值：没有任何接触时1.0，否则为0.0
    return 1.0 * zero_contact


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """根据净接触力违规程度惩罚接触力。
    
    与 undesired_contacts 不同，此函数不仅考虑是否超过阈值，
    还考虑超过的幅度，提供更精细的接触力控制。
    
    Args:
        env: 基于管理器的强化学习环境实例
        threshold: 接触力阈值
        sensor_cfg: 接触传感器配置
        
    Returns:
        torch.Tensor: 超过阈值的接触力幅度总和
    """
    # 提取所需的数量（用于类型提示）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 计算违规量（超过阈值的部分）
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # 计算惩罚（仅对正值违规进行惩罚）
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
速度跟踪奖励函数。
这些函数用于奖励智能体跟踪目标速度命令，包括线速度和角速度。
广泛用于移动机器人、无人机等需要精确速度控制的应用。
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核奖励线速度命令跟踪（xy轴）。
    
    指数核提供了更平滑的奖励信号，当误差较小时给予高奖励，
    误差较大时奖励迅速下降。适用于精确的速度控制任务。
    
    Args:
        env: 基于管理器的强化学习环境实例
        std: 标准差参数，控制奖励函数的带宽
        command_name: 命令管理器中的命令名称
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: xy轴线速度跟踪的指数奖励值
    """
    # 提取所需的数量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算误差（目标速度与实际速度的平方差之和）
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    # 返回指数奖励：误差越小，奖励越接近1.0
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核奖励角速度命令跟踪（yaw轴）。
    
    此函数专门用于跟踪绕z轴的旋转角速度（偏航角速度），
    常用于地面移动机器人的转向控制。
    
    Args:
        env: 基于管理器的强化学习环境实例
        std: 标准差参数，控制奖励函数的带宽
        command_name: 命令管理器中的命令名称
        asset_cfg: 场景实体配置，默认为"robot"
        
    Returns:
        torch.Tensor: z轴角速度跟踪的指数奖励值
    """
    # 提取所需的数量（用于类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算角速度误差（目标与实际的z轴角速度平方差）
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    # 返回指数奖励：误差越小，奖励越接近1.0
    return torch.exp(-ang_vel_error / std**2)
