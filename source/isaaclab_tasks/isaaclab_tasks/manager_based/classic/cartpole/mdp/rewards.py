# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    计算关节位置与目标值偏差的L2惩罚。
    
    该函数计算指定关节的当前位置与目标位置之间的平方差，用作奖励函数。
    对于倒立摆任务，这通常用于惩罚杆子偏离直立位置（0度）的程度。
    
    Args:
        env: 管理器基础强化学习环境实例
        target: 目标关节位置（弧度）
        asset_cfg: 场景实体配置，指定要监控的资产和关节
        
    Returns:
        torch.Tensor: 每个环境的L2距离惩罚值，形状为(num_envs,)
        
    Note:
        - 使用wrap_to_pi确保角度在(-π, π)范围内，避免角度跳跃问题
        - 返回值为正数，在奖励配置中通常使用负权重将其转换为惩罚
    """
    # 提取所需的数量（启用类型提示）
    # 从环境场景中获取指定的关节化资产（如倒立摆机器人）
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 将关节位置包装到(-π, π)范围内
    # 这对于角度计算很重要，避免了-π和π之间的跳跃问题
    # 例如，179度和-179度实际上只相差2度，而不是358度
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    
    # 计算奖励/惩罚值
    # 使用L2距离（平方差）来衡量当前位置与目标位置的偏差
    # torch.square(joint_pos - target): 计算每个关节的平方误差
    # torch.sum(..., dim=1): 对所有指定关节求和，返回每个环境的总误差
    return torch.sum(torch.square(joint_pos - target), dim=1)
