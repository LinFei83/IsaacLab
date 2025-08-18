# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)。
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv


def randomize_dead_zone(env: DirectRLEnv, env_ids: torch.Tensor | None):
    """随机化死区阈值。
    
    死区是一种控制策略，当控制信号的幅度小于某个阈值时，将其置零，
    以减少微小控制信号对系统的影响，提高控制的稳定性。
    
    Args:
        env: 直接强化学习环境实例
        env_ids: 需要随机化的环境ID列表，如果为None则随机化所有环境
    """
    # 为每个环境的每个自由度生成随机的死区阈值
    # 随机值范围在 [0, default_dead_zone] 之间
    env.dead_zone_thresholds = (
        torch.rand((env.num_envs, 6), dtype=torch.float32, device=env.device) * env.default_dead_zone
    )
