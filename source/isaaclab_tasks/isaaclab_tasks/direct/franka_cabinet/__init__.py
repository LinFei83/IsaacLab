# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka机器人与柜子交互的环境。
"""

import gymnasium as gym

from . import agents

##
# 注册Gym环境
##

gym.register(
    id="Isaac-Franka-Cabinet-Direct-v0",
    entry_point=f"{__name__}.franka_cabinet_env:FrankaCabinetEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置入口点
        "env_cfg_entry_point": f"{__name__}.franka_cabinet_env:FrankaCabinetEnvCfg",
        # RL Games配置入口点
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # RSL-RL配置入口点
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        # SKRL配置入口点
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
