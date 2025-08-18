# 版权所有 (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
人形机器人运动环境.

这个模块定义了人形机器人的运动环境配置，包括注册Gym环境和相关配置文件的入口点。
"""

# 导入gymnasium库用于创建强化学习环境
import gymnasium as gym

# 导入agents模块，包含不同强化学习框架的配置文件
from . import agents

##
# 注册Gym环境，使其可以通过gym.make()函数创建
##

# 注册人形机器人环境
# id: 环境的唯一标识符
# entry_point: 环境类的入口点，格式为"模块路径:类名"
# disable_env_checker: 禁用环境检查器以提高性能
# kwargs: 环境配置参数，包括各种配置文件的入口点
gym.register(
    id="Isaac-Humanoid-Direct-v0",
    entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置入口点
        "env_cfg_entry_point": f"{__name__}.humanoid_env:HumanoidEnvCfg",
        # RL-Games PPO算法配置入口点
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # RSL-RL PPO算法配置入口点
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        # SKRL PPO算法配置入口点
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
