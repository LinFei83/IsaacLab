# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)。
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# 导入代理模块，包含强化学习配置
from . import agents
# 导入 Forge 环境类
from .forge_env import ForgeEnv
# 导入三个任务配置类：插销插入、齿轮啮合、螺母螺纹
from .forge_env_cfg import ForgeTaskGearMeshCfg, ForgeTaskNutThreadCfg, ForgeTaskPegInsertCfg

##
# 注册 Gym 环境，使它们可以通过 gym.make() 创建
##

# 注册插销插入任务环境
gym.register(
    id="Isaac-Forge-PegInsert-Direct-v0",
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# 注册齿轮啮合任务环境
gym.register(
    id="Isaac-Forge-GearMesh-Direct-v0",
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# 注册螺母螺纹任务环境
gym.register(
    id="Isaac-Forge-NutThread-Direct-v0",
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
