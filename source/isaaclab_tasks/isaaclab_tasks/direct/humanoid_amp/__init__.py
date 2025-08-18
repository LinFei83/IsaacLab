# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP 人形机器人运动环境。
"""

import gymnasium as gym

from . import agents

##
# 注册 Gym 环境。
##

# 注册舞蹈动作环境
gym.register(
    id="Isaac-Humanoid-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

# 注册跑步动作环境
gym.register(
    id="Isaac-Humanoid-AMP-Run-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

# 注册走路动作环境
gym.register(
    id="Isaac-Humanoid-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)
