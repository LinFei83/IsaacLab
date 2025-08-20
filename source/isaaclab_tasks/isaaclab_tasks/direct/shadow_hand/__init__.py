# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand环境注册模块。
该模块负责注册Shadow Hand相关的Gym环境，包括基础环境、OpenAI风格环境和视觉环境。
"""

import gymnasium as gym

from . import agents

##
# 注册Gym环境。
##

# inhand manipulation任务入口
inhand_task_entry = "isaaclab_tasks.direct.inhand_manipulation"

# 注册基础Shadow Hand环境
gym.register(
    id="Isaac-Repose-Cube-Shadow-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# 注册OpenAI风格的前馈网络Shadow Hand环境
gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandAsymFFPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ff_ppo_cfg.yaml",
    },
)

# 注册OpenAI风格的LSTM网络Shadow Hand环境
gym.register(
    id="Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_env_cfg:ShadowHandOpenAIEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

### 视觉环境

# 注册视觉Shadow Hand环境
gym.register(
    id="Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
    entry_point=f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)

# 注册视觉Shadow Hand播放环境（用于推理）
gym.register(
    id="Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0",
    entry_point=f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_vision_env:ShadowHandVisionEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandVisionFFPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_vision_cfg.yaml",
    },
)
