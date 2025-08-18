# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
倒立摆平衡环境。

这个模块实现了经典的倒立摆（Cartpole）控制问题，包括多种不同的环境配置：
- 基础倒立摆环境（使用状态观测）
- 基于RGB相机的倒立摆环境
- 基于深度相机的倒立摆环境  
- 使用ResNet18特征提取的倒立摆环境
- 使用Theia-Tiny Transformer特征提取的倒立摆环境
"""

import gymnasium as gym

from . import agents

##
# 注册Gym环境
# 这里注册了多个倒立摆环境的变体，每个环境使用不同的观测类型和配置
##

# 基础倒立摆环境 - 使用关节位置和速度作为观测
# 这是最基本的倒立摆环境，智能体需要通过施加水平力来保持杆子直立
gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# RGB相机倒立摆环境 - 使用RGB图像作为观测
# 智能体需要从RGB图像中学习如何控制倒立摆，这增加了视觉感知的复杂性
gym.register(
    id="Isaac-Cartpole-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleRGBCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

# 深度相机倒立摆环境 - 使用深度图像作为观测
# 使用深度信息而不是RGB，可以提供距离信息但缺少颜色信息
gym.register(
    id="Isaac-Cartpole-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleDepthCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

# ResNet18特征提取倒立摆环境 - 使用预训练ResNet18提取的特征作为观测
# 通过预训练的卷积神经网络提取高级特征，可以减少训练时间并提高性能
gym.register(
    id="Isaac-Cartpole-RGB-ResNet18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleResNet18CameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)

# Theia-Tiny Transformer特征提取倒立摆环境 - 使用Theia-Tiny Transformer模型提取的特征
# 使用轻量级的视觉Transformer模型进行特征提取，提供了另一种视觉表示学习方法
gym.register(
    id="Isaac-Cartpole-RGB-TheiaTiny-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleTheiaTinyCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)
