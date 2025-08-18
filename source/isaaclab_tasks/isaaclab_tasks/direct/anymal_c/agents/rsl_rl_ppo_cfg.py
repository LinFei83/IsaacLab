# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AnymalCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境的步数
    num_steps_per_env = 24
    # 最大迭代次数
    max_iterations = 500
    # 保存间隔
    save_interval = 50
    # 实验名称
    experiment_name = "anymal_c_flat_direct"
    # 是否使用经验归一化
    empirical_normalization = False
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        # 初始化噪声标准差
        init_noise_std=1.0,
        # Actor网络隐藏层维度
        actor_hidden_dims=[128, 128, 128],
        # Critic网络隐藏层维度
        critic_hidden_dims=[128, 128, 128],
        # 激活函数
        activation="elu",
    )
    # 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值损失系数
        value_loss_coef=1.0,
        # 是否使用裁剪的价值损失
        use_clipped_value_loss=True,
        # 裁剪参数
        clip_param=0.2,
        # 熵系数
        entropy_coef=0.005,
        # 学习周期数
        num_learning_epochs=5,
        # 小批量数量
        num_mini_batches=4,
        # 学习率
        learning_rate=1.0e-3,
        # 学习率调度器
        schedule="adaptive",
        # 折扣因子
        gamma=0.99,
        # GAE参数
        lam=0.95,
        # 期望KL散度
        desired_kl=0.01,
        # 最大梯度范数
        max_grad_norm=1.0,
    )


@configclass
class AnymalCRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境的步数
    num_steps_per_env = 24
    # 最大迭代次数
    max_iterations = 1500
    # 保存间隔
    save_interval = 50
    # 实验名称
    experiment_name = "anymal_c_rough_direct"
    # 是否使用经验归一化
    empirical_normalization = False
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        # 初始化噪声标准差
        init_noise_std=1.0,
        # Actor网络隐藏层维度
        actor_hidden_dims=[512, 256, 128],
        # Critic网络隐藏层维度
        critic_hidden_dims=[512, 256, 128],
        # 激活函数
        activation="elu",
    )
    # 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值损失系数
        value_loss_coef=1.0,
        # 是否使用裁剪的价值损失
        use_clipped_value_loss=True,
        # 裁剪参数
        clip_param=0.2,
        # 熵系数
        entropy_coef=0.005,
        # 学习周期数
        num_learning_epochs=5,
        # 小批量数量
        num_mini_batches=4,
        # 学习率
        learning_rate=1.0e-3,
        # 学习率调度器
        schedule="adaptive",
        # 折扣因子
        gamma=0.99,
        # GAE参数
        lam=0.95,
        # 期望KL散度
        desired_kl=0.01,
        # 最大梯度范数
        max_grad_norm=1.0,
    )
