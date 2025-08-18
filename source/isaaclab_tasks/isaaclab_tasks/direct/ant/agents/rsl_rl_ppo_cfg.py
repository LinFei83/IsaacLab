# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AntPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # 每个环境的步数
    max_iterations = 1000  # 最大迭代次数
    save_interval = 50  # 保存间隔
    experiment_name = "ant_direct"  # 实验名称
    empirical_normalization = False  # 是否使用经验归一化
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始化噪声标准差
        actor_hidden_dims=[400, 200, 100],  # Actor网络隐藏层维度
        critic_hidden_dims=[400, 200, 100],  # Critic网络隐藏层维度
        activation="elu",  # 激活函数
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 价值损失系数
        use_clipped_value_loss=True,  # 是否使用裁剪的价值损失
        clip_param=0.2,  # PPO裁剪参数
        entropy_coef=0.0,  # 熵系数
        num_learning_epochs=5,  # 学习周期数
        num_mini_batches=4,  # 小批量数量
        learning_rate=5.0e-4,  # 学习率
        schedule="adaptive",  # 学习率调度策略
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE参数
        desired_kl=0.01,  # 期望的KL散度
        max_grad_norm=1.0,  # 最大梯度范数
    )
