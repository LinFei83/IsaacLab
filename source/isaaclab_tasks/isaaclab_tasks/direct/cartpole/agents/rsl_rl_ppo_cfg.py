# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16  # 每个环境的步数
    max_iterations = 150     # 最大迭代次数
    save_interval = 50      # 保存间隔
    experiment_name = "cartpole_direct"  # 实验名称
    empirical_normalization = False  # 是否使用经验归一化
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,          # 初始化噪声标准差
        actor_hidden_dims=[32, 32],   # Actor网络隐藏层维度
        critic_hidden_dims=[32, 32],  # Critic网络隐藏层维度
        activation="elu",             # 激活函数
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # 价值损失系数
        use_clipped_value_loss=True,  # 是否使用裁剪的价值损失
        clip_param=0.2,               # 裁剪参数
        entropy_coef=0.005,           # 熵系数
        num_learning_epochs=5,        # 学习周期数
        num_mini_batches=4,           # 小批量数量
        learning_rate=1.0e-3,         # 学习率
        schedule="adaptive",          # 学习率调度
        gamma=0.99,                   # 折扣因子
        lam=0.95,                     # GAE lambda参数
        desired_kl=0.01,              # 期望KL散度
        max_grad_norm=1.0,            # 最大梯度范数
    )
