# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 版权所有。
#
# SPDX-License-Identifier: BSD-3-Clause

# 导入配置类装饰器
from isaaclab.utils import configclass

# 导入RSL-RL库的相关配置类
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AllegroHandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境的步数
    num_steps_per_env = 16
    # 最大训练迭代次数
    max_iterations = 10000
    # 模型保存间隔（迭代次数）
    save_interval = 250
    # 实验名称
    experiment_name = "allegro_hand"
    # 是否使用经验归一化
    empirical_normalization = True
    
    # 策略网络配置（Actor-Critic架构）
    policy = RslRlPpoActorCriticCfg(
        # 初始化噪声标准差
        init_noise_std=1.0,
        # Actor网络隐藏层维度（策略网络）
        actor_hidden_dims=[1024, 512, 256, 128],
        # Critic网络隐藏层维度（价值网络）
        critic_hidden_dims=[1024, 512, 256, 128],
        # 激活函数类型
        activation="elu",
    )
    
    # PPO算法参数配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值损失系数
        value_loss_coef=1.0,
        # 是否使用裁剪的价值损失
        use_clipped_value_loss=True,
        # PPO裁剪参数 epsilon
        clip_param=0.2,
        # 熵损失系数（用于鼓励探索）
        entropy_coef=0.005,
        # 每次更新的学习轮数
        num_learning_epochs=5,
        # 小批量数量
        num_mini_batches=4,
        # 学习率
        learning_rate=5.0e-4,
        # 学习率调度策略
        schedule="adaptive",
        # 折扣因子 gamma
        gamma=0.99,
        # GAE参数 lambda
        lam=0.95,
        # 期望KL散度（用于自适应学习率调整）
        desired_kl=0.016,
        # 梯度范数上限（防止梯度爆炸）
        max_grad_norm=1.0,
    )
