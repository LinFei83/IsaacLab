# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand环境的RSL-RL PPO算法配置模块。
该模块定义了使用RSL-RL库训练Shadow Hand环境的PPO算法配置类。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ShadowHandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shadow Hand环境的RSL-RL PPO运行器配置类。"""
    
    # 训练参数
    num_steps_per_env = 16  # 每个环境的步数
    max_iterations = 10000  # 最大迭代次数
    save_interval = 250  # 保存间隔
    experiment_name = "shadow_hand"  # 实验名称
    empirical_normalization = True  # 是否使用经验归一化
    
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始化噪声标准差
        actor_hidden_dims=[512, 512, 256, 128],  # 演员网络隐藏层维度
        critic_hidden_dims=[512, 512, 256, 128],  # 评论家网络隐藏层维度
        activation="elu",  # 激活函数
    )
    
    # 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 价值损失系数
        use_clipped_value_loss=True,  # 是否使用裁剪的价值损失
        clip_param=0.2,  # PPO裁剪参数
        entropy_coef=0.005,  # 熵系数
        num_learning_epochs=5,  # 学习轮数
        num_mini_batches=4,  # 小批量数量
        learning_rate=5.0e-4,  # 学习率
        schedule="adaptive",  # 学习率调度
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE参数
        desired_kl=0.016,  # 期望KL散度
        max_grad_norm=1.0,  # 最大梯度范数
    )


@configclass
class ShadowHandAsymFFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shadow Hand非对称前馈PPO运行器配置类。"""
    
    # 训练参数
    num_steps_per_env = 16  # 每个环境的步数
    max_iterations = 10000  # 最大迭代次数
    save_interval = 250  # 保存间隔
    experiment_name = "shadow_hand_openai_ff"  # 实验名称
    empirical_normalization = True  # 是否使用经验归一化
    
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始化噪声标准差
        actor_hidden_dims=[400, 400, 200, 100],  # 演员网络隐藏层维度（前馈网络结构）
        critic_hidden_dims=[512, 512, 256, 128],  # 评论家网络隐藏层维度
        activation="elu",  # 激活函数
    )
    
    # 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 价值损失系数
        use_clipped_value_loss=True,  # 是否使用裁剪的价值损失
        clip_param=0.2,  # PPO裁剪参数
        entropy_coef=0.005,  # 熵系数
        num_learning_epochs=4,  # 学习轮数
        num_mini_batches=4,  # 小批量数量
        learning_rate=5.0e-4,  # 学习率
        schedule="adaptive",  # 学习率调度
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE参数
        desired_kl=0.01,  # 期望KL散度
        max_grad_norm=1.0,  # 最大梯度范数
    )


@configclass
class ShadowHandVisionFFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shadow Hand视觉前馈PPO运行器配置类。"""
    
    # 训练参数
    num_steps_per_env = 64  # 每个环境的步数
    max_iterations = 50000  # 最大迭代次数
    save_interval = 250  # 保存间隔
    experiment_name = "shadow_hand_vision"  # 实验名称
    empirical_normalization = True  # 是否使用经验归一化
    
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始化噪声标准差
        actor_hidden_dims=[1024, 512, 512, 256, 128],  # 演员网络隐藏层维度（适用于视觉任务的较大网络）
        critic_hidden_dims=[1024, 512, 512, 256, 128],  # 评论家网络隐藏层维度
        activation="elu",  # 激活函数
    )
    
    # 算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 价值损失系数
        use_clipped_value_loss=True,  # 是否使用裁剪的价值损失
        clip_param=0.2,  # PPO裁剪参数
        entropy_coef=0.005,  # 熵系数
        num_learning_epochs=5,  # 学习轮数
        num_mini_batches=4,  # 小批量数量
        learning_rate=5.0e-4,  # 学习率
        schedule="adaptive",  # 学习率调度
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE参数
        desired_kl=0.01,  # 期望KL散度
        max_grad_norm=1.0,  # 最大梯度范数
    )
