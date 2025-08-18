# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """倒立摆任务的PPO训练运行器配置。
    
    该配置专门为倒立摆任务调优，包括：
    - 网络架构参数（Actor-Critic网络结构）
    - PPO算法参数（学习率、裁剪系数等）
    - 训练进程控制（迭代次数、保存间隔等）
    """
    # 每个环境每次采样的步数 - 较小的值适合简单任务，可以提高更新频率
    num_steps_per_env = 16
    # 最大训练迭代次数 - 倒立摆任务相对简单，150次迭代通常足够收敛
    max_iterations = 150
    # 模型保存间隔 - 每50次迭代保存一次模型，防止训练中断
    save_interval = 50
    # 实验名称 - 用于文件命名和日志记录
    experiment_name = "cartpole"
    # 经验归一化 - 禁用基于经验的观测归一化，使用默认归一化
    empirical_normalization = False
    # Actor-Critic网络配置 - 定义策略网络和价值网络的结构
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,          # 初始噪声标准差 - 鼓励早期探索，逐渐减少
        actor_hidden_dims=[32, 32],  # Actor网络隐藏层维度 - 两层各32个神经元
        critic_hidden_dims=[32, 32], # Critic网络隐藏层维度 - 与Actor相同的结构
        activation="elu",            # 激活函数 - ELU可以缓解梯度消失问题
    )
    # PPO算法配置 - 定义PPO算法的所有超参数
    algorithm = RslRlPpoAlgorithmCfg(
        # 损失函数相关参数
        value_loss_coef=1.0,         # 价值损失系数 - 控制Critic网络的学习强度
        use_clipped_value_loss=True, # 启用价值损失裁剪 - 防止价值函数更新过大
        
        # PPO核心参数
        clip_param=0.2,              # PPO裁剪参数 - 防止策略更新过大，保持训练稳定
        entropy_coef=0.005,          # 熵系数 - 鼓励探索，防止过早收敛
        
        # 训练进程参数
        num_learning_epochs=5,       # 每次更新的学习轮数 - 对每批数据重复学习的次数
        num_mini_batches=4,          # 小批量数量 - 将数据分成4个小批量进行更新
        
        # 优化器参数
        learning_rate=1.0e-3,        # 学习率 - 控制参数更新的步长
        schedule="adaptive",         # 学习率调度 - 根据KL散度自适应调整
        
        # 强化学习核心参数
        gamma=0.99,                  # 折扣因子 - 对未来奖励的重视程度
        lam=0.95,                    # GAE参数 - 控制优势估计的偏差-方差权衡
        
        # 训练稳定性参数
        desired_kl=0.01,             # 期望KL散度 - 控制策略更新的幅度
        max_grad_norm=1.0,           # 最大梯度范数 - 防止梯度爆炸
    )
