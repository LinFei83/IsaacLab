# 版权所有 (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利.
#
# SPDX-License-Identifier: BSD-3-Clause

# 导入配置类装饰器
from isaaclab.utils import configclass

# 导入RSL-RL相关的配置类
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 训练配置参数
    # 每个环境的步数
    num_steps_per_env = 32
    # 最大训练迭代次数
    max_iterations = 1000
    # 模型保存间隔（迭代次数）
    save_interval = 50
    # 实验名称，用于日志和模型保存
    experiment_name = "humanoid_direct"
    # 是否使用经验归一化
    empirical_normalization = True
    
    # 策略网络配置（Actor-Critic架构）
    policy = RslRlPpoActorCriticCfg(
        # 初始化噪声标准差，用于动作探索
        init_noise_std=1.0,
        # Actor网络隐藏层维度（策略网络）
        actor_hidden_dims=[400, 200, 100],
        # Critic网络隐藏层维度（价值网络）
        critic_hidden_dims=[400, 200, 100],
        # 激活函数类型
        activation="elu",
    )
    
    # PPO算法配置
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值损失系数
        value_loss_coef=1.0,
        # 是否使用裁剪的价值损失
        use_clipped_value_loss=True,
        # PPO裁剪参数，限制策略更新的幅度
        clip_param=0.2,
        # 熵正则化系数，用于鼓励探索
        entropy_coef=0.0,
        # 每次更新的学习轮数
        num_learning_epochs=5,
        # 小批量数量，用于分割数据进行训练
        num_mini_batches=4,
        # 学习率
        learning_rate=1.0e-4,
        # 学习率调度策略
        schedule="adaptive",
        # 折扣因子，用于计算回报
        gamma=0.99,
        # GAE参数，用于优势估计
        lam=0.95,
        # 期望KL散度，用于自适应学习率调整
        desired_kl=0.008,
        # 梯度范数上限，用于梯度裁剪防止梯度爆炸
        max_grad_norm=1.0,
    )
