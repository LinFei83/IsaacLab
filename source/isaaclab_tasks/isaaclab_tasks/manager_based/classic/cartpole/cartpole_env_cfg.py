# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利。
#
# SPDX-许可证标识符: BSD-3-Clause

"""
倒立摆环境配置文件

这个文件定义了经典的倒立摆控制任务的完整配置。倒立摆是强化学习中的经典基准任务，
需要智能体学习控制小车的水平移动来保持杆子直立。

主要组件：
- CartpoleSceneCfg: 场景配置，包括机器人、地面和光照
- ActionsCfg: 动作空间配置，定义可执行的动作
- ObservationsCfg: 观测空间配置，定义状态信息
- RewardsCfg: 奖励函数配置，指导学习目标
- TerminationsCfg: 终止条件配置，定义episode结束条件
- CartpoleEnvCfg: 主环境配置，整合所有组件
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# 预定义配置
# 从资产库中导入预定义的倒立摆机器人配置
##
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip - 导入排序跳过标记


##
# 场景定义
# 定义倒立摆环境中的所有物体和光照
##


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """倒立摆场景的配置类。
    
    该类定义了倒立摆任务中所需的所有场景元素，包括：
    - 地面平面（提供物理支撑）
    - 倒立摆机器人（主要控制对象）
    - 环境光照（用于渲染和视觉效果）
    """

    # 地面平面 - 为倒立摆系统提供物理支撑和碰撞检测
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 倒立摆机器人 - 由小车和杆子组成的关节化系统
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 光照设置 - 为环境提供均匀的照明效果
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP（马尔科夫决策过程）设置
# 定义强化学习环境的所有组成部分：动作、观测、奖励、终止条件等
##


@configclass
class ActionsCfg:
    """动作配置类 - 定义MDP中的动作空间。
    
    在倒立摆任务中，智能体只能控制小车的水平移动，
    通过对slider_to_cart关节施加力矩来实现。
    """

    # 关节力矩动作 - 对小车的水平移动关节施加力矩
    # asset_name="robot": 指定要控制的机器人资产
    # joint_names=["slider_to_cart"]: 指定要控制的关节（小车的水平滑动关节）
    # scale=100.0: 动作缩放因子，将[-1,1]范围的动作信号放大到[-100,100]牛顿的力
    #              这个值决定了智能体动作的最大力度，需要根据机器人的物理特性调整
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    """观测配置类 - 定义MDP中的状态观测空间。
    
    在基础倒立摆环境中，智能体可以观测到：
    - 小车和杆子的关节位置（相对于目标位置）
    - 小车和杆子的关节速度（相对于目标速度）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略组的观测配置。
        
        这个组包含了智能体做决策所需的所有状态信息。
        """

        # 观测项（保持顺序） - 这些观测将按顺序连接成一个向量
        # 相对关节位置 - 获取小车和杆子相对于目标位置的偏差
        # 返回形状: (num_envs, 2) - [小车位置偏差, 杆子角度偏差]
        # 小车目标位置通常为0（中心位置），杆子目标角度为0（直立）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # 相对关节速度 - 获取小车和杆子相对于目标速度的偏差
        # 返回形状: (num_envs, 2) - [小车速度偏差, 杆子角速度偏差]
        # 目标速度通常为0（静止状态）
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            """后初始化方法 - 设置观测组的运行时参数。
            
            这个方法在PolicyCfg对象创建后自动调用，用于配置观测处理的细节。
            """
            # 初始化后的配置设置
            # 禁用观测噪声 - 保证观测数据的精确性
            # 在真实机器人应用中，可能需要启用噪声来提高鲁棒性
            self.enable_corruption = False
            # 启用观测项连接 - 将所有观测项合并为一个向量
            # 最终观测向量形状: (num_envs, 4) - [小车位置, 杆子角度, 小车速度, 杆子角速度]
            self.concatenate_terms = True

    # 观测组 - 将观测按用途分组（这里只有策略组）
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置类 - 定义环境中的各种事件。
    
    事件用于在特定时刻改变环境状态，主要用于：
    - 环境重置（设置初始状态）
    - 随机化初始条件（提高训练鲁棒性）
    """

    # 重置事件 - 在每个新episode开始时随机设置初始状态
    # 重置小车位置 - 在episode开始时随机设置小车的位置和速度
    # position_range: (-1.0, 1.0) 表示小车初始位置在-1到1米范围内随机
    # velocity_range: (-0.5, 0.5) 表示小车初始速度在-0.5到0.5m/s范围内随机
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # 重置杆子位置 - 在episode开始时随机设置杆子的角度和角速度
    # position_range: (-0.25*π, 0.25*π) 表示杆子初始角度在±45度范围内随机
    # velocity_range: (-0.25*π, 0.25*π) 表示杆子初始角速度在±45度/s范围内随机
    # 这样的设置保证了杆子不会一开始就倒下，但也不会太稳定
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """奖励配置类 - 定义MDP中的奖励函数。
    
    奖励函数的设计遵循以下原则：
    1. 主要目标：保持杆子直立（角度接近0）
    2. 辅助目标：减少不必要的运动（降低速度）
    3. 安全约束：避免小车超出边界或杆子倒下
    """

    # (1) 持续存活奖励 - 只要环境没有终止就给予正奖励
    # 鼓励智能体尽可能长时间保持倒立摆稳定
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) 失败惩罚 - 当环境终止时给予负奖励
    # 强化对失败状态的避免，促使智能体学习稳定的控制策略
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) 主要任务：保持杆子直立 - 这是倒立摆任务的核心目标
    # 使用L2距离计算杆子角度与目标角度(0度)的偏差，角度越接近0奖励越高
    # weight=-1.0 表示这是一个成本函数（偏差越大惩罚越大）
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) 形状奖励：降低小车速度 - 鼓励平稳的控制
    # 使用L1范数计算小车速度，鼓励智能体采用平稳的控制策略
    # 较小的权重(-0.01)表示这是次要目标，不会干扰主要任务
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) 形状奖励：降低杆子角速度 - 鼓励杆子稳定
    # 使用L1范数计算杆子角速度，鼓励杆子保持相对静止
    # 更小的权重(-0.005)表示这是最次要的目标
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    """终止条件配置类 - 定义MDP中的终止条件。
    
    终止条件决定了什么时候episode结束，包括：
    - 正常终止：达到最大时间步数
    - 失败终止：小车超出边界或杆子倒下
    """

    # (1) 超时终止 - 达到最大episode长度时的正常终止
    # 这是一个正常的终止条件，不会被视为失败
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) 小车超出边界 - 当小车移动超出允许范围时终止
    # 小车位置超出[-3.0, 3.0]米范围时终止episode
    # 这个边界设置防止小车无限制移动，保证任务的有界性
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


##
# 环境配置
# 将所有组件整合为完整的强化学习环境
##


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """倒立摆环境的主配置类。
    
    这个类整合了所有的MDP组件，包括：
    - 场景配置（机器人、地面、光照）
    - 动作空间（小车的水平移动控制）
    - 观测空间（关节位置和速度）
    - 奖励函数（保持直立、降低速度等）
    - 终止条件（超时、超出边界）
    """

    # 场景设置 - 配置物理世界的参数
    # num_envs=4096: 同时运行4096个并行环境实例，加速数据收集
    # env_spacing=4.0: 环境间距4米，防止相互干扰
    # clone_in_fabric=True: 优化GPU内存使用，提高性能
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    # 基本设置 - MDP的核心组件
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP组件 - 定义强化学习的数学框架
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 后初始化设置 - 在对象创建后调整参数
    def __post_init__(self) -> None:
        """后初始化方法 - 设置运行时参数。
        
        这个方法在对象创建后自动调用，用于设置：
        - 仿真参数（时间步长、渲染频率）
        - 视角参数（相机位置和朝向）
        - 训练参数（episode长度、动作频率）
        """
        # 通用设置 - 控制环境的基本行为
        # 动作重复次数 - 每个动作执行2个仿真步骤，平衡控制精度和效率
        self.decimation = 2
        # episode长度 - 每个episode最长5秒，提供足够的学习时间
        self.episode_length_s = 5
        # 视角设置 - 配置3D查看器的相机位置
        # 相机位置 - 设置在倒立摆正前方稍高的位置，便于观察
        self.viewer.eye = (8.0, 0.0, 5.0)
        # 仿真设置 - 控制物理仿真的精度和性能
        # 仿真时间步长 - 1/120秒，提供高精度的物理仿真
        self.sim.dt = 1 / 120
        # 渲染间隔 - 与动作重复次数保持一致，节约计算资源
        self.sim.render_interval = self.decimation
