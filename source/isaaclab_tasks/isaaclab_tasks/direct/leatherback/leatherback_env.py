from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from .waypoint import WAYPOINT_CFG
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG
from isaaclab.markers import VisualizationMarkers

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    """Leatherback环境的配置类。"""
    # 抽取（Decimation）因子，用于控制渲染和物理步骤的频率
    decimation = 4
    # 每轮（episode）的持续时间（秒）
    episode_length_s = 20.0
    # 动作空间维度
    action_space = 2
    # 观测空间维度
    observation_space = 8
    # 状态空间维度（未使用）
    state_space = 0
    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    # 机器人配置，指定机器人的prim路径
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # 路径点配置
    waypoint_cfg = WAYPOINT_CFG

    # 油门相关的自由度（DOF）名称
    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    # 转向相关的自由度（DOF）名称
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    # 环境之间的间距
    env_spacing = 32.0
    # 场景配置，定义了环境数量、间距和物理复制
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class LeatherbackEnv(DirectRLEnv):
    """Leatherback机器人导航任务的环境。"""
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化环境。"""
        super().__init__(cfg, render_mode, **kwargs)
        # 查找油门和转向自由度的索引
        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        # 初始化油门和转向状态张量
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        # 初始化目标点到达状态
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        # 任务完成状态
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        # 目标点数量
        self._num_goals = 10
        # 目标点位置
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        # 用于可视化的标记点位置
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        # 环境间距
        self.env_spacing = self.cfg.env_spacing
        # 赛道长度系数
        self.course_length_coefficient = 2.5
        # 赛道宽度系数
        self.course_width_coefficient = 2.0
        # 到达目标点的容忍距离
        self.position_tolerance = 0.15
        # 到达目标点的奖励
        self.goal_reached_bonus = 10.0
        # 位置进展奖励权重
        self.position_progress_weight = 1.0
        # 朝向系数，用于奖励计算
        self.heading_coefficient = 0.25
        # 朝向进展奖励权重
        self.heading_progress_weight = 0.05
        # 当前目标点的索引
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)

    def _setup_scene(self):
        """设置场景。"""
        # 创建一个大的地面平面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # 500m x 500m 的大地面
                color=(0.2, 0.2, 0.2),  # 深灰色
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # 设置场景的其余部分
        self.leatherback = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg) # 创建可视化标记点
        self.object_state = [] # 初始化对象状态
        
        # 克隆环境
        self.scene.clone_environments(copy_from_source=False) 
        # 过滤碰撞
        self.scene.filter_collisions(global_prim_paths=[]) 
        # 将机器人添加到场景中
        self.scene.articulations["leatherback"] = self.leatherback 

        # 添加光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """在物理步骤之前处理动作。"""
        throttle_scale = 10 # 油门缩放因子
        throttle_max = 50 # 油门最大值
        steering_scale = 0.1 # 转向缩放因子
        steering_max = 0.75 # 转向最大值

        # 油门动作：将单个油门值复制到4个轮子
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max)
        self._throttle_state = self._throttle_action
        
        # 转向动作：将单个转向值复制到2个前轮
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action

    def _apply_action(self) -> None:
        """应用动作到机器人。"""
        # 设置关节速度目标（油门）
        # 油门：使用速度控制（模拟电机驱动）
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        # 设置关节位置目标（转向）
        # 转向：使用位置控制（模拟转向角度）
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        """获取观测值。"""
        # 获取当前目标点位置
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        # 计算位置误差向量
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        # 保存上一次的位置误差
        self._previous_position_error = self._position_error.clone()
        # 计算当前位置误差（范数）
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        # 获取机器人当前朝向
        heading = self.leatherback.data.heading_w
        # 计算目标朝向
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        # 计算朝向误差
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # 组合观测值
        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1), # 位置误差
                torch.cos(self.target_heading_error).unsqueeze(dim=1), # 目标朝向余弦
                torch.sin(self.target_heading_error).unsqueeze(dim=1), # 目标朝向正弦
                self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1), # 机器人本体坐标系下的前向速度
                self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1), # 机器人本体坐标系下的侧向速度
                self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1), # 世界坐标系下的偏航角速度
                self._throttle_state[:, 0].unsqueeze(dim=1), # 油门状态
                self._steering_state[:, 0].unsqueeze(dim=1), # 转向状态
            ),
            dim=-1,
        )
        
        # 检查观测值是否包含NaN
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """计算奖励。"""
        # 位置进展奖励：衡量机器人是否更接近目标
        position_progress_rew = self._previous_position_error - self._position_error
        # 朝向奖励：鼓励机器人朝向目标点
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        # 检查是否到达目标点
        goal_reached = self._position_error < self.position_tolerance
        # 如果到达目标点，则更新目标点索引
        self._target_index = self._target_index + goal_reached
        # 检查任务是否完成（所有目标点都已到达）
        self.task_completed = self._target_index > (self._num_goals -1)
        # 循环目标点索引
        self._target_index = self._target_index % self._num_goals

        # 组合奖励
        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            goal_reached * self.goal_reached_bonus
        )

        # 可视化当前目标点
        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        # 检查奖励是否包含NaN
        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """获取完成状态。"""
        # 如果超过最大回合长度，则任务失败
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定索引的环境。"""
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        # 获取默认的机器人状态
        default_state = self.leatherback.data.default_root_state[env_ids]
        leatherback_pose = default_state[:, :7]
        leatherback_velocities = default_state[:, 7:]
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids]

        # 将机器人位置设置为环境原点
        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]
        leatherback_pose[:, 0] -= self.env_spacing / 2
        # 在赛道宽度内随机化机器人的起始侧向位置
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        # 随机化机器人的起始朝向
        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        # 将重置后的状态写入仿真
        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        # 重置目标点位置和标记点位置
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        # 生成新的目标点路径
        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        # 随机化目标点的y坐标
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        # 将目标点位置偏移到环境原点
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        # 重置目标点索引
        self._target_index[env_ids] = 0
        # 更新可视化标记点的位置
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        # 重置位置误差
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        # 重置朝向误差
        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()