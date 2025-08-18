# 版权所有 (c) 2022-2025, The Isaac Lab Project 开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import factory_control, factory_utils
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg


class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # 更新观测/状态的数量
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

    def _set_default_dynamics_parameters(self):
        """设置定义动力学交互的参数."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # 设置质量和摩擦系数.
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """初始化张量."""
        # 控制目标.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor  # 指数移动平均因子
        self.dead_zone_thresholds = None  # 死区阈值

        # 固定物体.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)  # 固定物体位置观测帧
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)  # 初始固定物体位置观测噪声

        # 计算身体索引.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")  # 左手指身体索引
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")  # 右手指身体索引
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")  # 指尖身体索引

        # 用于有限差分的张量.
        self.last_update_timestamp = 0.0  # 注意: 这是用于有限差分身体速度的.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)  # 上一时刻指尖位置
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )  # 上一时刻指尖四元数
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)  # 上一时刻关节位置

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)  # 回合是否成功
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)  # 回合成功时间

    def _setup_scene(self):
        """初始化仿真场景."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # 在场景中生成桌子的USD文件
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)  # 机器人
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)  # 固定物体
        self._held_asset = Articulation(self.cfg_task.held_asset)  # 持有物体
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)  # 小齿轮
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)  # 大齿轮

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # 我们需要为CPU仿真显式过滤碰撞
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """从原始张量计算值。这包括添加噪声."""
        # TODO: 这些值中的很多可能只需要设置一次?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins  # 固定物体位置
        self.fixed_quat = self._fixed_asset.data.root_quat_w  # 固定物体四元数

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins  # 持有物体位置
        self.held_quat = self._held_asset.data.root_quat_w  # 持有物体四元数

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins  # 指尖中点位置
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]  # 指尖中点四元数
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]  # 指尖中点线速度
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]  # 指尖中点角速度

        jacobians = self._robot.root_physx_view.get_jacobians()  # 获取雅可比矩阵

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]  # 左手指雅可比矩阵
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]  # 右手指雅可比矩阵
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5  # 指尖中点雅可比矩阵
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]  # 手臂质量矩阵
        self.joint_pos = self._robot.data.joint_pos.clone()  # 关节位置
        self.joint_vel = self._robot.data.joint_vel.clone()  # 关节速度

        # 有限差分得到更可靠的速度估计.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt  # 末端执行器线速度（有限差分）
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # 如果没有添加速度，则添加状态差异.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt  # 末端执行器角速度（有限差分）
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt  # 关节速度（有限差分）
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp  # 上次更新时间戳

    def _get_factory_obs_state_dict(self):
        """为策略和评论家填充字典."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise  # 带噪声的固定物体位置

        prev_actions = self.actions.clone()  # 上一动作

        # 观测字典
        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,  # 指尖位置
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,  # 指尖相对于固定物体的位置
            "fingertip_quat": self.fingertip_midpoint_quat,  # 指尖四元数
            "ee_linvel": self.ee_linvel_fd,  # 末端执行器线速度
            "ee_angvel": self.ee_angvel_fd,  # 末端执行器角速度
            "prev_actions": prev_actions,  # 上一动作
        }

        # 状态字典
        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,  # 指尖位置
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,  # 指尖相对于固定物体的位置
            "fingertip_quat": self.fingertip_midpoint_quat,  # 指尖四元数
            "ee_linvel": self.fingertip_midpoint_linvel,  # 末端执行器线速度
            "ee_angvel": self.fingertip_midpoint_angvel,  # 末端执行器角速度
            "joint_pos": self.joint_pos[:, 0:7],  # 关节位置
            "held_pos": self.held_pos,  # 持有物体位置
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,  # 持有物体相对于固定物体的位置
            "held_quat": self.held_quat,  # 持有物体四元数
            "fixed_pos": self.fixed_pos,  # 固定物体位置
            "fixed_quat": self.fixed_quat,  # 固定物体四元数
            "task_prop_gains": self.task_prop_gains,  # 任务比例增益
            "pos_threshold": self.pos_threshold,  # 位置阈值
            "rot_threshold": self.rot_threshold,  # 旋转阈值
            "prev_actions": prev_actions,  # 上一动作
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """使用非对称评论家获取演员/评论家输入."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])  # 观测张量
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])  # 状态张量
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """重置缓冲区."""
        self.ep_succeeded[env_ids] = 0  # 重置回合成功标志
        self.ep_success_times[env_ids] = 0  # 重置回合成功时间

    def _pre_physics_step(self, action):
        """应用带有平滑处理的策略动作."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        # 使用指数移动平均平滑动作
        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def close_gripper_in_place(self):
        """在夹爪闭合时保持夹爪在当前位置."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # 将动作解释为目标位置位移并设置位置目标
        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # 将动作解释为目标旋转（轴角）位移
        rot_actions = actions[:, 3:6]

        # 转换为四元数并设置旋转目标
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # 保持直立
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,  # 夹爪自由度位置目标
        )

    def _apply_action(self):
        """将策略动作应用为相对于当前位置的增量目标."""
        # 注意: 我们使用有限差分速度进行控制和观测.
        # 检查是否需要在减薄循环内重新计算速度.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # 将动作解释为目标位置位移并设置位置目标
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # 将动作解释为目标旋转（轴角）位移
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0] 单向旋转
        rot_actions = rot_actions * self.rot_threshold

        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # 为了加快学习速度，永远不允许策略移动超过5cm远离基座.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # 转换为四元数并设置旋转目标
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # 限制动作为直立.
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,  # 夹爪自由度位置目标
        )

    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """获取雅可比矩阵。设置 Franka 关节位置目标（手指）或关节扭矩（手臂）."""
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        # 为夹爪关节设置目标以使用 physx 的 PD 控制器
        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0  # 夹爪关节扭矩设为0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """检查哪些环境已终止.

        对于工厂重置逻辑，所有环境保持同步很重要
        (即，_get_dones 应该返回全真或全假).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1  # 超时终止
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """获取当前时间步的成功掩码."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # 获取持有物体的基座姿态和目标持有物体的基座姿态
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)  # XY平面距离
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]  # Z轴位移

        # 检查是否居中
        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # 高度阈值到目标
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold  # 高度阈值
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold  # 螺纹高度阈值
        else:
            raise NotImplementedError("Task not implemented")
        # 检查是否接近或低于
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        # 检查旋转
        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """跟踪回合统计信息并记录奖励."""
        # 仅在回合结束时记录回合成功率.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # 获取回合首次成功的时间.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # 仅记录成功回合.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        # 记录奖励
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _get_rewards(self):
        """更新奖励并计算成功统计信息."""
        # 获取当前时间步的成功和失败环境
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        # 计算奖励缓冲区
        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """计算当前时间步的奖励项."""
        rew_dict, rew_scales = {}, {}

        # 计算持有物体和固定物体在世界坐标系中的关键点位置
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        # 初始化关键点张量
        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        # 计算关键点位置
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)  # 关键点距离

        # 获取关键点系数
        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # 动作惩罚.
        action_penalty_ee = torch.norm(self.actions, p=2)  # 末端执行器动作惩罚
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)  # 动作梯度惩罚
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)  # 当前是否接触

        # 构建奖励字典和缩放字典
        rew_dict = {
            "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),  # 基线关键点奖励
            "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),   # 粗略关键点奖励
            "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),     # 精细关键点奖励
            "action_penalty_ee": action_penalty_ee,                          # 末端执行器动作惩罚
            "action_grad_penalty": action_grad_penalty,                      # 动作梯度惩罚
            "curr_engaged": curr_engaged.float(),                           # 当前接触奖励
            "curr_success": curr_successes.float(),                         # 当前成功奖励
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,    # 末端执行器动作惩罚缩放
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale, # 动作梯度惩罚缩放
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    def _reset_idx(self, env_ids):
        """我们假设所有环境将始终同时重置."""
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)  # 将资产设置为默认姿态
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)  # 将Franka设置为默认姿态
        self.step_sim_no_action()  # 不执行动作的情况下步进仿真

        self.randomize_initial_state(env_ids)  # 随机化初始状态

    def _set_assets_to_default_pose(self, env_ids):
        """在随机化之前将资产移动到默认姿态."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]  # 添加环境原点偏移
        held_state[:, 7:] = 0.0  # 速度设为0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]  # 添加环境原点偏移
        fixed_state[:, 7:] = 0.0  # 速度设为0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        """使用DLS IK设置机器人关节位置."""
        ik_time = 0.0
        while ik_time < 0.25:
            # 计算到目标的误差.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # 求解DLS问题.
            delta_dof_pos = factory_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # 更新自由度状态.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # 仿真并更新张量.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """获取帮助资产与指尖之间的默认相对姿态."""
        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height  # 设置Z轴位置
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length  # 减去手指垫长度
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
            held_asset_relative_pos[:, 0] += gear_base_offset[0]  # 设置X轴位置
            held_asset_relative_pos[:, 2] += gear_base_offset[2]  # 设置Z轴位置
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1  # 增加一半高度
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = factory_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        else:
            raise NotImplementedError("Task not implemented")

        # 设置相对四元数
        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        if self.cfg_task.name == "nut_thread":
            # 对于默认位置，沿框架的z轴旋转.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """将Franka返回到其默认关节位置."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25  # 计算夹爪宽度
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC 夹爪
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]  # 设置前7个关节位置
        joint_vel = torch.zeros_like(joint_pos)  # 关节速度设为0
        joint_effort = torch.zeros_like(joint_pos)  # 关节力设为0
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """在没有动作的情况下步进仿真。仅用于重置.

        此方法应仅在重置时调用，此时所有环境
        同时重置.
        """
        self.scene.write_data_to_sim()  # 将数据写入仿真
        self.sim.step(render=False)  # 步进仿真
        self.scene.update(dt=self.physics_dt)  # 更新场景
        self._compute_intermediate_values(dt=self.physics_dt)  # 计算中间值

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) iterative IK Method
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
