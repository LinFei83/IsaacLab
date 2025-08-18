# 版权所有 (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import numpy as np
import os
import torch

import carb
import isaacsim.core.utils.torch as torch_utils
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import axis_angle_from_quat

from . import automate_algo_utils as automate_algo
from . import automate_log_utils as automate_log
from . import factory_control as fc
from . import industreal_algo_utils as industreal_algo
from .assembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, AssemblyEnvCfg
from .soft_dtw_cuda import SoftDTW


class AssemblyEnv(DirectRLEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        """
        初始化装配环境
        
        Args:
            cfg: 环境配置
            render_mode: 渲染模式
            **kwargs: 其他参数
        """

        # 更新观测/状态数量
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        # 获取任务配置
        self.cfg_task = cfg.tasks[cfg.task_name]

        super().__init__(cfg, render_mode, **kwargs)

        # 设置刚体惯性
        self._set_body_inertias()
        # 初始化张量
        self._init_tensors()
        # 设置默认动力学参数
        self._set_default_dynamics_parameters()
        # 计算中间值
        self._compute_intermediate_values(dt=self.physics_dt)

        # 加载资产网格以用于基于SDF的密集奖励
        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.plug_mesh, self.plug_sample_points, self.socket_mesh = industreal_algo.load_asset_mesh_in_warp(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path,
            self.cfg_task.assembly_dir + self.cfg_task.fixed_asset_cfg.obj_path,
            self.cfg_task.num_mesh_sample_points,
            self.wp_device,
        )

        # 根据插头对象的边界框获取夹爪打开宽度
        self.gripper_open_width = automate_algo.get_gripper_open_width(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path
        )

        # 创建动态时间规整准则（稍后用于模仿奖励）
        self.soft_dtw_criterion = SoftDTW(use_cuda=True, gamma=self.cfg_task.soft_dtw_gamma)

        # 评估
        if self.cfg_task.if_logging_eval:
            self._init_eval_logging()

        if self.cfg_task.sample_from != "rand":
            self._init_eval_loading()

    def _init_eval_loading(self):
        """
        初始化评估加载
        从HDF5文件中加载评估数据，并根据采样方法创建高斯过程或高斯混合模型
        """
        eval_held_asset_pose, eval_fixed_asset_pose, eval_success = automate_log.load_log_from_hdf5(
            self.cfg_task.eval_filename
        )

        if self.cfg_task.sample_from == "gp":
            # 使用高斯过程建模成功样本
            self.gp = automate_algo.model_succ_w_gp(eval_held_asset_pose, eval_fixed_asset_pose, eval_success)
        elif self.cfg_task.sample_from == "gmm":
            # 使用高斯混合模型建模成功样本
            self.gmm = automate_algo.model_succ_w_gmm(eval_held_asset_pose, eval_fixed_asset_pose, eval_success)

    def _init_eval_logging(self):
        """
        初始化评估日志记录
        创建空的日志张量用于记录评估过程中的数据
        """

        # 创建空的日志张量
        # (position, quaternion) 位置和四元数
        self.held_asset_pose_log = torch.empty(
            (0, 7), dtype=torch.float32, device=self.device
        )
        self.fixed_asset_pose_log = torch.empty((0, 7), dtype=torch.float32, device=self.device)
        self.success_log = torch.empty((0, 1), dtype=torch.float32, device=self.device)

        # 在评估期间关闭SBC，以便所有插头都在插座外部初始化
        self.cfg_task.if_sbc = False

    def _set_body_inertias(self):
        """
        设置刚体惯性
        注意：这是为了考虑IGE中的asset_options.armature参数
        """
        # 获取机器人的惯性张量
        inertias = self._robot.root_physx_view.get_inertias()
        # 创建偏移量张量
        offset = torch.zeros_like(inertias)
        # 为惯性张量的对角线元素添加偏移量
        offset[:, :, [0, 4, 8]] += 0.01
        # 计算新的惯性张量
        new_inertias = inertias + offset
        # 设置机器人的惯性张量
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """
        设置定义动力学交互的参数
        包括默认增益、位置和旋转阈值，以及质量和摩擦系数
        """
        # 设置默认任务比例增益
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # 设置位置动作阈值
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        # 设置旋转动作阈值
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # 设置质量和摩擦系数
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """
        更新给定资产的材料属性
        设置静摩擦系数和动摩擦系数
        """
        # 获取资产的材料属性
        materials = asset.root_physx_view.get_material_properties()
        # 设置静摩擦系数
        materials[..., 0] = value  # Static friction.
        # 设置动摩擦系数
        materials[..., 1] = value  # Dynamic friction.
        # 获取环境ID
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        # 设置资产的材料属性
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """
        初始化张量
        一次性初始化所有需要的张量，包括控制目标、资产位置、关键点等
        """
        # 初始化单位四元数 (1, 0, 0, 0)
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # 控制目标张量
        # 关节位置控制目标
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        # 指尖中点位置控制目标
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # 指尖中点方向控制目标
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # 固定资产相关张量
        # 固定资产动作框架位置
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        # 固定资产观测框架位置
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        # 固定资产位置观测噪声初始化
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # 抓取资产相关张量
        # 抓取资产基座偏移量
        held_base_x_offset = 0.0
        held_base_z_offset = 0.0

        # 抓取资产局部位置
        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        # 抓取资产局部方向
        self.held_base_quat_local = self.identity_quat.clone().detach()

        # 抓取资产位置和方向
        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        # 加载装配信息
        self.plug_grasps, self.disassembly_dists = self._load_assembly_info()
        # 获取课程学习信息
        self.curriculum_height_bound, self.curriculum_height_step = self._get_curriculum_info(self.disassembly_dists)
        # 加载拆卸数据
        self._load_disassembly_data()

        # 根据装配ID从JSON文件加载抓取姿态
        # 抓取姿态张量
        # 手掌到手指中心的偏移量
        self.palm_to_finger_center = (
            torch.tensor([0.0, 0.0, -self.cfg_task.palm_to_finger_dist], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        # 机器人到夹爪的方向变换
        self.robot_to_gripper_quat = (
            torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        # 插头抓取位置局部坐标
        self.plug_grasp_pos_local = self.plug_grasps[: self.num_envs, :3]
        # 插头抓取方向局部坐标
        self.plug_grasp_quat_local = torch.roll(self.plug_grasps[: self.num_envs, 3:], -1, 1)

        # 计算机体索引
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # 有限差分相关张量
        # 上次更新时间戳 (用于有限差分计算刚体速度)
        self.last_update_timestamp = 0.0
        # 上一时刻指尖位置
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # 上一时刻指尖方向
        self.prev_fingertip_quat = self.identity_quat.clone()
        # 上一时刻关节位置
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # 关键点张量
        # 目标抓取资产基座位置
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # 目标抓取资产基座方向
        self.target_held_base_quat = self.identity_quat.clone().detach()

        # 计算关键点偏移量
        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        # 抓取资产关键点位置
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        # 固定资产关键点位置
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # 用于计算目标姿态的张量
        # 固定资产成功位置局部坐标
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_success_pos_local[:, 2] = 0.0

        # 环境成功状态记录
        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # 课程学习 (SBC)
        if self.cfg_task.if_sbc:
            # 如果启用SBC，当前最大位移为课程高度边界的第一列
            self.curr_max_disp = self.curriculum_height_bound[:, 0]
        else:
            # 如果不启用SBC，当前最大位移为课程高度边界的第二列
            self.curr_max_disp = self.curriculum_height_bound[:, 1]

    def _load_assembly_info(self):
        """
        为每个环境中的插头加载抓取姿态和拆卸距离
        从JSON文件中读取抓取姿态和拆卸距离信息
        """

        retrieve_file_path(self.cfg_task.plug_grasp_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.plug_grasp_json)) as f:
            plug_grasp_dict = json.load(f)
        plug_grasps = [plug_grasp_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        retrieve_file_path(self.cfg_task.disassembly_dist_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.disassembly_dist_json)) as f:
            disassembly_dist_dict = json.load(f)
        disassembly_dists = [disassembly_dist_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        return torch.as_tensor(plug_grasps).to(self.device), torch.as_tensor(disassembly_dists).to(self.device)

    def _get_curriculum_info(self, disassembly_dists):
        """
        计算每个环境中基于采样的课程学习(SBC)的范围和步长
        根据拆卸距离和配置参数计算课程学习的高度边界和步长
        """

        curriculum_height_bound = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        curriculum_height_step = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        curriculum_height_bound[:, 1] = disassembly_dists + self.cfg_task.curriculum_freespace_range

        curriculum_height_step[:, 0] = curriculum_height_bound[:, 1] / self.cfg_task.num_curriculum_step
        curriculum_height_step[:, 1] = -curriculum_height_step[:, 0] / 2.0

        return curriculum_height_bound, curriculum_height_step

    def _load_disassembly_data(self):
        """
        加载预收集的拆卸轨迹（仅包含末端执行器位置）
        从JSON文件中读取拆卸轨迹数据，用于模仿学习
        """

        retrieve_file_path(self.cfg_task.disassembly_path_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.disassembly_path_json)) as f:
            disassembly_traj = json.load(f)

        eef_pos_traj = []

        for i in range(len(disassembly_traj)):
            curr_ee_traj = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))
            curr_ee_goal = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))[0, :]

            # offset each trajectory to be relative to the goal
            eef_pos_traj.append(curr_ee_traj - curr_ee_goal)

        self.eef_pos_traj = torch.tensor(eef_pos_traj, dtype=torch.float32, device=self.device).squeeze()

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """
        初始化仿真场景
        设置地面、桌子、机器人、固定资产和抓取资产，并添加光源
        """
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = RigidObject(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.rigid_objects["held_asset"] = self._held_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """
        获取从原始张量计算出的值，包括添加噪声
        计算固定资产和抓取资产的位置、方向、速度等中间值
        """
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.gripper_goal_quat,
            self.gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_observations(self):
        """
        使用非对称评论器获取演员/评论器输入
        根据配置计算观测值和状态值
        """

        obs_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
        }

        state_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "joint_vel": self.joint_vel[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
        }
        # obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ['prev_actions']]
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order]
        obs_tensors = torch.cat(obs_tensors, dim=-1)

        # state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ['prev_actions']]
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order]
        state_tensors = torch.cat(state_tensors, dim=-1)

        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """
        重置缓冲区
        将环境的成功状态重置为0
        """
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """
        应用策略动作并进行平滑处理
        在每个物理步骤之前应用动作，并对动作进行指数移动平均平滑
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

    def move_gripper_in_place(self, ctrl_target_gripper_dof_pos):
        """
        在夹爪闭合时保持夹爪在当前位置
        通过设置夹爪的目标位置来实现夹爪的闭合控制
        """
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """
        应用策略动作作为从当前位置的增量目标
        将策略输出的动作转换为机器人末端执行器的位置和方向目标
        """
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """
        使用临界阻尼设置机器人增益
        计算比例增益和导数增益，用于机器人的力控制
        """
        # 设置任务比例增益
        self.task_prop_gains = prop_gains
        # 计算任务导数增益 (临界阻尼)
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        # 调整旋转导数增益
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """
        获取雅可比矩阵，设置Franka自由度位置目标（手指）或自由度力矩（手臂）
        计算关节力矩和施加的外力/力矩，用于控制机器人
        """
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use GYM's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """
        更新用于奖励和观测的中间值
        检查是否达到最大episode长度，如果是则结束episode
        """
        # 计算中间值
        self._compute_intermediate_values(dt=self.physics_dt)
        # 检查是否超时
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_rewards(self):
        """
        更新奖励并计算成功统计信息
        计算当前时间步的奖励值，包括SDF奖励、模仿奖励和成功奖励
        """
        # Get successful and failed envs at current timestep

        curr_successes = automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf,
        )

        rew_buf = self._update_rew_buf(curr_successes)
        self.ep_succeeded = torch.logical_or(self.ep_succeeded, curr_successes)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(self.ep_succeeded) / self.num_envs

            sbc_rwd_scale = automate_algo.get_curriculum_reward_scale(
                curr_max_disp=self.curr_max_disp,
                curriculum_height_bound=self.curriculum_height_bound,
            )

            rew_buf *= sbc_rwd_scale

            if self.cfg_task.if_sbc:

                self.curr_max_disp = automate_algo.get_new_max_disp(
                    curr_success=torch.count_nonzero(self.ep_succeeded) / self.num_envs,
                    cfg_task=self.cfg_task,
                    curriculum_height_bound=self.curriculum_height_bound,
                    curriculum_height_step=self.curriculum_height_step,
                    curr_max_disp=self.curr_max_disp,
                )

            self.extras["curr_max_disp"] = self.curr_max_disp

            if self.cfg_task.if_logging_eval:
                self.success_log = torch.cat([self.success_log, self.ep_succeeded.reshape((self.num_envs, 1))], dim=0)

                if self.success_log.shape[0] >= self.cfg_task.num_eval_trials:
                    automate_log.write_log_to_hdf5(
                        self.held_asset_pose_log,
                        self.fixed_asset_pose_log,
                        self.success_log,
                        self.cfg_task.eval_filename,
                    )
                    exit(0)

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """
        计算当前时间步的奖励
        根据SDF奖励、模仿奖励和成功奖励计算总奖励
        """
        rew_dict = dict({})

        # SDF-based reward.
        rew_dict["sdf"] = industreal_algo.get_sdf_reward(
            self.plug_mesh,
            self.plug_sample_points,
            self.held_pos,
            self.held_quat,
            self.fixed_pos,
            self.fixed_quat,
            self.wp_device,
            self.device,
        )

        rew_dict["curr_successes"] = curr_successes.clone().float()

        # Imitation Reward: Calculate reward
        curr_eef_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).reshape(
            -1, 3
        )  # relative position instead of absolute position
        rew_dict["imitation"] = automate_algo.get_imitation_reward_from_dtw(
            self.eef_pos_traj, curr_eef_pos, self.prev_fingertip_midpoint_pos, self.soft_dtw_criterion, self.device
        )

        self.prev_fingertip_midpoint_pos = torch.cat(
            (self.prev_fingertip_midpoint_pos[:, 1:, :], curr_eef_pos.unsqueeze(1).clone().detach()), dim=1
        )

        rew_buf = (
            self.cfg_task.sdf_rwd_scale * rew_dict["sdf"]
            + self.cfg_task.imitation_rwd_scale * rew_dict["imitation"]
            + rew_dict["curr_successes"]
        )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        重置环境索引
        我们假设所有环境将始终同时重置
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        if self.cfg_task.if_logging_eval:
            self.held_asset_pose_log = torch.cat(
                [self.held_asset_pose_log, torch.cat([self.held_pos, self.held_quat], dim=1)], dim=0
            )
            self.fixed_asset_pose_log = torch.cat(
                [self.fixed_asset_pose_log, torch.cat([self.fixed_pos, self.fixed_quat], dim=1)], dim=0
            )

        prev_fingertip_midpoint_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).unsqueeze(
            1
        )  # (num_envs, 1, 3)
        self.prev_fingertip_midpoint_pos = torch.repeat_interleave(
            prev_fingertip_midpoint_pos, self.cfg_task.num_point_robot_traj, dim=1
        )  # (num_envs, num_point_robot_traj, 3)

    def _set_assets_to_default_pose(self, env_ids):
        """
        在随机化之前将资产移动到默认姿态
        将抓取资产和固定资产重置到默认位置和方向
        """
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def _move_gripper_to_grasp_pose(self, env_ids):
        """
        定义插头的抓取姿态并将夹爪移动到该姿态
        计算夹爪的目标位置和方向，并移动夹爪到该位置
        """

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            gripper_goal_quat,
            gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = gripper_goal_pos.clone()

        # Set target rot
        self.ctrl_target_fingertip_midpoint_quat = gripper_goal_quat.clone()

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def set_pos_inverse_kinematics(self, env_ids):
        """
        使用DLS逆运动学设置机器人关节位置
        通过逆运动学计算机器人关节位置，使夹爪达到目标姿态
        """
        ik_time = 0.0
        while ik_time < 0.50:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _set_franka_to_default_pose(self, joints, env_ids):
        """
        将Franka机器人返回到默认关节位置
        设置机器人的关节位置和速度到默认值
        """
        gripper_width = self.gripper_open_width
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """
        在没有动作的情况下步进仿真，用于重置
        执行仿真步骤并更新场景数据
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_fixed_initial_state(self, env_ids):
        """
        随机化固定资产的初始状态
        设置固定资产的位置、方向和速度，并添加观测噪声
        """

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
        fixed_state[:, 2] += self.cfg_task.fixed_asset_z_offset

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
        self._fixed_asset.write_root_state_to_sim(fixed_state, env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids, pre_grasp):
        """
        随机化抓取资产的初始状态
        根据课程学习策略设置抓取资产的位置
        """

        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp
        if pre_grasp:
            self.curriculum_disp = self.curr_max_disp + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )

            if self.cfg_task.sample_from == "rand":

                rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
                held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                held_asset_init_pos_rand = torch.tensor(
                    self.cfg_task.held_asset_init_pos_noise, dtype=torch.float32, device=self.device
                )
                self.held_pos_init_rand = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)

            if self.cfg_task.sample_from == "gp":
                rand_sample = torch.rand((self.cfg_task.num_gp_candidates, 3), dtype=torch.float32, device=self.device)
                held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                held_asset_init_pos_rand = torch.tensor(
                    self.cfg_task.held_asset_init_pos_noise, dtype=torch.float32, device=self.device
                )
                held_asset_init_candidates = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)
                self.held_pos_init_rand, _ = automate_algo.propose_failure_samples_batch_from_gp(
                    self.gp, held_asset_init_candidates.cpu().detach().numpy(), len(env_ids), self.device
                )

            if self.cfg_task.sample_from == "gmm":
                self.held_pos_init_rand = automate_algo.sample_rel_pos_from_gmm(self.gmm, len(env_ids), self.device)

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0

        held_state[env_ids, 2] += self.curriculum_disp

        plug_in_freespace_idx = torch.argwhere(self.curriculum_disp > self.disassembly_dists)
        held_state[plug_in_freespace_idx, :2] += self.held_pos_init_rand[plug_in_freespace_idx, :2]

        self._held_asset.write_root_state_to_sim(held_state)
        self._held_asset.reset()

        self.step_sim_no_action()

    def randomize_initial_state(self, env_ids):
        """
        随机化初始状态并执行任何episode级别的随机化
        设置资产的初始位置和方向，并执行夹爪的抓取动作
        """
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        self.randomize_fixed_initial_state(env_ids)

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        self.randomize_held_initial_state(env_ids, pre_grasp=True)

        self._move_gripper_to_grasp_pose(env_ids)

        self.randomize_held_initial_state(env_ids, pre_grasp=False)

        # Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place(ctrl_target_gripper_dof_pos=0.0)
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
