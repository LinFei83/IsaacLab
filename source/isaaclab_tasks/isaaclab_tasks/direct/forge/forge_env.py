# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)。
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat

# 导入工厂任务的工具函数
from isaaclab_tasks.direct.factory import factory_utils
# 导入工厂环境基类
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

# 导入 Forge 任务的工具函数
from . import forge_utils
# 导入 Forge 环境配置类
from .forge_env_cfg import ForgeEnvCfg


class ForgeEnv(FactoryEnv):
    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化额外的随机化和日志张量。"""
        super().__init__(cfg, render_mode, **kwargs)

        # 成功预测相关参数
        self.success_pred_scale = 0.0
        # 记录首次预测成功的时刻
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 翻转四元数，用于处理手部姿态
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # 力传感器信息
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        # 平滑后的力传感器数据
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        # 世界坐标系下的平滑力传感器数据
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

        # 设置用于随机化的标称动力学参数
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

    def _compute_intermediate_values(self, dt):
        """为力感知向观察值添加噪声。"""
        super()._compute_intermediate_values(dt)

        # 为指尖位置添加噪声
        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        # 为指尖旋转添加噪声
        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        # 确保四元数的实部和虚部符号正确
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # 使用带噪声的指尖位置重复有限差分计算线速度
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        # 如果没有添加速度，则计算状态差异
        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        # 确保旋转差异四元数的符号正确
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        # 只在z轴方向有角速度
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        # 更新并平滑力值
        # 获取力传感器的世界坐标系下的力数据
        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self.force_sensor_body_idx
        ]

        # 使用指数移动平均平滑力传感器数据
        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        # 转换力传感器坐标系
        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # 计算带噪声的力值
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= self.cfg.obs_rand.ft_force
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def _get_observations(self):
        """添加额外的 FORGE 观察值。"""
        # 获取工厂环境的观察值和状态值
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        # 计算带噪声的固定物体位置
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        # 克隆动作并重置部分维度
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        # 更新观察值字典
        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,  # 指尖位置
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,  # 指尖相对于固定物体的位置
            "fingertip_quat": self.noisy_fingertip_quat,  # 指尖四元数
            "force_threshold": self.contact_penalty_thresholds[:, None],  # 接触力阈值
            "ft_force": self.noisy_force,  # 力传感器力值
            "prev_actions": prev_actions,  # 上一时刻的动作
        })

        # 更新状态值字典
        state_dict.update({
            "ema_factor": self.ema_factor,  # 指数移动平均因子
            "ft_force": self.force_sensor_smooth[:, 0:3],  # 平滑后的力传感器力值
            "force_threshold": self.contact_penalty_thresholds[:, None],  # 接触力阈值
            "prev_actions": prev_actions,  # 上一时刻的动作
        })

        # 将观察值和状态值字典折叠为张量
        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _apply_action(self):
        """FORGE 动作被定义为相对于固定资产的目标。"""
        # 如果需要，更新中间值
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # 步骤 (0): 将动作缩放到允许的范围内
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # 步骤 (1): 计算末端执行器框架中的期望姿态目标
        # (1.a) 位置。动作框架被假设为螺栓的顶部（带噪声的估计）。
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) 强制执行旋转动作约束，只允许绕z轴旋转
        rot_actions[:, 0:2] = 0.0

        # 假设关节限制在世界框架的 (+x, -y) 象限中
        # 将动作映射到关节限制范围内 [-180, 90] 度
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # 关节限制
        # (1.c) 获取期望的方向目标
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        # 计算从螺栓框架到末端执行器框架的四元数变换
        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        # 计算末端执行器的期望姿态（未裁剪）
        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # 步骤 (2): 如果目标距离当前末端执行器姿态太远，则裁剪目标
        # (2.a): 裁剪位置目标
        # 计算位置误差，用于后续的动作惩罚
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos
        # 根据位置阈值裁剪位置误差
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        # 计算裁剪后的位置目标
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) 裁剪方向目标。使用欧拉角。我们假设接近直立，
        # 因此裁剪偏航角将有效地导致缓慢运动。裁剪时还需要确保避免关节限制。

        # (2.b.i) 获取当前和期望的欧拉角
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) 纠正运动方向以避免关节限制
        # 将偏航角映射到 [-125, 235] 度之间（使角度在连续范围内，不受关节限制中断）
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) 在正确的方向上裁剪运动
        # 计算偏航角误差，用于后续的动作惩罚
        self.delta_yaw = desired_yaw - curr_yaw
        # 根据旋转阈值裁剪偏航角误差
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        # 计算裁剪后的偏航角目标
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) 裁剪滚转角和俯仰角
        # 处理滚转角的符号问题
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        # 处理俯仰角的符号问题
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        # 计算滚转角误差并裁剪
        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        # 处理俯仰角的符号问题
        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        # 计算俯仰角误差并裁剪
        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        # 从欧拉角计算裁剪后的四元数目标
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        # 生成控制信号
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_rewards(self):
        """FORGE 奖励包括接触惩罚和成功预测误差。"""
        # 使用与工厂环境相同的基线奖励
        rew_buf = super()._get_rewards()

        rew_dict, rew_scales = {}, {}
        # 计算相对于资产的动作空间的动作惩罚
        # 位置误差归一化
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        # 旋转误差归一化
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]
        # 接触惩罚：当接触力超过阈值时施加惩罚
        contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        # 添加成功预测奖励
        # 检查是否为螺母螺纹任务
        check_rot = self.cfg_task.name == "nut_thread"
        # 获取当前成功状态
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        # 将策略预测的成功概率从 [-1, 1] 缩放到 [0, 1]
        policy_success_pred = (self.actions[:, 6] + 1) / 2
        # 计算成功预测误差
        success_pred_error = (true_successes.float() - policy_success_pred).abs()
        # 延迟成功预测惩罚，直到发生一些成功事件
        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        # 添加新的 FORGE 奖励项
        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,  # 动作惩罚
            "contact_penalty": contact_penalty,  # 接触惩罚
            "success_pred_error": success_pred_error,  # 成功预测误差
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,  # 动作惩罚系数
            "contact_penalty": -self.cfg_task.contact_penalty_scale,  # 接触惩罚系数
            "success_pred_error": -self.success_pred_scale,  # 成功预测误差系数
        }
        # 累加所有奖励项
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        # 记录 FORGE 指标
        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _reset_idx(self, env_ids):
        """执行额外的随机化。"""
        super()._reset_idx(env_ids)

        # 计算初始动作以正确进行EMA计算
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # 计算相对于螺栓的偏航角
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        # 计算指尖相对于螺栓的四元数和偏航角
        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        # 规范化偏航角到 [-π, π] 范围
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        # 将偏航角映射到动作空间 [-1, 1]
        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        # 初始化成功预测动作
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA随机化
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        # 为回合设置初始增益
        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        # 为比例增益添加噪声
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        # 为位置阈值添加噪声
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        # 为旋转阈值添加噪声
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        # 计算导数增益
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        # 随机化接触惩罚阈值
        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        # 随机化死区阈值
        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        # 重置力传感器平滑数据
        self.force_sensor_world_smooth[:, :] = 0.0

        # 随机化翻转四元数
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reset_buffers(self, env_ids):
        """重置额外的日志指标。"""
        super()._reset_buffers(env_ids)
        # 重置成功预测指标
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """记录指标以评估成功预测性能。"""
        # 记录奖励项的平均值
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        # 遍历所有阈值，计算和记录成功预测相关指标
        for thresh, first_success_tx in self.first_pred_success_tx.items():
            # 当前预测成功的环境
            curr_predicted_success = policy_success_pred > thresh
            # 首次预测成功的索引（之前未记录过成功）
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            # 更新首次预测成功的时刻
            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            # 只在回合结束时记录
            if torch.any(self.reset_buf):
                # 记录预测延迟
                # 筛选出既实际成功又预测成功的环境
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                # 计算平均延迟时间
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                # 计算正确预测的延迟时间（预测时间晚于实际成功时间）
                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                # 记录提前终止成功率（对于所有我们已"停止"的回合，我们是否成功了？）
                # 筛选出已预测成功的回合
                pred_success_idxs = first_success_tx != 0

                # 筛选出真正成功的预测（实际成功且发生在预测之前）
                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,  # 实际上已成功
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],  # 成功发生在预测之前
                )

                # 计算精确率（预测成功的回合中真正成功的比例）
                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                # 计算召回率（实际成功的回合中被正确预测的比例）
                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
