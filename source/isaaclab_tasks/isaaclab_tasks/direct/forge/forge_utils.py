# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)。
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaacsim.core.utils.torch as torch_utils


def get_random_prop_gains(default_values, noise_levels, num_envs, device):
    """随机化控制器比例增益的辅助函数。
    
    通过添加噪声来随机化控制器的比例增益参数，以增加训练的鲁棒性。
    
    Args:
        default_values: 默认增益值
        noise_levels: 噪声水平，用于控制随机化的幅度
        num_envs: 环境数量
        device: 计算设备
    
    Returns:
        随机化后的比例增益值
    """
    # 生成随机噪声参数
    c_param_noise = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device)
    # 将噪声参数与噪声水平相乘
    c_param_noise = c_param_noise @ torch.diag(torch.tensor(noise_levels, dtype=torch.float32, device=device))
    # 计算增益乘数因子
    c_param_multiplier = 1.0 + c_param_noise
    # 随机决定是增加还是减少增益
    decrease_param_flag = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device) > 0.5
    # 根据标志位决定是增加还是减少增益
    c_param_multiplier = torch.where(decrease_param_flag, 1.0 / c_param_multiplier, c_param_multiplier)

    # 计算最终的比例增益值
    prop_gains = default_values * c_param_multiplier

    return prop_gains


def change_FT_frame(source_F, source_T, source_frame, target_frame):
    """将力/力矩读数从源坐标系转换到目标坐标系。
    
    使用现代机器人学中的变换公式来转换力/力矩传感器的读数。
    
    Args:
        source_F: 源坐标系下的力向量
        source_T: 源坐标系下的力矩向量
        source_frame: 源坐标系的位姿 (四元数, 位置)
        target_frame: 目标坐标系的位姿 (四元数, 位置)
    
    Returns:
        目标坐标系下的力和力矩向量
    """
    # Modern Robotics eq. 3.95
    # 计算源坐标系的逆变换
    source_frame_inv = torch_utils.tf_inverse(source_frame[0], source_frame[1])
    # 计算从源坐标系到目标坐标系的变换
    target_T_source_quat, target_T_source_pos = torch_utils.tf_combine(
        source_frame_inv[0], source_frame_inv[1], target_frame[0], target_frame[1]
    )
    # 变换力向量到目标坐标系
    target_F = torch_utils.quat_apply(target_T_source_quat, source_F)
    # 变换力矩向量到目标坐标系，考虑力臂的影响
    target_T = torch_utils.quat_apply(
        target_T_source_quat, (source_T + torch.cross(target_T_source_pos, source_F, dim=-1))
    )
    return target_F, target_T
