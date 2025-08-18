# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import sys
import torch
import trimesh

import warp as wp

print("Python Executable:", sys.executable)
print("Python Path:", sys.path)

from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
sys.path.append(base_dir)

from isaaclab.utils.assets import retrieve_file_path

"""
初始化/采样
"""


def get_prev_success_init(held_asset_pose, fixed_asset_pose, success, N, device):
    """
    随机选择N个held_asset_pose和对应的fixed_asset_pose，
    这些索引处的成功值为1，并将它们作为torch张量返回。

    参数:
        held_asset_pose (np.ndarray): 持有资产姿态的Numpy数组。
        fixed_asset_pose (np.ndarray): 固定资产姿态的Numpy数组。
        success (np.ndarray): 成功值的Numpy数组（1表示成功，0表示失败）。
        N (int): 要选择的成功索引数量。
        device: torch设备。

    返回:
        tuple: (held_asset_poses, fixed_asset_poses)作为torch张量，如果未找到成功项则返回None。
    """
    # 获取成功值为1的索引
    success_indices = np.where(success == 1)[0]

    if success_indices.size == 0:
        return None  # 未找到成功项

    # 从成功索引中随机选择最多N个索引
    selected_indices = np.random.choice(success_indices, min(N, len(success_indices)), replace=False)

    return torch.tensor(held_asset_pose[selected_indices], device=device), torch.tensor(
        fixed_asset_pose[selected_indices], device=device
    )


def model_succ_w_gmm(held_asset_pose, fixed_asset_pose, success):
    """
    使用高斯混合模型(GMM)将成功率分布建模为持有资产和固定资产之间相对位置的函数。

    参数:
        held_asset_pose (np.ndarray): 形状为(N, 7)的数组，表示持有资产的位置。
        fixed_asset_pose (np.ndarray): 形状为(N, 7)的数组，表示固定资产的位置。
        success (np.ndarray): 形状为(N, 1)的数组，表示成功与否。

    返回:
        GaussianMixture: 拟合的GMM模型。

    示例:
        gmm = model_succ_dist_w_gmm(held_asset_pose, fixed_asset_pose, success)
        relative_pose = held_asset_pose - fixed_asset_pose
        # 计算给定相对位置下每个组件的概率:
        probabilities = gmm.predict_proba(relative_pose)
    """
    # 计算相对位置（持有资产相对于固定资产）
    relative_pos = held_asset_pose[:, :3] - fixed_asset_pose[:, :3]

    # 展平成功数组作为样本权重。
    # 这样，成功率更高的样本对模型的贡献更大。
    sample_weights = success.flatten()

    # 使用指定的组件数量初始化高斯混合模型。
    gmm = GaussianMixture(n_components=2, random_state=0)

    # 使用成功率指标的样本权重，在相对位置上拟合GMM。
    gmm.fit(relative_pos, sample_weight=sample_weights)

    return gmm


def sample_rel_pos_from_gmm(gmm, batch_size, device):
    """
    从拟合的高斯混合模型中采样一批相对姿态（持有资产相对于固定资产）。

    参数:
        gmm (GaussianMixture): 在相对姿态数据上拟合的高斯混合模型。
        batch_size (int): 要生成的样本数量。

    返回:
        torch.Tensor: 形状为(batch_size, 3)的张量，包含采样的相对姿态。
    """
    # 从高斯混合模型中采样batch_size个样本。
    samples, _ = gmm.sample(batch_size)

    # 将numpy数组转换为torch张量。
    samples_tensor = torch.from_numpy(samples).to(device)

    return samples_tensor


def model_succ_w_gp(held_asset_pose, fixed_asset_pose, success):
    """
    使用高斯过程分类器，根据持有资产相对于固定资产的相对位置来建模成功率分布。

    参数:
        held_asset_pose (np.ndarray): 形状为(N, 7)的数组，表示持有资产的姿态。
                                      假设前3列是(x, y, z)位置。
        fixed_asset_pose (np.ndarray): 形状为(N, 7)的数组，表示固定资产的姿态。
                                      假设前3列是(x, y, z)位置。
        success (np.ndarray): 形状为(N, 1)的数组，表示成功结果（例如，0表示失败，
                              1表示成功）。

    返回:
        GaussianProcessClassifier: 训练好的GP分类器，用于建模成功率。
    """
    # 计算相对位置（仅使用平移分量）
    relative_position = held_asset_pose[:, :3] - fixed_asset_pose[:, :3]

    # 将成功数组从(N, 1)展平为(N,)
    y = success.ravel()

    # 创建并拟合高斯过程分类器
    # gp = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gp = GaussianProcessRegressor(random_state=42)
    gp.fit(relative_position, y)

    return gp


def propose_failure_samples_batch_from_gp(
    gp_model, candidate_points, batch_size, device, method="ucb", kappa=2.0, xi=0.01
):
    """
    使用三种采集函数之一从易失败区域提出一批候选样本：
    'ucb' (上置信界), 'pi' (改进概率), 或 'ei' (期望改进)。

    在这种公式中，较低的预测成功率（接近0）是期望的，
    因此我们反转了典型的采集公式。

    参数:
        gp_model: 训练好的高斯过程模型（例如，GaussianProcessRegressor），
                  支持通过'predict'方法进行带有不确定性的预测（使用return_std=True）。
        candidate_points (np.ndarray): 形状为(n_candidates, d)的数组，表示候选相对位置。
        batch_size (int): 要提出的候选样本数量。
        method (str): 要使用的采集函数：'ucb', 'pi', 或 'ei'。默认是'ucb'。
        kappa (float): UCB的探索参数。默认是2.0。
        xi (float): PI和EI的探索参数。默认是0.01。

    返回:
        best_candidates (np.ndarray): 形状为(batch_size, d)的数组，包含选定的候选点。
        acquisition (np.ndarray): 为每个候选点计算的采集值。
    """
    # 获取每个候选点的预测均值和标准差。
    mu, sigma = gp_model.predict(candidate_points, return_std=True)
    # mu, sigma = gp_model.predict(candidate_points)

    # 根据选择的方法计算采集值。
    if method.lower() == "ucb":
        # 反转：我们希望低成功率（即低mu）和高不确定性（sigma）具有吸引力。
        acquisition = kappa * sigma - mu
    elif method.lower() == "pi":
        # 改进概率：预测值低于目标=0.0的可能性。
        Z = (-mu - xi) / (sigma + 1e-9)
        acquisition = norm.cdf(Z)
    elif method.lower() == "ei":
        # 期望改进
        Z = (-mu - xi) / (sigma + 1e-9)
        acquisition = (-mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        # 在sigma接近零的地方将采集值设为0。
        acquisition[sigma < 1e-9] = 0.0
    else:
        raise ValueError("未知的采集方法。请选择'ucb', 'pi', 或 'ei'。")

    # 选择前batch_size个候选点的索引（采集值最高的）。
    sorted_indices = np.argsort(acquisition)[::-1]  # 按降序排序
    best_indices = sorted_indices[:batch_size]
    best_candidates = candidate_points[best_indices]

    # 将numpy数组转换为torch张量。
    best_candidates_tensor = torch.from_numpy(best_candidates).to(device)

    return best_candidates_tensor, acquisition


def propose_success_samples_batch_from_gp(
    gp_model, candidate_points, batch_size, device, method="ucb", kappa=2.0, xi=0.01
):
    """
    使用三种采集函数之一从高成功率区域提出一批候选样本：
    'ucb' (上置信界), 'pi' (改进概率), 或 'ei' (期望改进)。

    在这种公式中，较高的预测成功率是期望的。
    假设GP模型通过其'predict'方法提供带有不确定性的预测（使用return_std=True）。

    参数:
        gp_model: 训练好的高斯过程模型（例如，GaussianProcessRegressor），
                  支持带有不确定性的预测。
        candidate_points (np.ndarray): 形状为(n_candidates, d)的数组，表示候选相对位置。
        batch_size (int): 要提出的候选样本数量。
        method (str): 要使用的采集函数：'ucb', 'pi', 或 'ei'。默认是'ucb'。
        kappa (float): UCB的探索参数。默认是2.0。
        xi (float): PI和EI的探索参数。默认是0.01。

    返回:
        best_candidates (np.ndarray): 形状为(batch_size, d)的数组，包含选定的候选点。
        acquisition (np.ndarray): 为每个候选点计算的采集值。
    """
    # 获取每个候选点的预测均值和标准差。
    mu, sigma = gp_model.predict(candidate_points, return_std=True)

    # 根据选择的方法计算采集值。
    if method.lower() == "ucb":
        # 对于最大化，UCB定义为μ + kappa * σ。
        acquisition = mu + kappa * sigma
    elif method.lower() == "pi":
        # 改进概率（最大化公式）。
        Z = (mu - 1.0 - xi) / (sigma + 1e-9)
        acquisition = norm.cdf(Z)
    elif method.lower() == "ei":
        # 期望改进（最大化公式）。
        Z = (mu - 1.0 - xi) / (sigma + 1e-9)
        acquisition = (mu - 1.0 - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        # 处理接近零的sigma值。
        acquisition[sigma < 1e-9] = 0.0
    else:
        raise ValueError("未知的采集方法。请选择'ucb', 'pi', 或 'ei'。")

    # 按采集值降序排序候选点并选择前batch_size个。
    sorted_indices = np.argsort(acquisition)[::-1]
    best_indices = sorted_indices[:batch_size]
    best_candidates = candidate_points[best_indices]

    # 将numpy数组转换为torch张量。
    best_candidates_tensor = torch.from_numpy(best_candidates).to(device)

    return best_candidates_tensor, acquisition


"""
Util Functions
"""


def get_gripper_open_width(obj_filepath):
    """获取夹爪打开宽度。"""

    retrieve_file_path(obj_filepath, download_dir="./")
    obj_mesh = trimesh.load_mesh(os.path.basename(obj_filepath))
    # obj_mesh = trimesh.load_mesh(obj_filepath)
    aabb = obj_mesh.bounds

    return min(0.04, (aabb[1][1] - aabb[0][1]) / 1.25)


"""
Imitation Reward
"""


def get_closest_state_idx(ref_traj, curr_ee_pos):
    """在参考轨迹中找到最接近状态的索引。"""

    # ref_traj.shape = (num_trajs, traj_len, 3)
    traj_len = ref_traj.shape[1]
    num_envs = curr_ee_pos.shape[0]

    # dist_from_all_state.shape = (num_envs, num_trajs, traj_len, 1)
    dist_from_all_state = torch.cdist(ref_traj.unsqueeze(0), curr_ee_pos.reshape(-1, 1, 1, 3), p=2)

    # dist_from_all_state_flatten.shape = (num_envs, num_trajs * traj_len)
    dist_from_all_state_flatten = dist_from_all_state.reshape(num_envs, -1)

    # min_dist_per_env.shape = (num_envs)
    min_dist_per_env = torch.amin(dist_from_all_state_flatten, dim=-1)

    # min_dist_idx.shape = (num_envs)
    min_dist_idx = torch.argmin(dist_from_all_state_flatten, dim=-1)

    # min_dist_traj_idx.shape = (num_envs)
    # min_dist_step_idx.shape = (num_envs)
    min_dist_traj_idx = min_dist_idx // traj_len
    min_dist_step_idx = min_dist_idx % traj_len

    return min_dist_traj_idx, min_dist_step_idx, min_dist_per_env


def get_reward_mask(ref_traj, curr_ee_pos, tolerance):
    """获取奖励掩码。"""

    _, min_dist_step_idx, _ = get_closest_state_idx(ref_traj, curr_ee_pos)
    selected_steps = torch.index_select(
        ref_traj, dim=1, index=min_dist_step_idx
    )  # selected_steps.shape = (num_trajs, num_envs, 3)

    x_min = torch.amin(selected_steps[:, :, 0], dim=0) - tolerance
    x_max = torch.amax(selected_steps[:, :, 0], dim=0) + tolerance
    y_min = torch.amin(selected_steps[:, :, 1], dim=0) - tolerance
    y_max = torch.amax(selected_steps[:, :, 1], dim=0) + tolerance

    x_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 0], x_max), torch.gt(curr_ee_pos[:, 0], x_min))
    y_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 1], y_max), torch.gt(curr_ee_pos[:, 1], y_min))
    pos_in_range = torch.logical_and(x_in_range, y_in_range).int()

    return pos_in_range


def get_imitation_reward_from_dtw(ref_traj, curr_ee_pos, prev_ee_traj, criterion, device):
    """基于动态时间规整获取模仿奖励。"""

    soft_dtw = torch.zeros((curr_ee_pos.shape[0]), device=device)
    prev_ee_pos = prev_ee_traj[:, 0, :]  # 选择机器人轨迹中的第一个末端执行器位置
    min_dist_traj_idx, min_dist_step_idx, min_dist_per_env = get_closest_state_idx(ref_traj, prev_ee_pos)

    # 为每个环境计算DTW奖励
    for i in range(curr_ee_pos.shape[0]):
        traj_idx = min_dist_traj_idx[i]
        step_idx = min_dist_step_idx[i]
        curr_ee_pos_i = curr_ee_pos[i].reshape(1, 3)

        # 注意：在参考轨迹中，索引越大 -> 越接近目标
        traj = ref_traj[traj_idx, step_idx:, :].reshape((1, -1, 3))

        _, curr_step_idx, _ = get_closest_state_idx(traj, curr_ee_pos_i)

        # 如果当前步索引为0，则选择当前位置作为轨迹
        if curr_step_idx == 0:
            selected_pos = ref_traj[traj_idx, step_idx, :].reshape((1, 1, 3))
            selected_traj = torch.cat([selected_pos, selected_pos], dim=1)
        else:
            # 否则选择从当前步到下一步的轨迹段
            selected_traj = ref_traj[traj_idx, step_idx : (curr_step_idx + step_idx), :].reshape((1, -1, 3))
        # 将之前的末端执行器轨迹与当前末端执行器位置连接
        eef_traj = torch.cat((prev_ee_traj[i, 1:, :], curr_ee_pos_i)).reshape((1, -1, 3))
        # 计算软DTW距离
        soft_dtw[i] = criterion(eef_traj, selected_traj)

    # 计算任务进度权重
    w_task_progress = 1 - (min_dist_step_idx / ref_traj.shape[1])

    # 计算模仿奖励
    # imitation_rwd = torch.exp(-soft_dtw)
    imitation_rwd = 1 - torch.tanh(soft_dtw)

    return imitation_rwd * w_task_progress


"""
Sampling-Based Curriculum (SBC)
"""


def get_new_max_disp(curr_success, cfg_task, curriculum_height_bound, curriculum_height_step, curr_max_disp):
    """根据成功率更新回合开始时插头的最大向下位移。"""

    if curr_success > cfg_task.curriculum_success_thresh:
        # 如果成功率高于阈值，增加最小向下位移直到最大值
        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 0] < curriculum_height_bound[:, 1],
            curr_max_disp + curriculum_height_step[:, 0],
            curriculum_height_bound[:, 1],
        )
    elif curr_success < cfg_task.curriculum_failure_thresh:
        # 如果成功率低于阈值，减少最小向下位移直到最小值
        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 1] > curriculum_height_bound[:, 0],
            curr_max_disp + curriculum_height_step[:, 1],
            curriculum_height_bound[:, 0],
        )
    else:
        # 保持当前最大向下位移
        new_max_disp = curr_max_disp

    return new_max_disp


"""
Bonus and Success Checking
"""


def check_plug_close_to_socket(keypoints_plug, keypoints_socket, dist_threshold, progress_buf):
    """检查插头是否接近插座。"""

    # 计算插头和插座之间的关键点距离
    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)

    # 检查关键点距离是否低于阈值
    is_plug_close_to_socket = torch.where(
        torch.mean(keypoint_dist, dim=-1) < dist_threshold,
        torch.ones_like(progress_buf),
        torch.zeros_like(progress_buf),
    )

    return is_plug_close_to_socket


def check_plug_inserted_in_socket(
    plug_pos, socket_pos, disassembly_dist, keypoints_plug, keypoints_socket, close_error_thresh, progress_buf
):
    """检查插头是否插入插座。"""

    # 检查插头是否在装配状态的阈值距离内
    is_plug_below_insertion_height = plug_pos[:, 2] < socket_pos[:, 2] + disassembly_dist
    is_plug_above_table_height = plug_pos[:, 2] > socket_pos[:, 2]

    is_plug_height_success = torch.logical_and(is_plug_below_insertion_height, is_plug_above_table_height)

    # 检查插头是否接近插座
    # 注意：此检查解决了插头在装配状态的阈值距离内，
    # 但插头在插座外的边缘情况
    is_plug_close_to_socket = check_plug_close_to_socket(
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        dist_threshold=close_error_thresh,
        progress_buf=progress_buf,
    )

    # 结合两个检查结果
    is_plug_inserted_in_socket = torch.logical_and(is_plug_height_success, is_plug_close_to_socket)

    return is_plug_inserted_in_socket


def get_curriculum_reward_scale(curr_max_disp, curriculum_height_bound):
    """计算SBC的奖励缩放。"""

    # 计算训练开始时最大向下位移（最简单条件）
    # 和当前最大向下位移（基于当前课程阶段）之间的差异
    # 注意：随着课程变得 harder，这个数值会增加
    curr_stage_diff = curr_max_disp - curriculum_height_bound[:, 0]

    # 计算训练开始时最大向下位移（最简单条件）
    # 和最小向下位移（最难条件）之间的差异
    final_stage_diff = curriculum_height_bound[:, 1] - curriculum_height_bound[:, 0]

    # 计算奖励缩放
    reward_scale = curr_stage_diff / final_stage_diff + 1.0

    return reward_scale.mean()


"""
Warp内核
"""


# 将点从源坐标系变换到目标坐标系
@wp.kernel
def transform_points(src: wp.array(dtype=wp.vec3), dest: wp.array(dtype=wp.vec3), xform: wp.transform):
    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m


# 返回查询点（例如，当前姿态下的插头顶点）
# 和网格表面（例如，当前姿态下的插座网格）之间的相互穿透距离
@wp.kernel
def get_interpen_dist(
    queries: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    interpen_dists: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    # 声明不会被修改的wp.mesh_query_point()参数
    q = queries[tid]  # 查询点
    max_dist = 1.5  # 网格上查询点的最大距离

    # 声明将被修改的wp.mesh_query_point()参数
    sign = float(
        0.0
    )  # -1如果查询点在网格内部；0如果在网格上；+1如果在网格外部（注意：网格必须是水密的！）
    face_idx = int(0)  # 最近面的索引
    face_u = float(0.0)  # 最近点的重心u坐标
    face_v = float(0.0)  # 最近点的重心v坐标

    # 获取网格上最接近查询点的点
    closest_mesh_point_exists = wp.mesh_query_point(mesh, q, max_dist, sign, face_idx, face_u, face_v)

    # 如果点存在于max_dist范围内
    if closest_mesh_point_exists:
        # 根据面索引和重心坐标获取网格上点的3D位置
        p = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)

        # 获取查询点和网格点之间的有符号距离
        delta = q - p
        signed_dist = sign * wp.length(delta)

        # 如果有符号距离为负
        if signed_dist < 0.0:
            # 存储相互穿透距离
            interpen_dists[tid] = signed_dist
