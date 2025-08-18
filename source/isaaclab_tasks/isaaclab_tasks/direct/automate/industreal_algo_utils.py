# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""IndustReal: 算法模块.

包含实现模拟感知策略更新(SAPU)、基于SDF的奖励和基于采样的课程(SBC)的函数。

不打算作为独立脚本执行。
"""

# Force garbage collection for large arrays
import gc
import numpy as np
import os

# from pysdf import SDF
import torch
import trimesh
from trimesh.exchange.load import load

# from urdfpy import URDF
import warp as wp

from isaaclab.utils.assets import retrieve_file_path

"""
模拟感知策略更新(SAPU)
"""


def load_asset_mesh_in_warp(held_asset_obj, fixed_asset_obj, num_samples, device):
    """为所有环境在Warp中创建网格对象。"""
    # 下载并加载持有资产对象文件
    retrieve_file_path(held_asset_obj, download_dir="./")
    plug_trimesh = load(os.path.basename(held_asset_obj))
    # plug_trimesh = load(held_asset_obj)
    # 下载并加载固定资产对象文件
    retrieve_file_path(fixed_asset_obj, download_dir="./")
    socket_trimesh = load(os.path.basename(fixed_asset_obj))
    # socket_trimesh = load(fixed_asset_obj)

    # 创建插头的Warp网格对象
    plug_wp_mesh = wp.Mesh(
        points=wp.array(plug_trimesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(plug_trimesh.faces.flatten(), dtype=wp.int32, device=device),
    )

    # 在网格表面上采样点
    sampled_points, _ = trimesh.sample.sample_surface_even(plug_trimesh, num_samples)
    wp_mesh_sampled_points = wp.array(sampled_points, dtype=wp.vec3, device=device)

    # 创建插座的Warp网格对象
    socket_wp_mesh = wp.Mesh(
        points=wp.array(socket_trimesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(socket_trimesh.faces.flatten(), dtype=wp.int32, device=device),
    )

    return plug_wp_mesh, wp_mesh_sampled_points, socket_wp_mesh


"""
基于SDF的奖励
"""


def get_sdf_reward(
    wp_plug_mesh,
    wp_plug_mesh_sampled_points,
    plug_pos,
    plug_quat,
    socket_pos,
    socket_quat,
    wp_device,
    device,
):
    """计算基于SDF的奖励。"""

    num_envs = len(plug_pos)
    sdf_reward = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    # 为每个环境计算SDF奖励
    for i in range(num_envs):

        # 创建插头网格的副本
        mesh_points = wp.clone(wp_plug_mesh.points)
        mesh_indices = wp.clone(wp_plug_mesh.indices)
        mesh_copy = wp.Mesh(points=mesh_points, indices=mesh_indices)

        # 将插头网格从当前姿态变换到目标姿态
        # 注意：在源OBJ文件中，当插头和插座装配时，
        # 它们的姿态是相同的
        goal_transform = wp.transform(socket_pos[i], socket_quat[i])
        wp.launch(
            kernel=transform_points,
            dim=len(mesh_copy.points),
            inputs=[mesh_copy.points, mesh_copy.points, goal_transform],
            device=wp_device,
        )

        # 重建BVH（参见 https://nvidia.github.io/warp/_build/html/modules/runtime.html#meshes）
        mesh_copy.refit()

        # 创建采样点的副本
        sampled_points = wp.clone(wp_plug_mesh_sampled_points)

        # 将采样点从原始插头姿态变换到当前插头姿态
        curr_transform = wp.transform(plug_pos[i], plug_quat[i])
        wp.launch(
            kernel=transform_points,
            dim=len(sampled_points),
            inputs=[sampled_points, sampled_points, curr_transform],
            device=wp_device,
        )

        # 获取变换点的SDF值
        sdf_dist = wp.zeros((len(sampled_points),), dtype=wp.float32, device=wp_device)
        wp.launch(
            kernel=get_batch_sdf,
            dim=len(sampled_points),
            inputs=[mesh_copy.id, sampled_points, sdf_dist],
            device=wp_device,
        )
        sdf_dist = wp.to_torch(sdf_dist)

        # 钳位等值面外的值并取绝对值
        sdf_dist = torch.where(sdf_dist < 0.0, 0.0, sdf_dist)

        # 计算平均SDF距离作为奖励
        sdf_reward[i] = torch.mean(sdf_dist)

        # 清理临时变量
        del mesh_copy
        del mesh_points
        del mesh_indices
        del sampled_points

    # 应用对数变换以获得最终奖励
    sdf_reward = -torch.log(sdf_reward)

    gc.collect()  # 强制垃圾回收以释放内存
    return sdf_reward


"""
基于采样的课程(SBC)
"""


def get_curriculum_reward_scale(cfg_task, curr_max_disp):
    """计算SBC的奖励缩放。"""

    # 计算训练开始时最大向下位移（最简单条件）
    # 和当前最大向下位移（基于当前课程阶段）之间的差异
    # 注意：随着课程变得更难，这个数值会增加
    curr_stage_diff = cfg_task.curriculum_height_bound[1] - curr_max_disp

    # 计算训练开始时最大向下位移（最简单条件）
    # 和最小向下位移（最难条件）之间的差异
    final_stage_diff = cfg_task.curriculum_height_bound[1] - cfg_task.curriculum_height_bound[0]

    # 计算奖励缩放
    reward_scale = curr_stage_diff / final_stage_diff + 1.0

    return reward_scale


def get_new_max_disp(curr_success, cfg_task, curr_max_disp):
    """根据成功率更新回合开始时插头的最大向下位移。"""

    if curr_success > cfg_task.curriculum_success_thresh:
        # 如果成功率高于阈值，减少最大向下位移直到最小值
        # 注意：height_step[0]是负数
        new_max_disp = max(
            curr_max_disp + cfg_task.curriculum_height_step[0],
            cfg_task.curriculum_height_bound[0],
        )

    elif curr_success < cfg_task.curriculum_failure_thresh:
        # 如果成功率低于阈值，增加最大向下位移直到最大值
        # 注意：height_step[1]是正数
        new_max_disp = min(
            curr_max_disp + cfg_task.curriculum_height_step[1],
            cfg_task.curriculum_height_bound[1],
        )

    else:
        # 保持当前最大向下位移
        new_max_disp = curr_max_disp

    return new_max_disp


"""
奖励和成功检查
"""


def get_keypoint_offsets(num_keypoints, device):
    """获取沿单位长度线均匀间隔的关键点，以0为中心。"""

    # 初始化关键点偏移量数组
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    # 在z轴上设置均匀分布的关键点坐标
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5

    return keypoint_offsets


def check_plug_close_to_socket(keypoints_plug, keypoints_socket, dist_threshold, progress_buf):
    """检查插头是否接近插座。"""

    # 计算插头和插座之间的关键点距离
    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)

    # 检查关键点距离是否低于阈值
    is_plug_close_to_socket = torch.where(
        torch.sum(keypoint_dist, dim=-1) < dist_threshold,
        torch.ones_like(progress_buf),
        torch.zeros_like(progress_buf),
    )

    return is_plug_close_to_socket


def check_plug_inserted_in_socket(
    plug_pos, socket_pos, keypoints_plug, keypoints_socket, success_height_thresh, close_error_thresh, progress_buf
):
    """检查插头是否插入插座。"""

    # 检查插头是否在装配状态的阈值距离内
    is_plug_below_insertion_height = plug_pos[:, 2] < socket_pos[:, 2] + success_height_thresh

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
    is_plug_inserted_in_socket = torch.logical_and(is_plug_below_insertion_height, is_plug_close_to_socket)

    return is_plug_inserted_in_socket


def get_engagement_reward_scale(plug_pos, socket_pos, is_plug_engaged_w_socket, success_height_thresh, device):
    """计算奖励的缩放。如果插头未与插座啮合，缩放为零。
    如果插头啮合，缩放与插头和插座底部之间的距离成比例。"""

    # 将缩放的默认值设为零
    num_envs = len(plug_pos)
    reward_scale = torch.zeros((num_envs,), dtype=torch.float32, device=device)

    # 对于插头和插座啮合的环境，计算正缩放
    engaged_idx = np.argwhere(is_plug_engaged_w_socket.cpu().numpy().copy()).squeeze()
    height_dist = plug_pos[engaged_idx, 2] - socket_pos[engaged_idx, 2]
    # 注意：边缘情况：如果success_height_thresh大于0.1，
    # 分母可能为负
    reward_scale[engaged_idx] = 1.0 / ((height_dist - success_height_thresh) + 0.1)

    return reward_scale


"""
Warp函数
"""


@wp.func
def mesh_sdf(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    """计算网格的符号距离函数(SDF)。"""
    # 初始化网格查询参数
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    # 查询网格上最接近点的信息
    res = wp.mesh_query_point(mesh, point, max_dist, sign, face_index, face_u, face_v)
    if res:
        # 获取网格上最接近的点
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        # 返回有符号距离
        return wp.length(point - closest) * sign
    # 如果没有找到接近的点，返回最大距离
    return max_dist


"""
Warp内核
"""


@wp.kernel
def get_batch_sdf(
    mesh: wp.uint64,
    queries: wp.array(dtype=wp.vec3),
    sdf_dist: wp.array(dtype=wp.float32),
):
    """批量计算SDF值。"""
    tid = wp.tid()

    q = queries[tid]  # 查询点
    max_dist = 1.5  # 网格上查询点的最大距离
    # max_dist = 0.0

    # sdf_dist[tid] = wp.mesh_query_point_sign_normal(mesh, q, max_dist)
    # 使用自定义函数计算SDF值
    sdf_dist[tid] = mesh_sdf(mesh, q, max_dist)


# 将点从源坐标系变换到目标坐标系
@wp.kernel
def transform_points(src: wp.array(dtype=wp.vec3), dest: wp.array(dtype=wp.vec3), xform: wp.transform):
    """变换点坐标。"""
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
    """计算相互穿透距离。"""
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
