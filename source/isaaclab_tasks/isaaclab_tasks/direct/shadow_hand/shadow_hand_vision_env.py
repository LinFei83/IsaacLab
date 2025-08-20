# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand视觉环境模块。
该模块实现了基于视觉输入的Shadow Hand环境，使用CNN从图像中提取特征来辅助控制。
"""

from __future__ import annotations

import torch

# 从Isaac Sim 4.2开始，pxr.Semantics已弃用
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaacsim.core.utils.stage import get_current_stage

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale

from .feature_extractor import FeatureExtractor, FeatureExtractorCfg
from .shadow_hand_env_cfg import ShadowHandEnvCfg


@configclass
class ShadowHandVisionEnvCfg(ShadowHandEnvCfg):
    """Shadow Hand视觉环境配置类。"""
    
    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1225, env_spacing=2.0, replicate_physics=True)

    # 相机配置
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",  # 相机路径
        offset=TiledCameraCfg.OffsetCfg(pos=(0, -0.35, 1.0), rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"),  # 相机位置和旋转
        data_types=["rgb", "depth", "semantic_segmentation"],  # 数据类型：RGB、深度、语义分割
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)  # 相机参数
        ),
        width=120,  # 图像宽度
        height=120,  # 图像高度
    )
    
    # 特征提取器配置
    feature_extractor = FeatureExtractorCfg()

    # 环境参数
    observation_space = 164 + 27  # 状态观测 + 视觉CNN嵌入
    state_space = 187 + 27  # 非对称状态 + 视觉CNN嵌入


@configclass
class ShadowHandVisionEnvPlayCfg(ShadowHandVisionEnvCfg):
    """Shadow Hand视觉环境播放配置类（用于推理）。"""
    
    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    
    # CNN推理配置
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)


class ShadowHandVisionEnv(InHandManipulationEnv):
    """Shadow Hand视觉环境类。
    
    该环境使用相机捕获的图像来提取特征，辅助Shadow Hand进行在手操作任务。
    """
    
    cfg: ShadowHandVisionEnvCfg

    def __init__(self, cfg: ShadowHandVisionEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化Shadow Hand视觉环境。
        
        Args:
            cfg (ShadowHandVisionEnvCfg): 环境配置。
            render_mode (str | None): 渲染模式。
            **kwargs: 其他参数。
        """
        super().__init__(cfg, render_mode, **kwargs)
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(self.cfg.feature_extractor, self.device)
        # 隐藏目标立方体
        self.goal_pos[:, :] = torch.tensor([-0.2, 0.1, 0.6], device=self.device)
        # 关键点缓冲区
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        self.goal_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        """设置场景。"""
        # 添加手部、手中物体和目标物体
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        
        # 获取舞台
        stage = get_current_stage()
        # 为手中立方体添加语义信息
        prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")
        
        # 克隆和复制环境（此环境不需要过滤）
        self.scene.clone_environments(copy_from_source=False)
        # 将关节结构添加到场景中 - 必须注册到场景中才能使用EventManager进行随机化
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_image_observations(self):
        """计算图像观测。
        
        Returns:
            torch.Tensor: 图像观测张量。
        """
        # 生成手中立方体的真实关键点
        compute_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1), out=self.gt_keypoints)

        # 构造物体姿态（位置 + 角点坐标）
        object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)

        # 训练CNN回归关键点位置
        pose_loss, embeddings = self.feature_extractor.step(
            self._tiled_camera.data.output["rgb"],  # RGB图像
            self._tiled_camera.data.output["depth"],  # 深度图像
            self._tiled_camera.data.output["semantic_segmentation"][..., :3],  # 语义分割图像
            object_pose,  # 物体姿态
        )

        # 保存嵌入向量
        self.embeddings = embeddings.clone().detach()
        
        # 计算目标立方体的关键点
        compute_keypoints(
            pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=-1), out=self.goal_keypoints
        )

        # 构造观测（嵌入向量 + 目标关键点）
        obs = torch.cat(
            (
                self.embeddings,
                self.goal_keypoints.view(-1, 24),
            ),
            dim=-1,
        )

        # 记录CNN训练的姿态损失
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["pose_loss"] = pose_loss

        return obs

    def _compute_proprio_observations(self):
        """从物理系统计算本体感受观测。
        
        Returns:
            torch.Tensor: 本体感受观测张量。
        """
        obs = torch.cat(
            (
                # 手部观测
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),  # 关节位置
                self.cfg.vel_obs_scale * self.hand_dof_vel,  # 关节速度
                # 目标观测
                self.in_hand_pos,  # 手中物体位置
                self.goal_rot,  # 目标旋转
                # 指尖观测
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),  # 指尖位置
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),  # 指尖旋转
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),  # 指尖速度
                # 动作
                self.actions,  # 动作
            ),
            dim=-1,
        )
        return obs

    def _compute_states(self):
        """计算评论家（critic）的非对称状态。
        
        Returns:
            torch.Tensor: 评论家状态张量。
        """
        # 计算完整状态
        sim_states = self.compute_full_state()
        # 拼接嵌入向量
        state = torch.cat((sim_states, self.embeddings), dim=-1)
        return state

    def _get_observations(self) -> dict:
        """获取观测。
        
        Returns:
            dict: 包含策略观测和评论家状态的字典。
        """
        # 本体感受观测
        state_obs = self._compute_proprio_observations()
        # 视觉观测
        image_obs = self._compute_image_observations()
        # 拼接观测
        obs = torch.cat((state_obs, image_obs), dim=-1)
        
        # 非对称评论家状态
        self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        state = self._compute_states()

        observations = {"policy": obs, "critic": state}
        return observations


@torch.jit.script
def compute_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
):
    """计算立方体8个角点关键点的位置。
    
    Args:
        pose: 立方体中心的位置和方向。形状为(N, 7)
        num_keypoints: 要计算的关键点数量。默认为8
        size: 立方体X、Y、Z维度的长度。默认为[0.06, 0.06, 0.06]
        out: 存储关键点的缓冲区。如果为None，将创建新缓冲区。
        
    Returns:
        torch.Tensor: 关键点位置张量。
    """
    num_envs = pose.shape[0]
    if out is None:
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
        
    # 计算每个关键点的位置
    for i in range(num_keypoints):
        # 确定哪些维度需要取反
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * out[:, i, :]
        # 在世界坐标系中表达角点位置
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)

    return out
