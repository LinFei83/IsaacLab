# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
特征提取器模块。
该模块包含用于从图像数据中提取特征的CNN网络，主要用于回归手中立方体的关键点位置。
"""

import glob
import os
import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


class FeatureExtractorNetwork(nn.Module):
    """用于从图像数据回归手中立方体关键点位置的CNN架构。
    
    该网络接收7通道的输入图像（RGB+深度+分割），通过卷积层提取特征，
    最终输出27维的向量，表示立方体的位置和角点坐标。
    """

    def __init__(self):
        """初始化特征提取网络。"""
        super().__init__()
        # 输入通道数：3(RGB) + 1(深度) + 3(分割) = 7
        num_channel = 7
        self.cnn = nn.Sequential(
            # 第一层卷积：7->16通道，6x6卷积核，步长2
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 58, 58]),
            # 第二层卷积：16->32通道，4x4卷积核，步长2
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 28, 28]),
            # 第三层卷积：32->64通道，4x4卷积核，步长2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 13, 13]),
            # 第四层卷积：64->128通道，3x3卷积核，步长2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 6, 6]),
            # 全局平均池化
            nn.AvgPool2d(6),
        )

        # 全连接层：128->27，输出立方体位置和角点坐标
        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )

        # 图像标准化处理
        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        """前向传播函数。
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为(N, H, W, 7)
            
        Returns:
            torch.Tensor: 预测的立方体位置和角点坐标，形状为(N, 27)
        """
        # 调整维度顺序：(N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # 对RGB通道进行标准化处理
        x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        # 对分割图像进行标准化处理
        x[:, 4:7, :, :] = self.data_transforms(x[:, 4:7, :, :])
        # 卷积特征提取
        cnn_x = self.cnn(x)
        # 全连接层输出结果
        out = self.linear(cnn_x.view(-1, 128))
        return out


@configclass
class FeatureExtractorCfg:
    """特征提取器模型配置类。"""

    train: bool = True
    """是否在rollout过程中训练特征提取器模型。默认为False。"""

    load_checkpoint: bool = False
    """是否从检查点加载特征提取器模型。默认为False。"""

    write_image_to_file: bool = False
    """是否将相机传感器的图像写入文件。默认为False。"""


class FeatureExtractor:
    """从图像数据中提取特征的类。
    
    该类使用CNN从标准化的RGB、深度和分割图像中回归关键点位置。
    如果train标志设置为True，则在rollout过程中训练CNN。
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str):
        """初始化特征提取器模型。
        
        Args:
            cfg (FeatureExtractorCfg): 特征提取器模型的配置。
            device (str): 运行模型的设备。
        """

        self.cfg = cfg
        self.device = device

        # 特征提取器模型
        self.feature_extractor = FeatureExtractorNetwork()
        self.feature_extractor.to(self.device)

        self.step_count = 0
        # 日志目录
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 加载检查点
        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        # 根据训练标志设置模型状态
        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预处理输入图像。
        
        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
            depth_img (torch.Tensor): 深度图像张量。形状: (N, H, W, 1)。
            segmentation_img (torch.Tensor): 分割图像张量。形状: (N, H, W, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 预处理后的RGB、深度和分割图像
        """
        # RGB图像归一化到[0,1]范围
        rgb_img = rgb_img / 255.0
        # 处理深度图像
        depth_img[depth_img == float("inf")] = 0
        depth_img /= 5.0
        depth_img /= torch.max(depth_img)
        # 处理分割图像
        segmentation_img = segmentation_img / 255.0
        mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
        segmentation_img -= mean_tensor
        return rgb_img, depth_img, segmentation_img

    def _save_images(self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor):
        """将图像缓冲区写入文件。
        
        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
            depth_img (torch.Tensor): 深度图像张量。形状: (N, H, W, 1)。
            segmentation_img (torch.Tensor): 分割图像张量。形状: (N, H, W, 3)。
        """
        save_images_to_file(rgb_img, "shadow_hand_rgb.png")
        save_images_to_file(depth_img, "shadow_hand_depth.png")
        save_images_to_file(segmentation_img, "shadow_hand_segmentation.png")

    def step(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor, gt_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用图像提取特征，并在train标志设置为True时训练模型。
        
        Args:
            rgb_img (torch.Tensor): RGB图像张量。形状: (N, H, W, 3)。
            depth_img (torch.Tensor): 深度图像张量。形状: (N, H, W, 1)。
            segmentation_img (torch.Tensor): 分割图像张量。形状: (N, H, W, 3)。
            gt_pose (torch.Tensor): 真实姿态张量（位置和角点）。形状: (N, 27)。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 姿态损失和预测的姿态。
        """

        # 预处理图像
        rgb_img, depth_img, segmentation_img = self._preprocess_images(rgb_img, depth_img, segmentation_img)

        # 保存图像到文件
        if self.cfg.write_image_to_file:
            self._save_images(rgb_img, depth_img, segmentation_img)

        # 训练模式
        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    # 拼接图像输入
                    img_input = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)
                    self.optimizer.zero_grad()

                    # 预测姿态
                    predicted_pose = self.feature_extractor(img_input)
                    # 计算姿态损失
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    # 反向传播和优化
                    pose_loss.backward()
                    self.optimizer.step()

                    # 定期保存检查点
                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        # 推理模式
        else:
            # 拼接图像输入
            img_input = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)
            # 预测姿态
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
