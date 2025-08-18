# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

from .cartpole_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg

##
# 场景定义
# 在基础倒立摆场景的基础上添加相机传感器
##


@configclass
class CartpoleRGBCameraSceneCfg(CartpoleSceneCfg):
    """带RGB相机的倒立摆场景配置。
    
    继承基础倒立摆场景，并添加RGB相机传感器。
    """

    # 向场景中添加RGB相机 - 用于提供RGB视觉观测数据
    # 平铺相机配置 - 适用于多环境并行仿真
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        # 相机在USD场景中的路径，{ENV_REGEX_NS}会被替换为环境实例名
        prim_path="{ENV_REGEX_NS}/Camera",
        # 相机位置和朝向：位于倒立摆前方(-7m, 0m, 3m)，略微俯视
        # rot参数使用四元数(w,x,y,z)表示旋转，实现轻微下俯视角
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        # 数据类型：只采集RGB颜色信息
        data_types=["rgb"],
        # 针孔相机参数配置
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,           # 焦距24mm，控制视野范围
            focus_distance=400.0,        # 对焦距离400mm，适合观测距离
            horizontal_aperture=20.955,  # 水平孔径，影响景深和视野角度
            clipping_range=(0.1, 20.0)   # 裁剪范围：0.1-20米，只渲染这个范围内的物体
        ),
        # 图像分辨率：100x100像素，较低分辨率以提高训练效率
        width=100,
        height=100,
    )


@configclass
class CartpoleDepthCameraSceneCfg(CartpoleSceneCfg):
    """带深度相机的倒立摆场景配置。
    
    继承基础倒立摆场景，添加深度相机传感器。
    深度信息可以提供物体距离信息，但缺少颜色和纹理细节。
    """

    # 向场景中添加深度相机 - 用于提供距离信息观测
    # 深度相机配置 - 与RGB相机使用相同的位置和参数
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # 与RGB相机相同的位置和朝向设置
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        # 数据类型：深度信息（到相机的距离）
        data_types=["distance_to_camera"],
        # 相同的相机内参设置，保证一致的视野和成像质量
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )


##
# MDP设置
# 定义基于相机观测的强化学习环境的观测空间
##


@configclass
class RGBObservationsCfg:
    """RGB观测配置类 - 为MDP定义RGB图像观测。
    
    这个配置类定义了如何从RGB相机获取观测数据。
    """

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """使用RGB图像的策略组观测配置。
        
        与基础环境不同，这里的观测是原始RGB像素数据，
        需要通过卷积神经网络进行特征提取。
        """

        # RGB图像观测项 - 从平铺相机获取RGB数据
        # 返回的数据形状为(batch_size, height, width, 3)
        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})

        def __post_init__(self):
            # RGB图像观测的后处理配置
            # 禁用观测噪声 - 保持图像数据的原始性
            self.enable_corruption = False
            # 启用观测项连接 - 将图像数据展平为一个向量
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class DepthObservationsCfg:
    """深度观测配置类 - 为MDP定义深度图像观测。
    
    深度观测提供距离信息，对于空间定位和距离估计非常有用。
    """

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        """使用深度图像的策略组观测配置。
        
        深度数据是单通道的，每个像素表示该点到相机的距离。
        """

        # 深度图像观测项 - 从平铺相机获取深度数据
        # 返回的数据形状为(batch_size, height, width, 1)，值表示距离
        image = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "distance_to_camera"}
        )

    policy: ObsGroup = DepthCameraPolicyCfg()


@configclass
class ResNet18ObservationCfg:
    """ResNet18特征观测配置类 - 为MDP定义ResNet18特征提取观测。
    
    使用预训练的ResNet18模型从 RGB图像中提取高级特征，
    可以显著减少计算量并提高学习效率。
    """

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """使用冻结ResNet18提取的RGB图像特征的策略组观测配置。
        
        ResNet18是一个轻量级的卷积神经网络，适合实时特征提取。
        冻结权重意味着不更新预训练模型的参数。
        """

        # ResNet18特征提取观测项 - 使用预训练ResNet18模型处理RGB图像
        # 输出为512维特征向量（ResNet18最后一层的输出维度）
        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class TheiaTinyObservationCfg:
    """Theia-Tiny特征观测配置类 - 为MDP定义Theia-Tiny Transformer特征提取观测。
    
    Theia-Tiny是一个轻量级的视觉Transformer模型，提供了与卷积神经网络
    不同的视觉表示学习方法。
    """

    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """使用冻结Theia-Tiny Transformer提取的RGB图像特征的策略组观测配置。
        
        Theia-Tiny使用patch-based的注意力机制，能够捕捉全局和局部特征。
        """

        # Theia-Tiny特征提取观测项 - 使用视觉Transformer处理RGB图像
        # model_name指定使用的具体模型变体：patch16表示16x16的patch大小
        # model_device指定模型运行的GPU设备，确保高性能推理
        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv",  # 模型名称和配置
                "model_device": "cuda:0",                      # GPU设备指定
            },
        )

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


##
# 环境配置
# 定义各种基于相机观测的倒立摆环境变体
##


@configclass
class CartpoleRGBCameraEnvCfg(CartpoleEnvCfg):
    """带RGB相机的倒立摆环境配置。
    
    继承基础倒立摆环境，但使用RGB图像作为观测输入。
    适用于计算机视觉和端到端学习的研究。
    """

    # 使用带RGB相机的场景配置
    # num_envs=512: 较少的环境数量，因为图像处理需要更多计算资源
    # env_spacing=20: 更大的环境间距，防止相机视野中出现其他环境的干扰
    scene: CartpoleRGBCameraSceneCfg = CartpoleRGBCameraSceneCfg(num_envs=512, env_spacing=20)
    # 使用RGB图像观测配置替换默认的关节状态观测
    observations: RGBObservationsCfg = RGBObservationsCfg()

    def __post_init__(self):
        """后初始化 - 针对RGB相机环境的特殊设置。"""
        super().__post_init__()
        # 移除地面 - 防止地面阻挡相机视野，确保可以清晰观测倒立摆
        self.scene.ground = None
        # 视角设置 - 调整查看器相机位置以获得更好的观测视角
        self.viewer.eye = (7.0, 0.0, 2.5)      # 相机位置
        self.viewer.lookat = (0.0, 0.0, 2.5)   # 相机朝向


@configclass
class CartpoleDepthCameraEnvCfg(CartpoleEnvCfg):
    """带深度相机的倒立摆环境配置。
    
    使用深度信息而不是RGB信息，适用于研究空间感知和距离估计。
    """

    # 使用带深度相机的场景配置
    scene: CartpoleDepthCameraSceneCfg = CartpoleDepthCameraSceneCfg(num_envs=512, env_spacing=20)
    # 使用深度图像观测配置
    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self):
        """后初始化 - 针对深度相机环境的设置。"""
        super().__post_init__()
        # 移除地面 - 同样防止地面干扰深度检测
        self.scene.ground = None
        # 视角设置 - 与RGB相机环境相同
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class CartpoleResNet18CameraEnvCfg(CartpoleRGBCameraEnvCfg):
    """使用ResNet18特征作为观测的倒立摆环境配置。
    
    继承RGB相机环境，但使用预训练ResNet18提取的特征而不是原始像素。
    这可以显著减少观测维度并提高学习效率。
    """

    # 使用ResNet18特征提取观测配置替换RGB图像观测
    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()


@configclass
class CartpoleTheiaTinyCameraEnvCfg(CartpoleRGBCameraEnvCfg):
    """使用Theia-Tiny特征作为观测的倒立摆环境配置。
    
    使用轻量级视觉Transformer提取特征，提供与卷积网络不同的
    视觉表示学习方法。
    """

    # 使用Theia-Tiny Transformer特征提取观测配置
    observations: TheiaTinyObservationCfg = TheiaTinyObservationCfg()
