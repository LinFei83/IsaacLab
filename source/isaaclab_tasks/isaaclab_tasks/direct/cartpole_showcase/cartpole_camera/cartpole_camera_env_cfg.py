# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleRGBCameraEnvCfg as CartpoleCameraEnvCfg


def get_tiled_camera_cfg(data_type: str, width: int = 100, height: int = 100) -> TiledCameraCfg:
    """获取平铺相机配置
    
    Args:
        data_type (str): 数据类型，如 "rgb" 或 "depth"
        width (int): 相机图像宽度，默认为 100
        height (int): 相机图像高度，默认为 100
    
    Returns:
        TiledCameraCfg: 平铺相机配置对象
    """
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=[data_type],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=width,
        height=height,
    )


###
# 观测空间为 Box 类型
###


@configclass
class BoxBoxEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Box`` 类型，形状为 (height, width, 3))

        ===  ===
        索引  观测内容
        ===  ===
        -    RGB 图像
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车自由度施加的力矩比例: [-1, 1]
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class BoxDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Box`` 类型，形状为 (height, width, 3))

        ===  ===
        索引  观测内容
        ===  ===
        -    RGB 图像
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车自由度不施加力矩
        1    小车自由度施加负向最大力矩
        2    小车自由度施加正向最大力矩
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class BoxMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Box`` 类型，形状为 (height, width, 3))

        ===  ===
        索引  观测内容
        ===  ===
        -    RGB 图像
        ===  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车自由度不施加力矩
        1    小车自由度施加一半最大力矩
        2    小车自由度施加最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负向力矩 (一侧)
        1    正向力矩 (另一侧)
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# 观测空间为 Dict 类型
###


@configclass
class DictBoxEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Dict`` 类型，包含 2 个子空间)

        ================  ===
        键名              观测内容
        ================  ===
        joint-velocities  自由度速度
        camera            RGB 图像
        ================  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车自由度施加的力矩比例: [-1, 1]
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict({
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
    })  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class DictDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Dict`` 类型，包含 2 个子空间)

        ================  ===
        键名              观测内容
        ================  ===
        joint-velocities  自由度速度
        camera            RGB 图像
        ================  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车自由度不施加力矩
        1    小车自由度施加负向最大力矩
        2    小车自由度施加正向最大力矩
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict({
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
    })  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class DictMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Dict`` 类型，包含 2 个子空间)

        ================  ===
        键名              观测内容
        ================  ===
        joint-velocities  自由度速度
        camera            RGB 图像
        ================  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车自由度不施加力矩
        1    小车自由度施加一半最大力矩
        2    小车自由度施加最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负向力矩 (一侧)
        1    正向力矩 (另一侧)
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict({
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
    })  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# 观测空间为 Tuple 类型
###


@configclass
class TupleBoxEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Tuple`` 类型，包含 2 个子空间)

        ===  ===
        索引  观测内容
        ===  ===
        0    RGB 图像
        1    自由度速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车自由度施加的力矩比例: [-1, 1]
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class TupleDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Tuple`` 类型，包含 2 个子空间)

        ===  ===
        索引  观测内容
        ===  ===
        0    RGB 图像
        1    自由度速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车自由度不施加力矩
        1    小车自由度施加负向最大力矩
        2    小车自由度施加正向最大力矩
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class TupleMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * 观测空间 (``~gymnasium.spaces.Tuple`` 类型，包含 2 个子空间)

        ===  ===
        索引  观测内容
        ===  ===
        0    RGB 图像
        1    自由度速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车自由度不施加力矩
        1    小车自由度施加一半最大力矩
        2    小车自由度施加最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负向力矩 (一侧)
        1    正向力矩 (另一侧)
        ===  ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]
