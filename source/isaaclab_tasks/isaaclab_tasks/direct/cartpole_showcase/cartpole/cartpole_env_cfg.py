# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gymnasium import spaces

from isaaclab.utils import configclass

from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

###
# 观察空间为 Box 类型
###


@configclass
class BoxBoxEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Box`` 类型，形状为 (4,))

        ===  ===
        索引  观察值
        ===  ===
        0    杆关节位置
        1    杆关节速度
        2    小车关节位置
        3    小车关节速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车关节力矩比例: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # 或简化为: 4 或 [4]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # 或简化为: 1 或 [1]


@configclass
class BoxDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Box`` 类型，形状为 (4,))

        ===  ===
        索引  观察值
        ===  ===
        0    杆关节位置
        1    杆关节速度
        2    小车关节位置
        3    小车关节速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车关节零力矩
        1    小车关节负最大力矩
        2    小车关节正最大力矩
        ===  ===
    """

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # 或简化为: 4 或 [4]
    action_space = spaces.Discrete(3)  # 或简化为: {3}


@configclass
class BoxMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Box`` 类型，形状为 (4,))

        ===  ===
        索引  观察值
        ===  ===
        0    杆关节位置
        1    杆关节速度
        2    小车关节位置
        3    小车关节速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车关节零力矩
        1    小车关节半最大力矩
        2    小车关节最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负力矩 (一侧)
        1    正力矩 (另一侧)
        ===  ===
    """

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # 或简化为: 4 或 [4]
    action_space = spaces.MultiDiscrete([3, 2])  # 或简化为: [{3}, {2}]


###
# 观察空间为 Discrete 类型
###


@configclass
class DiscreteBoxEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Discrete`` 类型，包含 16 个元素)

        ===  ===
        编号  观察值 (值的符号: 杆位置, 小车位置, 杆速度, 小车速度)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车关节力矩比例: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Discrete(16)  # 或简化为: {16}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # 或简化为: 1 或 [1]


@configclass
class DiscreteDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Discrete`` 类型，包含 16 个元素)

        ===  ===
        编号  观察值 (值的符号: 杆位置, 小车位置, 杆速度, 小车速度)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车关节零力矩
        1    小车关节负最大力矩
        2    小车关节正最大力矩
        ===  ===
    """

    observation_space = spaces.Discrete(16)  # 或简化为: {16}
    action_space = spaces.Discrete(3)  # 或简化为: {3}


@configclass
class DiscreteMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Discrete`` 类型，包含 16 个元素)

        ===  ===
        编号  观察值 (值的符号: 杆位置, 小车位置, 杆速度, 小车速度)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
        ===  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车关节零力矩
        1    小车关节半最大力矩
        2    小车关节最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负力矩 (一侧)
        1    正力矩 (另一侧)
        ===  ===
    """

    observation_space = spaces.Discrete(16)  # 或简化为: {16}
    action_space = spaces.MultiDiscrete([3, 2])  # 或简化为: [{3}, {2}]


###
# 观察空间为 MultiDiscrete 类型
###


@configclass
class MultiDiscreteBoxEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 4 个离散空间)

        ===  ===
        编号  观察值 (离散空间 0)
        ===  ===
        0    负杆位置 (-)
        1    零或正杆位置 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 1)
        ===  ===
        0    负小车位置 (-)
        1    零或正小车位置 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 2)
        ===  ===
        0    负杆速度 (-)
        1    零或正杆速度 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 3)
        ===  ===
        0    负小车速度 (-)
        1    零或正小车速度 (+)
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车关节力矩比例: [-1, 1]
        ===  ===
    """

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # 或简化为: [{2}, {2}, {2}, {2}]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # 或简化为: 1 或 [1]


@configclass
class MultiDiscreteDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 4 个离散空间)

        ===  ===
        编号  观察值 (离散空间 0)
        ===  ===
        0    负杆位置 (-)
        1    零或正杆位置 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 1)
        ===  ===
        0    负小车位置 (-)
        1    零或正小车位置 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 2)
        ===  ===
        0    负杆速度 (-)
        1    零或正杆速度 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 3)
        ===  ===
        0    负小车速度 (-)
        1    零或正小车速度 (+)
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车关节零力矩
        1    小车关节负最大力矩
        2    小车关节正最大力矩
        ===  ===
    """

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # 或简化为: [{2}, {2}, {2}, {2}]
    action_space = spaces.Discrete(3)  # 或简化为: {3}


@configclass
class MultiDiscreteMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 4 个离散空间)

        ===  ===
        编号  观察值 (离散空间 0)
        ===  ===
        0    负杆位置 (-)
        1    零或正杆位置 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 1)
        ===  ===
        0    负小车位置 (-)
        1    零或正小车位置 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 2)
        ===  ===
        0    负杆速度 (-)
        1    零或正杆速度 (+)
        ===  ===

        ===  ===
        编号  观察值 (离散空间 3)
        ===  ===
        0    负小车速度 (-)
        1    零或正小车速度 (+)
        ===  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车关节零力矩
        1    小车关节半最大力矩
        2    小车关节最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负力矩 (一侧)
        1    正力矩 (另一侧)
        ===  ===
    """

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # 或简化为: [{2}, {2}, {2}, {2}]
    action_space = spaces.MultiDiscrete([3, 2])  # 或简化为: [{3}, {2}]


###
# 观察空间为 Dict 类型
###


@configclass
class DictBoxEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Dict`` 类型，包含 2 个子空间)

        ================  ===
        键名               观察值
        ================  ===
        joint-positions   关节位置
        joint-velocities  关节速度
        ================  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车关节力矩比例: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Dict({
        "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    })  # 或简化为: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # 或简化为: 1 或 [1]


@configclass
class DictDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Dict`` 类型，包含 2 个子空间)

        ================  ===
        键名               观察值
        ================  ===
        joint-positions   关节位置
        joint-velocities  关节速度
        ================  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车关节零力矩
        1    小车关节负最大力矩
        2    小车关节正最大力矩
        ===  ===
    """

    observation_space = spaces.Dict({
        "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    })  # 或简化为: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.Discrete(3)  # 或简化为: {3}


@configclass
class DictMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Dict`` 类型，包含 2 个子空间)

        ================  ===
        键名               观察值
        ================  ===
        joint-positions   关节位置
        joint-velocities  关节速度
        ================  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车关节零力矩
        1    小车关节半最大力矩
        2    小车关节最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负力矩 (一侧)
        1    正力矩 (另一侧)
        ===  ===
    """

    observation_space = spaces.Dict({
        "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    })  # 或简化为: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.MultiDiscrete([3, 2])  # 或简化为: [{3}, {2}]


###
# 观察空间为 Tuple 类型
###


@configclass
class TupleBoxEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Tuple`` 类型，包含 2 个子空间)

        ===  ===
        索引  观察值
        ===  ===
        0    关节位置
        1    关节速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Box`` 类型，形状为 (1,))

        ===  ===
        索引  动作
        ===  ===
        0    小车关节力矩比例: [-1, 1]
        ===  ===
    """

    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # 或简化为: (2, 2)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # 或简化为: 1 或 [1]


@configclass
class TupleDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Tuple`` 类型，包含 2 个子空间)

        ===  ===
        索引  观察值
        ===  ===
        0    关节位置
        1    关节速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.Discrete`` 类型，包含 3 个元素)

        ===  ===
        编号  动作
        ===  ===
        0    小车关节零力矩
        1    小车关节负最大力矩
        2    小车关节正最大力矩
        ===  ===
    """

    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # 或简化为: (2, 2)
    action_space = spaces.Discrete(3)  # 或简化为: {3}


@configclass
class TupleMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * 观察空间 (``~gymnasium.spaces.Tuple`` 类型，包含 2 个子空间)

        ===  ===
        索引  观察值
        ===  ===
        0    关节位置
        1    关节速度
        ===  ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete`` 类型，包含 2 个离散空间)

        ===  ===
        编号  动作 (离散空间 0)
        ===  ===
        0    小车关节零力矩
        1    小车关节半最大力矩
        2    小车关节最大力矩
        ===  ===

        ===  ===
        编号  动作 (离散空间 1)
        ===  ===
        0    负力矩 (一侧)
        1    正力矩 (另一侧)
        ===  ===
    """

    observation_space = spaces.Tuple((
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
    ))  # 或简化为: (2, 2)
    action_space = spaces.MultiDiscrete([3, 2])  # 或简化为: [{3}, {2}]
