# 版权所有 (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause
"""包含资产和传感器配置的包。"""

import os
import toml

# 为方便其他模块目录的相对路径
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""扩展源目录的路径。"""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""扩展数据目录的路径。"""

ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""从 extension.toml 文件解析的扩展元数据字典。"""

# 配置模块级变量
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]

from .robots import *
from .sensors import *
