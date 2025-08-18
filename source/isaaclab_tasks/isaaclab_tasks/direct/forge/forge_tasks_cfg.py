# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)。
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# 从工厂任务配置导入基础任务配置类
from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class ForgeTask(FactoryTask):
    """Forge 任务的基础配置类"""
    # 末端执行器动作惩罚系数（未使用）
    action_penalty_ee_scale: float = 0.0
    # 资产相对动作惩罚系数
    action_penalty_asset_scale: float = 0.001
    # 动作梯度惩罚系数
    action_grad_penalty_scale: float = 0.1
    # 接触力惩罚系数
    contact_penalty_scale: float = 0.05
    # 延迟成功预测惩罚的比例阈值
    delay_until_ratio: float = 0.25
    # 接触力惩罚阈值范围
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    """插销插入任务配置类"""
    # 插销插入任务的接触力惩罚系数
    contact_penalty_scale: float = 0.2


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    """齿轮啮合任务配置类"""
    # 齿轮啮合任务的接触力惩罚系数
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    """螺母螺纹任务配置类"""
    # 螺母螺纹任务的接触力惩罚系数
    contact_penalty_scale: float = 0.05
