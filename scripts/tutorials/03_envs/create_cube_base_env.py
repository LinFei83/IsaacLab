# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本创建了一个带有浮动立方体的简单环境。立方体由PD控制器控制，以跟踪任意目标位置。

在学习本教程时，我们建议您关注自定义动作项是如何定义的。动作项负责处理原始动作并将其应用于场景实体。

我们还定义了一个名为'randomize_scale'的事件项，该事件项随机化立方体的大小。此事件项的模式为'prestartup'，这意味着它在模拟开始前应用于USD阶段。此外，标志'replicate_physics'被设置为False，这意味着立方体不会在多个环境中复制，而是每个环境都有自己的立方体实例。

环境的其余部分与之前的教程类似。

.. code-block:: bash

    # 运行脚本
    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cube_base_env.py --num_envs 32

"""

from __future__ import annotations

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：创建浮动立方体环境。")
parser.add_argument("--num_envs", type=int, default=64, help="要生成的环境数量。")

# 添加 AppLauncher cli 参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来的所有内容。"""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# 自定义动作项
##

class CubeActionTerm(ActionTerm):
    """简单的动作项，实现PD控制器以跟踪目标位置。

    动作项应用于立方体资产。它包括两个步骤：

    1. **处理原始动作**：通常，这包括任何将原始动作转换到所需空间所需的变换。这在每个环境步骤中调用一次。
    2. **应用处理后的动作**：此步骤将处理后的动作应用于资产。它在每个模拟步骤中调用一次。

    在这种情况下，动作项简单地将原始动作应用于立方体资产。原始动作是立方体在环境框架中的期望目标位置。预处理步骤只是将原始动作复制到处理后的动作，因为不需要额外的处理。然后通过实现PD控制器来跟踪目标位置，将处理后的动作应用于立方体资产。
    """

    _asset: RigidObject
    """应用动作项的关节资产。"""

    def __init__(self, cfg: CubeActionTermCfg, env: ManagerBasedEnv):
        # 调用父类构造函数
        super().__init__(cfg, env)
        # 创建缓冲区
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        # 控制器增益
        self.p_gain = cfg.p_gain
        self.d_gain = cfg.d_gain

    """
    属性。
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    操作
    """

    def process_actions(self, actions: torch.Tensor):
        # 存储原始动作
        self._raw_actions[:] = actions
        # 不处理动作
        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):
        # 实现PD控制器以跟踪目标位置
        # 计算位置误差：目标位置 - 当前位置
        pos_error = self._processed_actions - (self._asset.data.root_pos_w - self._env.scene.env_origins)
        # 计算速度误差：期望速度为0，所以误差为负的当前速度
        vel_error = -self._asset.data.root_lin_vel_w
        # 设置速度目标
        self._vel_command[:, :3] = self.p_gain * pos_error + self.d_gain * vel_error
        # 将速度命令写入模拟器
        self._asset.write_root_velocity_to_sim(self._vel_command)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """立方体动作项的配置。"""

    class_type: type = CubeActionTerm
    """与动作项对应的类。"""

    p_gain: float = 5.0
    """PD控制器的比例增益。"""
    d_gain: float = 0.5
    """PD控制器的微分增益。"""


##
# 自定义观测项
##

def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """资产根框架中的根线性速度。"""
    # 提取使用的量（启用类型提示）
    asset: RigidObject = env.scene[asset_cfg.name]
    # 返回立方体相对于环境原点的位置
    return asset.data.root_pos_w - env.scene.env_origins


##
# 场景定义
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """示例场景配置。

    场景包括地面平面、光源和浮动立方体（禁用重力）。
    """

    # 添加地形
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # 添加立方体
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5)),
    )

    # 灯光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# 环境设置
##

@configclass
class ActionsCfg:
    """MDP的动作规范。"""

    # 使用自定义的立方体动作项
    joint_pos = CubeActionTermCfg(asset_name="cube")


@configclass
class ObservationsCfg:
    """MDP的观测规范。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略组的观测。"""

        # 立方体速度
        # 观测立方体相对于环境原点的位置
        position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置。"""

    # 此事件项重置立方体的基座位置。
    # 模式设置为'reset'，这意味着每当环境实例重置时（由于'TerminationCfg'中定义的终止条件），基座位置都会重置。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    # 此事件项随机化立方体的大小。
    # 模式设置为'prestartup'，这意味着在模拟开始前在USD阶段随机化大小。
    # 注意：USD级别的随机化需要将标志'replicate_physics'设置为False。
    randomize_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": {"x": (0.5, 1.5), "y": (0.5, 1.5), "z": (0.5, 1.5)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    # 此事件项随机化立方体的视觉颜色。
    # 与大小随机化类似，这也是一种USD级别的随机化，需要将标志'replicate_physics'设置为False。
    randomize_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="prestartup",
        params={
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "asset_cfg": SceneEntityCfg("cube"),
            "mesh_name": "geometry/mesh",
            "event_name": "rep_cube_randomize_color",
        },
    )


##
# 环境配置
##

@configclass
class CubeEnvCfg(ManagerBasedEnvCfg):
    """运动速度跟踪环境的配置。"""

    # 场景设置
    # 标志'replicate_physics'被设置为False，这意味着立方体不会在多个环境中复制，而是每个环境都有自己的立方体实例。
    # 这允许独立地修改每个环境的立方体属性。
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)

    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """后初始化。"""
        # 常规设置
        self.decimation = 2
        # 仿真设置
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.render_interval = 2  # 渲染间隔应该是decimation的倍数
        self.sim.device = args_cli.device
        # 查看器设置
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 2.0)


def main():
    """主函数。"""

    # 设置基础环境
    env_cfg = CubeEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # 设置目标位置命令
    # 生成随机目标位置，范围在[-1, 1]之间，然后乘以2得到[-2, 2]的范围
    target_position = torch.rand(env.num_envs, 3, device=env.device) * 2
    # 将z坐标增加2.0，使目标位置在空中
    target_position[:, 2] += 2.0
    # 偏移所有目标，使它们移动到世界原点
    target_position -= env.scene.env_origins

    # 模拟物理
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # 重置
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: 正在重置环境...")
            # 执行环境步骤
            obs, _ = env.step(target_position)
            # 打印目标位置和当前位置之间的均方位置误差
            error = torch.norm(obs["policy"] - target_position).mean().item()
            print(f"[步骤: {count:04d}]: 平均位置误差: {error:.4f}")
            # 更新计数器
            count += 1

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()