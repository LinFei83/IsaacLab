# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
本脚本演示了带有高度扫描传感器的四足机器人环境。

在此示例中，我们使用运动策略来控制机器人。机器人被命令以恒定速度向前移动。高度扫描传感器用于检测地形的高度。

.. code-block:: bash

    # 运行脚本
    ./isaaclab.sh -p scripts/tutorials/03_envs/create_quadruped_base_env.py --num_envs 32

"""

"""首先启动 Isaac Sim 模拟器。"""


import argparse

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="教程：创建四足机器人基础环境。")
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# 预定义配置
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


##
# 自定义观测项
##

def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """从命令生成器生成的命令。"""
    # 返回一个恒定的前进速度命令 [1, 0, 0]，表示x方向速度为1，y方向和角速度为0
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


##
# 场景定义
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """示例场景配置。"""

    # 添加地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # 添加机器人
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 传感器
    # 高度扫描仪，用于检测地形高度
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # 灯光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP 设置
##

@configclass
class ActionsCfg:
    """MDP的动作规范。"""

    # 关节位置控制，控制机器人的所有关节
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """MDP的观测规范。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略组的观测。"""

        # 观测项（保持顺序）
        # 基座线速度
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # 基座角速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 投影重力
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 速度命令
        velocity_commands = ObsTerm(func=constant_commands)
        # 相对关节位置
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 相对关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 上一时刻的动作
        actions = ObsTerm(func=mdp.last_action)
        # 高度扫描数据
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置。"""

    # 重置场景到默认状态
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# 环境配置
##

@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """运动速度跟踪环境的配置。"""

    # 场景设置
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """后初始化。"""
        # 常规设置
        self.decimation = 4  # 环境减速 -> 50 Hz 控制
        # 仿真设置
        self.sim.dt = 0.005  # 仿真时间步长 -> 200 Hz 物理
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device
        # 更新传感器更新周期
        # 我们根据最小更新周期（物理更新周期）来触发所有传感器
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz


def main():
    """主函数。"""
    # 设置基础环境
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # 加载预训练策略
    policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    # 检查策略文件是否存在
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"策略文件 '{policy_path}' 不存在。")
    file_bytes = read_file(policy_path)
    # jit 加载策略
    policy = torch.jit.load(file_bytes).to(env.device).eval()

    # 模拟物理
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # 重置
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: 正在重置环境...")
            # 推理动作
            action = policy(obs["policy"])
            # 执行环境步骤
            obs, _ = env.step(action)
            # 更新计数器
            count += 1

    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    simulation_app.close()