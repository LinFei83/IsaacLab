# 版权所有 (c) 2022-2025, The Isaac Lab Project 开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# 保留所有权利.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # 用于计算持有资产的质心.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # 用于夹爪宽度.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@configclass
class FactoryTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # 机器人
    hand_init_pos: list = [0.0, 0.0, 0.015]  # 相对于固定资产尖端.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # 动作
    unidirectional_rot: bool = False

    # 固定资产 (适用于所有任务)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # 持有资产 (适用于所有任务)
    held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # 夹爪中持有资产的噪声水平
    held_asset_rot_init: float = -90.0

    # 奖励
    ee_success_yaw: float = 0.0  # 仅适用于 nut_thread 任务.
    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # 奖励函数的详细信息可以在 https://arxiv.org/pdf/2408.04587 的附录B中找到.
    # 使用多尺度关键点来捕捉任务的不同阶段.
    # 每个奖励将关键点距离 x 通过一个压缩函数:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # 每个列表定义了 [a, b] 来控制压缩函数的斜率和最大值.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]  # 向固定物体的一般移动.
    keypoint_coef_coarse: list = [50, 2]  # 资产对齐的移动.
    keypoint_coef_fine: list = [100, 0]  # 用于螺纹连接或最后插入的小距离.
    # 固定资产高度的分数，用于奖励不同的奖金 (参见各个任务).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


@configclass
class PegInsert(FactoryTask):
    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    # 机器人
    hand_init_pos: list = [0.0, 0.0, 0.047]  # 相对于固定资产尖端.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # 固定资产 (适用于所有任务)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # 持有资产 (适用于所有任务)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # 夹爪中持有资产的噪声水平
    held_asset_rot_init: float = 0.0

    # 奖励
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # 插座高度的分数.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class GearBase(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_base.usd"
    height = 0.02
    base_height = 0.005
    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


@configclass
class MediumGear(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter = 0.03  # 用于夹爪宽度.
    height: float = 0.03
    mass = 0.012


@configclass
class GearMesh(FactoryTask):
    name = "gear_mesh"
    fixed_asset_cfg = GearBase()
    held_asset_cfg = MediumGear()
    duration_s = 20.0

    small_gear_usd = f"{ASSET_DIR}/factory_gear_small.usd"
    large_gear_usd = f"{ASSET_DIR}/factory_gear_large.usd"

    small_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SmallGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=small_gear_usd,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    large_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LargeGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=large_gear_usd,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    # 齿轮资产
    add_flanking_gears = True
    add_flanking_gears_prob = 1.0

    # 机器人
    hand_init_pos: list = [0.0, 0.0, 0.035]  # 相对于固定资产尖端.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # 固定资产 (适用于所有任务)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 15.0

    # 持有资产 (适用于所有任务)
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # 夹爪中持有资产的噪声水平
    held_asset_rot_init: float = -90.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # 齿轮销高度的分数.
    success_threshold: float = 0.05
    engage_threshold: float = 0.9

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


@configclass
class NutM16(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_nut_m16.usd"
    diameter = 0.024
    height = 0.01
    mass = 0.03
    friction = 0.01  # 与螺母相加意味着摩擦力为 (-0.25 + 0.75)/2 = 0.25


@configclass
class BoltM16(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_bolt_m16.usd"
    diameter = 0.024
    height = 0.025
    base_height = 0.01
    thread_pitch = 0.002


@configclass
class NutThread(FactoryTask):
    name = "nut_thread"
    fixed_asset_cfg = BoltM16()
    held_asset_cfg = NutM16()
    asset_size = 16.0
    duration_s = 30.0

    # 机器人
    hand_init_pos: list = [0.0, 0.0, 0.015]  # 相对于固定资产尖端.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 1.83]
    hand_init_orn_noise: list = [0.0, 0.0, 0.26]

    # 动作
    unidirectional_rot: bool = True

    # 固定资产 (适用于所有任务)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 120.0
    fixed_asset_init_orn_range_deg: float = 30.0

    # 持有资产 (适用于所有任务)
    held_asset_pos_noise: list = [0.0, 0.003, 0.003]  # 夹爪中持有资产的噪声水平
    held_asset_rot_init: float = -90.0

    # 奖励.
    ee_success_yaw = 0.0
    keypoint_coef_baseline: list = [100, 2]
    keypoint_coef_coarse: list = [500, 2]  # 100, 2
    keypoint_coef_fine: list = [1500, 0]  # 500, 0
    # 螺纹高度的分数.
    success_threshold: float = 0.375
    engage_threshold: float = 0.5
    keypoint_scale: float = 0.05

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
