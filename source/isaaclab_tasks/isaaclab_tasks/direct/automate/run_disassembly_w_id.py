# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import re
import subprocess
import sys


def update_task_param(task_cfg, assembly_id, disassembly_dir):
    # 读取文件行。
    with open(task_cfg) as f:
        lines = f.readlines()

    updated_lines = []

    # 正则表达式模式，用于捕获赋值行
    assembly_pattern = re.compile(r"^(.*assembly_id\s*=\s*).*$")
    disassembly_dir_pattern = re.compile(r"^(.*disassembly_dir\s*=\s*).*$")

    for line in lines:
        if "assembly_id =" in line:
            line = assembly_pattern.sub(rf"\1'{assembly_id}'", line)
        elif "disassembly_dir = " in line:
            line = disassembly_dir_pattern.sub(rf"\1'{disassembly_dir}'", line)

        updated_lines.append(line)

    # 将修改后的行写回文件。
    with open(task_cfg, "w") as f:
        f.writelines(updated_lines)


def main():
    parser = argparse.ArgumentParser(description="更新assembly_id并运行训练脚本。")
    parser.add_argument(
        "--disassembly_dir",
        type=str,
        help="包含输出拆卸轨迹的目录路径。",
        default="disassembly_dir",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="包含assembly_id的文件路径。",
        default="source/isaaclab_tasks/isaaclab_tasks/direct/automate/disassembly_tasks_cfg.py",
    )
    parser.add_argument("--assembly_id", type=str, default="00731", help="要设置的新装配ID。")
    parser.add_argument("--num_envs", type=int, default=128, help="并行环境的数量。")
    parser.add_argument("--seed", type=int, default=-1, help="随机种子。")
    parser.add_argument("--headless", action="store_true", help="以无头模式运行。")
    args = parser.parse_args()

    os.makedirs(args.disassembly_dir, exist_ok=True)

    update_task_param(
        args.cfg_path,
        args.assembly_id,
        args.disassembly_dir,
    )

    if sys.platform.startswith("win"):
        bash_command = "isaaclab.bat -p"
    elif sys.platform.startswith("linux"):
        bash_command = "./isaaclab.sh -p"

    bash_command += " scripts/reinforcement_learning/rl_games/train.py --task=Isaac-AutoMate-Disassembly-Direct-v0"

    bash_command += f" --num_envs={str(args.num_envs)}"
    bash_command += f" --seed={str(args.seed)}"

    if args.headless:
        bash_command += " --headless"

    # 运行bash命令
    subprocess.run(bash_command, shell=True, check=True)


if __name__ == "__main__":
    main()
