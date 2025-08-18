# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import re
import subprocess
import sys


def update_task_param(task_cfg, assembly_id, if_sbc, if_log_eval):
    # 读取文件行。
    with open(task_cfg) as f:
        lines = f.readlines()

    updated_lines = []

    # 正则表达式模式，用于捕获赋值行
    assembly_pattern = re.compile(r"^(.*assembly_id\s*=\s*).*$")
    if_sbc_pattern = re.compile(r"^(.*if_sbc\s*:\s*bool\s*=\s*).*$")
    if_log_eval_pattern = re.compile(r"^(.*if_logging_eval\s*:\s*bool\s*=\s*).*$")
    eval_file_pattern = re.compile(r"^(.*eval_filename\s*:\s*str\s*=\s*).*$")

    for line in lines:
        if "assembly_id =" in line:
            line = assembly_pattern.sub(rf"\1'{assembly_id}'", line)
        elif "if_sbc: bool =" in line:
            line = if_sbc_pattern.sub(rf"\1{str(if_sbc)}", line)
        elif "if_logging_eval: bool =" in line:
            line = if_log_eval_pattern.sub(rf"\1{str(if_log_eval)}", line)
        elif "eval_filename: str = " in line:
            line = eval_file_pattern.sub(r"\1'{}'".format(f"evaluation_{assembly_id}.h5"), line)

        updated_lines.append(line)

    # 将修改后的行写回文件。
    with open(task_cfg, "w") as f:
        f.writelines(updated_lines)


def main():
    parser = argparse.ArgumentParser(description="更新assembly_id并运行训练脚本。")
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="包含assembly_id的文件路径。",
        default="source/isaaclab_tasks/isaaclab_tasks/direct/automate/assembly_tasks_cfg.py",
    )
    parser.add_argument("--assembly_id", type=str, help="要设置的新装配ID。")
    parser.add_argument("--checkpoint", type=str, help="检查点路径。")
    parser.add_argument("--num_envs", type=int, default=128, help="并行环境的数量。")
    parser.add_argument("--seed", type=int, default=-1, help="随机种子。")
    parser.add_argument("--train", action="store_true", help="运行训练模式。")
    parser.add_argument("--log_eval", action="store_true", help="记录评估结果。")
    parser.add_argument("--headless", action="store_true", help="以无头模式运行。")
    parser.add_argument("--max_iterations", type=int, default=1500, help="策略学习的迭代次数。")
    args = parser.parse_args()

    update_task_param(args.cfg_path, args.assembly_id, args.train, args.log_eval)

    bash_command = None
    if sys.platform.startswith("win"):
        bash_command = "isaaclab.bat -p"
    elif sys.platform.startswith("linux"):
        bash_command = "./isaaclab.sh -p"
    if args.train:
        bash_command += " scripts/reinforcement_learning/rl_games/train.py --task=Isaac-AutoMate-Assembly-Direct-v0"
        bash_command += f" --seed={str(args.seed)} --max_iterations={str(args.max_iterations)}"
    else:
        if not args.checkpoint:
            raise ValueError("未提供用于评估的检查点。")
        bash_command += " scripts/reinforcement_learning/rl_games/play.py --task=Isaac-AutoMate-Assembly-Direct-v0"

    bash_command += f" --num_envs={str(args.num_envs)}"

    if args.checkpoint:
        bash_command += f" --checkpoint={args.checkpoint}"

    if args.headless:
        bash_command += " --headless"

    # 运行bash命令
    subprocess.run(bash_command, shell=True, check=True)


if __name__ == "__main__":
    main()
