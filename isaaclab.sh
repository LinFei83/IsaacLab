#!/usr/bin/env bash

# 版权所有 (c) 2022-2025, Isaac Lab 项目开发者 (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md)。
# 保留所有权利。
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# 配置部分
#==

# 发生错误时退出脚本
set -e

# 设置制表符宽度为4个空格
tabs 4

# 获取脚本所在的源代码目录路径
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# 辅助函数部分
#==

# 安装系统依赖项
install_system_deps() {
    # 检查 cmake 是否已经安装
    if command -v cmake &> /dev/null; then
        echo "[INFO] cmake 已经安装。"
    else
        # 检查是否以 root 用户身份运行
        if [ "$EUID" -ne 0 ]; then
            echo "[INFO] 正在安装系统依赖项..."
            sudo apt-get update && sudo apt-get install -y --no-install-recommends \
                cmake \
                build-essential
        else
            echo "[INFO] 正在安装系统依赖项..."
            apt-get update && apt-get install -y --no-install-recommends \
                cmake \
                build-essential
        fi
    fi
}

# 检查 Isaac Sim 版本是否为 4.5
is_isaacsim_version_4_5() {
    local python_exe
    python_exe=$(extract_python_exe)

    # 1) 尝试读取 VERSION 文件
    local sim_file
    sim_file=$("${python_exe}" -c "import isaacsim; print(isaacsim.__file__)" 2>/dev/null) || return 1
    local version_path
    version_path=$(dirname "${sim_file}")/../../VERSION
    if [[ -f "${version_path}" ]]; then
        local ver
        ver=$(head -n1 "${version_path}")
        [[ "${ver}" == 4.5* ]] && return 0
    fi

    # 2) 回退到使用 importlib.metadata，通过 here-doc 方式
    local ver
    ver=$("${python_exe}" <<'PYCODE' 2>/dev/null
from importlib.metadata import version, PackageNotFoundError
try:
    print(version("isaacsim"))
except PackageNotFoundError:
    import sys; sys.exit(1)
PYCODE
) || return 1

    [[ "${ver}" == 4.5* ]]
}

# 检查是否在 Docker 容器中运行
is_docker() {
    [ -f /.dockerenv ] || \
    grep -q docker /proc/1/cgroup || \
    [[ $(cat /proc/1/comm) == "containerd-shim" ]] || \
    grep -q docker /proc/mounts || \
    [[ "$(hostname)" == *"."* ]]
}

# 提取 Isaac Sim 路径
extract_isaacsim_path() {
    # 使用指向 Isaac Sim 目录的符号链接路径
    local isaac_path=${ISAACLAB_PATH}/_isaac_sim
    # 如果上述路径不可用，则尝试使用 Python 查找路径
    if [ ! -d "${isaac_path}" ]; then
        # 使用 Python 可执行文件获取路径
        local python_exe=$(extract_python_exe)
        # 通过导入 isaacsim 并获取环境路径来检索路径
        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            local isaac_path=$(${python_exe} -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])")
        fi
    fi
    # 检查是否存在路径
    if [ ! -d "${isaac_path}" ]; then
        # 如果未找到路径则抛出错误
        echo -e "[ERROR] 无法找到 Isaac Sim 目录: '${isaac_path}'" >&2
        echo -e "\t可能的原因如下:" >&2
        echo -e "\t1. Conda 环境未激活。" >&2
        echo -e "\t2. Isaac Sim pip 包 'isaacsim-rl' 未安装。" >&2
        echo -e "\t3. Isaac Sim 目录在默认路径 ${ISAACLAB_PATH}/_isaac_sim 不可用。" >&2
        # 退出脚本
        exit 1
    fi
    # 返回结果
    echo ${isaac_path}
}

# 从 Isaac Sim 中提取 Python 可执行文件
extract_python_exe() {
    # 检查是否使用 conda
    if ! [[ -z "${CONDA_PREFIX}" ]]; then
        # 使用 conda 的 Python
        local python_exe=${CONDA_PREFIX}/bin/python
    else
        # 使用 kit 的 Python
        local python_exe=${ISAACLAB_PATH}/_isaac_sim/python.sh

    if [ ! -f "${python_exe}" ]; then
            # 注意：我们需要检查系统 Python，例如在 Docker 中的情况
            # 在 Docker 内部，如果用户安装到系统 Python 中，我们需要使用它
            # 否则，使用 kit 中的 Python
            if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
                local python_exe=$(which python)
            fi
        fi
    fi
    # 检查是否存在 Python 路径
    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] 无法在路径 '${python_exe}' 找到任何 Python 可执行文件" >&2
        echo -e "\t可能的原因如下:" >&2
        echo -e "\t1. Conda 环境未激活。" >&2
        echo -e "\t2. Isaac Sim pip 包 'isaacsim-rl' 未安装。" >&2
        echo -e "\t3. Python 可执行文件在默认路径 ${ISAACLAB_PATH}/_isaac_sim/python.sh 不可用。" >&2
        exit 1
    fi
    # 返回结果
    echo ${python_exe}
}

# 从 Isaac Sim 中提取模拟器可执行文件
extract_isaacsim_exe() {
    # 获取 Isaac Sim 路径
    local isaac_path=$(extract_isaacsim_path)
    # 要使用的 Isaac Sim 可执行文件
    local isaacsim_exe=${isaac_path}/isaac-sim.sh
    # 检查是否存在 Python 路径
    if [ ! -f "${isaacsim_exe}" ]; then
        # 检查是否通过 Isaac Sim pip 安装
        # 注意：pip 安装的 Isaac Sim 只能来自直接的 Python 环境，所以我们可以直接在这里使用 'python'
        if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            # Isaac Sim - Python 包入口点
            local isaacsim_exe="isaacsim isaacsim.exp.full"
        else
            echo "[ERROR] 在路径 ${isaac_path} 未找到 Isaac Sim 可执行文件" >&2
            exit 1
        fi
    fi
    # 返回结果
    echo ${isaacsim_exe}
}

# 检查输入目录是否为 Python 扩展并安装模块
install_isaaclab_extension() {
    # 获取 Python 可执行文件
    python_exe=$(extract_python_exe)
    # 如果目录包含 setup.py 则安装 Python 模块
    if [ -f "$1/setup.py" ]; then
        echo -e "\t 模块: $1"
        ${python_exe} -m pip install --editable $1
    fi
}

# 为 Isaac Lab 设置 Anaconda 环境
setup_conda_env() {
    # 从输入获取环境名称
    local env_name=$1
    # 检查是否安装了 conda
    if ! command -v conda &> /dev/null
    then
        echo "[ERROR] 未找到 Conda。请安装 conda 后重试。"
        exit 1
    fi

    # 检查 _isaac_sim 符号链接是否存在且 isaacsim-rl 未通过 pip 安装
    if [ ! -L "${ISAACLAB_PATH}/_isaac_sim" ] && ! python -m pip list | grep -q 'isaacsim-rl'; then
        echo -e "[WARNING] 在 ${ISAACLAB_PATH}/_isaac_sim 未找到 _isaac_sim 符号链接"
        echo -e "\t如果您计划通过 pip 安装 Isaac Sim，可以忽略此警告。"
        echo -e "\t如果您使用的是 Isaac Sim 的二进制安装，请确保在设置 conda 环境之前创建符号链接。"
    fi

    # 检查环境是否存在
    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        echo -e "[INFO] 名为 '${env_name}' 的 conda 环境已存在。"
    else
        echo -e "[INFO] 正在创建名为 '${env_name}' 的 conda 环境..."
        echo -e "[INFO] 正在从 ${ISAACLAB_PATH}/environment.yml 安装依赖项"

        # 如需要则修补 Python 版本，但首先备份
        cp "${ISAACLAB_PATH}/environment.yml"{,.bak}
        if is_isaacsim_version_4_5; then
            echo "[INFO] 检测到 Isaac Sim 4.5 → 强制使用 python=3.10"
            sed -i 's/^  - python=3\.11/  - python=3.10/' "${ISAACLAB_PATH}/environment.yml"
        else
            echo "[INFO] Isaac Sim 5.0，安装 python=3.11"
        fi

        conda env create -y --file ${ISAACLAB_PATH}/environment.yml -n ${env_name}
        # (可选) 恢复原始 environment.yml:
        if [[ -f "${ISAACLAB_PATH}/environment.yml.bak" ]]; then
            mv "${ISAACLAB_PATH}/environment.yml.bak" "${ISAACLAB_PATH}/environment.yml"
        fi
    fi

    # 缓存当前路径以备后用
    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH
    # 清除任何现有文件
    rm -f ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    rm -f ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    # 设置目录以加载 Isaac Sim 变量
    mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
    mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d

    # 在激活期间向环境添加变量
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# 用于 Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# 如果非无头运行则显示图标' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    # 检查是否有 _isaac_sim 目录 -> 如果有则表示已安装二进制文件。
    # 我们需要设置 conda 变量以加载二进制文件
    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # 在激活期间向环境添加变量
        printf '%s\n' \
            '# 用于 Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi

    # 重新激活环境以加载变量
    # 需要这样做，因为 deactivate 会抱怨 Isaac Lab 别名，因为它否则不存在
    conda activate ${env_name}

    # 在停用期间从环境移除变量
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# 用于 Isaac Lab' \
        'unalias isaaclab &>/dev/null' \
        'unset ISAACLAB_PATH' \
        '' \
        '# 恢复路径' \
        'export PYTHONPATH='${cache_pythonpath}'' \
        'export LD_LIBRARY_PATH='${cache_ld_library_path}'' \
        '' \
        '# 用于 Isaac Sim' \
        'unset RESOURCE_NAME' \
        '' > ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh

    # 检查是否有 _isaac_sim 目录 -> 如果有则表示已安装二进制文件。
    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # 在激活期间向环境添加变量
        printf '%s\n' \
            '# 用于 Isaac Sim' \
            'unset CARB_APP_PATH' \
            'unset EXP_PATH' \
            'unset ISAAC_PATH' \
            '' >> ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    fi

    # 停用环境
    conda deactivate
    # 向用户添加关于别名的信息
    echo -e "[INFO] 已向 conda 环境添加 'isaaclab' 别名以指向 'isaaclab.sh' 脚本。"
    echo -e "[INFO] 已创建名为 '${env_name}' 的 conda 环境。\n"
    echo -e "\t\t1. 要激活环境，请运行:                conda activate ${env_name}"
    echo -e "\t\t2. 要安装 Isaac Lab 扩展，请运行:            isaaclab -i"
    echo -e "\t\t3. 要执行格式化，请运行:                      isaaclab -f"
    echo -e "\t\t4. 要停用环境，请运行:              conda deactivate"
    echo -e "\n"
}

# 从模板和 Isaac Sim 设置更新 VSCode 设置
update_vscode_settings() {
    echo "[INFO] 正在设置 VSCode 设置..."
    # 获取 Python 可执行文件
    python_exe=$(extract_python_exe)
    # setup_vscode.py 的路径
    setup_vscode_script="${ISAACLAB_PATH}/.vscode/tools/setup_vscode.py"
    # 在尝试运行之前检查文件是否存在
    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    else
        echo "[WARNING] 无法找到脚本 'setup_vscode.py'。正在中止 VSCode 设置。"
    fi
}

# 打印使用说明
print_help () {
    echo -e "\n用法: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] -- 管理 Isaac Lab 的实用程序。"
    echo -e "\n可选参数:"
    echo -e "\t-h, --help           显示帮助内容。"
    echo -e "\t-i, --install [LIB]  安装 Isaac Lab 内部的扩展和学习框架作为额外依赖项。默认为 'all'。"
    echo -e "\t-f, --format         运行 pre-commit 来格式化代码并检查语法。"
    echo -e "\t-p, --python         运行 Isaac Sim 或虚拟环境(如果激活)提供的 Python 可执行文件。"
    echo -e "\t-s, --sim            运行 Isaac Sim 提供的模拟器可执行文件 (isaac-sim.sh)。"
    echo -e "\t-t, --test           运行所有 Python pytest 测试。"
    echo -e "\t-o, --docker         运行 Docker 容器辅助脚本 (docker/container.sh)。"
    echo -e "\t-v, --vscode         从模板生成 VSCode 设置文件。"
    echo -e "\t-d, --docs           使用 sphinx 从源代码构建文档。"
    echo -e "\t-n, --new            从模板创建新的外部项目或内部任务。"
    echo -e "\t-c, --conda [NAME]   为 Isaac Lab 创建 conda 环境。默认名称为 'env_isaaclab'。"
    echo -e "\n" >&2
}


#==
# 主程序部分
#==

# 检查是否提供了参数
if [ -z "$*" ]; then
    echo "[Error] 未提供参数。" >&2;
    print_help
    exit 0
fi

# 传递参数
while [[ $# -gt 0 ]]; do
    # 读取键
    case "$1" in
        -i|--install)
            # 首先安装系统依赖项
            install_system_deps
            # 安装 IsaacLab/source 目录中的 Python 包
            echo "[INFO] 正在安装 Isaac Lab 仓库内的扩展..."
            python_exe=$(extract_python_exe)
            # 检查是否安装了 pytorch 及其版本
            # 安装支持 blackwell 的 pytorch with cuda 12.8
            if ${python_exe} -m pip list 2>/dev/null | grep -q "torch"; then
                torch_version=$(${python_exe} -m pip show torch 2>/dev/null | grep "Version:" | awk '{print $2}')
                echo "[INFO] 发现已安装 PyTorch 版本 ${torch_version}。"
                if [[ "${torch_version}" != "2.7.0+cu128" ]]; then
                    echo "[INFO] 正在卸载 PyTorch 版本 ${torch_version}..."
                    ${python_exe} -m pip uninstall -y torch torchvision torchaudio
                    echo "[INFO] 正在安装支持 CUDA 12.8 的 PyTorch 2.7.0..."
                    ${python_exe} -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
                else
                    echo "[INFO] PyTorch 2.7.0 已安装。"
                fi
            else
                echo "[INFO] 正在安装支持 CUDA 12.8 的 PyTorch 2.7.0..."
                ${python_exe} -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
            fi
            # 递归查找目录并安装它们
            # 这不会检查扩展之间的依赖关系
            export -f extract_python_exe
            export -f install_isaaclab_extension
            # 源代码目录
            find -L "${ISAACLAB_PATH}/source" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_isaaclab_extension "{}"' \;
            # 安装支持的强化学习框架的 Python 包
            echo "[INFO] 正在安装额外需求，如学习框架..."
            # 检查是否指定了要安装的 rl-framework
            if [ -z "$2" ]; then
                echo "[INFO] 正在安装所有 rl-framework..."
                framework_name="all"
            elif [ "$2" = "none" ]; then
                echo "[INFO] 不会安装任何 rl-framework。"
                framework_name="none"
                shift # 跳过参数
            else
                echo "[INFO] 正在安装 rl-framework: $2"
                framework_name=$2
                shift # 跳过参数
            fi
            # 安装指定的学习框架
            ${python_exe} -m pip install -e ${ISAACLAB_PATH}/source/isaaclab_rl["${framework_name}"]
            ${python_exe} -m pip install -e ${ISAACLAB_PATH}/source/isaaclab_mimic["${framework_name}"]

            # 检查我们是否在 Docker 容器内部或正在构建 Docker 镜像
            # 在这种情况下不要设置 VSCode，因为它会要求 EULA 协议从而触发用户交互
            if is_docker; then
                echo "[INFO] 正在 Docker 容器内运行。跳过 VSCode 设置。"
                echo "[INFO] 要设置 VSCode，请运行 'isaaclab -v'。"
            else
                # 更新 VSCode 设置
                update_vscode_settings
            fi

            # 取消设置局部变量
            unset extract_python_exe
            unset install_isaaclab_extension
            shift # 跳过参数
            ;;
        -c|--conda)
            # 如果未提供则使用默认名称
            if [ -z "$2" ]; then
                echo "[INFO] 使用默认 conda 环境名称: env_isaaclab"
                conda_env_name="env_isaaclab"
            else
                echo "[INFO] 使用 conda 环境名称: $2"
                conda_env_name=$2
                shift # 跳过参数
            fi
            # 为 Isaac Lab 设置 conda 环境
            setup_conda_env ${conda_env_name}
            shift # 跳过参数
            ;;
        -f|--format)
            # 重置 Python 路径以避免与 pre-commit 冲突
            # 这是必需的，因为 pre-commit 钩子安装在单独的虚拟环境中
            # 并且它使用系统 Python 来运行钩子
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi
            # 在仓库上运行格式化程序
            # 检查是否安装了 pre-commit
            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] 正在安装 pre-commit..."
                pip install pre-commit
                sudo apt-get install -y pre-commit
            fi
            # 始终在 Isaac Lab 目录内执行
            echo "[INFO] 正在格式化仓库..."
            cd ${ISAACLAB_PATH}
            pre-commit run --all-files
            cd - > /dev/null
            # 将 Python 路径设置回原始值
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi
            shift # 跳过参数
            # 整洁地退出
            break
            ;;
        -p|--python)
            # 运行 isaacsim 提供的 Python
            python_exe=$(extract_python_exe)
            echo "[INFO] 使用来自以下位置的 Python: ${python_exe}"
            shift # 跳过参数
            ${python_exe} "$@"
            # 整洁地退出
            break
            ;;
        -s|--sim)
            # 运行 isaacsim 提供的模拟器 exe
            isaacsim_exe=$(extract_isaacsim_exe)
            echo "[INFO] 正在运行来自以下位置的 isaac-sim: ${isaacsim_exe}"
            shift # 跳过参数
            ${isaacsim_exe} --ext-folder ${ISAACLAB_PATH}/source $@
            # 整洁地退出
            break
            ;;
        -n|--new)
            # 运行模板生成器脚本
            python_exe=$(extract_python_exe)
            shift # 跳过参数
            echo "[INFO] 正在安装模板依赖项..."
            ${python_exe} -m pip install -q -r ${ISAACLAB_PATH}/tools/template/requirements.txt
            echo -e "\n[INFO] 正在运行模板生成器...\n"
            ${python_exe} ${ISAACLAB_PATH}/tools/template/cli.py $@
            # 整洁地退出
            break
            ;;
        -t|--test)
            # 运行 isaacsim 提供的 Python
            python_exe=$(extract_python_exe)
            shift # 跳过参数
            ${python_exe} -m pytest ${ISAACLAB_PATH}/tools $@
            # 整洁地退出
            break
            ;;
        -o|--docker)
            # 运行 Docker 容器辅助脚本
            docker_script=${ISAACLAB_PATH}/docker/container.sh
            echo "[INFO] 正在运行来自以下位置的 Docker 实用程序脚本: ${docker_script}"
            shift # 跳过参数
            bash ${docker_script} $@
            # 整洁地退出
            break
            ;;
        -v|--vscode)
            # 更新 VSCode 设置
            update_vscode_settings
            shift # 跳过参数
            # 整洁地退出
            break
            ;;
        -d|--docs)
            # 构建文档
            echo "[INFO] 正在构建文档..."
            # 获取 Python 可执行文件
            python_exe=$(extract_python_exe)
            # 安装 pip 包
            cd ${ISAACLAB_PATH}/docs
            ${python_exe} -m pip install -r requirements.txt > /dev/null
            # 构建文档
            ${python_exe} -m sphinx -b html -d _build/doctrees . _build/current
            # 打开文档
            echo -e "[INFO] 要在默认浏览器中打开文档，请运行:"
            echo -e "\n\t\txdg-open $(pwd)/_build/current/index.html\n"
            # 整洁地退出
            cd - > /dev/null
            shift # 跳过参数
            # 整洁地退出
            break
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *) # 未知选项
            echo "[Error] 提供了无效参数: $1"
            print_help
            exit 1
            ;;
    esac
done
