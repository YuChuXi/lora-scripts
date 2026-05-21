#!/bin/bash

export HF_HOME=huggingface
export PYTHONUTF8=1

# 自动检查并初始化所有 submodule（含 Anima 后端 vendor/sd-scripts）
if [ ! -f "vendor/sd-scripts/anima_train_network.py" ]; then
    echo -e "\033[36m首次运行：正在初始化必要组件，请稍候...\033[0m"
    git submodule update --init --recursive
    if [ $? -ne 0 ]; then
        echo -e "\033[31m组件初始化失败，请检查网络连接后重新运行。\033[0m"
        exit 1
    fi
    echo -e "\033[32m初始化完成，继续启动...\033[0m"
fi

python gui.py "$@"

