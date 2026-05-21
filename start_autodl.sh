#!/bin/bash
# AutoDL 镜像专用启动脚本
# GUI: 6006 (AutoDL 默认开放) | 训练监控: 6008 (AutoDL 默认开放)

export HF_HOME="huggingface"
export PYTHONUTF8=1

python gui.py --port 6006 --listen --skip-prepare-environment --disable-tensorboard
