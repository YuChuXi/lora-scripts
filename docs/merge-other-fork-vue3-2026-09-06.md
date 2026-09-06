# merge other-fork/main（Vue 3 前端重写）任务笔记

面向维护者：记录本次将 `other-fork`（wochenlong/lora-scripts-next）合入 `main` 的背景、步骤与验证结果。任务日期 2026-09-06。

## 背景

- 本仓库 `frontend/` 是 vendored 的 VuePress 构建产物，历史上由 next fork 在产物上手工注入（假 hash、克隆 chunk、语义命名脚本），结构散装。
- 自分叉点 `80ac8bc9`（2026-06-06）以来，上游领先 401 个提交：Vue 3 + Vite 7 + TS 源码工程（`d6f40c43` 起）、90 个配套 mikazuki 后端提交、Anima/DoRA 链路 44 个修复（LoKr bf16 DoRA dtype、bucket 分辨率、预览链路等）。
- 本仓库对 `frontend/` 零独有提交；独有的 17 个提交全部为数据集/LoRA 配置与杂项，预期 merge 冲突极少。
- 总变更规模：469 文件，+43507/−13458。

## 执行步骤

1. [x] `git fetch other-fork`（需代理 127.0.0.1:7890；`git remote prune other-fork` 清理过 dev/builtin-services 引用冲突）
2. [x] `git merge other-fork/main` → merge commit `c187b8a6`
   - 唯一冲突 `requirements.txt`（3 处同模式：本地裸包名 vs 上游锁版本+注释），取上游侧——90 个后端提交以锁版本为开发基准
   - `build-scripts/*.ps1` 出现 CRLF 显示性差异，已 `git add --renormalize` 修复（`79856088`）
3. [x] 前端验证（Node 22.17.1 独立二进制置于 `~/.local/node-v22.17.1-linux-x64`，沙盒内 snap node 不可用）
   - `npm ci` 通过；`npm run typecheck` 通过；`npm run build` 通过（3.44s）
   - 本地构建产物 hash 与上游 CI 产物不同（环境性差异），已还原为上游提交的 dist
4. [x] 后端验证（用户选定轻量模式：不降级 .venv 已装包）
   - 仅补装 `.venv` 内缺失包：`diffusers==0.33.1`、`modelscope 1.39.1`、`huggingface-hub==0.36.2`
   - 发现并修复 `gui.py` tageditor 子进程 `-s` 隔离问题（见下）
   - 冒烟（默认端口布局）：`/` 200（title "Next Trainer | 训练 UI"）、`/api/tasks` 正常、`/proxy/tageditor` 200、日志 0 Traceback、Torch 2.12.0+cu130 与 3090 正常识别

## merge 后发现并修复的问题

### tageditor 子进程 `-s` 隔离与继承式环境冲突（`gui.py`）

- 上游 `b1d3a1c0` 为 Windows portable 内嵌 Python 加了 `python -s` 启动 tageditor 子进程，假设依赖全部位于环境内。
- 本机 .venv 为 `--system-site-packages` 继承式：torch 2.12.0+cu130、huggingface_hub 等位于 `~/.local`（用户 site-packages），`-s` 会使其不可见 → tageditor 导入链断裂。
- 训练子进程（`mikazuki/process.py`）不带 `-s`，不受影响。
- 修复：去掉该 `-s`，使子进程与主进程的 site 策略一致。

## 备注

- 上游 docs 里有 `anima-fast-merge-checklist.md` 等先例，本次 merge 后相关旧文档（VENDOR.md 描述的 VuePress 时代）仅具历史价值。
- 上游 VENDOR.md 声称的源码仓库 `Akegarasu/lora-scripts-frontend` 已不存在（404），修复路径以 next 的 Vue 3 工程为准。
- 默认启动（不带 `--skip-prepare-environment`）会按 requirements 锁版本自动降级 21 个包——本机选择轻量模式时启动请带 `--skip-prepare-environment`。
