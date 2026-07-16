# TT-VLA 环境与远程运行方式

## 结论

最终实验应在远程 Linux + NVIDIA CUDA 服务器执行。本机承担代码编辑、文档审阅、Git 版本管理和结果拉取；服务器承担依赖安装、模型/数据资产访问、训练与结果生成。

当前工作目录及其子目录均没有 `.git` 元数据，因此无法从现有文件验证 remote、branch 或 commit 历史。本机 `commit/push -> 服务器 pull` 是本次实验的目标工作流，不是可在当前目录直接执行的既有流程；开始前必须确定真实 Git clone 来源或初始化并关联远端。

## 仓库可确认的运行前提

| 前提 | 证据 | 结论 |
| --- | --- | --- |
| Python 3.10、Torch 2.2、CUDA 12.1 | `README.md:16-42` | 远程环境基线 |
| Editable 安装 OpenVLA、ManiSkill、SimplerEnv | `README.md:21-37` | 训练路径必需 |
| TTT 还需 VLAC | `rl4vla_env.sh:25`、`replay_buffer.py:216-234` | 原 VLAC 模式必需；随机模式可在代码上绕开 |
| GPU ManiSkill 环境 | `simpler_wrapper.py:15-28` | 不能把本机 macOS 当正式运行环境 |
| 视频工具 | `replay_buffer.py:280-283`、VLAC 视频压缩实现 | VLAC 模式需要 `ffmpeg` |

## 依赖风险：必须先在服务器探针验证

审核模板写 `rlvla`，根 README 写 `rlvla_env`，`rl4vla_env.sh` 创建 `rlvla_env3`。正式运行统一使用 `rlvla` 作为文档约定名称，但必须先确认服务器实际环境名。

| 组件 | 声明版本 | 风险 |
| --- | --- | --- |
| OpenVLA | `transformers==4.40.1`、`peft==0.11.1` | 与 VLAC 冲突 |
| VLAC | `transformers>=4.51`、`peft>=0.15.2`、`ms-swift==3.3` | 与 OpenVLA 冲突 |
| 安装脚本 | 混用 pip/conda，并自行提示版本不兼容 | 不能未经导入探针直接认定可复现 |

证据在 `openvla/pyproject.toml:32-56`、`VLAC/pyproject.toml:23-32`、`rl4vla_env.sh:1-45`。随机奖励若完成条件化 VLAC import/初始化，正式随机训练不会再调用 Critic，但 OpenVLA、ManiSkill 与 SimplerEnv 的远程环境仍须实测。

## 固定的部署与运行步骤

### 1. 本机：在真实 Git 工作树提交

当前目录不是 Git 工作树，以下命令只能在确认后的真实 clone 根目录执行：

```bash
git status
git add docs/ SimplerEnv/simpler_env/utils/replay_buffer.py SimplerEnv/simpler_env/train_ms3_ppo_ttt.py
git commit -m "Add random dense reward TT-VLA ablation"
git push origin <branch>
git rev-parse HEAD
```

不得提交 datasets、checkpoint、W&B run、视频或模型缓存。

### 2. 服务器：同步同一 commit

```bash
cd /path/to/TTT_VLARL-main
git fetch origin
git switch <branch>
git pull --ff-only origin <branch>
git rev-parse HEAD
```

将该 SHA 写入实验结果文件。它必须与本机提交 SHA 一致。

### 3. 服务器：创建/核对环境

不要直接执行 `rl4vla_env.sh` 中的 `conda remove` 行。按 README 的 Python 3.10、CUDA 12.1 与 editable 安装顺序创建单一环境；安装后先执行：

```bash
conda activate rlvla
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import prismatic, mani_skill, simpler_env; print('core-imports-ok')"
ffmpeg -version
```

原 VLAC 路线额外执行：

```bash
python -c "import evo_vlac; print('vlac-import-ok')"
```

只有所有探针通过后才可开始单 episode 冒烟。若 VLAC 与 OpenVLA 不能同环境导入，记录准确的版本冲突；不得把未验证的 `rl4vla_env.sh` 视为可运行证据。

### 4. 服务器：运行与取回结果

从 `SimplerEnv` 目录运行 [推理评测线路](TT-VLA-推理评测线路.md) 中的命令。每个任务/seed 完成后，收集：

- Git SHA、conda package snapshot、GPU/driver 信息；
- `glob/config.yaml` 与新增结构化结果文件；
- 逐 episode 视频；
- 终端日志或作业调度日志。

本机只拉取这些轻量结果和日志摘要；模型、数据和 checkpoint 保留在服务器。

## GPU 注意事项

- 多 GPU 时既有代码将 VLAC 固定为逻辑 `cuda:0`、OpenVLA 为逻辑 `cuda:1`；见 `train_ms3_ppo_ttt.py:129-140`、`replay_buffer.py:226-230`。
- 单卡不是代码层面禁止，但 VLAC、OpenVLA 与 GPU 仿真会竞争显存；根 README 的 40G/80G 说明针对普通 RL，不构成 TTT 显存保证。
- 随机奖励模式绕开 VLAC Critic 后可显著降低显存与视频推理开销；仍应从一 episode 开始验证。

## 已验证与未验证

| 层级 | 状态 |
| --- | --- |
| 静态代码与安装脚本 | 已验证 |
| 当前目录 Git remote/branch | 未验证，且当前无 `.git` |
| 服务器 OS、GPU、驱动、CUDA、资产、W&B/HF 权限 | 未验证 |
| 同环境 OpenVLA + VLAC 共存 | 未验证，存在明确冲突 |
| 随机奖励训练、三类最小评测及成功率 | 未运行 |
