# TT-VLA：从 Git 克隆到远程运行指南

> 适用对象：当前仓库的 TT-VLA / VLAC TTT 路线。
>
> 最终运行环境：远程 Linux + NVIDIA GPU 服务器。本机用于修改、提交和查看结果。

## 0. 开始前必须准备

准备以下实际值，后续命令中的尖括号内容必须替换：

| 变量 | 含义 |
| --- | --- |
| `<repo-ssh-url>` | 你的 Git SSH 仓库地址，例如 `git@github.com:owner/repo.git` |
| `<branch>` | 要运行的分支名 |
| `<openvla-path>` | OpenVLA base model 或 warm-up checkpoint 的本地路径/Hugging Face ID |
| `<lora-path>` | 可选 LoRA adapter 路径；无 adapter 时按模型加载约定留空 |
| `<vlac-model-path>` | 原 VLAC reward model 的本地目录；当前未改代码的 TTT 路线需要它 |

服务器需要 Linux x86_64、可用 NVIDIA GPU、CUDA 驱动、conda、Git 和 `ffmpeg`。本机 macOS 不作为正式训练/评测环境。

## 1. 本机：克隆、创建分支并推送

```bash
git clone <repo-ssh-url>
cd TTT_VLARL-main
git switch -c <branch>
```

在本机完成代码或文档修改后：

```bash
git status
git add docs/ SimplerEnv/simpler_env/utils/replay_buffer.py SimplerEnv/simpler_env/train_ms3_ppo_ttt.py
git commit -m "Add random dense reward TT-VLA ablation"
git push -u origin <branch>
git rev-parse HEAD
```

记录最后一条命令输出的 commit SHA。不要提交 `datasets/`、模型 checkpoint、W&B run、视频或下载缓存。

## 2. 服务器：克隆同一分支

```bash
git clone <repo-ssh-url>
cd TTT_VLARL-main
git switch <branch>
git rev-parse HEAD
```

此 SHA 必须等于本机记录的 SHA。之后每次本机推送新提交，在服务器执行：

```bash
cd /path/to/TTT_VLARL-main
git fetch origin
git pull --ff-only origin <branch>
git rev-parse HEAD
```

## 3. 服务器：创建基础 conda 环境

统一使用环境名 `rlvla`。仓库中还出现 `rlvla_env` 与 `rlvla_env3`，不要混用名称。

```bash
conda create -n rlvla -y python=3.10
conda activate rlvla

conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U tyro
pip install datasets==3.3.2
```

进入仓库根目录后安装三个核心本地包：

```bash
cd /path/to/TTT_VLARL-main
pip install -e openvla
pip install -e ManiSkill
pip install -e SimplerEnv
```

安装与本机服务器 CUDA/Python ABI 匹配的 FlashAttention。仓库 README 给出的 wheel 只适用于 Python 3.10、Torch 2.2、CUDA 12 和 Linux x86_64：

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## 4. VLAC：先处理版本冲突，再继续

当前原始 TTT 代码会 import 并初始化 VLAC Critic。VLAC 要求较新的 `transformers`/`peft`，而 OpenVLA 固定旧版本；两者在同一环境是否可用必须实测，不能假定安装成功即兼容。

原 VLAC 路线尝试安装：

```bash
cd /path/to/TTT_VLARL-main
pip install -e VLAC
```

然后先做导入探针；失败时停止，不要直接启动长训练：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import prismatic, mani_skill, simpler_env; print('core-imports-ok')"
python -c "import evo_vlac; print('vlac-import-ok')"
ffmpeg -version
```

随机奖励版本如果已经按审核文档实现为“跳过 VLAC import、Critic 初始化和 progress 推理”，则运行随机模式不应依赖 `evo_vlac` 或 `<vlac-model-path>`；在代码尚未完成该改动前，仍按原 VLAC 路线准备模型和依赖。

## 5. 模型与资产检查

正式运行前确认：

```bash
test -e <openvla-path> && echo openvla-ok
test -e <vlac-model-path> && echo vlac-ok
```

若使用 LoRA adapter，也确认：

```bash
test -e <lora-path> && echo lora-ok
```

OpenVLA checkpoint、VLAC 模型和 ManiSkill/Bridge 资产通常应缓存在服务器本地磁盘，不要放进 Git。

## 6. 单 episode 冒烟运行

以下先验证原始 VLAC TTT 路线的端到端可用性。必须从 `SimplerEnv` 目录执行，且首次使用 `--num_envs 1`、`--max_episodes 1`。

```bash
conda activate rlvla
cd /path/to/TTT_VLARL-main/SimplerEnv

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo_ttt.py \
  --name ttt-vlac-smoke-seed0 \
  --env_id PutOnPlateInScene25PositionChangeTo-v1 \
  --vla_path <openvla-path> \
  --vla_unnorm_key bridge_orig \
  --vla_load_path <lora-path> \
  --reward_model_path <vlac-model-path> \
  --seed 0 \
  --num_envs 1 \
  --episode_len 160 \
  --alg_name ppo \
  --ttt 1 \
  --tt_steps 8 \
  --max_episodes 1 \
  --obj_set test \
  --no_wandb
```

检查终端是否打印 `success rate`，并检查 W&B run 相邻的 `glob/` 是否出现 `config.yaml` 和 MP4 视频。失败时保存完整 traceback；当前 runner 会无限重试异常，因此发现重复异常后应手动停止作业并先修复依赖/路径。

## 7. 随机奖励版本与三类最小任务

随机奖励代码改完并通过 100,000 次分布验证后，复用上述命令：

- 将 `--name` 改为能标识 `random-dense`、任务与 seed 的名称。
- 使用新增的随机 reward mode 参数；具体参数名以实施后的 `Args` 定义为准。
- 随机模式不应再传 `--reward_model_path`，前提是 VLAC 已被条件化绕过。
- 将 `--max_episodes 1` 改为正式数量，并逐任务、逐 seed 运行。

最小三任务：

| 类别 | `--env_id` |
| --- | --- |
| Execution | `PutOnPlateInScene25PositionChangeTo-v1` |
| Vision | `PutOnPlateInScene25VisionImage-v1` |
| Semantics | `PutOnPlateInScene25MultiCarrot-v1` |

类别是代码语义映射；提交结果前仍要对照论文确认对应 baseline 表格行列。

## 8. 取回与整理结果

每个任务至少保留：Git SHA、环境包版本、GPU 信息、任务、seed、checkpoint、随机奖励参数、每 episode success、总成功率、日志路径和视频路径。

服务器运行完成后，将结构化结果与必要日志摘要拉回本机；模型、数据和完整视频可保留在服务器。详细的奖励改动边界、评测口径和审核状态见：

- [训练与随机奖励改动线路](TT-VLA-训练与随机奖励改动线路.md)
- [推理评测线路](TT-VLA-推理评测线路.md)
- [环境与远程运行方式](TT-VLA-环境与远程运行方式.md)
