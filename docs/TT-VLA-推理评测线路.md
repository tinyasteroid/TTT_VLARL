# TT-VLA 推理与最小评测线路

## 结论

随机奖励版本的在线运行入口是 `SimplerEnv/simpler_env/train_ms3_ppo_ttt.py`。它每个 episode 重新加载初始策略、运行 TTT 更新并读取真实 `success`；因此它才是随机奖励最小实验的结果来源。`--only_render` 的普通渲染只评估初始加载策略，不可替代 TTT 运行结果。

## 模型、环境与动作线路

```text
OpenVLA base / LoRA checkpoint
  -> OpenVLAPolicy 加载与动作统计
  -> predict_action_batch
  -> token 解码、反归一化为 7D 动作
  -> ManiSkill GPU SimplerEnv.step
  -> RGB 观测、真实 success、truncated
  -> TTT 窗口内随机 dense reward -> PPO
```

- `SimplerEnv/simpler_env/policies/openvla/openvla_train.py:31-72`：加载 OpenVLA、processor/tokenizer 与可选 LoRA adapter；adapter 路径还用于获取动作归一化统计。
- `openvla_train.py:149-165`：推理调用 `predict_action_batch`，评测温度默认 `0.6`。
- `SimplerEnv/simpler_env/env/simpler_wrapper.py:12-29`：环境强制使用 ManiSkill GPU backend、`rgb+segmentation`、第三人称 RGB、`sim_freq=500`、`control_freq=5`。
- `simpler_wrapper.py:51-91`：OpenVLA token 解码为 7 维连续机器人控制动作。

## 三类最小评测任务

仓库没有显式出现论文的 `Execution`、`Vision`、`Semantics` 三个总类名。下表是依据 README 的任务语义和对应环境实现做出的**代码语义映射**；论文 baseline 的表格行列仍需人工核对。

| 类别 | 最小建议任务 | 代码证据 | 覆盖含义 |
| --- | --- | --- | --- |
| Execution | `PutOnPlateInScene25PositionChangeTo-v1` | `README.md:323-325`；`ManiSkill/.../put_on_in_scene_multi.py:2559-2626` | episode 中目标物重定位 |
| Vision | `PutOnPlateInScene25VisionImage-v1` | `README.md:310-314`；`put_on_in_scene_multi.py:942-979` | 未见桌面图像 |
| Semantics | `PutOnPlateInScene25MultiCarrot-v1` | `README.md:319-320`；`put_on_in_scene_multi.py:1691-1760` | 多物体与未见目标物 |

完整候选集合：

- Vision：`VisionImage`、`VisionTexture03/05`、`VisionWhole03/05`。
- Semantics：`Carrot`、`Plate`、`Instruct`、`MultiCarrot`、`MultiPlate`。
- Execution：`Position`、`EEPose`、`PositionChangeTo`。

## 远程最小运行命令

以下命令是静态代码审计得到的单任务 TTT 冒烟线路。模型路径、环境名和 Git commit 必须以服务器实测填实。

```bash
conda activate rlvla
cd /path/to/TTT_VLARL-main/SimplerEnv

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo_ttt.py \
  --name random-dense-execution-seed0 \
  --env_id PutOnPlateInScene25PositionChangeTo-v1 \
  --vla_path /path/to/openvla-7b-rlvla-warmup \
  --vla_unnorm_key bridge_orig \
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

完成冒烟后，将 `--max_episodes` 改为预定值，并分别仅替换 `--name` 与 `--env_id`：

- Vision：`PutOnPlateInScene25VisionImage-v1`
- Semantics：`PutOnPlateInScene25MultiCarrot-v1`

历史 TTT 命令的可比参数依据为 `SimplerEnv/scratch.py:107-154`。随机模式应不传 `--reward_model_path`，前提是实施已使随机分支跳过 VLAC 的导入/初始化。

## 结果口径与收集

| 产物 | 当前路径/行为 | 是否足够 |
| --- | --- | --- |
| 运行配置 | `wandb/.../glob/config.yaml` | 需要新增 reward mode、随机 seed 与分布字段 |
| 逐 episode 视频 | `wandb/.../glob/{episode}_suc_{0|1}_ttt_...mp4` | 可人工复核 success |
| 成功率 | 终端累计打印 | 不足，需要写结构化 JSON/YAML/CSV |
| README 普通评测 stats | `vis_0_train/test/stats.yaml` | 不适用于在线 TTT 最终策略 |

原因是 TTT `run()` 不调用 `render()`，而 `SimplerEnv/scripts/calc_statistics.py` 只处理 `render()` 的 `stats.yaml`。最终实现应在 TTT episode 汇总处写入包含任务、seed、checkpoint、reward mode、交互步数、每 episode success、累计成功率、视频与日志路径的结构化结果。

## 已验证层级

- 已静态验证：入口、参数、checkpoint 加载、环境交互、真实 success、TTT 更新与现有产物路径。
- 未验证：checkpoint 可加载性、资产可用性、GPU 显存、实际成功率、论文 baseline 具体行列。
