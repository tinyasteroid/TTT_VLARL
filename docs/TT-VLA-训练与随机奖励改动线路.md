# TT-VLA 训练与随机奖励改动线路

## 结论

随机 dense reward 的集中改动点是 `SimplerEnv/simpler_env/utils/replay_buffer.py` 中 `SeparatedReplayBuffer_vlac.compute_returns_ppo()`；不是环境 wrapper 的 `get_reward()`。

当前 TTT 路线先将环境返回的 shaping reward 写入 buffer，但在每个 TTT 窗口训练前，VLAC progress 的相邻差分会覆写同一段 `self.rewards`。因此，按审核模板替换“reward model 数值”必须替换这一次覆写。

## 实际训练调用链

```text
train_ms3_ppo_ttt.py:main
  -> Runner.run()
    -> collect() -> OpenVLAPolicy.get_action()
    -> SimlerWrapper.step()
      -> ManiSkill 环境交互、动作执行、success/终止信息
      -> 临时环境 shaping reward
    -> Runner.insert() -> replay buffer
    -> 每 tt_steps 步 Runner.train()
      -> OpenVLAPPO.train_ppo_vlac()
      -> SeparatedReplayBuffer_vlac.compute_returns_ppo()
      -> VLAC progress 差分覆写训练 reward
      -> GAE / advantage -> PPO 更新 OpenVLA
```

| 环节 | 代码证据 | 审核含义 |
| --- | --- | --- |
| TTT 入口与参数 | `SimplerEnv/simpler_env/train_ms3_ppo_ttt.py:31-99,470-497` | 使用该文件，不是根 README 中的普通 `train_ms3_ppo.py` |
| 交互和 buffer 初写入 | `train_ms3_ppo_ttt.py:382-387` | 环境 reward 先写入，但不是最终训练 reward |
| TTT 定时更新 | `train_ms3_ppo_ttt.py:394-399` | 仅在每个完整 `tt_steps` 窗口后训练 |
| PPO 回报计算入口 | `SimplerEnv/simpler_env/policies/openvla/openvla_train.py:547-570` | 先执行 buffer 的 `compute_returns_ppo()` |
| VLAC reward 覆写 | `SimplerEnv/simpler_env/utils/replay_buffer.py:238-269` | 唯一应替换的位置 |

## 应修改与不得修改的范围

### 集中修改：两个文件

| 文件 | 最小职责 |
| --- | --- |
| `SimplerEnv/simpler_env/utils/replay_buffer.py` | 增加显式 seed 的随机奖励采样器；在 `compute_returns_ppo()` 的 241-249 行区段禁用 VLAC progress 推理，并对最近 TTT 窗口写入随机 reward；随机模式还必须跳过 `GAC_model` 导入与初始化（现有代码在 216-234 行无条件加载 Critic）。 |
| `SimplerEnv/simpler_env/train_ms3_ppo_ttt.py` | 在 `Args` 增加 reward mode、随机 reward seed/分布配置和结果记录字段，使它们进入既有 W&B config 与 `glob/config.yaml`。 |

建议保持默认 `reward_mode=vlac`，新增 `reward_mode=random_dense`，使原有路线不受改变。随机模式使用独立的 `numpy.random.Generator(seed)`，逐 reward 独立采样：正数概率 `0.3620`、整数分 `1..53`；负数概率 `0.0953`、整数分 `-33..-1`；零概率 `0.5427`、值 `0`；最后除以 `100`。这保证写入值精确为两位小数且不由浮点格式化改变类别。

### 不得修改

| 文件/路径 | 原因 |
| --- | --- |
| `SimplerEnv/simpler_env/env/simpler_wrapper.py` | `get_reward()`（38-49 行）是临时环境 shaping reward；改它会偏离“替换 reward model 输出”的审核边界。 |
| `ManiSkill/.../put_on_in_scene_multi.py` | 真实任务 success 的定义与环境任务逻辑。 |
| `SimplerEnv/simpler_env/policies/openvla/openvla_train.py` | PPO 更新、returns/advantages 的既有算法流程。 |
| `VLAC/evo_vlac/...` | 随机实验不需要改变 reward model 本体。 |

## success、终止与 reward 的隔离

- `SimlerWrapper.step()` 从 ManiSkill 获得 `success` 与 `truncated`，见 `simpler_wrapper.py:107-123`。
- wrapper 当前丢弃 `_terminated`，并将 `truncated` 作为训练 `done`；该既有语义必须保留。
- `Runner.insert()` 将该 `done` 转为 PPO mask，见 `train_ms3_ppo_ttt.py:192-203`。
- 最终成功率读取真实的 `env_info["success"]`，见 `train_ms3_ppo_ttt.py:408-417`。

所以随机 reward 只能写入 `self.rewards` 的训练窗口；不能写入 `success`、done/mask、动作或环境终止字段。

## 实施前的最小验证

1. 以固定 seed 采样至少 100,000 次，记录三类经验频率、最小/最大值，以及所有值乘 100 后为整数。
2. 运行一 episode、`--num_envs 1`、`--tt_steps 8` 的随机奖励冒烟任务，验证 `buffer/reward_mean`、视频、配置和 success 输出。
3. 核验随机模式未调用 `get_progress_from_vlac()`、未初始化 Critic，也未改动 `success` 与 `truncated`。
4. `episode_len` 必须能被 `tt_steps` 整除；否则末尾不完整窗口不会进入 TTT 更新。

## 已知限制

- 当前 TTT buffer 只取第 0 个环境轨迹，PPO 批处理亦固定使用 `instruction[0]`；随机奖励实验必须传 `--num_envs 1`，依据 `replay_buffer.py:271-274`、`openvla_train.py:572-591`。
- 当前 TTT runner 捕获任意异常后无限重试（`train_ms3_ppo_ttt.py:362-406`）。远程正式运行前必须先完成单 episode 冒烟，避免路径或依赖错误造成卡死。
- TTT 默认只打印累计成功率并写视频/config；需要在 `train_ms3_ppo_ttt.py:410-417` 补写每 episode 结构化结果，才能满足审核模板的结果表要求。
