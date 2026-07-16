# TT-VLA 随机 Dense Reward 实验：代码审核结论

> 审核依据：[TT-VLA-随机奖励任务-审核模板](TT-VLA-随机奖励任务-审核模板.md)
>
> 审核方式：仓库静态代码、配置与历史命令审计；未启动 conda、GPU、模型或远程服务器。

## 审核结论

**条件通过实施。** 模板规定的核心边界与代码实际结构一致：训练用 VLAC dense reward 可在同一 replay buffer 写入位置替换，真实 `success`、动作执行、环境交互和训练主框架无需重构。

实施前必须满足三个前置条件：

1. 在服务器验证 OpenVLA/ManiSkill/SimplerEnv 的导入和单 episode 运行；若仍保留 VLAC 模式，还要验证 OpenVLA 与 VLAC 的版本共存。
2. 将随机模式实现为绕过 VLAC 的 import、Critic 初始化和 progress 推理，而不是只删除 reward 赋值。
3. 确定真实 Git clone/branch。当前目录没有 `.git`，不能证明或直接执行“本机 push、服务器 pull”。

## 对模板关键项的审核

| 模板项 | 审核结果 | 代码证据/说明 |
| --- | --- | --- |
| 仅替换 reward model 数值 | 通过 | `replay_buffer.py:241-249` 从 VLAC progress 差分写入最终训练 reward。 |
| 不改整体训练框架 | 通过 | 后续 GAE、advantage 与 PPO 更新保持 `replay_buffer.py:250-269`、`openvla_train.py:547-591` 原样。 |
| 不改 success、动作、终止和统计口径 | 通过 | success/done 在 `simpler_wrapper.py:107-123` 与 `train_ms3_ppo_ttt.py:408-417` 独立存在。 |
| 每条 reward 独立随机、两位小数、可复现 | 待实施验证 | 需在 buffer 改动中使用独立 RNG 和整数分采样，并完成 100,000 次统计。 |
| 三类各一个最小任务 | 条件通过 | 推荐任务见 [推理评测线路](TT-VLA-推理评测线路.md)；类别是代码语义映射，论文 baseline 行列待人工核对。 |
| 在同一 rlvla 环境补 VLAC | 有风险 | OpenVLA 与 VLAC 声明的 Transformers/PEFT 版本直接冲突，必须远程 probe。随机模式应不依赖 VLAC。 |
| 记录 seed、checkpoint、日志和结果 | 当前不足 | config 已会写入；TTT 当前只打印成功率和视频，需追加结构化结果文件。 |

## 实施改动面

| 优先级 | 文件 | 操作 |
| --- | --- | --- |
| 必须 | `SimplerEnv/simpler_env/utils/replay_buffer.py` | 用随机采样替代 VLAC progress 差分；随机模式条件化/延迟 VLAC import 与 Critic 初始化。 |
| 必须 | `SimplerEnv/simpler_env/train_ms3_ppo_ttt.py` | 添加 reward mode 与随机 seed 配置；写入逐 episode 结构化实验结果。 |
| 不改 | 环境 wrapper、ManiSkill task、PPO 算法、VLAC 源码 | 保证实验只改变训练 dense reward。 |

详细调用链与边界见 [训练与随机奖励改动线路](TT-VLA-训练与随机奖励改动线路.md)。

## 最终运行方式

```text
本机真实 Git 工作树：修改代码与 docs -> commit/push
远程 Linux CUDA 服务器：pull 同一 SHA -> 环境探针 -> 单 episode 冒烟
远程服务器：Execution / Vision / Semantics 各运行随机奖励任务
远程服务器：写配置、结构化结果、视频和日志
本机：拉取结果摘要，与论文 baseline 汇总对比
```

具体命令、环境冲突和远程核验清单见 [环境与远程运行方式](TT-VLA-环境与远程运行方式.md)。

## 当前验收状态

- [x] 随机 reward 的正确替换位置已定位。
- [x] success、done 与动作执行的隔离已确认。
- [x] TTT 训练、推理评测与结果产物线路已梳理。
- [x] 环境、GPU 与 Git 的静态风险已标明。
- [ ] 随机奖励 100,000 次分布验证。
- [ ] 服务器依赖 probe 与单 episode 冒烟。
- [ ] 三类任务运行、结果汇总和论文 baseline 对照。
