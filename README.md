# ppo-param-exp

PPO 参数敏感性实验：每次只改一个参数，记录它对训练指标的影响。

实验对象是开源仓库 `rsl_rl_isrc` 的 `algorithms/ppo.py`，`ppo.py` 本身不做改动，参数在构建 `PPO` 时传入。本仓库只放实验脚本、记录与原始输出，不含该仓库的源码。

## 环境

- 系统：Windows 11 + WSL2（Ubuntu 24.04）
- Python 与 PyTorch 装在虚拟环境 `~/venvs/torch`
- 被测仓库：`~/code/rsl_rl_isrc`

## 运行

```bash
t                                            # 激活虚拟环境
cp ppo_probe.py ~/code/rsl_rl_isrc/          # 脚本需放进被测仓库目录运行
cd ~/code/rsl_rl_isrc
python ppo_probe.py                          # 基线
python ppo_probe.py --clip 1.0               # 只改 clip_param
python ppo_probe.py --epochs 16              # 只改 num_learning_epochs
```

脚本默认值等于仓库自带训练配置（`tests/test_ppo_runner.py` 的 `train_cfg`），命令行只覆盖指定项。

## 实验设置

- 环境：自建小车环境，4 维观测、1 维动作，每步奖励 1，单回合最多 200 步
- 运行：4 个并行环境，每轮每环境 100 步，共 300 轮，随机种子 0，单次运行
- 指标：第 300 轮的平均回合回报、策略标准差均值、值函数损失

## 结果摘要

| 参数 | 三档取值 | 第 300 轮回报 |
|---|---|---|
| `clip_param` | 0.01 / 0.2 / 1.0 | 2.30 / 48.30 / 26.65 |
| `learning_rate` | 1e-4 / 3e-4 / 1e-3 | 5.40 / 48.30 / 5.35 |
| `num_learning_epochs` | 1 / 4 / 16 | 5.40 / 48.30 / 195.00 |
| `num_mini_batches` | 1 / 4 / 16 | 11.45 / 48.30 / 162.45 |
| `entropy_coef` | 0.0 / 0.01 / 0.1 | 53.70 / 48.30 / 24.85 |
| `gamma` | 0.95 / 0.99 / 0.999 | 152.90 / 48.30 / 34.40 |

每行中间一档为基线。结果为单种子、单次运行，只说明该设置下的趋势。

## 文件

| 文件 | 内容 |
|---|---|
| `report.md` | 完整实验记录：条件、方法、结果、逐项说明、限制 |
| `ppo_probe.py` | 实验脚本，把 `act` → 环境步进 → `update` 串成一个循环并打印指标 |
| `sweep_results.txt` | 全部 15 组运行的原始输出 |
