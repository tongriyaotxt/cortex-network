# CORTEX / AGI-CORTEX 实操训练指南

本文档提供从零开始训练 CORTEX 的**完整实操步骤**，包括数据准备、分阶段训练命令、监控方法。

---

## 一、环境准备

```bash
# 基础依赖（必须）
pip install torch numpy

# 数据相关（推荐）
pip install datasets transformers pyyaml

# 实验跟踪（可选）
pip install wandb tensorboard
```

验证安装：
```bash
python tests/test_cortex.py        # 9/9 通过
python tests/test_agi_modules.py   # 16/16 通过
```

---

## 二、数据准备（一键完成）

### 2.1 自动生成所有训练数据

```bash
python scripts/prepare_data.py --all
```

这会生成以下数据目录：

```
data/
├── wikitext/          # WikiText-2（需要 datasets 库）
│   └── train.txt
├── reasoning/         # 推理数据（bAbI 失败时自动生成替代数据）
│   └── synthetic_train.jsonl   # 10000 条逻辑推理链
├── metacognition/     # 元认知数据
│   └── train.jsonl              # 5000 条不确定性校准样本
├── continual/         # 持续学习多领域数据
│   ├── math.txt
│   ├── geo.txt
│   ├── code.txt
│   └── bio.txt
└── rl_trajectories.txt  # 需要单独生成（见 2.3）
```

### 2.2 各数据类型说明

| 数据类型 | 文件 | 用途 | 格式 |
|----------|------|------|------|
| 基础语料 | `wikitext/train.txt` | 语言建模预训练 | 纯文本 |
| 推理数据 | `reasoning/*.jsonl` | M1 符号推理训练 | JSONL |
| 元认知 | `metacognition/*.jsonl` | M2 自我建模训练 | JSONL |
| 多领域 | `continual/*.txt` | M5 抗遗忘测试 | 纯文本 |
| RL 轨迹 | `rl_trajectories.txt` | M3 动作策略训练 | 纯文本 |

### 2.3 生成 RL 轨迹数据

```bash
# 使用 mock 环境（无需安装 gym）
python scripts/generate_rl_trajectories.py --env mock --episodes 1000

# 或使用真实环境（需要 gymnasium）
pip install gymnasium
python scripts/generate_rl_trajectories.py --env CartPole-v1 --episodes 500
```

---

## 三、分阶段训练（推荐路线）

### 阶段 0：基础架构热身（1-2 小时）

**目标**：验证核心组件（DCU、GNW、脉冲编码）能正常工作。

```bash
python examples/train_agi_cortex.py \
    --config configs/stage0_base.yaml \
    --data_path data/reasoning/synthetic_train.jsonl \
    --total_steps 10000
```

**检查指标**：
- loss 下降
- spike_rate 在 0.1-0.3 之间
- 训练不崩溃

---

### 阶段 1：语言模型预训练（1-2 周）

**目标**：让 CORTEX 获得基础语言能力。

```bash
# 如果有 WikiText-2
python examples/train_agi_cortex.py \
    --config configs/stage1_lm.yaml \
    --data_path data/wikitext/train.txt \
    --total_steps 100000

# 如果没有 WikiText，用推理数据替代
python examples/train_agi_cortex.py \
    --config configs/stage1_lm.yaml \
    --data_path data/reasoning/synthetic_train.jsonl \
    --total_steps 50000
```

**产出**：`outputs/stage1_lm/best.pt`

---

### 阶段 2：符号推理 + 自我建模（1 周）

**目标**：在阶段 1 基础上开启 M1/M2，训练结构化推理和元认知。

```bash
python examples/train_agi_cortex.py \
    --config configs/stage2_reasoning.yaml \
    --resume outputs/stage1_lm/best.pt \
    --data_path data/reasoning/synthetic_train.jsonl \
    --total_steps 50000
```

**关键观察**：
```
Step 1000 | loss=3.85 | base=3.82 | [symbolic=0.45 | self=0.12]
```
- `symbolic` 损失应 < 0.5（符号一致性良好）
- `self` 损失应稳定下降（自我状态稳定）
- `certainty` 日志值应随任务难度变化

---

### 阶段 3：具身交互 + 层次规划（1 周）

**目标**：加入 M3/M4，让模型学会动作输出和目标分解。

```bash
# 先确保 RL 轨迹已生成
python scripts/generate_rl_trajectories.py --env mock --episodes 1000

python examples/train_agi_cortex.py \
    --config configs/stage3_rl.yaml \
    --resume outputs/stage2_reasoning/best.pt \
    --data_path data/rl_trajectories.txt \
    --total_steps 50000
```

---

### 阶段 4：全模块联合训练（2-4 周）

**目标**：开启 M5/M6，端到端优化所有模块。

```bash
python examples/train_agi_cortex.py \
    --config configs/stage4_full.yaml \
    --resume outputs/stage3_rl/best.pt \
    --data_path data/wikitext/train.txt \
    --total_steps 100000
```

---

## 四、一键运行完整管道

```bash
# 完整管道（自动按顺序执行 5 个阶段）
python scripts/train_pipeline.py --all

# 跳过数据准备（如果数据已经生成）
python scripts/train_pipeline.py --all --skip_data_prep

# 仅运行单个阶段
python scripts/train_pipeline.py --stage 2
```

---

## 五、自定义训练（快速实验）

### 5.1 最小可运行示例

```bash
# 用 JSONL 推理数据训练小模型（10 分钟）
python examples/train_agi_cortex.py \
    --data_path data/reasoning/synthetic_train.jsonl \
    --vocab_size 256 \
    --d_model 66 \
    --n_layers 2 \
    --batch_size 4 \
    --seq_len 64 \
    --total_steps 1000 \
    --log_interval 50
```

### 5.2 全 AGI 模块小模型实验

```bash
python examples/train_agi_cortex.py \
    --data_path data/metacognition/train.jsonl \
    --vocab_size 256 \
    --d_model 66 \
    --n_layers 2 \
    --batch_size 2 \
    --seq_len 32 \
    --total_steps 500 \
    --use_symbolic \
    --use_self_modeling \
    --use_embodied \
    --use_hierarchical \
    --use_continual \
    --use_causal
```

### 5.3 从 checkpoint 恢复

```bash
python examples/train_agi_cortex.py \
    --config configs/stage2_reasoning.yaml \
    --resume outputs/stage2_reasoning/step_5000.pt \
    --total_steps 100000
```

---

## 六、训练监控

### 6.1 日志输出解读

```
Step 1000/100000 | loss=4.1234 | base=4.1000 | [symbolic=0.45 | self=0.02 | action=-0.15 | plasticity=0.00] | cert=0.65 | load=0.42 | lr=8.50e-05
```

| 字段 | 含义 | 健康范围 |
|------|------|----------|
| `loss` | 总损失（base + 模块损失） | 持续下降 |
| `base` | 语言建模交叉熵 | 2-5（取决于任务） |
| `symbolic` | M1 符号一致性损失 | < 0.5 |
| `self` | M2 自我一致性损失 | < 0.1 |
| `action` | M3 动作正则化 | -0.5 ~ 0 |
| `plasticity` | M5 可塑性正则化 | < 0.01 |
| `cert` | 自我确定度 | 0.3-0.9（太低=不自信，太高=盲目） |
| `load` | 认知负荷 | 0.2-0.7 |

### 6.2 关键检查清单

每 1000 steps 检查一次：

- [ ] `loss` 没有突然飙升（如果飙升，降低学习率或检查数据）
- [ ] `spike_rate` 不在 0 或 1 上停留（检查 DCU threshold）
- [ ] `ignition_prob` 有波动（不是恒定的 0.5）
- [ ] `module_losses` 中所有模块都有非零值（确认模块被激活）
- [ ] checkpoint 文件正常保存

### 6.3 问题排查

| 现象 | 原因 | 解决 |
|------|------|------|
| loss = NaN | 学习率太高或梯度爆炸 | 降低 `--lr` 到 1e-5，或增大 `--max_grad_norm` |
| spike_rate = 0 | DCU threshold 太高 | 暂时关闭 `use_spike`，或修改 `threshold=0.5` |
| 模块损失始终为 0 | 模块未启用或权重为 0 | 确认 `--use_xxx` 已开启，检查 `lambda_xxx` |
| 显存溢出 | batch_size 或 seq_len 太大 | 降低 `--batch_size` 或 `--seq_len` |
| 训练极慢 | 数据文件太大，每次加载全文件 | 将数据预处理为更小的 chunk |

---

## 七、数据增强建议

### 7.1 提高推理能力

编辑 `scripts/prepare_data.py` 中的 `templates` 列表，添加更多逻辑规则：

```python
templates.append({
    "template": "如果 {A} 那么 {B}。如果 {B} 那么 {C}。{A}。因此？",
    "answer": "{C}",
    "rule": "chain_syllogism"
})
```

然后重新运行：
```bash
python scripts/prepare_data.py --reasoning
```

### 7.2 添加领域特定数据

创建 `data/domain/my_domain.txt`，然后：

```bash
python examples/train_agi_cortex.py \
    --resume outputs/stage4_full/best.pt \
    --data_path data/domain/my_domain.txt \
    --lr 0.00001 \
    --total_steps 10000
```

### 7.3 连接真实环境（M3）

```python
from cortex import CORTEXEnvWrapper, EnvironmentWrapper

# 安装 gymnasium: pip install gymnasium
env = EnvironmentWrapper('LunarLander-v3')
wrapper = CORTEXEnvWrapper(model, env, device='cuda')

# 训练循环
for episode in range(1000):
    result = wrapper.run_episode()
    print(f"Episode {episode}: reward={result['episode_reward']:.2f}")
```

---

## 八、产出物

训练完成后，你会得到：

```
outputs/
├── stage0_base/
│   ├── config.json       # 训练配置
│   ├── train.log         # 完整日志
│   ├── best.pt           # 验证集上最好的模型
│   └── final.pt          # 最终模型
├── stage1_lm/
│   └── ...
├── stage2_reasoning/
│   └── ...
├── stage3_rl/
│   └── ...
└── stage4_full/
    ├── best.pt           # 最终推荐的模型文件
    ├── final.pt
    └── step_XXXXX.pt     # 中间 checkpoint
```

**使用训练好的模型**：

```python
import torch
from cortex import AGICORTEXModel

model = AGICORTEXModel(vocab_size=32000, d_model=512, n_layers=12, ...)
model.load_state_dict(torch.load('outputs/stage4_full/best.pt')['model'])
model.eval()

# 生成文本（AGI 模块参与）
input_ids = torch.tensor([[1, 2, 3]])
output = model.generate(input_ids, max_new_tokens=50, use_agi_modules=True)
```

---

## 九、硬件建议

| 阶段 | 推荐 GPU | 显存 | 训练时间 |
|------|----------|------|----------|
| 阶段 0 | GTX 1660 | 6GB | 1-2 小时 |
| 阶段 1 | RTX 3060 | 12GB | 2-3 天 |
| 阶段 2-4 | RTX 3090 / A100 | 24GB+ | 1-2 周 |

**小模型快速实验配置**（适合 6GB 显存）：
```bash
--d_model 252 --n_layers 4 --batch_size 4 --seq_len 128
```
