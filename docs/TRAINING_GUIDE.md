# AGI-CORTEX 训练策略与实战指南

> **目标**：从"能跑"到"能训好"的完整训练方案，包含分阶段策略、数据配方、监控指标和故障排查。

---

## 一、核心训练哲学

CORTEX 不是标准 Transformer，它的训练需要**多目标优化** + **课程学习** + **认知架构理解**。

### 1.1 为什么 CORTEX 难训？

| 问题 | 传统 Transformer | CORTEX |
|------|-----------------|--------|
| 目标函数 | 单一 next-token 预测 | LM + 符号 + 自我 + 动作 + 因果 |
| 动力学 | 纯前馈 | RNN 状态 + 脉冲神经元 + 工作空间竞争 |
| 梯度流 | 简单反向传播 | 多时间尺度 + VQ 直通估计器 |
| 评估 | Perplexity | 模块独立能力 + 系统涌现行为 |

### 1.2 训练成功的关键原则

1. **先独立，后联合**：每个 AGI 模块先单独验证能学，再逐步耦合
2. **课程学习是必需**：不能一开始就全开，必须渐进启用模块
3. **小 batch 优于大 batch**：GNW 点火是随机过程，大 batch 会平均掉
4. **监控模块指标**：只看 loss 不够，必须看 certainty、spike_rate、cognitive_load

---

## 二、分阶段训练策略

### 阶段 0：架构验证（0-2 小时）

**目标**：确认核心组件没有 bug，梯度能流动

```yaml
# configs/stage0_base.yaml
d_model: 66          # 极小模型
n_layers: 2
n_modules: 2
batch_size: 4
seq_len: 32
total_steps: 500

# AGI 模块：全部关闭
use_symbolic: false
use_self_modeling: false
use_embodied: false
use_hierarchical: false
use_continual: false
use_causal: false
```

**通过标准**：
- [ ] Loss 从 ~5.5 降到 < 3.0
- [ ] 所有参数都有梯度（`param.grad is not None`）
- [ ] 没有 NaN/Inf

**命令**：
```bash
python scripts/train_pipeline.py --quick --from_stage 0 --to_stage 0
```

---

### 阶段 1：语言建模热身（1-3 天）

**目标**：让基础 CORTEX 学会基本的语言规律

```yaml
# configs/stage1_lm.yaml
d_model: 252
n_layers: 4
n_modules: 4
batch_size: 16
seq_len: 128
total_steps: 10000

# 启用脉冲，但无 AGI 模块
use_symbolic: false
use_self_modeling: false
```

**数据**：基础语料（WikiText-2 / 自制语料）

**通过标准**：
- [ ] Val loss 稳定下降
- [ ] Spike rate 在 0.1-0.3 之间（不是 0 或 1）
- [ ] 能生成连贯的短句

---

### 阶段 2：符号 + 元认知（M1 + M2）（2-5 天）

**目标**：引入符号推理和自我建模能力

```yaml
# configs/stage2_reasoning.yaml
use_symbolic: true
use_self_modeling: true
lambda_symbolic: 0.05
lambda_self: 0.05
```

**数据配比**：
- 70% 基础语料（防止遗忘语言）
- 20% 推理数据（bAbI / 合成逻辑链）
- 10% 元认知数据（确定性校准）

**关键监控指标**：

| 指标 | 健康值 | 危险信号 |
|------|--------|----------|
| symbolic_loss | 0.5-2.0 | >5.0 → 降低 lambda_symbolic |
| self_loss | 0.01-0.5 | >2.0 → 检查 self_state 漂移 |
| certainty | 0.3-0.9 | 恒=0.99 → 过度自信 |
| cognitive_load | 0.2-0.7 | >0.9 → 任务太难 |

**通过标准**：
- [ ] 推理数据准确率 > 60%
- [ ] 低确定性样本的 certainty < 0.3
- [ ] 高确定性样本的 certainty > 0.8

---

### 阶段 3：具身 + 层次规划（M3 + M4）（3-7 天）

**目标**：让模型能与环境交互并分解目标

```yaml
# configs/stage3_rl.yaml
use_embodied: true
use_hierarchical: true
lambda_action: 0.1
```

**数据**：
- 继续语言数据（50%）
- RL 轨迹（30%）：CartPole、简单导航任务
- 层次任务（20%）：多步骤指令

**关键技巧**：
- 使用 `CORTEXEnvWrapper` 连接环境
- 动作损失权重先高（0.1）后低（0.02）
- 监控 episode reward 是否提升

---

### 阶段 4：全模块联合（M1-M6）（2-4 周）

**目标**：所有模块协同工作

```yaml
# configs/stage4_full.yaml
use_symbolic: true
use_self_modeling: true
use_embodied: true
use_hierarchical: true
use_continual: true
use_causal: true

# 损失权重（均衡）
lambda_symbolic: 0.02
lambda_self: 0.02
lambda_action: 0.05
lambda_plasticity: 0.001
```

**数据**：
- 混合所有数据类型
- 加入持续学习序列（math → geo → code → bio → physics）
- 加入因果推断数据（结构化时间序列）

**课程学习**：
```python
# 三阶段课程
阶段 A (0-33%): 关闭 M5/M6，低权重 M1/M2/M3/M4
阶段 B (33-66%): 启用 M5，中等权重
阶段 C (66-100%): 全模块，正常权重
```

---

## 三、数据生成与配比

### 3.1 一键生成全部数据

```bash
python scripts/generate_enriched_data.py --all --viz
```

生成 6 类数据 + 可视化报告：
- `data/corpus/` — 基础语料（中英文混合）
- `data/reasoning/` — 符号推理链
- `data/metacognition/` — 元认知样本
- `data/rl/` — RL 轨迹
- `data/continual/` — 多领域持续学习数据
- `data/causal/` — 因果系统观测

### 3.2 数据配比建议

| 训练阶段 | 基础语料 | 推理 | 元认知 | RL | 持续学习 | 因果 |
|----------|---------|------|--------|-----|----------|------|
| 阶段 0-1 | 100% | 0% | 0% | 0% | 0% | 0% |
| 阶段 2 | 70% | 20% | 10% | 0% | 0% | 0% |
| 阶段 3 | 50% | 15% | 10% | 25% | 0% | 0% |
| 阶段 4 | 40% | 15% | 10% | 15% | 10% | 10% |

### 3.3 数据质量检查

运行后查看 `outputs/data_viz/data_distribution.png`：
- 语言分布是否均衡（中英文比例）
- 推理规则是否覆盖全面（MP、MT、传递、因果、类比）
- 元认知确定性分布是否合理（低/中/高都有）

---

## 四、实时监控与可视化

### 4.1 训练时实时监控

```bash
# 终端 1：启动训练
python examples/train_agi_cortex.py --config configs/stage2_reasoning.yaml

# 终端 2：实时监控（每 5 秒更新）
python scripts/visualize_training.py --log_dir outputs/stage2_reasoning --watch
```

仪表盘包含：
1. **Loss 曲线**：总 loss + 基础 loss
2. **模块损失分解**：symbolic / self / action / plasticity
3. **自我状态**：certainty（应随训练提升）、cognitive_load（应稳定在中等）
4. **学习率**：CosineAnnealing 调度
5. **验证对比**：训练 vs 验证 loss
6. **训练统计**：总步数、最新 loss、模块活跃度

### 4.2 实验对比

```bash
# 对比两个阶段的训练曲线
python scripts/visualize_training.py \
    --compare outputs/stage1_lm outputs/stage2_reasoning
```

### 4.3 关键指标解读

| 指标 | 期望行为 | 异常诊断 |
|------|----------|----------|
| loss 突然上升 | 不应发生 | 学习率太高 → 减半 lr |
| spike_rate = 0 | 错误 | threshold 太高 → 调低 |
| spike_rate > 0.5 | 太密集 | threshold 太低 → 调高 |
| certainty 恒=0.5 | 未学习 | 检查 M2 是否激活 |
| certainty 恒=0.99 | 过度自信 | 增加陷阱样本（低 cert） |
| cognitive_load > 0.9 | 过载 | 降低任务难度或 seq_len |
| symbolic_loss 不降 | 未学 | 增加推理数据比例 |
| val_loss >> train_loss | 过拟合 | 增加 dropout / weight_decay |

---

## 五、常见故障排查

### 5.1 RuntimeError: backward through graph a second time

**原因**：`self._prev_self_state` 保留了梯度，跨 batch 使用。

**修复**：已在 `agi_cortex_model.py` 中修复，保存时 detach：
```python
self._prev_self_state = SelfState(
    goal_embedding=self_state.goal_embedding.detach(),
    certainty=self_state.certainty.detach() if tensor else self_state.certainty,
    ...
)
```

### 5.2 Loss = NaN

**排查步骤**：
1. 检查学习率：`lr > 0.001` 且 d_model 小 → 可能爆炸
2. 检查梯度：是否有 inf？
3. 检查 VQ 量化：`symbolic.py` 中码本是否发散？
4. 临时关闭 AMP：可能混合精度溢出

### 5.3 模块损失始终为 0

**排查**：
1. 确认模块已激活：`use_symbolic=True` 等
2. 检查 lambda 权重：是否 > 0？
3. 检查数据：是否有对应类型的数据？

### 5.4 GPU OOM

**解决方案**（按优先级）：
1. `--batch_size 2`
2. `--seq_len 64`
3. `--d_model 126`
4. `--use_amp`
5. 减少 n_layers

---

## 六、快速上手命令

### 最速验证（10 分钟）

```bash
# 1. 生成数据
python scripts/generate_enriched_data.py --all --viz --n_corpus 1000 --n_reasoning 500

# 2. 快速训练
python examples/train_agi_cortex.py \
    --data_path data/corpus/train.txt \
    --output_dir outputs/quick_test \
    --vocab_size 256 --d_model 66 --n_layers 2 \
    --seq_len 32 --batch_size 2 --total_steps 200 \
    --use_symbolic --use_self_modeling \
    --device cuda

# 3. 可视化
python scripts/visualize_training.py --render --log_dir outputs/quick_test
```

### 完整训练管道（一键）

```bash
python scripts/train_pipeline.py --full --viz --quick
```

### 生产级训练

```bash
# 阶段 0-1：基础 LM（2-5 天）
python examples/train_agi_cortex.py --config configs/stage1_lm.yaml

# 阶段 2：推理 + 元认知（3-7 天）
python examples/train_agi_cortex.py --config configs/stage2_reasoning.yaml \
    --resume outputs/stage1_lm/best.pt

# 阶段 3：具身 + 层次（3-7 天）
python examples/train_agi_cortex.py --config configs/stage3_rl.yaml \
    --resume outputs/stage2_reasoning/best.pt

# 阶段 4：全模块（2-4 周）
python examples/train_agi_cortex.py --config configs/stage4_full.yaml \
    --resume outputs/stage3_rl/best.pt
```

---

## 七、扩展与定制

### 7.1 添加新数据类型

编辑 `scripts/generate_enriched_data.py`，在 `generate_xxx_data()` 函数中添加新数据生成逻辑。

### 7.2 调整损失权重

在训练脚本中动态调整：
```python
# 训练到一半时降低符号损失权重
if trainer.global_step > 5000:
    trainer.model.lambda_symbolic = 0.01
```

### 7.3 接入真实环境

```python
from cortex.env_wrapper import CORTEXEnvWrapper
import gymnasium as gym

env = gym.make("CartPole-v1")
wrapper = CORTEXEnvWrapper(model, env)
results = wrapper.run_episode(max_steps=500)
```

---

## 八、总结：训练检查清单

每次开始训练前确认：

- [ ] 数据已生成且通过质量检查
- [ ] `d_model % n_timescales == 0`
- [ ] GPU 可用且显存足够
- [ ] 配置文件参数合理
- [ ] 监控脚本已准备好
- [ ] Checkpoint 保存路径存在

训练过程中每 1000 步检查：

- [ ] Loss 在下降
- [ ] 没有 NaN/Inf
- [ ] 各模块损失都有值
- [ ] 自我状态指标合理
- [ ] 验证集 loss 没有发散
- [ ] Checkpoint 文件正常

---

> **记住**：CORTEX 的训练是一个**认知系统的培养过程**，不是简单的参数优化。耐心观察模块的涌现行为，比单纯追求低 loss 更重要。
