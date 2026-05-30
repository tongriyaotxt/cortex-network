# CORTEX 训练指南

**作者：** 田晓潼 / tongriyao

---

## 1. 环境准备

### 1.1 基础依赖

```bash
pip install torch>=2.0 numpy tqdm
```

### 1.2 可选依赖（推荐）

```bash
# 用于实验跟踪
pip install wandb tensorboard

# 用于数据集处理
pip install datasets transformers

# 用于混合精度训练
# PyTorch 2.0+ 自带 torch.cuda.amp，无需额外安装
```

---

## 2. 快速开始：最小训练示例

```python
import torch
from cortex import CORTEXModel

# 1. 创建模型
model = CORTEXModel(
    vocab_size=32000,
    d_model=512,
    n_layers=8,
    n_modules=8,
    workspace_dim=256,
    n_branches=4,
    n_timescales=3,
    max_seq_len=2048,
    dropout=0.1,
    consciousness_output=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 3. 简单训练循环
model.train()
for step in range(1000):
    # 模拟一批数据
    x = torch.randint(0, 32000, (4, 128)).to(device)
    y = torch.randint(0, 32000, (4, 128)).to(device)
    
    optimizer.zero_grad()
    outputs = model(x, labels=y)
    loss = outputs['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

---

## 3. CORTEX 特有的训练要点

### 3.1 多目标损失函数

CORTEX 的训练损失由多个部分组成：

```python
loss_total = loss_task + λ₁ * loss_pred + λ₂ * loss_consciousness + λ₃ * loss_sparse
```

| 损失项 | 作用 | 推荐权重 |
|--------|------|----------|
| `loss_task` | 主任务损失（交叉熵/MSE） | 1.0 |
| `loss_pred` | 预测编码损失 | 0.1 ~ 0.3 |
| `loss_consciousness` | 意识状态平滑损失 | 0.01 ~ 0.05 |
| `loss_sparse` | 脉冲稀疏性正则化 | 0.001 ~ 0.01 |

**注意**：在训练初期，建议将 `loss_pred` 和 `loss_sparse` 的权重设为 0，待主任务损失稳定后再逐步增加。

### 3.2 课程学习策略（三阶段训练）

由于 CORTEX 包含脉冲门控、工作空间竞争等非标准组件，建议采用**课程学习**：

#### 阶段一：纯连续训练（Warm-up）

```python
# 关闭脉冲编码，关闭工作空间竞争
model = CORTEXModel(
    ...,
    use_spike=False,      # 禁用脉冲路径
    consciousness_output=False,
)
```

- 训练 10-20% 的总步数
- 仅使用 `loss_task`
- 让模型的连续路径先收敛

#### 阶段二：引入脉冲稀疏性

```python
# 启用脉冲，但降低稀疏性惩罚
for block in model.layers:
    block.use_spike = True
```

- 逐步增加 `loss_sparse` 权重
- 目标是将平均脉冲率控制在 10-30%

#### 阶段三：完整联合训练

```python
# 启用所有组件
# 全目标损失 + 工作空间竞争
```

- 使用完整的多目标损失
- 微调超参数

### 3.3 梯度裁剪至关重要

由于脉冲替代梯度和工作空间竞争机制，CORTEX 的梯度动态比 Transformer 更不稳定。**务必使用梯度裁剪**：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3.4 学习率调度

推荐采用 **Warm-up + Cosine Annealing**：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# Warm-up 阶段
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps
)

# 主训练阶段
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=1e-6
)
```

**推荐学习率**：
- 小型模型（d_model <= 256）：`3e-4 ~ 1e-3`
- 中型模型（d_model 256-512）：`1e-4 ~ 3e-4`
- 大型模型（d_model >= 512）：`5e-5 ~ 1e-4`

---

## 4. 不同任务的训练方法

### 4.1 语言建模（自回归）

```python
# 数据：输入序列，标签为输入右移一位
input_ids = tokens[:, :-1]
labels = tokens[:, 1:]

outputs = model(input_ids, labels=labels)
loss = outputs['loss']
```

**关键配置**：
- `vocab_size`：根据 tokenizer 设定
- `max_seq_len`：取决于显存，通常 512-4096
- `tie_weights=True`：共享词嵌入和输出层权重

### 4.2 文本分类

```python
# 使用 num_classes 参数创建分类模型
model = CORTEXModel(
    vocab_size=32000,
    num_classes=10,  # 分类类别数
    d_model=256,
    n_layers=4,
    ...
)

# 标签是类别索引
outputs = model(input_ids, labels=class_labels)
loss = outputs['loss']  # 内部自动使用 CrossEntropyLoss
```

**注意**：分类任务中，模型会对序列做 mean pooling 后送入分类头。

### 4.3 序列到序列（Seq2Seq）

CORTEX 目前主要设计用于编码器（类似 GPT/BERT）。如需 Seq2Seq：

```python
# 使用两个 CORTEXModel：Encoder + Decoder
# Encoder 处理输入序列
encoder_outputs = encoder(input_ids)
memory = encoder_outputs['consciousness']  # 用意识状态作为记忆

# Decoder 自回归生成
# （需要自行实现交叉注意力机制，当前版本未内置）
```

---

## 5. 数据准备

### 5.1 使用 HuggingFace Datasets

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_function, batched=True)
```

### 5.2 自定义数据集

```python
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokens = [tokenizer.encode(t, max_length=max_length, truncation=True) 
                       for t in texts]
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx])
```

### 5.3 数据整理（Data Collator）

```python
def collate_fn(batch, pad_token_id=0):
    max_len = max(len(x) for x in batch)
    padded = []
    masks = []
    for x in batch:
        pad_len = max_len - len(x)
        padded.append(torch.cat([x, torch.full((pad_len,), pad_token_id)]))
        masks.append(torch.cat([torch.ones(len(x)), torch.zeros(pad_len)]))
    return torch.stack(padded), torch.stack(masks)
```

---

## 6. 高级训练技巧

### 6.1 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs, labels=labels)
        loss = outputs['loss']
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 6.2 分布式训练 (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend='nccl')

# 包装模型
model = DDP(model, device_ids=[local_rank])

# 使用 DistributedSampler
dataloader = DataLoader(
    dataset,
    sampler=DistributedSampler(dataset),
    ...
)
```

### 6.3 Checkpoint 保存与恢复

```python
# 保存
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'step': step,
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# 恢复
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 6.4 监控脉冲率

```python
# 定期检查脉冲率
stats = model.get_spike_statistics(input_ids)
print(f"Mean spike rate: {stats['mean_spike_rate']:.4f}")

# 如果脉冲率过高，增加稀疏性损失权重
if stats['mean_spike_rate'] > 0.5:
    sparse_weight *= 1.1
```

---

## 7. 常见问题

### Q1: 训练 loss 不下降？

- 检查是否启用了梯度裁剪（CORTEX 必须启用）
- 降低学习率到 `1e-4` 以下
- 先禁用脉冲编码（`use_spike=False`），确认连续路径能训练

### Q2: 脉冲率太高（>50%）？

- 增加 `loss_sparse` 权重
- 提高 DCU 的 threshold（如从 1.0 提高到 1.5）
- 降低 `spike_dim_ratio`（如从 0.5 降到 0.3）

### Q3: 显存不足？

- 减少 `d_model`（建议能被 `n_timescales` 整除）
- 减少 `n_layers` 或 `n_modules`
- 减少 `max_seq_len`
- 使用 `torch.cuda.amp` 混合精度
- 减少 `batch_size` 并增大 `gradient_accumulation_steps`

### Q4: 工作空间竞争失效（所有模块权重相同）？

- 检查 `competition_temp` 是否过高（默认 0.5，可降至 0.1）
- 检查模块的 `saliency_estimator` 是否在学习（查看梯度是否存在）

---

## 8. 完整训练脚本

见 `examples/train_cortex.py` —— 一个生产级的通用训练脚本，支持：
- YAML 配置文件
- 多任务类型（LM / Classification）
- 自动课程学习
- WandB / TensorBoard 日志
- Checkpoint 自动保存
- 验证和早停

```bash
python examples/train_cortex.py --config configs/lm_base.yaml
```
