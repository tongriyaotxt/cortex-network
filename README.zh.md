# CORTEX：意识编排递归变换器兴奋性整合网络

**作者：** 田晓潼

---

## 概述

CORTEX 是一个从根本上全新的神经网络架构，它架起了三个历史上相互割裂的领域之间的桥梁：**深度学习工程**、**计算神经科学**和**认知科学**。它是首个系统性地整合了以下要素的架构：

- **状态空间模型**（如 Mamba/Mamba-2）用于线性时间序列建模
- **脉冲神经网络（SNN）** 用于事件驱动的稀疏计算
- **全局工作空间理论** 来自认知神经科学的意识研究
- **生物合理计算**：树突分支、预测编码和多时间尺度动态

与传统模型将神经网络视为纯粹的函数逼近器不同，CORTEX 被设计为一个**动态系统**，它维持并更新一个显式的内部世界模型，这更接近生物大脑处理信息的方式。

---

## 核心创新

### 1. 全局神经工作空间层（GNW Layer）
受 Dehaene & Changeux 的意识全局神经工作空间理论启发，该层实现了**竞争性选择**、**阈值点火（ignition）**和**全局广播**。信息经历从局部无意识处理到全局可用意识内容的相变过程。

### 2. 树突计算单元（DCU）
受锥体神经元电生理学启发，每个 DCU 包含多个具有**独立非线性**的树突分支（兴奋性、抑制性、整流性、调制性）。这使得非线性空间聚合的表达能力严格优于标准 MLP。

### 3. 脉冲-连续混合编码
一种全新的训练-推理解耦设计：
- **训练时**：使用可微分的软脉冲（概率）保持梯度流
- **推理时**：可以作为真正的稀疏脉冲部署到神经形态硬件（如 Intel Loihi）
- 结合了 SNN 的能效优势和 ANN 的可训练性

### 4. 集成预测编码
每一层都实现了带精度加权的**预测-误差分解**，受 Friston 自由能原理启发。模型不仅学习将输入映射到输出，还学习预测上一层的状态，利用预测误差驱动学习。

### 5. 多时间尺度异质处理
不同的神经元子群在不同时间尺度上运行（快速 ~25ms、中等 ~100ms、慢速 ~500ms），通过跨时间尺度连接耦合。这自然地统一了短时工作记忆和长时上下文记忆，无需外部记忆机制。

### 6. 意识门控注意力
注意力权重受全局工作空间意识状态 `C_t` 的调制，实现**自上而下的认知控制**。模型"正在思考什么"会影响它"关注什么"。

---

## 安装

```bash
pip install torch numpy
```

---

## 快速开始

```python
import torch
from cortex import CORTEXModel

# 创建 CORTEX 模型
model = CORTEXModel(
    d_model=512,
    n_layers=12,
    n_modules=8,            # 局部处理模块数
    workspace_dim=256,      # 工作空间维度
    n_branches=4,           # 每单元树突分支数
    n_timescales=3,         # 时间尺度数量
    vocab_size=50000,
    max_seq_len=4096,
    consciousness_output=True
)

# 前向传播
x = torch.randint(0, 50000, (2, 1024))  # (batch, seq_len)
outputs = model(x, return_consciousness=True)

logits = outputs['logits']
consciousness = outputs['consciousness']  # 提取模型的"意识状态"

# 文本生成
generated = model.generate(
    input_ids=x[:, :8],
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

---

## 架构文档

详见 [ARCHITECTURE.md](ARCHITECTURE.md) 获取完整的数学公式、生物学动机和设计原理。

---

## 项目结构

```
cortex/
├── __init__.py
├── dendritic.py              # 树突计算单元
├── spike_encoding.py         # 脉冲-连续混合编码
├── workspace.py              # 全局神经工作空间层
├── predictive_coding.py      # 预测编码层
├── multiscale_state.py       # 多时间尺度状态管理
├── cortex_block.py           # CORTEX 构建块
└── cortex_model.py           # 完整序列模型

tests/
└── test_cortex.py            # 单元测试

examples/
├── train_language_model.py   # 语言建模示例
└── consciousness_analysis.py # 意识状态分析示例
```

---

## 核心设计哲学

| 传统深度学习 | CORTEX |
|---|---|
| 函数逼近 | 维持内部模型的动态系统 |
| 自注意力（全对全） | 全局工作空间（竞争 + 广播） |
| 单一时间尺度 | 异质多时间尺度耦合 |
| 密集计算 | 脉冲门控稀疏计算 |
| 仅端到端反向传播 | 局部预测误差信号 + 全局任务损失 |

---

## 引用

如果您在研究中使用了 CORTEX，请引用：

```bibtex
@article{cortex2025,
  title={CORTEX: A Biologically-Motivated Neural Architecture Integrating Global Workspace Theory, 
         Dendritic Computation, and Hybrid Spike-Continuous Dynamics},
  author={田晓潼},
  year={2025}
}
```

---

## 致谢

本架构综合了以下领域的洞见：
- **Transformer/SSM 研究**（Vaswani et al., Gu & Dao, Dao & Gu）
- **全局工作空间理论**（Baars, Dehaene & Changeux）
- **预测加工理论**（Friston, Rao & Ballard）
- **神经形态计算**（Maass, Indiveri, Zenke）
- **树突计算**（Spruston, Larkum, Major）

---

*由 田晓潼 设计并实现。*
