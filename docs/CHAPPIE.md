# CHAPPIE: Non-Training CORTEX

**C**onscious **H**euristic **A**daptive **P**rogramming via **P**redictive **I**nstant **E**ncoding

---

## 核心思想

传统深度学习把智能看作**函数逼近问题**：用海量数据拟合一个从输入到输出的映射。

CHAPPIE 提出另一种范式：智能是**结构化知识 + 自组织可塑性 + 元认知自修改**的产物。就像《超能查派》中的机器人——
- 观看一次就能学会格斗
- 被上传意识后直接获得人格
- 能感知自身损伤并自我修复
- 一次惊吓就能形成永久记忆

这些都不需要梯度下降。

---

## 五大非训练机制

### 1. KnowledgeCompiler —— 知识直接编译

**问题**：如何让模型"知道"重力会导致坠落？
- 传统方法：在数百万文本中隐式学习
- CHAPPIE 方法：直接把规则编译进权重

```python
compiler = KnowledgeCompiler(d_model=512, n_branches=4)

# 编译一条规则：if human AND falling -> catch
compiler.compile_rule(
    premises=["human", "falling"],
    conclusion="catch",
    confidence=0.95
)

# 编译一个事实：gravity causes falling
compiler.compile_fact(
    subject="gravity",
    relation="causes",
    object="falling"
)

# 编译一套动作：catch_human procedure
compiler.compile_procedure(
    name="catch_human",
    steps=["detect_fall", "move_fast", "extend_arms", "absorb_impact"],
    motor_outputs=[0.9, 0.95, 0.8, 0.7]
)

# 直接注入 DCU 权重 —— 零训练，毫秒完成
compiler.apply_to_dcu(dcu_layer)
```

**技术原理**：
- 使用超向量（Hypervector）编码符号
- 通过 FFT 绑定实现符号关联
- 将结构化知识投影为 DCU 树突分支的权重模式
- 每个分支变成一个"逻辑检测器"

### 2. HebbianDCU —— 单次交互学习

**问题**：如何只看一次演示就学会？
- 传统方法：需要大量标注数据和反向传播
- CHAPPIE 方法：赫布学习，一次激活就足够

```python
# 包装现有 DCU
hebbian_dcu = HebbianDCU(base_dcu, rule="oja", lr=0.01)

# 展示一次
output = hebbian_dcu(demonstration_input)
# 权重已自动更新 —— 没有 backward()，没有 optimizer

# 查看分支专业化程度
specs = hebbian_dcu.get_branch_specialization()
```

**支持的规则**：
| 规则 | 公式 | 特性 |
|------|------|------|
| Plain Hebbian | Δw = η · post ⊗ pre | 简单快速 |
| Oja | Δw = η · (post ⊗ pre − post² · w) | 自动归一化 |
| BCM | Δw = η · post · (post − θ) ⊗ pre | 选择性增强 |
| STDP | 基于脉冲时序 | 生物精确 |

### 3. WorkspaceBootstrapper —— 想象自举

**问题**：没有外部数据时如何学习？
- 传统方法：无法学习
- CHAPPIE 方法：模型自己想象场景，从预测误差中学习

```python
bootstrapper = WorkspaceBootstrapper(workspace_dim=256, vocab_size=1000)

# 白日梦：从随机种子想象 100 步
imagined = bootstrapper.daydream(seed_tokens, model, steps=100)

# 从想象学习
bootstrapper.learn_from_imagination(imagined, model, hebbian_layers)

# 睡眠周期：重放和巩固
bootstrapper.sleep_cycle(model, hebbian_layers, cycles=5)
```

**核心机制**：
- 内部世界模型预测下一步工作空间状态
- 实际与预测的差距 = 惊讶度 = 学习信号
- 好奇心驱动：只从"惊讶"的事件学习
- 睡眠重放：像生物睡眠一样巩固记忆

### 4. SelfModifyingInterface —— 元认知自修改

**问题**：模型能感知和修复自己的问题吗？
- 传统方法：需要人类工程师诊断
- CHAPPIE 方法：模型自检、自诊、自修

```python
interface = SelfModifyingInterface(model)

# 全面诊断
recommendations = interface.diagnose()
# -> [{"issue": "3 dead neurons", "fix": "prune_dead"}, ...]

# 自动修复
result = interface.self_heal()
# -> 发现问题 -> 执行修复 -> 验证 -> 回滚（如果失败）

# 有目标的进化
interface.evolve(objective="efficiency")  # 剪枝冗余
interface.evolve(objective="capacity")    # 增加分支
interface.evolve(objective="stability")   # 平衡激活
```

**可执行的操作**：
- `prune_dead_branches`: 剪除死亡分支
- `strengthen_path`: 增强特定通路
- `balance_activation`: 稳态可塑性调节
- `add_branch_to_dcu`: 神经发生式新增分支
- `normalize_weights`: 权重归一化防爆炸

### 5. OneShotConsolidator —— 单次记忆巩固

**问题**：一次惊吓如何形成终身记忆？
- 传统方法：需要重复曝光（epoch）
- CHAPPIE 方法：重要性门控的即时结构修改

```python
consolidator = OneShotConsolidator(dcu_layer, importance_threshold=0.7)

# 捕获一个重要事件（比如看到枪）
consolidator.capture(
    input_pattern=perception,
    workspace_state=consciousness,
    importance=0.98,  # 高重要性 -> 立即巩固
    auto_consolidate=True
)

# 之后检索
memories = consolidator.retrieve(query_pattern, top_k=3)

# 注入当前工作空间（回忆影响当前思维）
new_workspace = consolidator.retrieve_and_inject(query, workspace)
```

**记忆层级**：
```
事件发生
    -> 捕获到 pending_engrams（瞬间）
    -> 若 importance > 0.9: 立即巩固（闪光记忆）
    -> 否则: 等待睡眠周期统一巩固
    -> 检索时: 相似度匹配 +  rehearsal 强化
```

---

## 整合使用：ChappieCORTEX

```python
from cortex import AGICORTEXModel
from cortex.chappie import ChappieCORTEX

# 1. 创建基础 CORTEX 模型（无需预训练！）
base = AGICORTEXModel(d_model=512, n_layers=8, ...)

# 2. 包装为 CHAPPIE
chappie = ChappieCORTEX(base)

# 3. 上传知识（意识转移）
chappie.upload_knowledge({
    "rules": [...],
    "facts": [...],
    "procedures": [...]
})

# 4. 看一次演示就学会
chappie.watch_demonstration(input=demo_in, output=demo_out, importance=0.95)

# 5. 自我进化
chappie.self_heal()
chappie.daydream(steps=100)

# 6. 生成（带记忆检索）
output = chappie.generate_with_memory(prompt, max_new_tokens=50)
```

---

## 与 v0.1-baseline 的兼容性

CHAPPIE 是**纯扩展层**：
- 不修改任何现有 CORTEX 组件的接口
- 通过包装器（wrapper）添加能力
- 每个模块可独立启用/禁用
- 基础模型可以是随机初始化的（零预训练）

---

## 生物学依据

| CHAPPIE 机制 | 生物对应 | 神经科学文献 |
|-------------|---------|-------------|
| 知识编译 | 先天连接模式 | Innate knowledge (Spelke) |
| 赫布学习 | LTP/LTD | Bliss & Lømo (1973) |
| 想象自举 | 默认模式网络 | Raichle (2001) |
| 自修改 | 神经可塑性 + 元认知 | Flavell (1979) |
| 单次巩固 | 闪光记忆 | Brown & Kulik (1977) |
| 睡眠重放 | 海马-皮层重放 | Wilson & McNaughton (1994) |

---

## 运行 Demo

```bash
python examples/chappie_demo.py
```

包含 5 个场景：
1. **知识上传**：规则/事实/技能编译进权重
2. **单次学习**：一次演示，赫布可塑性
3. **自我修复**：诊断并修复模型问题
4. **闪光记忆**：重要性门控的即时巩固
5. **完整系统**：整合所有能力的端到端演示

---

*"You can't train consciousness. You can only create the conditions for it to emerge."*
