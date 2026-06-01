# AGI-CORTEX 扩展路线图 v1.0

**Status:** Design Phase  
**Baseline:** v0.1-baseline (tagged)  
**Goal:** 将 CORTEX 从序列处理器扩展为具备 AGI 雏形的认知架构

---

## 1. 总体架构愿景

### 1.1 核心设计理念

> **AGI-CORTEX 不是推翻现有架构，而是在保持 O(N) 复杂度和生物合理性的前提下，通过六层扩展逐步叠加上层认知功能。**

现有 CORTEX 的每个组件都有明确的"认知对应物"：
- **DCU** → 皮层锥体神经元
- **GNW** → 前额-顶叶工作空间
- **多时间尺度** → 不同频率的神经振荡
- **预测编码** → 自由能最小化

AGI 扩展遵循同样的原则：每个新增组件必须有认知神经科学的理论依据，且与现有组件正交可插拔。

### 1.2 扩展后的总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGI-CORTEX 系统全景                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT LAYER                    OUTPUT LAYER                                │
│  ┌──────────────┐              ┌─────────────────────────────┐             │
│  │ 感知 LPM     │              │ 认知输出 (token/logits)     │             │
│  │ (视觉/语言)  │              ├─────────────────────────────┤             │
│  ├──────────────┤              │ 行动输出 (action tokens)    │  ← 具身交互 │
│  │ 内感受 LPM   │──────────────┤ 工具调用 (tool calls)       │             │
│  │ (reward/     │              ├─────────────────────────────┤             │
│  │  confusion)  │              │ 自我报告 (certainty/needs)  │  ← 自我建模 │
│  ├──────────────┤              │ 符号产出 (logic expressions)│  ← 符号推理 │
│  │ 自我 LPM     │              └─────────────────────────────┘             │
│  │ (self_repr)  │                                                           │
│  └──────────────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    嵌套全局工作空间 (H-GNW)                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ 符号缓冲区   │  │ 连续缓冲区   │  │ 假设空间     │  ← 因果推断 │   │
│  │  │ Symbolic WS  │  │ Continuous WS│  │ Counterfact. │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ 目标栈 (Goal Stack)                                        │   │   │
│  │  │ [Parent: write_thesis] → [Child: write_ch1] → [Sub: para1] │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ├── DCU (Branch-Isolated: 任务分区租用)                          │   │
│         ├── DCU (SEM-Branch: 结构方程因果分支)                           │   │
│         ├── DCU (Symbolic Branch: 向量量化符号分支)                      │   │
│         ├── Multi-Timescale (自传记体封存 + 跨尺度耦合)                  │   │
│         └── Predictive Coding (环境前向模型 + 精度元估计)                │   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 记忆子系统                                                          │   │
│  │  ├── 工作记忆 (快尺度 τ=25ms)                                      │   │
│  │  ├── 情节记忆 (中尺度 τ=100ms, 事件封存)                           │   │
│  │  ├── 自传体记忆 (慢尺度 τ=500ms, 自我叙事)                         │   │
│  │  └── 记忆柜 (Memory Cabinet, 外部索引)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 与 v0.1-baseline 的兼容性保证

所有扩展遵循以下兼容性原则：

| 原则 | 说明 |
|------|------|
| **向后兼容** | v0.1-baseline 的 `CORTEXModel` 接口保持不变，新增能力通过可选参数启用 |
| **渐进启用** | 每个扩展都有 `use_xxx=False` 开关，可以独立开启/关闭 |
| **不破坏复杂度** | 单模块启用时保持 O(N) 时间复杂度，多模块组合时最坏 O(N log N) |
| **模块化替换** | 现有组件可逐步被扩展版本替换，无需整体重写 |

---

## 2. 六大扩展模块：统一接口规范

每个模块定义以下五个部分：
1. **职责声明** — 该模块解决什么 AGI 能力缺口
2. **核心抽象** — 新增的关键类/接口
3. **输入输出契约** — 数据类型、shape、语义
4. **与现有系统的集成点** — 钩入 CORTEX 的哪个位置
5. **数据流** — 信息如何在该模块内部及与外部模块间流动

---

## 2.1 模块 M1：符号推理 (Symbolic Reasoning)

### 2.1.1 职责声明
将 GNW 中传播的连续向量部分离散化为显式符号，使 CORTEX 能执行结构化逻辑操作（替换、否定、组合），而不只是统计模式匹配。

### 2.1.2 核心抽象

```python
# === 抽象接口，非实现 ===

class SymbolicToken:
    """符号原子，GNW 中传播的最小离散单位。"""
    token_id: int           # 在码本中的索引
    embedding: Tensor       # 对应的连续嵌入 (d_embed,)
    confidence: float       # 离散化的置信度 [0,1]
    source_module: int      # 来自哪个 LPM

class SymbolicWorkspace:
    """符号工作空间：符号缓冲区的抽象。"""
    
    def __init__(self, vocab_size: int, d_embed: int):
        """初始化符号码本。"""
        
    def quantize(self, continuous: Tensor) -> List[SymbolicToken]:
        """
        将连续向量离散化为符号。
        Input:  continuous (batch, seq_len, d_model)
        Output: List[SymbolicToken] 每个位置一个符号
        """
        
    def compose(self, tokens: List[SymbolicToken]) -> SymbolicExpression:
        """
        符号组合：将多个符号组合成结构化表达式。
        例如：[RED, APPLE] → Color(Object=APPLE, Value=RED)
        """
        
    def apply_rule(self, expr: SymbolicExpression, 
                   rule: RewriteRule) -> SymbolicExpression:
        """
        符号重写：应用逻辑规则。
        例如：A ∧ True → A
        """
        
    def broadcast(self, expr: SymbolicExpression) -> Tensor:
        """
        符号广播：将符号表达式解码回连续向量，供现有 LPM 接收。
        Output: (batch, seq_len, d_model)
        """
```

### 2.1.3 输入输出契约

| 接口 | 输入 | 输出 | 语义 |
|------|------|------|------|
| `quantize` | `(B, N, d)` 连续表示 | `List[SymbolicToken]` | 最近邻码本查找 |
| `compose` | `List[SymbolicToken]` | `SymbolicExpression` | 基于 attention 的符号组合 |
| `apply_rule` | `SymbolicExpression` + `RewriteRule` | `SymbolicExpression` | 模式匹配 + 替换 |
| `broadcast` | `SymbolicExpression` | `(B, N, d)` | 符号嵌入 + 残差拼接 |

### 2.1.4 与现有系统的集成点

```
现有 GNW 流程：
  LPM outputs (continuous) → softmax competition → W_t (continuous) → broadcast

扩展后 GNW 流程：
  LPM outputs (continuous) ─┬─→ softmax competition → W_continuous (continuous)
                            │
                            └─→ SymbolicWorkspace.quantize() ─┬─→ Symbolic WS Buffer
                                                                ├─→ compose()
                                                                ├─→ apply_rule()
                                                                └─→ broadcast() ──→
  
  W_combined = W_continuous + λ_sym * broadcast(symbolic_result)
```

**集成位置：** `GlobalWorkspaceLayer.forward()` 内部，在 `weighted aggregation` 之后、`ignition` 之前插入符号处理分支。

**DCU 集成：** 新增 `branch_type='symbolic'`，该分支内部执行 VQ-VAE 式的码本查找，输出离散的符号 token。

### 2.1.5 数据流

```
[感知输入] → [LPM 处理] → [连续向量 z] ─┬─→ [GNW 连续竞争]
                                        │
                                        └─→ [SymbolicBranch.quantize]
                                                ↓
                                            [SymbolicToken]
                                                ↓
                                            [SymbolicWorkspace.compose]
                                                ↓
                                            [SymbolicExpression]
                                                ↓
                                            [apply_rule / 逻辑运算]
                                                ↓
                                            [SymbolicWorkspace.broadcast]
                                                ↓
                                            [连续残差向量]
                                                ↓
                                            [与 W_continuous 加权融合]
                                                ↓
                                            [GNW 点火 & 广播]
```

### 2.1.6 依赖关系
- **无前置依赖** — 可独立实现
- **被 M4（层次化规划）依赖** — 目标栈的分解/组合需要符号操作
- **被 M6（因果推断）依赖** — 因果图需要符号节点和边

---

## 2.2 模块 M2：自我建模 (Self-Modeling)

### 2.2.1 职责声明
使 CORTEX 维持一个关于"自身状态"的显式表征，包括：当前目标、认知负荷、不确定度、情绪状态。这是元认知（metacognition）和主观体验的基础。

### 2.2.2 核心抽象

```python
# === 抽象接口 ===

class SelfState:
    """自我状态向量，模型对'我是谁、我在干什么'的表征。"""
    goal_embedding: Tensor        # (d_goal,) 当前目标嵌入
    certainty: float              # [0,1] 对当前预测的确定度
    cognitive_load: float         # [0,1] 认知负荷（脉冲率 + 工作空间活动）
    emotional_valence: float      # [-1,1] 情绪效价（基于预测误差和奖励）
    autobiographical_context: Tensor  # (d_auto,) 当前激活的自传体记忆片段

class SelfModule:
    """自我 LPM：永不关闭，只监控内部状态。"""
    
    def __init__(self, d_model: int, d_goal: int, d_auto: int):
        """
        输入不是外部数据，而是系统内部信号：
        - workspace_state: 当前工作空间状态
        - goal_stack: 当前目标栈
        - prediction_errors: 各层预测误差
        - spike_rates: 各层脉冲率
        - reward_signals: 外部/内部奖励
        """
        
    def forward(self, internal_signals: InternalSignals) -> Tuple[SelfState, float]:
        """
        处理内部信号，输出自我状态和显著性分数。
        显著性分数衡量"自我状态的变化有多重要"。
        Output: (SelfState, saliency)
        """

class AutobiographicalMemory:
    """自传体记忆：用慢时间尺度存储'我经历过的事件'。"""
    
    def __init__(self, d_event: int, capacity: int):
        """capacity: 最大存储事件数。"""
        
    def encode_event(self, consciousness: Tensor, 
                     action: Optional[Tensor],
                     outcome: float,      # 结果好坏 [-1,1]
                     timestamp: int) -> EventRecord:
        """
        将当前意识状态编码为事件记录。
        Input: 意识向量 + 行动 + 结果
        Output: EventRecord (可存入慢时间尺度)
        """
        
    def retrieve_similar(self, query: Tensor, 
                         k: int = 3) -> List[EventRecord]:
        """
        基于相似度检索历史事件。
        Input: 查询向量（当前意识状态）
        Output: k 个最相似的历史事件
        """
        
    def consolidate(self, workspace_replay: List[Tensor]):
        """
        记忆巩固：工作空间重播白天的内容，写入慢尺度。
        由 GNW 在'离线'阶段触发。
        """

class MetacognitiveMonitor:
    """元认知监控：评估模型自身的认知状态。"""
    
    def estimate_uncertainty(self, prediction: Tensor, 
                            precision: Tensor) -> float:
        """
        基于预测编码的 precision 估计不确定度。
        precision 高 → 不确定度低（我很确定）
        precision 低 → 不确定度高（我不确定）
        """
        
    def detect_confusion(self, error_history: List[float]) -> bool:
        """
        检测持续高预测误差 → "我很困惑"。
        返回 True 时，SelfModule 的显著性升高，请求 GNW 关注。
        """
        
    def request_information(self, gap: Tensor) -> InformationRequest:
        """
        信息缺口检测：当预测需要但缺失某类信息时，发出请求。
        例如："我需要更多关于 X 的上下文"。
        """
```

### 2.2.3 输入输出契约

| 接口 | 输入 | 输出 | 语义 |
|------|------|------|------|
| `SelfModule.forward` | `InternalSignals` (工作空间 + 误差 + 脉冲率 + 奖励) | `(SelfState, saliency)` | 更新自我表征 |
| `AutobiographicalMemory.encode_event` | `(consciousness, action, outcome, timestamp)` | `EventRecord` | 写入记忆 |
| `AutobiographicalMemory.retrieve_similar` | `query (d_model,)` | `List[EventRecord]` | 检索相似经历 |
| `MetacognitiveMonitor.estimate_uncertainty` | `(prediction, precision)` | `float [0,1]` | 输出"我有多确定" |

### 2.2.4 与现有系统的集成点

```
现有 GNW：
  [LPM_0: 视觉] ─┐
  [LPM_1: 语言] ─┼→ [softmax 竞争] → W_t → broadcast
  [LPM_2: ... ] ─┘

扩展后 GNW：
  [LPM_0: 视觉] ─┐
  [LPM_1: 语言] ─┤
  [LPM_2: ... ] ─┤
  [SelfModule] ──┼→ [softmax 竞争] → W_t → broadcast
                 │
                 │   SelfModule 的特殊性：
                 │   1. 它的输入不是外部 x，而是 internal_signals
                 │   2. 它永不关闭（always_active=True）
                 │   3. 它的输出参与竞争，但不参与 broadcast
                 │      （自我状态只影响注意力调制，不覆盖感知内容）
                 ↓
  [内部信号收集器]  ←── 从各层收集：
                      - workspace_state: GNW 的 W_t
                      - layer_errors: 预测编码误差
                      - spike_rates: DCU 脉冲率
                      - reward: 外部奖励信号
                      - goal_stack: 当前目标（来自 M4）
```

**多时间尺度集成：**
- **快尺度 (τ=25ms)**：当前自我状态（"我现在很确定"）
- **中尺度 (τ=100ms)**：短期情绪趋势（"过去几秒我一直在困惑"）
- **慢尺度 (τ=500ms)**：自传体记忆（"上次遇到类似情况，我做了 X"）

### 2.2.5 数据流

```
[外部输入] → [各 LPM 处理]
                 │
                 ├─→ [GNW 竞争]
                 │       ↓
                 │   [W_t 工作空间状态]
                 │       ↓
                 └─→ [内部信号收集器]
                         ↓
                    ┌─────────────┐
                    │ SelfModule  │
                    │ - 读取 W_t  │
                    │ - 读取误差  │
                    │ - 读取脉冲率│
                    │ - 读取奖励  │
                    └─────────────┘
                         ↓
                    [SelfState]
                         ↓
            ┌────────────┼────────────┐
            ↓            ↓            ↓
      [Autobiographical] [Metacognitive] [GNW 显著性]
      [Memory.retrieve]  [Monitor]       [参与竞争]
            ↓            ↓            ↓
      [历史相关事件]   [不确定度]    [自我状态进入意识]
            ↓            ↓            ↓
      [影响当前决策]   [影响行动]    [调制注意力]
```

### 2.2.6 新增损失函数

```python
L_self_consistency = ||self_t - self_{t-k}||²      # 自我连续性
L_self_narrative = -cos(self_t, autobiography_t)    # 自我叙事一致性
L_metacognitive = -E[precision * log(confidence)]   # 元认知校准（不确定度与精度匹配）
```

### 2.2.7 依赖关系
- **无前置依赖** — 可独立实现
- **被 M4（层次化规划）依赖** — 目标栈需要自我模块来跟踪当前意图
- **与 M1（符号推理）协同** — 自我状态可以用符号表达（"I am confused"）

---

## 2.3 模块 M3：具身交互 (Embodied Interaction)

### 2.3.1 职责声明
打破 CORTEX "只读"的限制，使模型能输出动作、改变环境、观察结果，形成感知-行动闭环。这是智能体（agent）而非模型（model）的分水岭。

### 2.3.2 核心抽象

```python
# === 抽象接口 ===

class ActionSpace:
    """动作空间定义：模型可以执行的所有动作。"""
    n_actions: int              # 离散动作数
    n_continuous: int           # 连续参数维度
    action_types: List[str]     # 动作语义标签

class ActionLPM:
    """行动 LPM：将 GNW 的意图转化为具体动作。"""
    
    def __init__(self, d_model: int, action_space: ActionSpace):
        """
        输入：来自 GNW 广播的意图向量
        输出：动作概率分布 + 连续参数
        """
        
    def forward(self, intention: Tensor) -> ActionDistribution:
        """
        Input: intention (batch, d_model) 来自工作空间的意图
        Output: ActionDistribution
          - discrete_logits (batch, n_actions)
          - continuous_params (batch, n_continuous)
          - saliency (batch,) 行动的迫切程度
        """

class ForwardModel:
    """前向模型：预测"执行动作后环境会变成什么样"。"""
    
    def __init__(self, d_state: int, d_action: int):
        """
        学习环境的转移动力学：s_{t+1} = f(s_t, a_t)
        """
        
    def predict_next_state(self, current_state: Tensor, 
                           action: Tensor) -> Tuple[Tensor, float]:
        """
        Input: (current_state, action)
        Output: (predicted_next_state, prediction_uncertainty)
        """
        
    def imagine_trajectory(self, initial_state: Tensor, 
                           action_sequence: List[Tensor],
                           horizon: int) -> List[Tensor]:
        """
        想象：在脑中模拟执行一系列动作后的结果。
        用于规划（M4）和因果推断（M6）。
        Output: List[predicted_states] 长度 = horizon
        """

class EnvironmentInterface:
    """环境接口：CORTEX 与外部世界的桥梁。"""
    
    def step(self, action: Action) -> Tuple[Tensor, float, bool, dict]:
        """
        执行一步交互：
        Input: Action
        Output: (new_observation, reward, done, info)
        与 OpenAI Gym 接口兼容。
        """
        
    def reset(self) -> Tensor:
        """重置环境。"""

class InteroceptionChannel:
    """内感受通道：将身体/系统状态作为感知输入。"""
    
    def __init__(self, d_intero: int):
        """
        内感受信号包括：
        - 预测误差累积（"困惑"）
        - 脉冲能耗（"疲劳"）
        - 外部奖励（"愉悦/痛苦"）
        - 目标进度（"成就感"）
        """
        
    def compute_signals(self, system_state: SystemState) -> Tensor:
        """
        将系统状态编码为内感受向量。
        Output: (batch, d_intero)
        """
```

### 2.3.3 输入输出契约

| 接口 | 输入 | 输出 | 语义 |
|------|------|------|------|
| `ActionLPM.forward` | `intention (B, d)` | `ActionDistribution` | 意图 → 动作 |
| `ForwardModel.predict_next_state` | `(state, action)` | `(next_state, uncertainty)` | 一步环境预测 |
| `ForwardModel.imagine_trajectory` | `(state, actions, H)` | `List[state]` | H 步想象 |
| `EnvironmentInterface.step` | `Action` | `(obs, reward, done, info)` | 真实环境交互 |
| `InteroceptionChannel.compute_signals` | `SystemState` | `(B, d_intero)` | 身体信号编码 |

### 2.3.4 与现有系统的集成点

```
现有 CORTEXModel 输出：
  x → [layers] → [GNW] → [output_head] → logits

扩展后 CORTEXModel 输出（多头）：
  x → [layers] → [GNW] ─┬─→ [cognitive_head] → logits (原有)
                        ├─→ [action_head] → ActionDistribution (新增)
                        └─→ [self_report_head] → (certainty, needs) (M2)

感知-行动闭环：
  [环境] ──obs──→ [CORTEX] ──action──→ [环境]
     ↑                                    │
     └────────── reward/result ───────────┘

内感受融合：
  [外部感知] ─┐
              ├─→ [LPMs] → [GNW]
  [内感受] ───┘
  
  内感受 LPM 与普通 LPM 的区别：
  - 输入不是 token，而是 interoception 向量
  - 它的显著性天然较高（疼痛/困惑自动触发全局警报）
```

**预测编码扩展：**
- 现有：`prediction = predictor(h_current)` → 预测下一层表示
- 扩展：`prediction = predictor(h_current, action)` → 预测执行 action 后的下一层表示
- 这使得预测编码从"被动预测"变为"主动想象"

### 2.3.5 数据流

```
┌─────────┐      obs       ┌──────────┐     action     ┌─────────┐
│ 环境    │───────────────→│ CORTEX   │───────────────→│ 环境    │
│ (World) │←───────────────│ (Agent)  │←───────────────│ (World) │
└─────────┘    reward      └──────────┘    result      └─────────┘
                               │
                               │ 内部数据流
                               ↓
                    ┌─────────────────────┐
                    │ GNW 广播的意图向量  │
                    └─────────────────────┘
                               │
                    ┌──────────┼──────────┐
                    ↓          ↓          ↓
              [认知头]   [行动头]   [自我报告头]
                    ↓          ↓          ↓
              [logits]  [ActionDist]  [不确定度]
                    │          │          │
                    │    ┌─────┘          │
                    │    ↓                 │
                    │ [环境.step(action)]  │
                    │    ↓                 │
                    │ [new_obs + reward]   │
                    │    │                 │
                    └────┼─────────────────┘
                         ↓
                    [ForwardModel]
                    predict_next_state
                         ↓
                    [预测下一状态]
                    （与真实结果比较 → 误差 → 学习）
```

### 2.3.6 依赖关系
- **被 M4（层次化规划）依赖** — 规划需要执行动作，行动头是基础
- **被 M6（因果推断）依赖** — 因果推断需要干预环境，行动头是干预手段
- **与 M2（自我建模）协同** — 内感受信号直接输入 SelfModule

---

## 2.4 模块 M4：层次化规划 (Hierarchical Planning)

### 2.4.1 职责声明
将扁平的 GNW 扩展为嵌套结构，支持目标的层次化分解（写论文 → 第一章 → 第一节 → 一段）。这是复杂任务解决和长程规划的基础。

### 2.4.2 核心抽象

```python
# === 抽象接口 ===

class Goal:
    """目标节点。"""
    goal_id: str
    description: str                    # 目标描述（可用符号表达）
    embedding: Tensor                   # (d_goal,) 目标嵌入
    parent: Optional[Goal]              # 父目标
    children: List[Goal]                # 子目标
    status: Literal['pending', 'active', 'completed', 'failed']
    priority: float                     # [0,1] 动态优先级
    deadline: Optional[int]             # 时间步截止

class GoalStack:
    """目标栈：用慢时间尺度状态模拟调用栈。"""
    
    def __init__(self, max_depth: int = 5):
        """max_depth: 最大嵌套深度。"""
        
    def push(self, goal: Goal) -> bool:
        """
        压入新目标。
        Return: True if success, False if stack overflow.
        触发条件：当前工作空间检测到"目标复杂，需要分解"。
        """
        
    def pop(self) -> Optional[Goal]:
        """
        弹出完成的目标。
        触发条件：子工作空间返回"任务完成"。
        """
        
    def peek(self) -> Goal:
        """查看栈顶目标（当前焦点）。"""
        
    def decompose(self, goal: Goal, 
                  strategy: DecompositionStrategy) -> List[Goal]:
        """
        目标分解：将复杂目标拆分为子目标。
        Input: Goal + DecompositionStrategy
        Output: List[SubGoal]
        例如："写论文" → ["写摘要", "写引言", "写方法", "写结果", "写讨论"]
        """

class HierarchicalWorkspace:
    """嵌套工作空间：工作空间可以包含子工作空间。"""
    
    def __init__(self, d_model: int, n_modules: int, 
                 max_depth: int = 3):
        """
        max_depth: 最大嵌套层级。
        """
        
    def create_child_workspace(self, 
                               goal: Goal,
                               parent_workspace: WorkspaceNode) -> WorkspaceNode:
        """
        为子目标创建子工作空间。
        子工作空间有自己的 LPM 集合（可继承父 LPM，也可专用）。
        """
        
    def close_child_workspace(self, 
                              child: WorkspaceNode) -> Tensor:
        """
        关闭子工作空间，返回结果摘要。
        结果摘要进入父工作空间的广播。
        """
        
    def escalate(self, child: WorkspaceNode, 
                 problem: ProblemDescription):
        """
        子工作空间遇到无法解决的问题，上报父工作空间。
        例如：子目标"搜索文献"失败 → 上报 → 父目标调整策略。
        """

class SubroutineLPM:
    """子程序 LPM：编码可复用的技能。"""
    
    def __init__(self, d_model: int, 
                 subroutine_name: str,
                 skill_embedding: Tensor):
        """
        每个子程序 LPM 是一个"可调用函数"。
        例如：SortLPM, SearchLPM, CompareLPM, SummarizeLPM
        """
        
    def forward(self, input_state: Tensor, 
                context: WorkspaceContext) -> Tensor:
        """
        执行子程序。
        Input: 输入状态 + 上下文
        Output: 执行结果
        子程序有自己的局部工作空间，执行时不干扰主工作空间。
        """
        
    def estimate_cost(self, input_state: Tensor) -> float:
        """
        估计执行该子程序的计算成本。
        用于工作空间的成本-效益竞争。
        """
```

### 2.4.3 输入输出契约

| 接口 | 输入 | 输出 | 语义 |
|------|------|------|------|
| `GoalStack.push` | `Goal` | `bool` | 压栈 |
| `GoalStack.decompose` | `Goal + Strategy` | `List[Goal]` | 目标分解 |
| `HierarchicalWorkspace.create_child_workspace` | `Goal + Parent` | `WorkspaceNode` | 创建子空间 |
| `HierarchicalWorkspace.close_child_workspace` | `Child` | `Tensor` | 关闭并返回结果 |
| `SubroutineLPM.forward` | `(state, context)` | `Tensor` | 执行技能 |
| `SubroutineLPM.estimate_cost` | `state` | `float` | 成本估计 |

### 2.4.4 与现有系统的集成点

```
现有 GlobalWorkspaceLayer：
  单一工作空间，所有 LPM 竞争同一个广播信道

扩展后 HierarchicalWorkspace：
  WorkspaceNode (root) ──┬─→ [LPM_0, LPM_1, ..., LPM_n]
                         │     ↑ 竞争
                         │     ↓ 广播
                         │   [WorkspaceState_root]
                         │
                         ├──→ WorkspaceNode (child_1: "写论文")
                         │      ├──→ [专用 LPM + 继承 LPM]
                         │      └──→ 竞争 → 广播 → 子结果
                         │
                         ├──→ WorkspaceNode (child_2: "读文献")
                         │      └──→ ...
                         │
                         └──→ WorkspaceNode (child_3: "做实验")
                                └──→ ...

栈状态存储：
  [快尺度 τ=25ms]  → 当前工作空间活动
  [中尺度 τ=100ms] → 当前层级上下文
  [慢尺度 τ=500ms] → 完整目标栈 + 历史目标

目标分解触发条件：
  当 GNW 的 ignition 触发后，"目标分解 LPM"评估当前目标复杂度。
  如果复杂度 > 阈值 → 调用 GoalStack.decompose() → 创建子工作空间。
```

### 2.4.5 数据流

```
[顶层输入] → [Root Workspace]
                 ↓
         [ignition: "写论文"]
                 ↓
         [GoalStack.decompose]
                 ↓
         [Goal("写论文")]
            ├──→ Child WS("写摘要")
            │       ↓
            │   [执行] → [结果] → [返回 Root]
            │
            ├──→ Child WS("写方法")
            │       ↓
            │   [执行中...]
            │   [遇到子问题：需要引用某论文]
            │       ↓
            │   [escalate 到父 WS]
            │       ↓
            │   [父 WS 调整：新增子目标"下载论文"]
            │       ↓
            │   [创建新 Child WS("下载论文")]
            │       ↓
            │   [完成后返回] → [继续"写方法"]
            │
            └──→ Child WS("写结果")
                    ↓
                [等待"做实验"完成]
                    ↓
                [收到完成信号] → [继续执行]
```

### 2.4.6 依赖关系
- **依赖 M1（符号推理）** — 目标描述、分解策略需要符号表达
- **依赖 M2（自我建模）** — 目标栈需要自我模块来跟踪当前意图和进度
- **依赖 M3（具身交互）** — 子目标需要执行动作
- **被 M5（持续学习）依赖** — 子程序 LPM 需要持续学习新技能

---

## 2.5 模块 M5：持续学习 (Continual Learning)

### 2.5.1 职责声明
使 CORTEX 能按顺序学习多个任务而不遗忘旧知识。利用树突分支的结构特性，实现"神经分区租用"——不同任务使用不同的分支子集。

### 2.5.2 核心抽象

```python
# === 抽象接口 ===

class BranchMask:
    """分支掩码：标记哪些分支参与当前任务。"""
    task_id: str
    mask: Tensor                # (n_branches,) bool
    usage_count: int            # 该掩码被使用的次数
    performance_history: List[float]  # 任务历史性能

class BranchAllocator:
    """分支分配器：管理 DCU 分支的租用。"""
    
    def __init__(self, n_branches: int, n_tasks_max: int):
        """
        n_tasks_max: 最大并发任务数。
        """
        
    def allocate(self, task_id: str, 
                 required_branches: int = 2) -> BranchMask:
        """
        为新任务分配空闲分支。
        策略：选择使用率最低的分支组合。
        Output: BranchMask
        """
        
    def release(self, task_id: str):
        """释放任务占用的分支（软释放：降低学习率而非冻结）。"""
        
    def update_usage(self, task_id: str, performance: float):
        """
        更新任务性能记录。
        性能下降 → 触发"重训练"或"分支重新分配"。
        """

class ElasticPlasticity:
    """弹性可塑性：动态调整各分支的学习率。"""
    
    def __init__(self):
        """
        每个分支的可塑性系数：[0, 1]
        0 = 完全冻结（保护旧知识）
        1 = 完全可塑（学习新知识）
        """
        
    def compute_plasticity(self, branch_id: int, 
                          usage_stats: UsageStats) -> float:
        """
        计算分支可塑性：
        - 常用分支 → 低可塑性（保护）
        - 空闲分支 → 高可塑性（学习）
        - 性能下降分支 → 临时提高可塑性（修复）
        """
        
    def consolidate_signal(self, workspace_state: Tensor) -> Tensor:
        """
        由 GNW 的 ignition 触发的巩固信号。
        高显著性信息 → 广播到慢时间尺度 → 降低相关分支的可塑性。
        """

class MemoryCabinet:
    """记忆柜：显式封存和检索长期记忆。"""
    
    def __init__(self, d_memory: int, capacity: int):
        """
        外部记忆库，可类比为"海马体索引"。
        """
        
    def archive(self, slow_state: Tensor, 
                context_tag: str, 
                importance: float) -> MemoryKey:
        """
        封存慢时间尺度状态。
        Input: 慢状态 + 上下文标签 + 重要性
        Output: MemoryKey（检索用的键）
        """
        
    def retrieve(self, query: Tensor, 
                 context_hint: Optional[str] = None,
                 k: int = 3) -> List[MemoryRecord]:
        """
        基于相似度 + 上下文标签检索记忆。
        """
        
    def replay(self, task_id: str, 
               n_samples: int) -> List[Tensor]:
        """
        重播：为特定任务检索历史样本，用于防止遗忘。
        在"睡眠/离线"阶段触发。
        """

class OfflineConsolidation:
    """离线巩固：模拟睡眠中的记忆重放。"""
    
    def __init__(self, model: CORTEXModel, 
                 memory_cabinet: MemoryCabinet):
        """"""
        
    def sleep_phase(self, duration_steps: int):
        """
        睡眠阶段：
        1. 关闭外部输入
        2. 从记忆柜随机采样历史事件
        3. 通过工作空间重播
        4. 慢时间尺度写入
        5. 相关分支可塑性降低（巩固）
        """
        
    def wake_phase(self):
        """唤醒：恢复正常操作。"""
```

### 2.5.3 输入输出契约

| 接口 | 输入 | 输出 | 语义 |
|------|------|------|------|
| `BranchAllocator.allocate` | `task_id, required_branches` | `BranchMask` | 为新任务分配分支 |
| `ElasticPlasticity.compute_plasticity` | `branch_id, usage_stats` | `float [0,1]` | 计算可塑性 |
| `MemoryCabinet.archive` | `(slow_state, tag, importance)` | `MemoryKey` | 封存记忆 |
| `MemoryCabinet.replay` | `task_id, n_samples` | `List[Tensor]` | 重播防遗忘 |
| `OfflineConsolidation.sleep_phase` | `duration_steps` | `None` | 离线巩固 |

### 2.5.4 与现有系统的集成点

```
现有 DCU：
  all_branches = [B0, B1, B2, B3]
  weights = softmax(all_branch_weights)
  output = sum(w_i * branch_i)

扩展后 DCU（带分支隔离）：
  mask = BranchAllocator.get_mask(current_task)  # [1, 1, 0, 0]
  effective_branches = [B0, B1]  # B2, B3 被隔离
  weights = softmax(mask * branch_weights)  # B2, B3 权重归零
  output = sum(w_i * branch_i)
  
  梯度传播：
  - B0, B1: 正常反向传播
  - B2, B3: 梯度被 mask 阻断（或乘以一个极小的 plasticity 系数）

预测编码集成：
  预测误差不仅驱动当前任务学习，还触发 MemoryCabinet.archive()：
  - 高误差事件 → "印象深刻" → 高重要性 → 封存
  - 低误差事件 → "意料之中" → 低重要性 → 可能不封存

GNW 巩固信号：
  当 ignition 触发时，GNW 同时发送 ConsolidationSignal：
  - 目标：慢时间尺度 + 被激活的分支
  - 效果：降低这些分支的可塑性（"这个很重要，保护好"）
```

### 2.5.5 数据流

```
[新任务 A 到来]
       ↓
[BranchAllocator.allocate("task_A", 2)]
       ↓
[BranchMask_A = [1,1,0,0]]  ← 分配 B0, B1 给任务 A
       ↓
[训练任务 A]
   ├──→ 前向：只用 B0, B1
   ├──→ 反向：梯度只更新 B0, B1
   └──→ B2, B3 被保护
       ↓
[任务 A 完成]
       ↓
[BranchAllocator.update_usage("task_A", performance)]
       ↓
[新任务 B 到来]
       ↓
[BranchAllocator.allocate("task_B", 2)]
       ↓
[BranchMask_B = [0,0,1,1]]  ← 分配 B2, B3 给任务 B
       ↓
[训练任务 B]
   ├──→ 前向：只用 B2, B3
   ├──→ 反向：梯度只更新 B2, B3
   └──→ B0, B1 被保护（任务 A 的知识保留）
       ↓
[任务 B 中遇到与 A 相关的输入]
       ↓
[MemoryCabinet.retrieve(query, context_hint="task_A")]
       ↓
[检索到任务 A 的相关记忆]
       ↓
[记忆被注入慢时间尺度]
       ↓
[同时激活 B0, B1 和 B2, B3]
       ↓
[知识迁移：任务 A 的知识帮助任务 B]
       ↓
[睡眠阶段]
       ↓
[OfflineConsolidation.sleep_phase()]
   ├──→ 关闭外部输入
   ├──→ 重播高重要性记忆
   ├──→ 慢时间尺度写入
   └──→ 相关分支可塑性永久降低
```

### 2.5.6 依赖关系
- **依赖 M1（符号推理）** — 分支掩码可以符号化表达（"哪些分支负责语言任务"）
- **依赖 M4（层次化规划）** — 不同子目标可以租用不同分支
- **被 M6（因果推断）依赖** — 因果学习需要持续积累干预经验

---

## 2.6 模块 M6：因果推断 (Causal Inference)

### 2.6.1 职责声明
将预测编码从"相关性预测"升级为"因果模拟"，使 CORTEX 能回答反事实问题（"如果我当时做了 X，结果会怎样？"）。

### 2.6.2 核心抽象

```python
# === 抽象接口 ===

class CausalVariable:
    """因果变量：符号化的可干预节点。"""
    var_id: str
    embedding: Tensor                   # 连续嵌入
    possible_values: List[SymbolicToken]  # 离散可能值
    parents: List[str]                  # 父节点 ID
    children: List[str]                 # 子节点 ID

class CausalGraph:
    """因果图：有向无环图（DAG）的显式表示。"""
    
    def __init__(self):
        self.nodes: Dict[str, CausalVariable] = {}
        self.edges: Dict[Tuple[str, str], float] = {}  # 因果强度
        
    def add_edge(self, cause: str, effect: str, 
                 strength: float):
        """添加因果边。"""
        
    def remove_edge(self, cause: str, effect: str):
        """删除因果边（反事实：假设这个因果关系不存在）。"""
        
    def do(self, var_id: str, value: SymbolicToken) -> 'CausalGraph':
        """
        Pearl's do-operator: 干预。
        将变量 var_id 设为固定值 value，切断其所有父节点的影响。
        Output: 干预后的因果图（用于模拟）。
        """
        
    def query(self, intervention: Dict[str, SymbolicToken],
              outcome_var: str) -> Distribution:
        """
        因果查询：P(outcome | do(intervention))
        在干预后的图上进行前向传播，预测结果分布。
        """

class StructuralEquationBranch:
    """结构方程分支：DCU 的新分支类型，显式建模因果关系。"""
    
    def __init__(self, d_in: int, d_out: int):
        """
        标准分支：h = σ(Wx + b)  (相关性)
        结构方程分支：h = f(parents) + g(noise)  (因果)
        其中 f 是父节点的确定性函数，g 是外生噪声。
        """
        
    def forward(self, parent_values: Tensor, 
                noise: Tensor) -> Tensor:
        """
        结构方程：子节点 = 父节点函数 + 噪声
        """
        
    def intervene(self, forced_value: Tensor) -> Tensor:
        """
        干预：忽略父节点，直接设定值。
        这是 do-operator 在 DCU 级别的实现。
        """

class CounterfactualWorkspace:
    """反事实工作空间：并行运行多个"假设世界"。"""
    
    def __init__(self, base_workspace: GlobalWorkspaceLayer,
                 n_counterfactuals: int = 2):
        """
        创建 n 个副本工作空间。
        副本共享基础 LPM，但接受不同的干预设定。
        """
        
    def run_counterfactual(self, 
                           intervention: Dict[str, SymbolicToken],
                           base_state: Tensor) -> Tensor:
        """
        在假设世界中运行一步。
        Input: 干预设定 + 基础状态
        Output: 假设世界的演化结果
        """
        
    def compare(self, factual_result: Tensor,
                counterfactual_results: List[Tensor]) -> CausalEffect:
        """
        比较事实结果与反事实结果，估计因果效应。
        Effect = E[outcome | do(X=x)] - E[outcome | do(X=x')]
        """

class CausalDiscoveryLPM:
    """因果发现 LPM：从时间序列中自动发现因果关系。"""
    
    def __init__(self, d_model: int, max_lag: int = 5):
        """
        max_lag: 考虑的最大时间延迟。
        """
        
    def forward(self, history: List[Tensor]) -> CausalGraph:
        """
        输入：变量的时间序列历史
        输出：发现的因果图
        使用格兰杰因果检验 + 注意力机制发现时间上的因果依赖。
        """
        
    def estimate_causal_strength(self, cause: Tensor, 
                                 effect: Tensor) -> float:
        """估计特定因果对的强度。"""
```

### 2.6.3 输入输出契约

| 接口 | 输入 | 输出 | 语义 |
|------|------|------|------|
| `CausalGraph.do` | `(var_id, value)` | `CausalGraph` | 干预（切断父节点） |
| `CausalGraph.query` | `(intervention, outcome_var)` | `Distribution` | 因果查询 |
| `StructuralEquationBranch.intervene` | `forced_value` | `Tensor` | 节点级干预 |
| `CounterfactualWorkspace.run_counterfactual` | `(intervention, state)` | `Tensor` | 反事实模拟 |
| `CounterfactualWorkspace.compare` | `(factual, counterfactuals)` | `CausalEffect` | 效应估计 |
| `CausalDiscoveryLPM.forward` | `List[history]` | `CausalGraph` | 自动发现因果 |

### 2.6.4 与现有系统的集成点

```
现有预测编码：
  prediction = predictor(h_current)  → 预测 h_next
  error = h_next - prediction

扩展后因果预测编码：
  # 事实世界（观察）
  prediction_factual = predictor(h_current, action=None)
  
  # 反事实世界（假设我做了 action X）
  prediction_counter = predictor(
      h_current, 
      action=intervention,
      graph=causal_graph.do(intervention)
  )
  
  # 因果效应 = 事实 - 反事实
  causal_effect = prediction_counter - prediction_factual
  
  # 学习：不仅最小化预测误差，还最小化因果效应估计的方差

DCU 集成：
  新增 branch_type='structural_equation'
  
  标准分支输出：h = σ(Wx + b)
  结构方程分支输出：h = f(parent_selection) + ε
    - parent_selection: 从输入中选择哪些维度是"父节点"
    - f: 父节点的确定性组合
    - ε: 外生噪声（可学习的噪声模型）
  
  干预时：h = forced_value（忽略 f(parents)）

GNW 集成：
  CounterfactualWorkspace 包含多个 GNW 副本：
  - W_real: 观察世界（所有变量自由演化）
  - W_cf1: 假设世界 1（变量 X = a）
  - W_cf2: 假设世界 2（变量 X = b）
  
  副本共享基础 LPM（观察结果相同），
  但各自接受不同的干预设定。
  
  比较器从所有副本收集结果，计算因果效应。
```

### 2.6.5 数据流

```
[观察数据] → [CausalDiscoveryLPM]
                 ↓
            [CausalGraph]
                 ↓
            [do(X = a)] ─┬─→ [CounterfactualWorkspace_1]
            [do(X = b)] ─┼─→ [CounterfactualWorkspace_2]
            [无干预] ─────┘→ [FactualWorkspace]
                 ↓
            [并行运行 H 步]
                 ↓
            [收集结果]
                 ↓
            [CounterfactualWorkspace.compare]
                 ↓
            [因果效应估计]
                 ↓
            [更新 CausalGraph 的边权重]
                 ↓
            [用于未来决策]

反事实推理用于决策：
  [当前状态] → [ForwardModel.imagine]
                 ↓
            [生成候选 action 序列]
                 ↓
            [对每个候选：]
              ├──→ run_counterfactual(action=候选)
              ├──→ 预测 outcome
              └──→ 计算 expected reward
                 ↓
            [选择最优 action]
```

### 2.6.6 依赖关系
- **依赖 M1（符号推理）** — 因果变量是符号化的，需要 SymbolicToken
- **依赖 M3（具身交互）** — 因果推断需要干预环境，行动是干预手段
- **依赖 M4（层次化规划）** — 复杂因果推理需要层次化子目标
- **依赖 M5（持续学习）** — 因果图需要持续积累干预经验来修正

---

## 3. 模块间依赖关系总图

```
                    ┌─────────────────────────────────────┐
                    │           AGI-CORTEX 总图            │
                    └─────────────────────────────────────┘

        M1 符号推理          M2 自我建模           M3 具身交互
        ┌─────────┐          ┌─────────┐          ┌─────────┐
        │ Symbolic│          │ Self    │          │ Action  │
        │ WS      │          │ Module  │          │ Head    │
        │ VQ-Code │          │ AutoBio │          │ Forward │
        │ books   │          │ Memory  │          │ Model   │
        └───┬─────┘          └───┬─────┘          └───┬─────┘
            │                    │                    │
            ├────────────────────┼────────────────────┤
            │     提供符号目标   │  提供意图向量    │  提供行动能力
            ↓                    ↓                    ↓
        ┌─────────────────────────────────────────────────┐
        │              M4 层次化规划                       │
        │         Hierarchical Workspace                  │
        │         Goal Stack + Subroutine LPM             │
        └────────────────────────┬────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ↓                    ↓                    ↓
        ┌─────────┐          ┌─────────┐          ┌─────────┐
        │ M5 持续 │          │ M6 因果 │          │ 现有    │
        │ 学习    │          │ 推断    │          │ CORTEX  │
        │ Branch  │          │ Causal  │          │ Core    │
        │ Isolation│         │ Graph   │          │ DCU/SSM │
        │ Memory  │          │ Counter │          │ GNW/PC  │
        │ Cabinet │          │ factual │          │ MT/Spike│
        └─────────┘          └─────────┘          └─────────┘

依赖关系：
  M1 ──→ M4, M5, M6       (符号推理是上层模块的基础)
  M2 ──→ M4               (自我建模提供目标跟踪)
  M3 ──→ M4, M6           (具身交互提供行动和干预能力)
  M4 ──→ M5, M6           (层次规划需要持续学习和因果推断)
  M5 ──→ M6               (因果图需要持续积累干预经验)
```

---

## 4. 实施路线图

### Phase 1：基础扩展（Week 1-2）
目标：实现 M1 + M2，验证与现有系统的兼容性。

| 任务 | 优先级 | 文件 |
|------|--------|------|
| 实现 `SymbolicBranch` + `SymbolicWorkspace` | P0 | `cortex/symbolic.py` |
| 扩展 `GlobalWorkspaceLayer` 支持符号缓冲区 | P0 | `cortex/workspace.py` |
| 实现 `SelfModule` + `AutobiographicalMemory` | P0 | `cortex/self_modeling.py` |
| 扩展 `CORTEXModel` 支持 `return_symbolic` | P0 | `cortex/cortex_model.py` |
| 添加 `L_self_consistency` 损失 | P1 | `cortex/cortex_model.py` |
| 单元测试：符号量化/反量化 | P0 | `tests/test_symbolic.py` |
| 单元测试：自我状态连续性 | P0 | `tests/test_self.py` |

### Phase 2：交互能力（Week 3-4）
目标：实现 M3，让 CORTEX 成为 Agent。

| 任务 | 优先级 | 文件 |
|------|--------|------|
| 实现 `ActionLPM` + `ActionDistribution` | P0 | `cortex/action.py` |
| 实现 `ForwardModel` | P0 | `cortex/forward_model.py` |
| 实现 `InteroceptionChannel` | P0 | `cortex/interoception.py` |
| 扩展 `CORTEXModel` 支持 `action_head` | P0 | `cortex/cortex_model.py` |
| 预测编码扩展：`predictor(h, action)` | P0 | `cortex/predictive_coding.py` |
| 环境接口示例：Gym 兼容 | P1 | `examples/embodied_agent.py` |
| 单元测试：前向模型预测精度 | P0 | `tests/test_forward_model.py` |

### Phase 3：认知控制（Week 5-6）
目标：实现 M4，支持复杂任务规划。

| 任务 | 优先级 | 文件 |
|------|--------|------|
| 实现 `Goal` + `GoalStack` | P0 | `cortex/goal.py` |
| 实现 `HierarchicalWorkspace` | P0 | `cortex/hierarchical_workspace.py` |
| 实现 `SubroutineLPM` | P1 | `cortex/subroutine.py` |
| 扩展多时间尺度：栈状态存储 | P0 | `cortex/multiscale_state.py` |
| 集成：M1 符号 + M4 目标分解 | P0 | `cortex/hierarchical_workspace.py` |
| 示例：层次化任务解决 | P1 | `examples/hierarchical_task.py` |

### Phase 4：终身学习（Week 7-8）
目标：实现 M5，解决灾难性遗忘。

| 任务 | 优先级 | 文件 |
|------|--------|------|
| 实现 `BranchMask` + `BranchAllocator` | P0 | `cortex/branch_isolation.py` |
| 扩展 DCU：`branch_type='isolated'` | P0 | `cortex/dendritic.py` |
| 实现 `ElasticPlasticity` | P0 | `cortex/plasticity.py` |
| 实现 `MemoryCabinet` | P0 | `cortex/memory_cabinet.py` |
| 实现 `OfflineConsolidation` | P1 | `cortex/consolidation.py` |
| 示例：多任务持续学习 | P1 | `examples/continual_learning.py` |

### Phase 5：因果智能（Week 9-10）
目标：实现 M6，支持反事实推理。

| 任务 | 优先级 | 文件 |
|------|--------|------|
| 实现 `CausalGraph` + `do-operator` | P0 | `cortex/causal.py` |
| 实现 `StructuralEquationBranch` | P0 | `cortex/dendritic.py` |
| 实现 `CounterfactualWorkspace` | P0 | `cortex/counterfactual.py` |
| 实现 `CausalDiscoveryLPM` | P1 | `cortex/causal_discovery.py` |
| 因果预测编码损失 | P1 | `cortex/predictive_coding.py` |
| 示例：因果决策 | P1 | `examples/causal_reasoning.py` |

### Phase 6：整合与基准测试（Week 11-12）
目标：全部模块联调，发布 v1.0-ag。

| 任务 | 优先级 |
|------|--------|
| 端到端集成测试 | P0 |
| 长程任务基准（BabyAI / ALFWorld） | P1 |
| 持续学习基准（Split-CIFAR / Permuted MNIST） | P1 |
| 因果推断基准（CausalWorld / Bandit） | P1 |
| 性能回归测试（确保未破坏 O(N) 复杂度） | P0 |
| 文档更新 + 教程 | P1 |

---

## 5. 统一数据协议

为确保模块间通信一致，定义以下统一数据格式：

### 5.1 WorkspacePacket（工作空间数据包）

```python
class WorkspacePacket:
    """GNW 中传播的统一数据结构。"""
    
    # 连续内容
    continuous: Tensor              # (batch, d_model)
    
    # 符号内容（M1）
    symbolic: Optional[List[SymbolicToken]] = None
    symbolic_expression: Optional[SymbolicExpression] = None
    
    # 自我内容（M2）
    self_state: Optional[SelfState] = None
    autobiographical_hint: Optional[Tensor] = None
    
    # 行动内容（M3）
    action_distribution: Optional[ActionDistribution] = None
    imagined_outcome: Optional[Tensor] = None
    
    # 规划内容（M4）
    goal_context: Optional[Goal] = None
    subroutine_call: Optional[str] = None
    
    # 记忆内容（M5）
    memory_key: Optional[MemoryKey] = None
    consolidation_request: bool = False
    
    # 因果内容（M6）
    causal_graph: Optional[CausalGraph] = None
    intervention: Optional[Dict[str, SymbolicToken]] = None
    counterfactual_tag: Optional[str] = None
    
    # 元数据
    timestamp: int
    source_module: str
    saliency: float
    ignition: bool
```

### 5.2 InternalSignals（内部信号）

```python
class InternalSignals:
    """SelfModule 接收的内部监控信号。"""
    
    workspace_state: Tensor         # (d_model,) 当前工作空间状态
    goal_stack: List[Goal]          # 当前目标栈
    prediction_errors: Dict[str, float]  # 各层预测误差
    spike_rates: Dict[str, float]   # 各层脉冲率
    reward: float                   # 外部奖励
    energy_consumption: float       # 计算能耗估计
    memory_retrieval_hits: int      # 记忆检索次数
    branch_usage: Dict[str, int]    # 分支使用情况
```

### 5.3 Action（统一动作）

```python
class Action:
    """统一动作表示，兼容离散和连续动作。"""
    
    discrete_action: Optional[int] = None           # 离散动作 ID
    continuous_params: Optional[Tensor] = None      # 连续参数
    action_type: str                                # 动作语义标签
    confidence: float = 1.0                         # 动作置信度
    expected_outcome: Optional[Tensor] = None       # 预期结果（前向模型）
    causal_effect_estimate: Optional[float] = None  # 估计因果效应（M6）
```

---

## 6. 版本管理策略

### 6.1 分支策略

```
main
├── v0.1-baseline (tag)
│
└── agi-dev
    ├── feature/M1-symbolic
    ├── feature/M2-self-modeling
    ├── feature/M3-embodied
    ├── feature/M4-hierarchical
    ├── feature/M5-continual
    ├── feature/M6-causal
    │
    └── integration (每两周合并一次 feature 分支)
```

### 6.2 发布里程碑

| 版本 | 内容 | 时间 |
|------|------|------|
| v0.2-alpha | M1 + M2 | Week 2 |
| v0.3-alpha | + M3 | Week 4 |
| v0.4-beta | + M4 | Week 6 |
| v0.5-beta | + M5 | Week 8 |
| v0.9-rc | + M6 | Week 10 |
| v1.0-ag | 全部集成 + 基准测试 | Week 12 |

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| 符号-连续耦合导致梯度断裂 | 训练失败 | 使用直通估计 + 残差连接 |
| 层次化规划导致栈溢出 | 推理崩溃 | 最大深度限制 + 循环检测 |
| 分支隔离导致表达力下降 | 性能退化 | 动态分支复用 + 迁移学习 |
| 反事实工作空间内存爆炸 | OOM | 限制并行副本数 + 共享基础 LPM |
| 元认知循环导致不稳定 | 训练发散 | 自我状态动量平滑 + 梯度裁剪 |
| 六模块组合复杂度失控 | 无法维护 | 每模块独立测试 + 渐进集成 |

---

## 8. 结论

本文档定义了从 CORTEX v0.1-baseline 到 AGI-CORTEX v1.0 的完整路线图。六个模块不是相互独立的插件，而是围绕**嵌套工作空间**这一核心逐渐生长的认知层次：

1. **符号推理**给工作空间装上语法
2. **自我建模**给工作空间装上"自我"视角
3. **具身交互**让工作空间能改变世界
4. **层次化规划**让工作空间能自我分解
5. **持续学习**让工作空间能终身成长
6. **因果推断**让工作空间能理解"为什么"

最终目标是让 CORTEX 从**被动预测器**进化为**主动认知者**——一个能思考、能行动、能学习、能反思的系统。

---

*Document Version: 1.0*  
*Last Updated: 2026-05-30*  
*Author: AGI-CORTEX Architecture Team*
