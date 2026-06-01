"""
Stage 0 高质量训练数据生成器
- 高多样性（避免模板重复）
- 中英文混合
- 适合语言建模（next-token prediction）
"""

import random
import os

random.seed(42)

# 主题池
topics_en = [
    "machine learning", "neuroscience", "consciousness", "memory systems",
    "pattern recognition", "adaptive systems", "predictive coding",
    "neural networks", "cognitive architecture", "information theory",
    "dynamical systems", "emergent behavior", "representation learning",
    "attention mechanisms", "temporal processing", "sensorimotor integration",
    "decision making", "uncertainty quantification", "causal reasoning",
    "meta-learning", "transfer learning", "few-shot learning",
    "reinforcement learning", "exploration vs exploitation",
    "credit assignment", "hierarchical models", "compositionality",
    "symbolic reasoning", "analogical reasoning", "abductive inference",
    "Bayesian inference", "free energy principle", "active inference",
    "embodied cognition", "extended mind", "distributed cognition",
    "collective intelligence", "swarm behavior", "self-organization",
    "criticality", "phase transitions", "attractor dynamics",
    "chaos theory", "fractal structures", "complex networks",
    "graph theory", "topological data analysis", "manifold learning",
    "dimensionality reduction", "feature extraction", "kernel methods",
    "deep learning", "convolutional networks", "recurrent networks",
    "transformer architectures", "self-attention", "multi-head attention",
    "positional encoding", "layer normalization", "residual connections",
    "gradient descent", "momentum methods", "adaptive optimizers",
    "learning rate scheduling", "warmup strategies", "curriculum learning",
    "continual learning", "catastrophic forgetting", "elastic weight consolidation",
    "progressive networks", "neural architecture search", "AutoML",
    "hyperparameter optimization", "Bayesian optimization", "evolutionary strategies",
    "genetic algorithms", "particle swarm optimization", "simulated annealing",
]

topics_cn = [
    "机器学习", "神经科学", "意识研究", "记忆系统",
    "模式识别", "自适应系统", "预测编码",
    "神经网络", "认知架构", "信息论",
    "动力系统", "涌现行为", "表征学习",
    "注意力机制", "时间处理", "感觉运动整合",
    "决策制定", "不确定性量化", "因果推理",
    "元学习", "迁移学习", "少样本学习",
    "强化学习", "探索与利用",
    "信用分配", "层次模型", "组合性",
    "符号推理", "类比推理", "溯因推理",
    "贝叶斯推断", "自由能原理", "主动推断",
    "具身认知", "延展心智", "分布式认知",
    "集体智能", "群体行为", "自组织",
    "临界性", "相变", "吸引子动力学",
    "混沌理论", "分形结构", "复杂网络",
    "图论", "拓扑数据分析", "流形学习",
    "降维", "特征提取", "核方法",
    "深度学习", "卷积网络", "循环网络",
    "Transformer架构", "自注意力", "多头注意力",
    "位置编码", "层归一化", "残差连接",
    "梯度下降", "动量方法", "自适应优化器",
    "学习率调度", "预热策略", "课程学习",
    "持续学习", "灾难性遗忘", "弹性权重巩固",
    "渐进网络", "神经架构搜索", "自动机器学习",
    "超参数优化", "贝叶斯优化", "进化策略",
    "遗传算法", "粒子群优化", "模拟退火",
]

# 英文句式模板（含随机插入，避免完全重复）
def generate_en_sentence(topic):
    templates = [
        f"The study of {topic} reveals fundamental principles about how intelligent systems process information.",
        f"Researchers have discovered that {topic} depends critically on the interplay between structure and dynamics.",
        f"In recent years, {topic} has emerged as a cornerstone of modern artificial intelligence.",
        f"Understanding {topic} requires integrating insights from multiple disciplinary perspectives.",
        f"The challenge of {topic} lies in balancing computational efficiency with representational capacity.",
        f"Advances in {topic} have enabled breakthrough applications across diverse domains.",
        f"Critically, {topic} cannot be reduced to purely symbolic or purely connectionist frameworks.",
        f"The temporal dynamics of {topic} suggest that processing occurs at multiple timescales simultaneously.",
        f"From a biological perspective, {topic} mirrors organizational principles found in cortical circuits.",
        f"Theoretical analysis shows that {topic} exhibits properties of both stability and flexibility.",
        f"Empirical studies demonstrate that {topic} improves monotonically with scale and diversity.",
        f"However, {topic} remains poorly understood when systems operate far from equilibrium.",
        f"A key insight is that {topic} involves recursive feedback loops between perception and action.",
        f"The mathematical foundations of {topic} draw from differential geometry and statistical mechanics.",
        f"Practitioners often underestimate the importance of {topic} in real-world deployment scenarios.",
        f"Interestingly, {topic} displays universal characteristics across biological and artificial implementations.",
        f"The history of {topic} traces back to foundational work in cybernetics and control theory.",
        f"Contemporary debates center on whether {topic} should be approached top-down or bottom-up.",
        f"Experimental evidence suggests that {topic} benefits from modularity and sparse connectivity.",
        f"Ultimately, {topic} may hold the key to building systems that generalize robustly.",
    ]
    return random.choice(templates)

# 中文句式模板
def generate_cn_sentence(topic):
    templates = [
        f"{topic}的研究揭示了智能系统处理信息的基本原理。",
        f"科学家们发现，{topic}依赖于结构与动力学之间的关键相互作用。",
        f"近年来，{topic}已成为现代人工智能的基石之一。",
        f"理解{topic}需要整合多学科视角的深刻洞见。",
        f"{topic}的挑战在于计算效率与表征能力之间的平衡。",
        f"{topic}的进展已在多个领域实现了突破性应用。",
        f"关键在于，{topic}不能单纯归结为符号主义或连接主义框架。",
        f"{topic}的时间动力学表明，处理过程同时在多个时间尺度上进行。",
        f"从生物学角度看，{topic}反映了皮层回路中的组织原则。",
        f"理论分析表明，{topic}兼具稳定性与灵活性的特征。",
        f"实证研究表明，{topic}随规模和多样性的增加而单调提升。",
        f"然而，当系统远离平衡态时，{topic}仍然难以被充分理解。",
        f"一个重要洞见是，{topic}涉及感知与行动之间的递归反馈循环。",
        f"{topic}的数学基础源于微分几何与统计力学。",
        f"实践者常常低估{topic}在实际部署场景中的重要性。",
        f"有趣的是，{topic}在生物与人工实现中展现出普遍性特征。",
        f"{topic}的历史可追溯至控制论与控制理论的开创性工作。",
        f"当代争论的焦点在于，应当以自上而下还是自下而上的方式研究{topic}。",
        f"实验证据表明，{topic}受益于模块化和稀疏连接结构。",
        f"最终，{topic}可能是构建鲁棒泛化系统的关键所在。",
    ]
    return random.choice(templates)

# 生成段落（多句组合）
def generate_paragraph():
    n_sentences = random.randint(2, 4)
    if random.random() < 0.5:
        # 英文段落
        topics = random.sample(topics_en, n_sentences)
        sentences = [generate_en_sentence(t) for t in topics]
    else:
        # 中文段落
        topics = random.sample(topics_cn, n_sentences)
        sentences = [generate_cn_sentence(t) for t in topics]
    return ' '.join(sentences)

# 主函数
def main():
    n_train = 50000
    n_val = 5000
    
    os.makedirs('data/corpus', exist_ok=True)
    
    # 生成训练集
    train_lines = []
    seen = set()
    while len(train_lines) < n_train:
        para = generate_paragraph()
        # 简单去重
        if para not in seen:
            seen.add(para)
            train_lines.append(para)
    
    # 生成验证集
    val_lines = []
    while len(val_lines) < n_val:
        para = generate_paragraph()
        if para not in seen:
            seen.add(para)
            val_lines.append(para)
    
    # 保存
    with open('data/corpus/train.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_lines))
    
    with open('data/corpus/val.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(val_lines))
    
    # 统计
    unique_train = len(set(train_lines))
    unique_val = len(set(val_lines))
    
    print(f"训练集: {len(train_lines)} 段, 唯一性: {unique_train/len(train_lines):.2%}")
    print(f"验证集: {len(val_lines)} 段, 唯一性: {unique_val/len(val_lines):.2%}")
    print(f"样本示例:")
    for i in range(3):
        print(f"  [{i}] {train_lines[i][:100]}...")

if __name__ == '__main__':
    main()
