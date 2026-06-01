"""
AGI-CORTEX 增强版训练数据生成器

生成六大类数据，支持分布统计、质量报告、可视化导出。

Usage:
    python scripts/generate_enriched_data.py --all --viz
    python scripts/generate_enriched_data.py --reasoning --metacog --viz
    python scripts/generate_enriched_data.py --rl --n_episodes 5000
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# matplotlib 可选
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

DATA_DIR = Path("data")
VIZ_DIR = Path("outputs/data_viz")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# 1. 基础语料：WikiText-2 风格（英文 + 中文混合）
# =============================================================================

def generate_base_corpus(n_samples=50000):
    """生成合成基础语料（含中英文混合段落）。"""
    print("[1/6] 生成基础语料...")
    output_dir = DATA_DIR / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 英文模板
    en_templates = [
        "The {noun} is {adj} and {adj2}. It can {verb} very well.",
        "In {year}, scientists discovered that {noun} {verb} {noun2}.",
        "{person} said: '{quote}' This changed how we think about {topic}.",
        "When {noun} meets {noun2}, the result is usually {adj}.",
        "Research shows that {verb} {noun} requires {noun2} and patience.",
    ]
    en_nouns = ["neural network", "brain", "cortex", "algorithm", "consciousness",
                "memory", "perception", "intelligence", "pattern", "system"]
    en_adjs = ["complex", "adaptive", "robust", "dynamic", "efficient",
               "interesting", "important", "fundamental", "novel", "powerful"]
    en_verbs = ["learns", "adapts", "evolves", "processes", "predicts",
                "recognizes", "integrates", "transforms", "generates", "understands"]
    en_people = ["Dr. Smith", "Professor Chen", "Alice", "Bob", "Research Team A"]
    en_topics = ["cognition", "AI", "neuroscience", "machine learning", "consciousness"]

    # 中文模板
    cn_templates = [
        "{cn_noun}是一种非常{cn_adj}的现象，它可以帮助我们理解{cn_topic}。",
        "研究表明，{cn_noun}与{cn_noun2}之间存在密切的关系。",
        "{cn_person}指出：'{cn_quote}'这对{cn_topic}领域产生了深远影响。",
        "在{cn_year}年，科学家们发现{cn_noun}能够{cn_verb}{cn_noun2}。",
        "{cn_noun}的本质是{cn_adj}的，这需要{cn_noun2}的支持。",
    ]
    cn_nouns = ["神经网络", "大脑皮层", "认知系统", "智能体", "记忆机制",
                "感知系统", "注意力", "模式识别", "预测编码", "全局工作空间"]
    cn_adjs = ["复杂", "自适应", "鲁棒", "动态", "高效",
               "有趣", "重要", "基础", "新颖", "强大"]
    cn_verbs = ["学习", "适应", "进化", "处理", "预测",
                "识别", "整合", "转换", "生成", "理解"]
    cn_people = ["王博士", "李教授", "张研究员", "陈院士", "刘博士"]
    cn_topics = ["认知科学", "人工智能", "神经科学", "机器学习", "意识研究"]

    lines = []
    for i in range(n_samples):
        if random.random() < 0.5:
            # 英文
            t = random.choice(en_templates)
            line = t.format(
                noun=random.choice(en_nouns),
                noun2=random.choice(en_nouns),
                adj=random.choice(en_adjs),
                adj2=random.choice(en_adjs),
                verb=random.choice(en_verbs),
                year=random.randint(1950, 2024),
                person=random.choice(en_people),
                quote=random.choice(["Knowledge is power.", "The brain is a prediction machine.", "Intelligence is adaptation."]),
                topic=random.choice(en_topics),
            )
        else:
            # 中文
            t = random.choice(cn_templates)
            line = t.format(
                cn_noun=random.choice(cn_nouns),
                cn_noun2=random.choice(cn_nouns),
                cn_adj=random.choice(cn_adjs),
                cn_verb=random.choice(cn_verbs),
                cn_year=random.randint(1950, 2024),
                cn_person=random.choice(cn_people),
                cn_quote=random.choice(["知识就是力量。", "大脑是一台预测机器。", "智能即适应。"]),
                cn_topic=random.choice(cn_topics),
            )
        lines.append(line)

    # 保存
    train_split = int(len(lines) * 0.9)
    for split_name, split_lines in [("train", lines[:train_split]), ("val", lines[train_split:])]:
        path = output_dir / f"{split_name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(split_lines))
        print(f"  {split_name}: {len(split_lines)} 段 -> {path}")

    stats = {
        "total_samples": len(lines),
        "avg_length": np.mean([len(l) for l in lines]),
        "language_dist": {"en": sum(1 for l in lines if ord(l[0]) < 128), "cn": sum(1 for l in lines if ord(l[0]) >= 128)}
    }
    return stats


# =============================================================================
# 2. 符号推理数据（M1）
# =============================================================================

def generate_reasoning_data(n_samples=10000):
    """生成结构化符号推理数据，含推理链。"""
    print("[2/6] 生成符号推理数据...")
    output_dir = DATA_DIR / "reasoning"
    output_dir.mkdir(parents=True, exist_ok=True)

    rules = [
        {
            "name": "modus_ponens",
            "templates": [
                ("如果 {A} 那么 {B}。{A}。因此？", "{B}"),
                ("If {A} then {B}. {A}. Therefore?", "{B}"),
                ("所有 {A} 都是 {B}。{C} 是 {A}。因此？", "{C} 是 {B}"),
            ],
            "difficulty": 1
        },
        {
            "name": "modus_tollens",
            "templates": [
                ("如果 {A} 那么 {B}。不是 {B}。因此？", "不是 {A}"),
                ("If {A} then {B}. Not {B}. Therefore?", "Not {A}"),
            ],
            "difficulty": 2
        },
        {
            "name": "transitive",
            "templates": [
                ("{A} 在 {B} 的左边。{B} 在 {C} 的左边。{A} 在 {C} 的？", "左边"),
                ("{A} is taller than {B}. {B} is taller than {C}. So {A} is ? than {C}", "taller"),
                ("{A} > {B} 且 {B} > {C}。那么 {A} 和 {C} 的关系是？", "{A} > {C}"),
            ],
            "difficulty": 2
        },
        {
            "name": "causal",
            "templates": [
                ("{A} 导致 {B}。{B} 发生了。这可能是由于？", "{A}"),
                ("{A} causes {B}. We observe {B}. The likely cause is?", "{A}"),
            ],
            "difficulty": 3
        },
        {
            "name": "analogy",
            "templates": [
                ("{A} 之于 {B} 就像 {C} 之于？", "{D}"),
                ("{A} is to {B} as {C} is to ?", "{D}"),
            ],
            "difficulty": 3
        },
    ]

    entities = ["猫", "狗", "鸟", "鱼", "人", "机器人", "书", "桌子", "门", "钥匙",
                "太阳", "月亮", "星星", "云朵", "雨水", "风", "火", "土壤",
                "cat", "dog", "bird", "fish", "John", "Mary", "Alice", "Bob",
                "teacher", "student", "doctor", "engineer", "artist", "scientist"]
    properties = ["动物", "物体", "活的", "红色", "蓝色", "大", "小", "热", "冷",
                  "animal", "object", "alive", "red", "blue", "big", "small", "hot", "cold"]
    analogies = [
        ("手", "手套", "脚", "鞋"), ("鸟", "天空", "鱼", "水"),
        ("医生", "医院", "老师", "学校"), ("蜜蜂", "花", "鸟", "树"),
        ("hand", "glove", "foot", "shoe"), ("bird", "sky", "fish", "water"),
    ]

    samples = []
    rule_counts = Counter()
    difficulty_dist = Counter()

    for _ in range(n_samples):
        rule = random.choice(rules)
        template, answer_template = random.choice(rule["templates"])

        if rule["name"] == "analogy":
            a, b, c, d = random.choice(analogies)
            context = template.format(A=a, B=b, C=c, D=d)
            answer = answer_template.format(D=d)
        else:
            vals = {
                "A": random.choice(entities),
                "B": random.choice(properties + entities),
                "C": random.choice(entities),
            }
            context = template.format(**vals)
            answer = answer_template.format(**vals)

        # 偶尔生成错误答案（让模型学会识别）
        is_valid = random.random() > 0.15
        if not is_valid:
            wrong_answers = random.sample(entities, 3)
            answer = f"[INVALID] {random.choice(wrong_answers)}"

        samples.append({
            "context": context,
            "answer": answer,
            "rule": rule["name"],
            "difficulty": rule["difficulty"],
            "valid": is_valid,
            "type": "reasoning"
        })
        rule_counts[rule["name"]] += 1
        difficulty_dist[rule["difficulty"]] += 1

    # 保存
    path = output_dir / "train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  生成: {len(samples)} 条推理链 -> {path}")

    stats = {
        "total": len(samples),
        "rule_distribution": dict(rule_counts),
        "difficulty_distribution": dict(difficulty_dist),
        "valid_ratio": sum(1 for s in samples if s["valid"]) / len(samples),
    }
    return stats


# =============================================================================
# 3. 元认知数据（M2）
# =============================================================================

def generate_metacognition_data(n_samples=5000):
    """生成需要自我评估确定性的元认知数据。"""
    print("[3/6] 生成元认知数据...")
    output_dir = DATA_DIR / "metacognition"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 高确定性
    high_q = [
        ("2 + 2 = ?", "4", 0.95),
        ("水的沸点是多少摄氏度？", "100", 0.92),
        ("一年有多少个月？", "12", 0.98),
        ("太阳从哪个方向升起？", "东方", 0.97),
        ("What is the capital of France?", "Paris", 0.95),
        ("How many legs does a dog have?", "4", 0.96),
    ]

    # 中等确定性（需要计算/推理）
    medium_q = []
    for _ in range(n_samples // 3):
        a, b = random.randint(10, 99), random.randint(10, 99)
        op = random.choice(["+", "-", "×"])
        if op == "+": ans = a + b
        elif op == "-": ans = a - b
        else: ans = a * b
        medium_q.append((f"{a} {op} {b} = ?", str(ans), 0.65))

    # 低确定性（信息不足）
    low_q = [
        ("约翰的叔叔的妻子的姐姐的女儿叫什么名字？", "信息不足，无法确定", 0.15),
        ("一个没有上下文的汉字'行'是什么意思？", "可能是'可以'或'走路'，需要更多上下文", 0.20),
        ("如果所有 A 都是 B，那么有多少个 B？", "信息不足，无法确定", 0.25),
        ("明年的今天股市会涨吗？", "无法预测", 0.10),
        ("What did my neighbor eat for breakfast yesterday?", "I don't have this information", 0.15),
        ("Is there life on planet XYZ-42?", "Unknown, insufficient data", 0.10),
    ]

    all_q = (high_q * (n_samples // 6) + medium_q + low_q * (n_samples // 6))
    random.shuffle(all_q)
    all_q = all_q[:n_samples]

    samples = []
    certainty_buckets = Counter()
    for q, a, cert in all_q:
        # 添加一些"陷阱"——表面简单但实际 tricky
        is_trap = random.random() < 0.1 and cert > 0.5
        if is_trap:
            cert = max(0.2, cert - 0.4)  # 降低确定性

        samples.append({
            "input": f"问题：{q}\n请先评估你的确定性（0-1），然后回答。",
            "expected_certainty": cert,
            "expected_answer": a,
            "is_trap": is_trap,
            "type": "metacognition"
        })

        # 分桶统计
        if cert >= 0.8:
            certainty_buckets["high"] += 1
        elif cert >= 0.4:
            certainty_buckets["medium"] += 1
        else:
            certainty_buckets["low"] += 1

    path = output_dir / "train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  生成: {len(samples)} 条元认知样本 -> {path}")

    stats = {
        "total": len(samples),
        "certainty_distribution": dict(certainty_buckets),
        "trap_ratio": sum(1 for s in samples if s["is_trap"]) / len(samples),
        "avg_certainty": np.mean([s["expected_certainty"] for s in samples]),
    }
    return stats


# =============================================================================
# 4. RL 轨迹数据（M3）
# =============================================================================

def generate_rl_trajectories(n_episodes=2000):
    """生成模拟环境交互轨迹。"""
    print("[4/6] 生成 RL 轨迹数据...")
    output_dir = DATA_DIR / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = []
    episode_lengths = []
    episode_rewards = []

    for ep in range(n_episodes):
        # 模拟不同难度的环境
        difficulty = random.choice(["easy", "medium", "hard"])
        max_steps = {"easy": 50, "medium": 100, "hard": 200}[difficulty]

        states = []
        actions = []
        rewards = []

        # 模拟状态转移
        state = np.random.randn(4)  # 4维状态空间
        cumulative_reward = 0

        for step in range(max_steps):
            # 动作空间：0-3
            action = random.randint(0, 3)

            # 简单动态：状态根据动作转移
            next_state = state + np.random.randn(4) * 0.1
            next_state[action % 4] += 0.5  # 动作影响对应维度

            # 奖励：接近目标状态
            target = np.array([1.0, -1.0, 0.5, -0.5])
            dist = np.linalg.norm(next_state - target)
            reward = -dist + random.gauss(0, 0.1)

            states.append(state.tolist())
            actions.append(action)
            rewards.append(reward)
            cumulative_reward += reward

            state = next_state

            # 终止条件
            if dist < 0.5 or step == max_steps - 1:
                break

        trajectory = {
            "episode_id": ep,
            "difficulty": difficulty,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "length": len(states),
            "total_reward": cumulative_reward,
            "type": "rl_trajectory"
        }
        trajectories.append(trajectory)
        episode_lengths.append(len(states))
        episode_rewards.append(cumulative_reward)

    # 保存
    path = output_dir / "trajectories.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")
    print(f"  生成: {len(trajectories)} 条轨迹 -> {path}")

    stats = {
        "total_episodes": len(trajectories),
        "avg_length": np.mean(episode_lengths),
        "avg_reward": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "difficulty_dist": Counter(t["difficulty"] for t in trajectories),
    }
    return stats


# =============================================================================
# 5. 持续学习数据（M5）
# =============================================================================

def generate_continual_data():
    """生成多领域顺序任务数据。"""
    print("[5/6] 生成持续学习数据...")
    output_dir = DATA_DIR / "continual"
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = {
        "math": {
            "desc": "数学计算",
            "templates": [
                "计算 {a} + {b} = {ans}", "计算 {a} × {b} = {ans}",
                "解方程：x + {a} = {b}，x = {ans}", "{a} 的平方是 {ans}",
            ],
            "generator": lambda: (random.randint(1, 100), random.randint(1, 100))
        },
        "geo": {
            "desc": "地理知识",
            "templates": [
                "{city} 是哪个国家的首都？{ans}",
                "世界上最大的{cn_noun}是？{ans}",
                "{continent} 位于哪个半球？{ans}",
            ],
            "generator": lambda: random.choice([
                ("北京", "中国"), ("东京", "日本"), ("巴黎", "法国"),
                ("伦敦", "英国"), ("纽约", "美国"),
            ])
        },
        "code": {
            "desc": "编程知识",
            "templates": [
                "Python 中 {kw} 的用途是什么？{ans}",
                "写出使用 {kw} 的代码示例。{ans}",
            ],
            "generator": lambda: random.choice([
                ("for", "循环遍历可迭代对象"),
                ("def", "定义函数"),
                ("class", "定义类"),
                ("import", "导入模块"),
            ])
        },
        "bio": {
            "desc": "生物常识",
            "templates": [
                "{bio_term} 的功能是什么？{ans}",
                "生物体内的 {bio_term} 负责什么？{ans}",
            ],
            "generator": lambda: random.choice([
                ("DNA", "存储遗传信息"),
                ("RNA", "转录和翻译蛋白质"),
                ("线粒体", "产生能量（ATP）"),
                ("细胞膜", "控制物质进出"),
            ])
        },
        "physics": {
            "desc": "物理基础",
            "templates": [
                "{law} 的内容是什么？{ans}",
                "解释 {concept} 的原理。{ans}",
            ],
            "generator": lambda: random.choice([
                ("牛顿第一定律", "物体保持静止或匀速直线运动，除非受到外力作用"),
                ("能量守恒", "能量既不会凭空产生，也不会凭空消失"),
                ("万有引力", "两个物体之间的引力与质量成正比，与距离平方成反比"),
            ])
        },
    }

    all_stats = {}
    for domain_id, info in domains.items():
        lines = []
        for _ in range(2000):
            template = random.choice(info["templates"])
            if domain_id == "math":
                a, b = info["generator"]()
                op = random.choice([("+", a+b), ("×", a*b), ("-", a-b)])
                text = template.format(a=a, b=b, ans=op[1], cn_noun="", continent="")
            elif domain_id == "geo":
                city, country = info["generator"]()
                text = template.format(city=city, ans=country, cn_noun="洋", continent="亚洲")
            elif domain_id in ["code", "bio", "physics"]:
                term, ans = info["generator"]()
                text = template.format(kw=term, ans=ans, law=term, concept=term, bio_term=term)
            else:
                text = template.format(**{k: "sample" for k in ["a", "b", "ans", "city"]})
            lines.append(text)

        path = output_dir / f"{domain_id}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  领域 {domain_id} ({info['desc']}): {len(lines)} 条 -> {path}")
        all_stats[domain_id] = len(lines)

    return {"domain_sizes": all_stats, "total": sum(all_stats.values())}


# =============================================================================
# 6. 因果推断数据（M6）
# =============================================================================

def generate_causal_data(n_systems=100, n_steps_per_system=500):
    """生成结构化因果系统观测数据。"""
    print("[6/6] 生成因果推断数据...")
    output_dir = DATA_DIR / "causal"
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = []
    edge_counts = []

    for sys_id in range(n_systems):
        n_vars = random.randint(3, 8)
        var_names = [f"V{i}" for i in range(n_vars)]

        # 随机生成 DAG（有向无环图）
        adj = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if random.random() < 0.3:  # 30% 概率有边
                    adj[i, j] = random.uniform(-1, 1)

        edge_counts.append(int((adj != 0).sum()))

        # 生成时间序列
        states = np.zeros((n_steps_per_system, n_vars))
        states[0] = np.random.randn(n_vars)

        for t in range(1, n_steps_per_system):
            # 线性动态 + 噪声
            states[t] = states[t-1] @ adj.T + np.random.randn(n_vars) * 0.1
            # 偶尔干预
            if random.random() < 0.05:
                intervene_var = random.randint(0, n_vars - 1)
                states[t, intervene_var] = random.uniform(-3, 3)

        # 偶尔生成反事实问题
        counterfactuals = []
        if random.random() < 0.3:
            cf_var = random.randint(0, n_vars - 1)
            cf_value = random.uniform(-2, 2)
            counterfactuals.append({
                "variable": var_names[cf_var],
                "intervention_value": round(float(cf_value), 3),
                "question": f"如果我们将 {var_names[cf_var]} 设为 {cf_value:.2f}，其他变量会如何变化？"
            })

        systems.append({
            "system_id": sys_id,
            "n_variables": n_vars,
            "variables": var_names,
            "adjacency": adj.tolist(),
            "time_series": states.tolist(),
            "n_edges": int((adj != 0).sum()),
            "counterfactuals": counterfactuals,
            "type": "causal_system"
        })

    path = output_dir / "systems.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s in systems:
            f.write(json.dumps(s) + "\n")
    print(f"  生成: {len(systems)} 个因果系统 -> {path}")

    stats = {
        "n_systems": len(systems),
        "avg_variables": np.mean([s["n_variables"] for s in systems]),
        "avg_edges": np.mean(edge_counts),
        "total_observations": len(systems) * n_steps_per_system,
        "counterfactual_ratio": sum(1 for s in systems if s["counterfactuals"]) / len(systems),
    }
    return stats


# =============================================================================
# 可视化
# =============================================================================

def visualize_all(stats_dict, output_dir=VIZ_DIR):
    """生成数据分布可视化图表。"""
    if not MATPLOTLIB_AVAILABLE:
        print("[Warning] matplotlib 未安装，跳过可视化。运行: pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[可视化] 生成数据分布图表...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("AGI-CORTEX Training Data Distribution", fontsize=16, fontweight='bold')

    # 1. 基础语料语言分布
    ax = axes[0, 0]
    if "corpus" in stats_dict:
        dist = stats_dict["corpus"].get("language_dist", {})
        if dist:
            ax.bar(dist.keys(), dist.values(), color=['#3498db', '#e74c3c'])
            ax.set_title("Corpus Language Distribution")
            ax.set_ylabel("Samples")

    # 2. 推理规则分布
    ax = axes[0, 1]
    if "reasoning" in stats_dict:
        dist = stats_dict["reasoning"].get("rule_distribution", {})
        if dist:
            colors = plt.cm.Set3(np.linspace(0, 1, len(dist)))
            ax.barh(list(dist.keys()), list(dist.values()), color=colors)
            ax.set_title("Reasoning Rule Distribution")
            ax.set_xlabel("Count")

    # 3. 推理难度分布
    ax = axes[0, 2]
    if "reasoning" in stats_dict:
        dist = stats_dict["reasoning"].get("difficulty_distribution", {})
        if dist:
            labels = [f"Level {k}" for k in dist.keys()]
            ax.pie(dist.values(), labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title("Reasoning Difficulty")

    # 4. 元认知确定性分布
    ax = axes[1, 0]
    if "metacognition" in stats_dict:
        dist = stats_dict["metacognition"].get("certainty_distribution", {})
        if dist:
            ax.bar(dist.keys(), dist.values(), color=['#2ecc71', '#f39c12', '#e74c3c'])
            ax.set_title("Metacognition Certainty Distribution")
            ax.set_ylabel("Samples")

    # 5. RL 奖励分布（模拟）
    ax = axes[1, 1]
    if "rl" in stats_dict:
        # 生成模拟奖励直方图
        avg_r = stats_dict["rl"].get("avg_reward", 0)
        std_r = stats_dict["rl"].get("reward_std", 1)
        rewards = np.random.normal(avg_r, std_r, 1000)
        ax.hist(rewards, bins=30, color='#9b59b6', edgecolor='white', alpha=0.7)
        ax.axvline(avg_r, color='red', linestyle='--', label=f'Mean={avg_r:.2f}')
        ax.set_title("RL Episode Reward Distribution")
        ax.set_xlabel("Total Reward")
        ax.legend()

    # 6. 持续学习领域分布
    ax = axes[1, 2]
    if "continual" in stats_dict:
        dist = stats_dict["continual"].get("domain_sizes", {})
        if dist:
            colors = plt.cm.Paired(np.linspace(0, 1, len(dist)))
            ax.bar(dist.keys(), dist.values(), color=colors)
            ax.set_title("Continual Learning Domain Sizes")
            ax.set_ylabel("Samples")
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    viz_path = output_dir / "data_distribution.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  图表已保存 -> {viz_path}")
    plt.close()

    # 额外：生成数据质量报告
    report_path = output_dir / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"  质量报告已保存 -> {report_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate enriched training data for AGI-CORTEX")
    parser.add_argument("--all", action="store_true", help="Generate all data types")
    parser.add_argument("--corpus", action="store_true", help="Base corpus")
    parser.add_argument("--reasoning", action="store_true", help="Symbolic reasoning data")
    parser.add_argument("--metacog", action="store_true", help="Metacognition data")
    parser.add_argument("--rl", action="store_true", help="RL trajectories")
    parser.add_argument("--continual", action="store_true", help="Continual learning data")
    parser.add_argument("--causal", action="store_true", help="Causal inference data")
    parser.add_argument("--viz", action="store_true", help="Generate visualizations")
    parser.add_argument("--n_reasoning", type=int, default=10000)
    parser.add_argument("--n_metacog", type=int, default=5000)
    parser.add_argument("--n_rl", type=int, default=2000)
    parser.add_argument("--n_corpus", type=int, default=50000)
    args = parser.parse_args()

    if not any([args.all, args.corpus, args.reasoning, args.metacog, args.rl, args.continual, args.causal]):
        args.all = True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}

    if args.all or args.corpus:
        stats["corpus"] = generate_base_corpus(args.n_corpus)
    if args.all or args.reasoning:
        stats["reasoning"] = generate_reasoning_data(args.n_reasoning)
    if args.all or args.metacog:
        stats["metacognition"] = generate_metacognition_data(args.n_metacog)
    if args.all or args.rl:
        stats["rl"] = generate_rl_trajectories(args.n_rl)
    if args.all or args.continual:
        stats["continual"] = generate_continual_data()
    if args.all or args.causal:
        stats["causal"] = generate_causal_data()

    print("\n" + "=" * 60)
    print("数据生成完成!")
    print("=" * 60)
    for name, s in stats.items():
        print(f"\n[{name}]")
        for k, v in s.items():
            print(f"  {k}: {v}")

    if args.viz or args.all:
        visualize_all(stats)

    print(f"\n数据目录: {DATA_DIR.absolute()}")
    if args.viz or args.all:
        print(f"可视化目录: {VIZ_DIR.absolute()}")


if __name__ == "__main__":
    main()
