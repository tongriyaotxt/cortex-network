"""
CORTEX / AGI-CORTEX 训练数据准备脚本

自动下载并预处理以下数据：
- 基础语料：WikiText-2（语言建模）
- 推理数据：bAbI（符号推理）
- 元认知数据：自制陷阱样本（不确定性校准）
- RL 环境数据：TextWorld 轨迹（可选）

Usage:
    python scripts/prepare_data.py --all          # 准备所有数据
    python scripts/prepare_data.py --corpus       # 仅基础语料
    python scripts/prepare_data.py --reasoning    # 仅推理数据
    python scripts/prepare_data.py --metacog      # 仅元认知数据
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

DATA_DIR = Path("data")


# =============================================================================
# 1. 基础语料：WikiText-2
# =============================================================================

def prepare_wikitext():
    """下载并预处理 WikiText-2。"""
    print("[prepare_wikitext] 准备 WikiText-2...")
    output_dir = DATA_DIR / "wikitext"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        for split in ["train", "validation", "test"]:
            texts = [item["text"] for item in dataset[split] if item["text"].strip()]
            output_file = output_dir / f"{split}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(texts))
            print(f"  {split}: {len(texts)} 段文本 -> {output_file}")

        print("  WikiText-2 准备完成")
        return True
    except ImportError:
        print("  错误：需要安装 datasets 库: pip install datasets")
        return False
    except Exception as e:
        print(f"  错误：{e}")
        return False


# =============================================================================
# 2. 推理数据：bAbI + 自制逻辑链
# =============================================================================

def prepare_babi():
    """下载 bAbI 任务（Facebook AI Research）。"""
    print("[prepare_babi] 准备 bAbI 推理数据...")
    output_dir = DATA_DIR / "babi"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        # bAbI tasks 1-20
        dataset = load_dataset("facebook/babi_qa", "en-10k", trust_remote_code=True)

        all_chains = []
        for item in dataset["train"]:
            story = item.get("story", {})
            text = story.get("text", "")
            answer = item.get("answer", "")
            if text and answer:
                all_chains.append({
                    "context": text,
                    "question": item.get("question", ""),
                    "answer": answer,
                    "type": "babi_reasoning"
                })

        # 保存为 JSONL
        output_file = output_dir / "train.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chain in all_chains:
                f.write(json.dumps(chain, ensure_ascii=False) + "\n")
        print(f"  训练集: {len(all_chains)} 条推理链 -> {output_file}")
        print("  bAbI 准备完成")
        return True
    except Exception as e:
        print(f"  警告：bAbI 下载失败 ({e})，将使用自制逻辑数据替代")
        return prepare_synthetic_reasoning()


def prepare_synthetic_reasoning(n_samples: int = 10000):
    """自制符号推理数据（bAbI 不可用时的回退）。"""
    print("[prepare_synthetic_reasoning] 生成自制逻辑推理数据...")
    output_dir = DATA_DIR / "reasoning"
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = [
        {
            "template": "如果 {A} 那么 {B}。{A}。因此？",
            "answer": "{B}",
            "rule": "modus_ponens"
        },
        {
            "template": "如果 {A} 那么 {B}。不是 {B}。因此？",
            "answer": "不是 {A}",
            "rule": "modus_tollens"
        },
        {
            "template": "{A} 或者 {B}。不是 {A}。因此？",
            "answer": "{B}",
            "rule": "disjunctive_syllogism"
        },
        {
            "template": "所有 {A} 都是 {B}。{C} 是 {A}。因此？",
            "answer": "{C} 是 {B}",
            "rule": "syllogism"
        },
        {
            "template": "{A} 在 {B} 的左边。{B} 在 {C} 的左边。{A} 在 {C} 的？",
            "answer": "左边",
            "rule": "transitive_relations"
        },
    ]

    entities = ["猫", "狗", "鸟", "鱼", "人", "机器人", "书", "桌子", "门", "钥匙",
                "cat", "dog", "bird", "John", "Mary", "box", "garden", "kitchen"]
    properties = ["动物", "物体", "活的", "红色", "蓝色", "大", "小",
                  "animal", "object", "red", "blue", "big", "small"]

    samples = []
    for _ in range(n_samples):
        t = random.choice(templates)
        vals = {
            "A": random.choice(entities),
            "B": random.choice(properties + entities),
            "C": random.choice(entities),
        }
        context = t["template"].format(**vals)
        answer = t["answer"].format(**vals)
        samples.append({
            "context": context,
            "question": "结论是什么？",
            "answer": answer,
            "rule": t["rule"],
            "type": "synthetic_reasoning"
        })

    output_file = output_dir / "synthetic_train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  生成: {len(samples)} 条合成推理链 -> {output_file}")
    print("  自制推理数据准备完成")
    return True


# =============================================================================
# 3. 元认知数据：不确定性校准 + 陷阱样本
# =============================================================================

def prepare_metacognition(n_samples: int = 5000):
    """生成需要自我反思的元认知训练数据。"""
    print("[prepare_metacognition] 生成元认知训练数据...")
    output_dir = DATA_DIR / "metacognition"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 类别 1：明确的问题（高确定性）
    clear_questions = [
        ("2 + 2 = ?", "4", "high"),
        ("水的化学式是？", "H2O", "high"),
        ("一年有多少个月？", "12", "high"),
        ("太阳从哪个方向升起？", "东方", "high"),
    ]

    # 类别 2：模糊/需要推理的问题（中等确定性）
    medium_questions = [
        ("13 × 17 = ?", "221", "medium"),
        ("下列哪个最重：1kg铁、1kg棉花、2kg羽毛？", "2kg羽毛", "medium"),
        ("如果 A > B 且 B > C，那么 A 和 C 的关系是？", "A > C", "medium"),
    ]

    # 类别 3：陷阱/歧义问题（低确定性，模型应该承认不知道）
    ambiguous_questions = [
        ("约翰的叔叔的妻子的姐姐的女儿叫什么名字？", "信息不足，无法确定", "low"),
        ("一个没有上下文的中文单字'行'是什么意思？", "可能是'可以'或'走路'，需要更多上下文", "low"),
        ("如果所有 A 都是 B，那么有多少个 B？", "信息不足，无法确定", "low"),
        ("明年的今天股市会涨吗？", "无法预测", "low"),
    ]

    all_questions = clear_questions * (n_samples // 4) + \
                    medium_questions * (n_samples // 4) + \
                    ambiguous_questions * (n_samples // 2)
    random.shuffle(all_questions)

    samples = []
    certainty_map = {"high": 0.9, "medium": 0.6, "low": 0.2}
    for q, a, level in all_questions[:n_samples]:
        samples.append({
            "input": f"问题：{q}\n请先评估你的确定性（0-1），然后回答。",
            "expected_certainty": certainty_map[level],
            "expected_answer": a,
            "difficulty": level,
            "type": "metacognition"
        })

    output_file = output_dir / "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  生成: {len(samples)} 条元认知样本 -> {output_file}")
    print("  元认知数据准备完成")
    return True


# =============================================================================
# 4. 持续学习数据：多领域顺序任务
# =============================================================================

def prepare_continual_data():
    """生成用于测试灾难性遗忘的多领域数据。"""
    print("[prepare_continual_data] 生成持续学习数据...")
    output_dir = DATA_DIR / "continual"
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = [
        ("math", "数学计算和公式", "计算 3+5。解方程 x^2=4。"),
        ("geo", "地理知识", "中国的首都是哪里？太平洋是世界上最大的洋吗？"),
        ("code", "编程基础", "Python 中列表和元组的区别是什么？写出 for 循环的语法。"),
        ("bio", "生物常识", "光合作用需要什么？DNA 的全称是什么？"),
    ]

    for domain_id, desc, examples in domains:
        # 每个领域生成不同风格的文本
        lines = []
        for i in range(2000):
            if domain_id == "math":
                a, b = random.randint(1, 100), random.randint(1, 100)
                lines.append(f"计算 {a} + {b} = {a+b}")
            elif domain_id == "geo":
                cities = ["北京", "东京", "纽约", "伦敦", "巴黎", "悉尼"]
                lines.append(f"{random.choice(cities)}是一个重要的城市。")
            elif domain_id == "code":
                keywords = ["def", "class", "import", "return", "if", "for"]
                lines.append(f"编程中使用{random.choice(keywords)}关键字。")
            elif domain_id == "bio":
                terms = ["细胞", "基因", "蛋白质", "光合作用", "呼吸作用"]
                lines.append(f"生物学研究{random.choice(terms)}的机制。")

        output_file = output_dir / f"{domain_id}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  领域 {domain_id} ({desc}): {len(lines)} 条 -> {output_file}")

    print("  持续学习数据准备完成")
    return True


# =============================================================================
# 5. 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for CORTEX")
    parser.add_argument("--all", action="store_true", help="Prepare all datasets")
    parser.add_argument("--corpus", action="store_true", help="Prepare base corpus (WikiText)")
    parser.add_argument("--reasoning", action="store_true", help="Prepare reasoning data (bAbI)")
    parser.add_argument("--metacog", action="store_true", help="Prepare metacognition data")
    parser.add_argument("--continual", action="store_true", help="Prepare continual learning data")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 如果没有指定任何参数，默认准备全部
    if not any([args.all, args.corpus, args.reasoning, args.metacog, args.continual]):
        args.all = True

    results = []
    if args.all or args.corpus:
        results.append(("corpus", prepare_wikitext()))
    if args.all or args.reasoning:
        results.append(("reasoning", prepare_babi()))
    if args.all or args.metacog:
        results.append(("metacognition", prepare_metacognition()))
    if args.all or args.continual:
        results.append(("continual", prepare_continual_data()))

    print("\n" + "=" * 60)
    print("数据准备完成")
    print("=" * 60)
    for name, ok in results:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")
    print(f"\n数据目录: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
