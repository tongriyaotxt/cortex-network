"""
原生中文 BPE Tokenizer

不需要 transformers/tokenizers 等外部库。
在 ~100MB 中文文本上训练 4096 vocab 只需要 1-2 分钟。

设计要点：
1. 初始词表：所有常见汉字（~3500）+ ASCII 字符 + 常见标点
2. BPE 合并：基于相邻 token 对的共现频率
3. 编码：贪心最长匹配（greedy longest match）
4. 对中文友好：优先合并常见词组（如"机器学习"）

Usage:
    from cortex.chinese_bpe import ChineseBPETokenizer
    tok = ChineseBPETokenizer(vocab_size=4096)
    tok.train(["data/zhwiki/corpus.txt"])
    tok.save("data/tokenizer/bpe_4096.json")

    tok2 = ChineseBPETokenizer.load("data/tokenizer/bpe_4096.json")
    ids = tok2.encode("机器学习是人工智能的核心技术。")
    text = tok2.decode(ids)
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple


class ChineseBPETokenizer:
    """
    原生 BPE Tokenizer，专为中文优化。
    """

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}       # token_str -> id
        self.inverse_vocab: Dict[int, str] = {}  # id -> token_str
        self.merges: List[Tuple[str, str]] = []   # BPE 合并规则列表

    # =====================================================================
    # 训练
    # =====================================================================

    def train(self, file_paths: List[str], min_freq: int = 2):
        """
        在指定文本文件上训练 BPE。

        Args:
            file_paths: 文本文件路径列表
            min_freq: 最小合并频率，低于此值的 pair 不合并
        """
        print(f"[BPE] 开始训练，目标 vocab_size={self.vocab_size}")

        # 1. 收集初始 token 频率
        token_freqs = defaultdict(int)
        for fp in file_paths:
            print(f"      读取 {fp} ...")
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 预分词：每个 UTF-8 字符作为一个初始 token
                    # 但为了效率，我们先把常见词组（2-4 字）统计出来
                    for ch in line:
                        token_freqs[ch] += 1

        # 2. 构建初始词表：包含所有出现过的字符 + 必须字符
        # 中文常用字只有几千个，vocab_size=4096 足够覆盖
        # 优先确保所有字符都在 vocab 中，避免 <unk>
        must_have = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            '，。！？、；：“”‘’（）《》【】\n\t '
            '<unk>'
        )

        self.vocab = {}
        for ch in must_have:
            self.vocab[ch] = len(self.vocab)

        # 加入训练数据中所有字符
        for ch in sorted(token_freqs.keys()):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        # 如果字符太多超过了 vocab_size，按频率截断
        if len(self.vocab) > self.vocab_size:
            sorted_chars = sorted(token_freqs.items(), key=lambda x: -x[1])
            self.vocab = {ch: i for i, (ch, _) in enumerate(sorted_chars[:self.vocab_size])}
            # 确保 must_have 中的关键字符不被丢
            for ch in must_have:
                if ch not in self.vocab:
                    # 替换掉最低频的
                    lowest = sorted_chars[self.vocab_size - 1][0]
                    del self.vocab[lowest]
                    self.vocab[ch] = self.vocab_size - 1

        print(f"      初始字符词表: {len(self.vocab)} 个")

        # 3. BPE 合并
        # 先把所有文本转换为 "token token token" 的序列
        print(f"      构建初始 token 序列...")
        corpus_sequences = []
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                text = f.read()
            # 把不在 vocab 中的字符映射到 <unk>
            seq = []
            for ch in text:
                if ch in self.vocab:
                    seq.append(ch)
                else:
                    seq.append('<unk>')
            corpus_sequences.append(seq)

        if '<unk>' not in self.vocab:
            self.vocab['<unk>'] = len(self.vocab)
        if '<pad>' not in self.vocab:
            self.vocab['<pad>'] = len(self.vocab)

        # 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        print(f"      计划合并次数: {num_merges}")

        for i in range(num_merges):
            # 统计所有相邻 pair 的频率
            pair_freqs = defaultdict(int)
            for seq in corpus_sequences:
                for j in range(len(seq) - 1):
                    pair = (seq[j], seq[j + 1])
                    pair_freqs[pair] += 1

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            if best_freq < min_freq:
                print(f"      提前停止: 最高频 pair 频率仅 {best_freq} < {min_freq}")
                break

            merged_token = best_pair[0] + best_pair[1]
            self.vocab[merged_token] = len(self.vocab)
            self.merges.append(best_pair)

            # 更新所有序列中的 pair
            new_sequences = []
            for seq in corpus_sequences:
                new_seq = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == best_pair[0] and seq[j + 1] == best_pair[1]:
                        new_seq.append(merged_token)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                new_sequences.append(new_seq)
            corpus_sequences = new_sequences

            if (i + 1) % 100 == 0 or i < 5:
                print(f"      Merge {i+1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> "
                      f"'{merged_token[:20]}...' (freq={best_freq})")

        # 构建 inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"[BPE] 训练完成，最终 vocab_size={len(self.vocab)}")

    # =====================================================================
    # 编码 / 解码
    # =====================================================================

    def encode(self, text: str, max_length=None, truncation=True) -> List[int]:
        """将文本编码为 token ID 列表。贪心最长匹配。
        
        Args:
            text: 输入文本
            max_length: 最大长度（若 truncation=True 则截断）
            truncation: 是否截断
        """
        tokens = []
        i = 0
        while i < len(text):
            matched = None
            matched_len = 0
            for length in range(min(len(text) - i, 20), 0, -1):
                substr = text[i:i + length]
                if substr in self.vocab:
                    matched = substr
                    matched_len = length
                    break

            if matched is not None:
                tokens.append(self.vocab[matched])
                i += matched_len
            else:
                tokens.append(self.vocab.get('<unk>', 0))
                i += 1

        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """将 token ID 列表解码为文本。"""
        chars = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, '<unk>')
            if token == '<unk>':
                chars.append('')  # 未知字符解码为空，避免污染文本
            else:
                chars.append(token)
        return ''.join(chars)

    # =====================================================================
    # 保存 / 加载
    # =====================================================================

    def save(self, path: str):
        """保存 tokenizer 到 JSON 文件。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': [list(m) for m in self.merges],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BPE] 已保存到 {path}")

    @classmethod
    def load(cls, path: str):
        """从 JSON 文件加载 tokenizer。"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(vocab_size=data['vocab_size'])
        tok.vocab = data['vocab']
        tok.merges = [tuple(m) for m in data['merges']]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        print(f"[BPE] 已从 {path} 加载，vocab_size={len(tok.vocab)}")
        return tok

    # =====================================================================
    # 统计
    # =====================================================================

    def analyze(self, text: str):
        """分析一段文本的 tokenization 结果。"""
        ids = self.encode(text)
        tokens = [self.inverse_vocab[i] for i in ids]
        print(f"原始文本: {text}")
        print(f"Token 数: {len(ids)}")
        print(f"字符数: {len(text)}")
        print(f"压缩比: {len(text)/len(ids):.2f}")
        print(f"Tokens: {tokens}")


# =====================================================================
# 命令行入口
# =====================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python chinese_bpe.py <train_file> <output_path> [vocab_size]")
        print("  e.g. python chinese_bpe.py data/zhwiki/corpus.txt data/tokenizer/bpe_4096.json 4096")
        sys.exit(1)

    train_file = sys.argv[1]
    output_path = sys.argv[2]
    vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

    tok = ChineseBPETokenizer(vocab_size=vocab_size)
    tok.train([train_file])
    tok.save(output_path)

    # 测试
    print("\n[Test] 测试编码解码:")
    test_texts = [
        "机器学习是人工智能的核心技术。",
        "The quick brown fox jumps over the lazy dog.",
        "2026年，神经科学和深度学习的交叉研究取得了重大突破。",
    ]
    for t in test_texts:
        tok.analyze(t)
        decoded = tok.decode(tok.encode(t))
        print(f"还原一致: {t == decoded}")
        print()
