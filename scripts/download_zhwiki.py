"""
下载中文维基百科最新 dump 并提取纯文本。

Usage:
    python scripts/download_zhwiki.py

输出：
    data/zhwiki/raw/          -- 下载的原始 XML
    data/zhwiki/extracted/    -- 提取的纯文本 .txt 文件
    data/zhwiki/corpus.txt    -- 合并后的总语料
"""

import os
import sys
import bz2
import re
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

# =============================================================================
# 配置
# =============================================================================
RAW_DIR = Path("data/zhwiki/raw")
EXTRACT_DIR = Path("data/zhwiki/extracted")
CORPUS_PATH = Path("data/zhwiki/corpus.txt")
VAL_PATH = Path("data/zhwiki/val.txt")

# 中文维基百科 dump 地址
# 如果官方地址慢，可以换成镜像
DUMP_URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
# 备选镜像（国内可用时）：
# DUMP_URL = "https://mirror.bjtu.edu.cn/wikimedia-dumps/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"

CHUNK_SIZE = 1024 * 1024  # 1MB


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "zhwiki-latest-pages-articles.xml.bz2"

    if raw_path.exists():
        size_mb = raw_path.stat().st_size / (1024 * 1024)
        print(f"[Download] 文件已存在: {raw_path} ({size_mb:.1f} MB)")
        print("           如需重新下载，请删除该文件")
        return raw_path

    print(f"[Download] 正在下载中文维基百科 dump...")
    print(f"           URL: {DUMP_URL}")
    print(f"           目标: {raw_path}")
    print(f"           预计大小: ~2-3 GB，可能需要 10-30 分钟")

    try:
        req = urllib.request.Request(DUMP_URL, headers={
            'User-Agent': 'CORTEX-Training/1.0'
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            with open(raw_path, 'wb') as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r           {downloaded/1024/1024:.1f} / {total/1024/1024:.1f} MB ({pct:.1f}%)", end='')
                    else:
                        print(f"\r           {downloaded/1024/1024:.1f} MB downloaded", end='')
        print()
        print(f"[Download] 完成: {raw_path.stat().st_size / 1024 / 1024:.1f} MB")
        return raw_path
    except Exception as e:
        print(f"\n[Download] 失败: {e}")
        if raw_path.exists():
            raw_path.unlink()
        sys.exit(1)


def clean_text(text: str) -> str:
    """清理维基百科文本。"""
    # 移除模板 {{...}}
    text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    # 移除引用 <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除 URL
    text = re.sub(r'https?://\S+', '', text)
    # 移除文件/图片标记
    text = re.sub(r'\[\[File:.*?\]\]', '', text)
    text = re.sub(r'\[\[Image:.*?\]\]', '', text)
    # 处理内部链接 [[目标|显示文本]] -> 显示文本
    text = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    # 移除强调标记
    text = re.sub(r"''+'", '', text)
    # 合并多个空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除行首空格
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    return text.strip()


def extract_articles(xml_bz2_path: Path):
    """从 bz2 压缩的 XML 中提取文章纯文本。"""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Extract] 正在解压并提取文本...")
    print(f"          源文件: {xml_bz2_path}")

    articles = []
    total_articles = 0
    total_chars = 0

    # 逐块解压，避免内存爆炸
    with bz2.open(xml_bz2_path, 'rb') as f:
        # MediaWiki XML 的 namespace
        ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}

        # 由于 XML 很大，我们用 iterparse 流式处理
        context = ET.iterparse(f, events=('end',))
        context = iter(context)
        event, root = next(context)

        for event, elem in context:
            if elem.tag.endswith('page'):
                title_elem = elem.find('mw:title', ns)
                text_elem = elem.find('.//mw:text', ns)

                if title_elem is not None and text_elem is not None:
                    title = title_elem.text or ''
                    raw_text = text_elem.text or ''

                    # 跳过重定向、消歧义页、列表页
                    if (raw_text.startswith('#REDIRECT') or
                        raw_text.startswith('#重定向') or
                        '消歧义' in title or
                        '列表' in title[-10:] or
                        title.startswith('Wikipedia:') or
                        title.startswith('File:') or
                        title.startswith('Template:') or
                        title.startswith('Category:') or
                        title.startswith('Help:')):
                        elem.clear()
                        root.clear()
                        continue

                    cleaned = clean_text(raw_text)
                    if len(cleaned) > 100:  # 只保留有实质内容的
                        articles.append((title, cleaned))
                        total_articles += 1
                        total_chars += len(cleaned)

                elem.clear()
                root.clear()

                if total_articles % 1000 == 0:
                    print(f"\r          已提取 {total_articles} 篇文章, "
                          f"{total_chars/1024/1024:.1f} MB 文本", end='')

    print(f"\n[Extract] 完成: {total_articles} 篇文章, {total_chars/1024/1024:.1f} MB")
    return articles


def save_corpus(articles):
    """保存为训练语料，按 9:1 切分 train/val。"""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # 按文章保存（方便调试）
    for i, (title, text) in enumerate(articles):
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)[:50]
        path = EXTRACT_DIR / f"{i:06d}_{safe_title}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(text)
            f.write('\n\n')

    # 合并为总语料（shuffle 后切分）
    import random
    random.seed(42)
    random.shuffle(articles)

    split_idx = int(len(articles) * 0.95)
    train_articles = articles[:split_idx]
    val_articles = articles[split_idx:]

    def merge(arts, path):
        with open(path, 'w', encoding='utf-8') as f:
            for title, text in arts:
                f.write(f"{title}\n{text}\n\n")

    merge(train_articles, CORPUS_PATH)
    merge(val_articles, VAL_PATH)

    train_chars = sum(len(t) for _, t in train_articles)
    val_chars = sum(len(t) for _, t in val_articles)

    print(f"[Save] Train: {CORPUS_PATH} ({len(train_articles)} 篇, {train_chars/1024/1024:.1f} MB)")
    print(f"[Save] Val:   {VAL_PATH} ({len(val_articles)} 篇, {val_chars/1024/1024:.1f} MB)")


def main():
    print("=" * 60)
    print("中文维基百科语料准备工具")
    print("=" * 60)

    raw_path = download()
    articles = extract_articles(raw_path)

    if not articles:
        print("[Error] 没有提取到任何文章")
        sys.exit(1)

    save_corpus(articles)

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"训练语料: {CORPUS_PATH}")
    print(f"验证语料: {VAL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
