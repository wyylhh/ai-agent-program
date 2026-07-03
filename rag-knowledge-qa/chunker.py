"""
文本分块器：三种策略实现（纯 Python，零依赖）
"""
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def fixed_window_split(text: str, chunk_size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    固定窗口分块 — 最基础策略
    原理：按固定字符数滑动窗口切分
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def _split_by_separator(text: str, separator: str) -> list[str]:
    """按分隔符切分，保留分隔符"""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _merge_splits(splits: list[str], chunk_size: int, overlap: int) -> list[str]:
    """将小片段合并成不超过 chunk_size 的块，带重叠"""
    chunks = []
    current = ""
    for split in splits:
        if len(current) + len(split) <= chunk_size:
            current += split
        else:
            if current:
                chunks.append(current)
            # 重叠：从上一块的末尾取 overlap 字符作为新块的开头
            if overlap > 0 and chunks:
                prev = chunks[-1]
                current = prev[-overlap:] if len(prev) > overlap else prev
                current += split
            else:
                current = split
    if current:
        chunks.append(current)
    # 去重：如果前后块内容相同，保留一个
    result = []
    for c in chunks:
        if not result or c != result[-1]:
            result.append(c)
    return result


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    递归字符分块 — 生产推荐策略
    原理：优先按段落(\\n\\n) → 换行 → 句子(。！？) → 逗号 → 空格 → 字符
         逐级切分，尽可能保证语义完整性
    """
    splits = [text]
    for sep in _SEPARATORS:
        new_splits = []
        for s in splits:
            if len(s) <= chunk_size:
                new_splits.append(s)
            else:
                new_splits.extend(_split_by_separator(s, sep))
        splits = new_splits
    return _merge_splits(splits, chunk_size, overlap)


def semantic_split(text: str, min_sentences: int = 5,
                   max_sentences: int = 15) -> list[str]:
    """
    语义分块（简易版）— 按句子边界合并
    原理：先按句号/问号/感叹号切句子，再将相邻句子合并成块，
         尽量让每块的句子数在 [min, max] 范围内，保持语义完整
    """
    sentences = re.split(r'(?<=[。！？\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    i = 0
    while i < len(sentences):
        block = []
        char_count = 0
        while i < len(sentences) and len(block) < max_sentences:
            s = sentences[i]
            # 如果当前块还小，继续加；否则开启新块
            if len(block) < min_sentences or char_count + len(s) < CHUNK_SIZE * 2:
                block.append(s)
                char_count += len(s)
                i += 1
            else:
                break
        if block:
            chunks.append(" ".join(block))
    return chunks


STRATEGIES = {
    "fixed":    fixed_window_split,
    "recursive": recursive_split,
    "semantic": semantic_split,
}
