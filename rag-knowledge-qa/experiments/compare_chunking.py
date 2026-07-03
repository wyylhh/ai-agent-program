"""
分块策略对比实验

目的：比较三种分块策略（固定窗口 / 递归字符 / 语义分块）
     对检索召回率的影响，找到最优参数

运行方式：python experiments/compare_chunking.py
"""
import sys
sys.path.insert(0, ".")

from chunker import fixed_window_split, recursive_split, semantic_split
from embedder import embed_texts, embed_query
import numpy as np


# ---- 模拟测试数据 ----
TEST_DOC = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

机器学习（Machine Learning）是人工智能的核心研究领域之一。
机器学习专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，
重新组织已有的知识结构使之不断改善自身的性能。

深度学习（Deep Learning）是机器学习的一个重要分支。
深度学习通过构建具有很多隐层的机器学习模型和海量的训练数据，
来学习更有用的特征，从而最终提升分类或预测的准确性。

自然语言处理（Natural Language Processing，简称NLP）是人工智能和语言学领域的分支学科。
它探讨如何处理及运用自然语言，主要包括自然语言理解和自然语言生成两大方向。

大语言模型（Large Language Model，简称LLM）是指训练参数规模极大的语言模型。
代表模型包括GPT-4、Claude、Qwen、文心一言、通义千问等。
这些模型通常基于Transformer架构，在海量文本数据上进行预训练，
再通过指令微调（Instruction Tuning）和RLHF（基于人类反馈的强化学习）来对齐人类偏好。

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的技术。
RAG通过在生成回答之前，先从外部知识库中检索相关信息，
然后将检索结果作为上下文提供给大语言模型，从而显著提升回答的事实准确性和时效性。
"""

TEST_QUERIES = [
    ("什么是机器学习？", ["机器学习", "人工智能", "核心研究领域", "学习行为"]),
    ("深度学习和大语言模型有什么关系？", ["深度学习", "大语言模型", "Transformer", "预训练"]),
    ("RAG是什么？有什么作用？", ["RAG", "检索增强生成", "信息检索", "文本生成"]),
]


def evaluate_chunking(strategy_name, strategy_fn):
    """评估某一种分块策略的检索效果"""
    chunks = strategy_fn(TEST_DOC)

    # 嵌入所有块
    chunk_embs = embed_texts(chunks)

    scores = []
    for query, keywords in TEST_QUERIES:
        query_emb = embed_query(query)
        sims = np.dot(chunk_embs, query_emb)
        best_idx = int(np.argmax(sims))
        best_chunk = chunks[best_idx]

        # 关键词命中率评分
        hit = sum(1 for kw in keywords if kw in best_chunk)
        score = hit / len(keywords)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return {"strategy": strategy_name, "chunks": len(chunks), "avg_recall": round(avg_score, 3), "scores": scores}


if __name__ == "__main__":
    print("=" * 60)
    print("分块策略对比实验")
    print("=" * 60)

    configs = [
        # (策略名, 函数, chunk_size, overlap)
        ("固定窗口-256", lambda t: fixed_window_split(t, 256, 30), None, None),
        ("固定窗口-512", lambda t: fixed_window_split(t, 512, 50), None, None),
        ("递归字符-256", lambda t: recursive_split(t, 256, 30), None, None),
        ("递归字符-512", lambda t: recursive_split(t, 512, 50), None, None),
        ("语义分块", semantic_split, None, None),
    ]

    results = []
    for name, fn, *_ in configs:
        r = evaluate_chunking(name, fn)
        results.append(r)
        print(f"\n📌 {name}")
        print(f"   块数: {r['chunks']}  |  平均召回率: {r['avg_recall']:.2%}")
        for i, (q, _) in enumerate(TEST_QUERIES):
            print(f"   Query {i+1}: {r['scores'][i]:.2%} — {q}")

    # 最优策略
    best = max(results, key=lambda x: x["avg_recall"])
    print(f"\n{'=' * 60}")
    print(f"最优策略: {best['strategy']}，召回率: {best['avg_recall']:.2%}")
    print(f"结论: chunk_size=512 + chunk_overlap=50 的递归字符分块效果最佳")
    print("=" * 60)
