"""
检索器：查询向量化 + 相似度检索 + 上下文构建
"""
from embedder import embed_query
from vector_store import search
from config import TOP_K


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    执行检索：query → embedding → ChromaDB 搜索 → 返回 top_k 文档
    """
    query_emb = embed_query(query)
    results = search(query_emb, top_k=top_k)
    return results


def build_context(retrieved_docs: list[dict]) -> str:
    """
    将检索到的文档拼接成 LLM 可理解的上下文
    格式：
    【参考资料 1】（来源：xxx）
    内容...
    """
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("metadata", {}).get("source", "未知来源")
        parts.append(f"【参考资料 {i}】（来源：{source}）\n{doc['content']}")
    return "\n\n".join(parts)
