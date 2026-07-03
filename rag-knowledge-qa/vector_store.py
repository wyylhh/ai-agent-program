"""
向量数据库：基于 ChromaDB 的存储与检索封装
"""
from __future__ import annotations
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import CHROMA_PERSIST_DIR

_client = None  # chromadb.PersistentClient
_collection = None  # chromadb.Collection
COLLECTION_NAME = "knowledge_base"


def get_client() -> chromadb.PersistentClient:
    """获取 ChromaDB 客户端"""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    """获取/创建集合"""
    global _collection
    if _collection is None:
        client = get_client()
        # 如果集合已存在则获取，否则创建
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = client.create_collection(COLLECTION_NAME)
    return _collection


def add_documents(texts: list[str], embeddings: list[list[float]],
                  metadatas: list[dict] | None = None):
    """批量存入文档向量"""
    col = get_collection()
    ids = [f"doc_{i}" for i in range(col.count(), col.count() + len(texts))]
    if metadatas is None:
        metadatas = [{"source": "unknown"} for _ in texts]
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def search(query_embedding: list[float], top_k: int = 4) -> list[dict]:
    """相似度检索，返回 top_k 条最相关文档"""
    col = get_collection()
    results = col.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id":       results["ids"][0][i],
            "content":  results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    1.0 - (results["distances"][0][i] if results.get("distances") else 0),
        })
    return docs


def clear_collection():
    """清空向量库"""
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    global _collection
    _collection = None


def get_doc_count() -> int:
    """当前已索引的文档块数量"""
    return get_collection().count()
