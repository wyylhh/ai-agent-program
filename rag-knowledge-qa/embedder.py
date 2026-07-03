"""
向量嵌入器：将文本转为向量

支持两种模式（通过 config.py 中 EMBED_BACKEND 切换）:
- "sklearn": TfidfVectorizer（离线，无需下载，立即可用）
- "bge":    BGE 模型（需联网下载，质量更高）
"""
import os
import numpy as np
from config import EMBED_MODEL_NAME, EMBED_BACKEND

_embedder = None
_embed_dim = 0


def _init_sklearn():
    """使用 sklearn TfidfVectorizer，完全离线"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=512,
    )


def _init_bge():
    """使用 BGE 模型，需联网下载"""
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_embedder():
    """获取嵌入模型（懒加载）"""
    global _embedder, _embed_dim
    if _embedder is None:
        print(f"[Embedder] 初始化后端: {EMBED_BACKEND}")
        if EMBED_BACKEND == "bge":
            print(f"[Embedder] 正在下载模型 {EMBED_MODEL_NAME} ...")
            _embedder = _init_bge()
            _embed_dim = _embedder.get_sentence_embedding_dim()
        else:
            _embedder = _init_sklearn()
            _embed_dim = _embedder.max_features
        print(f"[Embedder] 就绪，向量维度: {_embed_dim}")
    return _embedder


def get_embedding_dim() -> int:
    """获取当前嵌入维度"""
    if _embedder is None:
        get_embedder()
    return _embed_dim


def embed_texts(texts: list[str]) -> list[list[float]]:
    """将文本列表转为向量列表"""
    model = get_embedder()
    if EMBED_BACKEND == "bge":
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()
    else:
        # sklearn: 先 fit 再 transform
        embeddings = model.fit_transform(texts).toarray()
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """将查询文本转为向量"""
    model = get_embedder()
    if EMBED_BACKEND == "bge":
        embedding = model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
    else:
        # sklearn: transform 单个查询
        vec = model.transform([query]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
