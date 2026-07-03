"""
RAG 完整管线：文档加载 → 分块 → 嵌入 → 存储 → 检索 → 生成
"""
from document_loader import load_document, load_directory
from chunker import recursive_split
from embedder import embed_texts
from vector_store import add_documents, clear_collection, get_doc_count
from retriever import retrieve, build_context
from generator import generate
from config import CHUNK_SIZE, CHUNK_OVERLAP


class RAGPipeline:
    """
    RAG 管线，封装完整的文档索引与问答流程

    使用示例:
        pipeline = RAGPipeline()
        pipeline.ingest_file("data/sample_docs/ai_intro.txt")
        answer = pipeline.ask("什么是机器学习？")
    """

    def __init__(self):
        self.history: list[dict] = []  # 对话历史

    # ---------- 文档索引 ----------

    def ingest_file(self, file_path: str) -> int:
        """索引单个文件，返回索引的块数"""
        content = load_document(file_path)
        chunks = recursive_split(content, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print("[RAG] 文档为空，跳过")
            return 0
        embeddings = embed_texts(chunks)
        metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
        add_documents(chunks, embeddings, metadatas)
        print(f"[RAG] 已索引 {len(chunks)} 个文本块 (来源: {file_path})")
        return len(chunks)

    def ingest_directory(self, dir_path: str) -> int:
        """批量索引目录，返回总块数"""
        docs = load_directory(dir_path)
        total = 0
        for doc in docs:
            chunks = recursive_split(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                continue
            embeddings = embed_texts(chunks)
            metadatas = [{"source": doc["name"], "chunk_index": i} for i in range(len(chunks))]
            add_documents(chunks, embeddings, metadatas)
            total += len(chunks)
            print(f"[RAG] {doc['name']}: {len(chunks)} 块")
        print(f"[RAG] 总计索引 {total} 块，当前库总量: {get_doc_count()}")
        return total

    # ---------- 问答 ----------

    def ask(self, question: str, top_k: int = 4) -> dict:
        """执行一次 RAG 问答，返回完整结果"""
        # 1. 检索
        docs = retrieve(question, top_k=top_k)
        # 2. 构建上下文
        context = build_context(docs)
        # 3. 生成
        answer = generate(question, context, history=self.history)
        # 4. 更新对话历史
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": docs,
            "context": context,
        }

    def reset_history(self):
        """重置对话历史"""
        self.history = []

    def reset_knowledge_base(self):
        """清空向量库"""
        clear_collection()
        self.reset_history()
