"""
FastAPI 后端：提供 RESTful API 接口
启动方式：uvicorn app:app --reload --port 7860
"""
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import RAGPipeline

app = FastAPI(title="RAG 知识库问答系统", version="1.0")
pipeline = RAGPipeline()


class IngestRequest(BaseModel):
    file_path: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


class AskResponse(BaseModel):
    question: str
    answer: str
    references: list[dict]


@app.post("/ingest")
def api_ingest(req: IngestRequest):
    """上传文档进行索引"""
    count = pipeline.ingest_file(req.file_path)
    return {"status": "ok", "chunks_indexed": count}


@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
    """RAG 问答"""
    result = pipeline.ask(req.question, top_k=req.top_k)
    refs = [
        {"source": d["metadata"].get("source", ""), "snippet": d["content"][:200]}
        for d in result["retrieved_docs"]
    ]
    return AskResponse(question=result["question"], answer=result["answer"], references=refs)


@app.post("/reset")
def api_reset():
    """重置对话历史"""
    pipeline.reset_history()
    return {"status": "ok"}


@app.get("/status")
def api_status():
    """查看系统状态"""
    from vector_store import get_doc_count
    return {
        "indexed_chunks": get_doc_count(),
        "history_turns": len(pipeline.history) // 2,
    }
