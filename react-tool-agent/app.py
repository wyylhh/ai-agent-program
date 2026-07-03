"""
FastAPI 后端：ReAct Agent API
启动方式：uvicorn app:app --reload --port 7861
"""
from fastapi import FastAPI
from pydantic import BaseModel
from agent import ReActAgent

app = FastAPI(title="ReAct Agent 工具调用系统", version="1.0")
agent = ReActAgent()


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    query: str
    answer: str
    iterations: int
    tool_calls: list[dict]
    trajectory: list[dict]


@app.post("/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
    """向 Agent 提问"""
    result = agent.run(req.query)
    return AskResponse(**result)


@app.get("/tools")
def api_list_tools():
    """列出所有可用工具"""
    return [
        {"name": t.name, "description": t.description}
        for t in agent.tools.list_tools()
    ]


@app.get("/health")
def api_health():
    return {"status": "ok", "model": agent.model}
