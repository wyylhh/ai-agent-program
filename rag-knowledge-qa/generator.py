"""
生成器：调用 LLM 基于检索到的上下文生成回答
"""
from openai import OpenAI
from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL, TEMPERATURE


def get_llm_client() -> OpenAI:
    """获取 LLM 客户端（兼容任何 OpenAI 兼容 API）"""
    return OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)


SYSTEM_PROMPT = """你是一个基于知识的问答助手。请严格根据以下【参考资料】回答用户问题。

规则：
1. 如果参考资料中有答案，请准确引用，并在括号中注明来源编号，如【1】
2. 如果参考资料不足以回答问题，请如实说"根据现有资料无法回答"，不要编造
3. 回答简洁、条理清晰，用中文回复"""


def generate(query: str, context: str,
             history: list[dict] | None = None,
             model: str = LLM_MODEL) -> str:
    """
    基于上下文和对话历史生成回答

    Args:
        query: 用户当前问题
        context: 检索到的参考资料
        history: 最近 N 轮对话 [{"role": "user/assistant", "content": "..."}]
        model: 模型名称
    """
    client = get_llm_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 加入最近对话历史
    if history:
        messages.extend(history[-10:])

    # 拼接上下文与问题
    user_msg = f"{context}\n\n【用户问题】{query}"
    messages.append({"role": "user", "content": user_msg})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def generate_without_rag(query: str, model: str = LLM_MODEL) -> str:
    """直接调用 LLM（无 RAG），用于对比实验"""
    client = get_llm_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=TEMPERATURE,
        max_tokens=1024,
    )
    return resp.choices[0].message.content
