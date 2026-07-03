"""
Gradio 可视化界面：在浏览器中与 ReAct Agent 交互
启动方式：python ui.py
"""
import gradio as gr
from agent import ReActAgent

agent = ReActAgent()


def chat(query, history):
    if not query.strip():
        return "", history
    result = agent.run(query)

    # 构建可读的回答
    answer = f"### 最终回答\n{result['answer']}\n\n"
    answer += f"---\n### 推理过程（共 {result['iterations']} 步）\n\n"

    for step in result["trajectory"]:
        answer += f"**第 {step['iteration']} 步**\n"
        answer += f"> Thought: {step.get('thought', '')[:200]}\n\n"
        if step.get("action") and step["action"] != "Final Answer":
            answer += f"Action: `{step['action']}`\n\n"
            obs = step.get("observation", "")[:500]
            answer += f"Observation: {obs}\n\n"
            answer += "---\n"

    history.append((query, answer))
    return "", history


with gr.Blocks(title="ReAct Agent 工具调用系统") as demo:
    gr.Markdown("""# ReAct Agent 智能助手

    这是一个基于 **ReAct（Reasoning + Acting）范式**的 AI Agent。
    它能自主选择工具来完成任务：

    - **搜索引擎** — 查找实时信息
    - **计算器** — 精确数值计算
    - **代码解释器** — 执行 Python 代码
    - **天气查询** — 查询全球城市天气

    **试试这些例子**:
    - "北京和上海今天哪个更热？"
    - "计算 1 到 100 的质数之和"
    - "搜索最新的 AI 新闻，然后总结三条"
    """)

    chatbot = gr.Chatbot(label="对话")
    msg = gr.Textbox(label="输入你的问题", placeholder="例如：搜索最新的GPT-5消息，并总结关键信息")
    clear = gr.Button("清除对话")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], []), None, [msg, chatbot])

demo.launch(server_name="127.0.0.1", server_port=7861)
