"""
Gradio 可视化界面：在浏览器中直接使用 RAG 问答
启动方式：python ui.py
"""
import gradio as gr
from pipeline import RAGPipeline

pipeline = RAGPipeline()


def ingest_and_respond(file_obj, question, top_k):
    """处理文件上传 + 提问"""
    if file_obj is not None:
        count = pipeline.ingest_file(file_obj.name)
        ingest_msg = f"已索引 {count} 个文本块\n"
    else:
        ingest_msg = ""

    if not question.strip():
        return ingest_msg or "请输入问题", "", ""

    result = pipeline.ask(question, top_k=int(top_k))
    refs = "\n\n".join(
        f"【{i+1}】来源: {d['metadata'].get('source', '?')}\n{d['content'][:300]}..."
        for i, d in enumerate(result["retrieved_docs"])
    )
    return ingest_msg + result["answer"], refs, result["context"]


with gr.Blocks(title="RAG 知识库问答系统") as demo:
    gr.Markdown("# RAG 知识库问答系统")
    gr.Markdown("上传文档 → 自动索引 → 提问获取答案")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传文档 (PDF/TXT/MD/DOCX)", file_types=[".pdf", ".txt", ".md", ".docx"])
            question = gr.Textbox(label="输入问题", placeholder="例如：什么是机器学习？", lines=2)
            top_k = gr.Slider(1, 10, value=4, step=1, label="检索文档数 (top_k)")
            btn = gr.Button("提问", variant="primary")
            reset_btn = gr.Button("重置对话历史")

        with gr.Column(scale=2):
            answer = gr.Textbox(label="回答", lines=8)
            with gr.Accordion("检索到的参考资料", open=False):
                references = gr.Textbox(label="参考资料", lines=10)
            with gr.Accordion("完整上下文 (发送给 LLM 的内容)", open=False):
                full_context = gr.Textbox(label="上下文", lines=10)

    btn.click(ingest_and_respond, [file_input, question, top_k], [answer, references, full_context])
    reset_btn.click(pipeline.reset_history)

demo.launch(server_name="127.0.0.1", server_port=7860)
