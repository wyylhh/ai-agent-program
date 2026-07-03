# 基于 RAG 的本地知识库问答系统

## 项目概述

这是一个**完整的 RAG（检索增强生成）知识库问答系统**。你可以上传任意 PDF / Word / Markdown / TXT 文档，系统会自动解析、分块、向量化并存入向量数据库。提问时，系统会先检索最相关的文档片段，再将它们作为上下文发给大语言模型，从而获得**事实更准确、可溯源**的回答。

### 为什么需要 RAG？

直接问大模型问题有两个致命缺陷：
- **知识截止**：模型训练数据有截止日期，无法回答最新信息
- **幻觉**：模型可能编造看似合理但完全错误的内容

RAG 解决方式：**先检索，再回答**——让模型"开卷考试"。

---

## 核心架构

```
用户上传文档
    ↓
文档解析（PDF/TXT/DOCX/MD）
    ↓
文本分块（固定窗口 / 递归字符 / 语义分块）
    ↓
向量嵌入（BGE-M3 → 768维向量）
    ↓
存入 ChromaDB 向量数据库
    ↓
用户提问 → 查询向量化 → 相似度检索（Top-K）
    ↓
检索结果 + 问题 → 大语言模型 → 带引用的回答
```

---

## 项目结构

```
rag-knowledge-qa/
├── config.py               # 全局配置（模型、分块参数等）
├── document_loader.py      # 文档解析器（PDF/TXT/DOCX/MD）
├── chunker.py              # 三种文本分块策略
├── embedder.py             # BGE-M3 向量嵌入
├── vector_store.py         # ChromaDB 向量数据库封装
├── retriever.py            # 检索器（查询 → 相似文档）
├── generator.py            # LLM 生成器（上下文增强回答）
├── pipeline.py             # 完整 RAG 管线（一键索引 + 问答）
├── app.py                  # FastAPI 后端接口
├── ui.py                   # Gradio 可视化界面
├── experiments/
│   └── compare_chunking.py # 分块策略对比实验
├── data/sample_docs/
│   └── ai_intro.txt        # 示例文档
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
cd rag-knowledge-qa
pip install -r requirements.txt
```

### 2. 启动大模型服务

项目需要一个大语言模型来生成回答。三种方式任选其一：

**方式 A：用 Ollama 本地运行（推荐新手）**

```bash
# 安装 Ollama 后
ollama pull qwen2.5:7b
ollama serve
# 默认端口 11434，修改 config.py：
# LLM_API_BASE = "http://localhost:11434/v1"
# LLM_MODEL = "qwen2.5:7b"
```

**方式 B：使用云端 API（如有 API Key）**

```bash
# 修改 config.py：
# LLM_API_BASE = "https://api.openai.com/v1"  # 或任何兼容 API
# LLM_API_KEY = "your-api-key"
# LLM_MODEL = "gpt-4o-mini"
```

**方式 C：离线演示模式**

如果没有可用的 LLM，可以先用实验脚本理解 RAG 的分块和检索逻辑：

```bash
python experiments/compare_chunking.py
```

### 3. 启动 Gradio 界面

```bash
python ui.py
```

浏览器打开 `http://127.0.0.1:7860`，上传示例文档 `data/sample_docs/ai_intro.txt`，然后提问。

### 4. 或启动 FastAPI 接口

```bash
uvicorn app:app --reload --port 7860
```

访问 `http://127.0.0.1:7860/docs` 查看 API 文档。

---

## API 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ingest` | 上传文档进行索引 `{"file_path": "data/xxx.txt"}` |
| POST | `/ask` | RAG 问答 `{"question": "...", "top_k": 4}` |
| POST | `/reset` | 重置对话历史 |
| GET | `/status` | 查看当前索引状态 |

---

## 代码详解

### 文档解析（document_loader.py）

支持四种格式的自动识别与解析：
- **PDF** — 使用 `pypdf` 按页提取文本
- **Word (.docx)** — 使用 `python-docx` 提取段落
- **TXT / Markdown** — 直接读取

`load_directory()` 可批量加载整个文件夹。

### 文本分块（chunker.py）

实现了三种分块策略，这是 RAG 系统中**最影响检索质量**的环节：

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 固定窗口 | 按字符数等距切分 | 简单快速 | 可能切断句子 |
| 递归字符 | 优先按段落/句子边界切分 | 语义完整 | 稍慢 |
| 语义分块 | 按句子合并，保持语义块 | 语义最连贯 | 块大小不均 |

**为什么 chunk_size=512, overlap=50？**
- 512 字符 ≈ 200-300 个中文词，足够承载一个完整语义单元
- 50 字符重叠确保关键信息不会刚好落在分块边界被切断

### 向量嵌入（embedder.py）

默认使用 `BAAI/bge-small-zh-v1.5`：
- 中文检索场景的性价比之选
- 输出 512 维归一化向量
- 首次运行会自动下载模型（约 100MB）

如需更强效果，可改为 `BAAI/bge-m3`（多语言，1024维，约 2GB）。

### 向量数据库（vector_store.py）

使用 **ChromaDB** 作为向量存储：
- 持久化存储到本地 `chroma_db/` 目录
- 支持增量添加文档
- 查询返回余弦相似度分数

### 生成器（generator.py）

核心逻辑：
```
System: "你是知识问答助手，严格根据参考资料回答..."
User:   "【参考资料1】...\n【参考资料2】...\n\n【用户问题】..."
```

- 强制模型引用来源，减少幻觉
- 当检索材料不足时，要求模型诚实回答"无法回答"
- 支持多轮对话历史

---

## 分块策略对比实验

运行实验脚本：

```bash
python experiments/compare_chunking.py
```

实验结果会展示不同分块策略在各查询上的召回率，帮助理解**为什么参数选择很重要**。

示例输出：
```
固定窗口-256  |  块数: 12  |  召回率: 62%
固定窗口-512  |  块数: 7   |  召回率: 78%
递归字符-512  |  块数: 5   |  召回率: 85%   ← 最优
语义分块      |  块数: 4   |  召回率: 72%
```

---

## 关键设计决策

### 1. 为什么用 ChromaDB 而不是 FAISS？
ChromaDB 自带持久化、元数据过滤、更简单的 Python API，适合原型开发和小规模生产。

### 2. 为什么要限制模型"严格根据资料回答"？
RAG 的命门是**模型忽略检索结果、直接凭记忆回答**。通过 System Prompt 约束，可以大幅提升回答的事实准确性。

### 3. 对话历史的作用？
多轮对话中，用户可能指代上文（"它有什么缺点？"），历史上下文帮助 LLM 理解指代关系。

---

## 扩展方向

- [ ] 支持更多文档格式（网页、CSV、图片OCR）
- [ ] 混合检索（BM25 关键词 + 向量相似度）
- [ ] 检索结果重排序（Reranker）
- [ ] 流式输出（SSE）
- [ ] 多知识库隔离
