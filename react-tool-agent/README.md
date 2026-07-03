# 基于 ReAct 范式的工具调用型 Agent

## 项目概述

这是一个**从零实现的 ReAct（Reasoning + Acting）AI Agent**。它能够在「思考 → 行动 → 观察」的循环中自主推理，根据用户需求智能选择和编排多个外部工具，完成复杂的多步任务。

与传统的"一问一答"不同，ReAct Agent 会在执行过程中**自我思考、调用工具、观察结果、调整计划**，就像人类解决复杂问题时的思维方式。

### 什么场景需要 Agent？

单次 LLM 调用的局限：
- 无法获取实时信息（知识截止）
- 无法进行精确计算
- 无法执行代码验证
- 面对复杂任务缺乏规划能力

Agent 的解决方案：**让 LLM 学会"使用工具"**——遇到不知道的就去查，需要算的就去算，不确定的就去验证。

---

## 核心架构

```
用户提问: "北京和东京今天哪个更热？温差多少？"
    ↓
┌─────────────────────────────────────────┐
│  ReAct 推理循环（最多 10 步）             │
│                                          │
│  Step 1: Thought → 需要查两个城市天气     │
│          Action → get_weather("Beijing") │
│          Observation → 北京: 28°C        │
│                                          │
│  Step 2: Thought → 再查东京              │
│          Action → get_weather("Tokyo")  │
│          Observation → 东京: 22°C        │
│                                          │
│  Step 3: Thought → 计算温差              │
│          Action → calculator("28-22")   │
│          Observation → 6°C              │
│                                          │
│  Step 4: Thought → 信息足够，输出答案     │
│          Final Answer → 北京更热，差6°C   │
└─────────────────────────────────────────┘
    ↓
返回: 完整推理轨迹 + 最终答案
```

---

## 项目结构

```
react-tool-agent/
├── agent.py                 # ReAct Agent 核心引擎（从零实现推理循环）
├── config.py                # 全局配置
├── tools/
│   ├── __init__.py          # 工具注册入口
│   ├── base.py              # 工具基类 + 注册中心（统一接口设计）
│   ├── search.py            # 搜索引擎工具（DuckDuckGo）
│   ├── calculator.py        # 数学计算器（安全沙箱）
│   ├── code_interpreter.py  # Python 代码解释器（沙箱执行）
│   └── weather.py           # 天气查询工具（免费 API）
├── app.py                   # FastAPI 后端接口
├── ui.py                    # Gradio 可视化界面
├── experiments/
│   └── compare_paradigms.py # Agent 范式对比实验
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
cd react-tool-agent
pip install -r requirements.txt
```

### 2. 启动大模型服务

同 RAG 项目，需要 LLM 服务。推荐 Ollama：

```bash
ollama pull qwen2.5:7b
ollama serve
```

修改 `config.py` 中的 API 地址：
```python
LLM_API_BASE = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:7b"
```

### 3. 启动 Gradio 界面

```bash
python ui.py
```

浏览器打开 `http://127.0.0.1:7861`，试试下面的例子。

### 4. 或启动 FastAPI 接口

```bash
uvicorn app:app --reload --port 7861
```

访问 `http://127.0.0.1:7861/docs` 查看 API 文档。

---

## API 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ask` | 向 Agent 提问 `{"query": "..."}` |
| GET | `/tools` | 列出所有可用工具 |
| GET | `/health` | 健康检查 |

---

## 推荐体验用例

向 Agent 发送以下问题，观察它的推理过程：

```
1. "北京今天天气怎么样？"
   → 演示单工具调用

2. "计算斐波那契数列第20项，然后判断它是不是质数"
   → 演示多工具编排

3. "搜索2024年诺贝尔文学奖得主，并查一下TA所在城市的天气"
   → 演示信息检索 + 条件查询

4. "用Python生成一个1到100的随机数列表，计算它们的平均值和标准差"
   → 演示代码解释器 + 计算器组合
```

---

## 代码详解

### 工具基类（tools/base.py）

所有工具继承 `BaseTool`，必须实现：
- `name`：工具名称（Agent 通过名称选择工具）
- `description`：工具描述（告诉 Agent 什么时候该用此工具）
- `parameters`：参数定义（JSON Schema 格式）
- `execute(**kwargs)`：核心执行逻辑

`ToolRegistry` 是工具注册中心，负责：
- 注册/管理所有工具
- 生成工具描述文本（注入 System Prompt）
- 生成 OpenAI Function Calling 格式

`BaseTool.run()` 是所有工具的统一入口，自动提供了：
- **超时保护**：防止工具卡死
- **自动重试**：失败时最多重试 N 次
- **结果统一**：返回标准化的 `ToolResult`

### ReAct Agent 核心（agent.py）

这是整个项目的心脏，实现了完整的 ReAct 推理循环：

```
while 未达到最大迭代次数:
    1. 调用 LLM，获取 Thought + Action
    2. 如果 LLM 输出 Final Answer → 结束
    3. 解析 Action 和 Action Input
    4. 调用对应工具，获取 Observation
    5. 将 Observation 追加到对话历史
    6. 循环回到第 1 步
```

关键设计：
- **System Prompt 注入**：教 LLM 理解 ReAct 格式和可用工具
- **正则解析**：从 LLM 自由文本中提取结构化信息
- **强制终止**：达到最大步数时强制 LLM 基于已有信息回答
- **完整轨迹**：每步的 Thought-Action-Observation 全部记录

### 四个内置工具

| 工具 | 功能 | 实现方式 |
|------|------|----------|
| web_search | 搜索互联网 | DuckDuckGo（离线模式可降级为模拟） |
| calculator | 数学计算 | 安全 eval + 函数白名单 |
| code_interpreter | 执行 Python 代码 | exec 沙箱 + stdout 捕获 |
| get_weather | 查询天气 | Open-Meteo 免费 API（无需 Key） |

### 异常处理与降级机制

Agent 中设计了多层容错：

1. **工具级别**：`BaseTool.run()` 内建重试逻辑
2. **Agent 级别**：当工具返回错误时，Observation 包含错误信息，LLM 可据此调整策略
3. **网络降级**：天气和搜索工具在网络不可用时会返回模拟数据
4. **最大步数保护**：防止 Agent 陷入死循环

---

## Agent 范式对比实验

运行实验脚本：

```bash
python experiments/compare_paradigms.py
```

此实验对比三种范式在相同多步推理任务上的表现：

| 范式 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **ReAct** | 思考→行动→观察 循环 | 可解释、可纠错、适合多步 | 较慢，需要多次 LLM 调用 |
| **Few-shot CoT** | 一次性推理链 | 快速、单次 LLM 调用 | 无法验证中间步骤 |
| **Function Calling** | LLM 自主决定工具调用 | 开发简单、框架原生支持 | 多步编排能力弱 |

---

## 关键设计决策

### 1. 为什么要从零实现 ReAct，而不直接用 LangChain Agent？

从零实现可以深入理解 Agent 的底层机制，而非仅仅调用框架 API。掌握原理后，使用任何框架（LangChain、LlamaIndex、CrewAI）都会事半功倍。

### 2. 为什么用文本解析而不是 Function Calling？

ReAct 标签格式（`Thought: ... Action: ...`）的优势是**完全透明的推理过程**。你可以看到 LLM 每步在想什么，便于调试和优化。Function Calling 虽然开发更快，但过程是黑盒的。

### 3. 工具沙箱安全设计

- `calculator`：使用 eval 但限制到安全函数白名单，不能执行任意代码
- `code_interpreter`：使用受限的 `__builtins__`，隔离文件系统和网络访问
- 所有工具都有超时保护

---

## ReAct vs 其他 Agent 范式

| | ReAct | Plan-and-Execute | Multi-Agent |
|------|------|------|------|
| 推理方式 | 边想边做 | 先计划后执行 | 多 Agent 协作 |
| 灵活性 | 高 | 中 | 高 |
| 可解释性 | 强 | 强 | 弱 |
| 适用场景 | 工具调用、信息检索 | 结构化任务 | 复杂协作任务 |

---

## 扩展方向

- [ ] 添加记忆模块（长期/短期记忆）
- [ ] 支持 Plan-and-Execute 范式
- [ ] 多 Agent 协作（CrewAI 风格）
- [ ] 工具执行结果缓存
- [ ] 用户确认机制（Human-in-the-loop）
- [ ] 流式输出推理过程
