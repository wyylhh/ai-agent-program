"""
ReAct Agent 核心引擎：从零实现 ReAct（Reasoning + Acting）推理循环

ReAct 范式流程:
    用户输入 → Thought(思考) → Action(选择工具) → Observation(观察结果)
    → Thought → Action → Observation → ... → 满足条件 → Final Answer

每轮迭代，Agent 自主完成:
    1. 思考（Thought）：分析当前状态，决定下一步
    2. 行动（Action）：选择工具并生成参数
    3. 观察（Observation）：获取工具执行结果
    4. 决策：是否需要继续迭代，还是输出最终答案
"""
import json
import re
from datetime import datetime
from openai import OpenAI
from tools import ToolRegistry, ToolResult, get_default_tools
from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL,
    MAX_ITERATIONS, TEMPERATURE,
)


# ReAct System Prompt — 核心：教 LLM 如何进行思考-行动-观察循环
SYSTEM_PROMPT = """你是一个具备工具调用能力的 AI 助手。你通过「思考 → 行动 → 观察」的循环来解决复杂问题。

## 可用工具
{tool_descriptions}

## 工作流程
对于每个问题，按以下格式逐步推理：

Thought: 分析当前情况，决定下一步要做什么。如果可以直接回答，说明原因。
Action: 要调用的工具名称
Action Input: 工具参数的 JSON 对象
Observation: 工具返回的结果（由系统自动填入）

... (可重复 Thought/Action/Action Input/Observation 多次) ...

Thought: 我已获得足够信息来回答用户问题
Final Answer: 基于所有观察结果的最终回答

## 规则
1. 每次只调用一个工具
2. Action Input 必须是有效的 JSON
3. 充分利用工具获取信息，不要凭空猜测
4. 如果工具调用失败，尝试其他方法或工具
5. 当你确信能回答用户时，以 "Final Answer:" 开头给出最终答案
6. 用中文思考和回答

当前日期: {current_date}"""


class ReActAgent:
    """
    ReAct Agent 实现

    核心原理：
        ReAct = Reasoning（推理）+ Acting（行动）
        将 LLM 的推理能力和工具调用能力交替结合，
        让模型在"想"和"做"之间迭代，直到完成任务。

    使用示例:
        agent = ReActAgent()
        result = agent.run("北京今天天气怎么样？然后计算 35 * 28 + 100")
        print(result)
    """

    def __init__(self, tools: ToolRegistry | None = None, model: str = LLM_MODEL):
        self.tools = tools or get_default_tools()
        self.model = model
        self.client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        self.max_iterations = MAX_ITERATIONS
        self.trajectory: list[dict] = []  # 完整推理轨迹

    def run(self, user_query: str) -> dict:
        """
        执行 ReAct 循环

        Returns:
            {
                "query": "...",
                "answer": "...",
                "iterations": 3,
                "trajectory": [...],  # 完整的 Thought-Action-Observation 轨迹
                "tool_calls": [...],  # 工具调用记录
            }
        """
        self.trajectory = []
        tool_calls_log = []

        # 构建初始消息
        tool_desc = self.tools.get_tool_descriptions()
        system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=tool_desc,
            current_date=datetime.now().strftime("%Y年%m月%d日"),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        iteration = 0
        final_answer = ""

        print(f"\n{'='*60}")
        print(f"[Agent] 用户输入: {user_query}")
        print(f"{'='*60}")

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- 第 {iteration} 步 ---")

            # 1. 调用 LLM 获取 Thought + Action
            response = self._call_llm(messages)
            full_text = response.strip()
            print(f"[LLM 输出]\n{full_text}")

            # 2. 解析 LLM 输出
            parsed = self._parse_output(full_text)

            # 3. 如果 LLM 直接给出 Final Answer，结束
            if parsed["final_answer"]:
                final_answer = parsed["final_answer"]
                self.trajectory.append({"iteration": iteration, "thought": parsed["thought"], "action": "Final Answer", "result": final_answer})
                print(f"[Agent] 最终回答: {final_answer}")
                break

            # 4. 如果有 Action，执行工具调用
            if parsed["action"] and parsed["action_input"]:
                tool_name = parsed["action"]
                tool_input = parsed["action_input"]

                tool = self.tools.get(tool_name)
                if not tool:
                    observation = f"错误: 未找到工具 '{tool_name}'。可用工具: {list(self.tools._tools.keys())}"
                else:
                    print(f"[Agent] 调用工具: {tool_name}({tool_input})")
                    result: ToolResult = tool.run(**tool_input)
                    if result.success:
                        observation = result.output
                        print(f"[工具结果] {observation[:300]}...")
                    else:
                        observation = f"工具调用失败: {result.error}"
                        print(f"[工具错误] {observation}")

                tool_calls_log.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "input": tool_input,
                    "success": result.success if tool else False,
                    "output": observation[:500],
                })

                # 将 Observation 追加到对话
                messages.append({"role": "assistant", "content": full_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

                self.trajectory.append({
                    "iteration": iteration,
                    "thought": parsed["thought"],
                    "action": tool_name,
                    "action_input": parsed["action_input"],
                    "observation": observation,
                })
                continue

            # 5. 如果既没有 Final Answer 也没有 Action，要求 LLM 继续
            messages.append({"role": "assistant", "content": full_text})
            messages.append({"role": "user", "content": "请继续，你需要选择 Action 或给出 Final Answer。"})

        # 如果达到最大迭代次数仍未完成
        if not final_answer:
            final_answer = self._force_final_answer(messages)
            print(f"[Agent] 达到最大迭代次数，强制生成回答")

        return {
            "query": user_query,
            "answer": final_answer,
            "iterations": iteration,
            "trajectory": self.trajectory,
            "tool_calls": tool_calls_log,
        }

    def _call_llm(self, messages: list[dict]) -> str:
        """调用 LLM"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    def _parse_output(self, text: str) -> dict:
        """解析 LLM 输出，提取 Thought / Action / Action Input / Final Answer"""
        result = {
            "thought": "",
            "action": None,
            "action_input": None,
            "final_answer": None,
        }

        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final)|$)', text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 检查是否有 Final Answer
        final_match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # 提取 Action
        action_match = re.search(r'Action:\s*(.+?)(?=\n|$)', text)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # 提取 Action Input (JSON)
        input_match = re.search(r'Action Input:\s*(\{.+?\})', text, re.DOTALL)
        if input_match:
            try:
                result["action_input"] = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                result["action_input"] = {}

        return result

    def _force_final_answer(self, messages: list[dict]) -> str:
        """达到最大迭代次数时，强制要求 LLM 基于已有信息给出回答"""
        messages.append({
            "role": "user",
            "content": "你已达到最大迭代次数。请基于目前收集到的所有信息，以 'Final Answer:' 开头，给出你最好的回答。"
        })
        resp = self._call_llm(messages)
        match = re.search(r'Final Answer:\s*(.+)', resp, re.DOTALL)
        return match.group(1).strip() if match else resp.strip()

    def format_trajectory(self) -> str:
        """将推理轨迹格式化为可读文本"""
        lines = []
        for step in self.trajectory:
            lines.append(f"## 第 {step['iteration']} 步")
            lines.append(f"**Thought**: {step['thought']}")
            if step.get('action') and step['action'] != 'Final Answer':
                lines.append(f"**Action**: {step['action']}")
                lines.append(f"**Action Input**: {json.dumps(step.get('action_input', {}), ensure_ascii=False)}")
                obs = step.get('observation', '')
                lines.append(f"**Observation**: {obs[:300]}")
            else:
                lines.append(f"**Final Answer**: {step.get('result', '')}")
            lines.append("")
        return "\n".join(lines)
