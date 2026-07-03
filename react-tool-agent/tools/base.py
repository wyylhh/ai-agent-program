"""
工具基类：定义统一的 Tool 接口

所有外部工具必须继承 BaseTool 并实现 execute() 方法。
这种设计让 Agent 无需关心每个工具的内部实现，统一通过接口调用。
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from config import TOOL_TIMEOUT, MAX_RETRIES


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    output: str
    tool_name: str = ""
    error: str | None = None
    duration_ms: float = 0.0


class BaseTool(ABC):
    """
    工具基类

    子类只需实现:
    - name: 工具名称（LLM 通过名称选择工具）
    - description: 工具功能描述（告诉 LLM 什么时候用这个工具）
    - parameters: 参数说明
    - execute(**kwargs): 具体执行逻辑
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def run(self, **kwargs) -> ToolResult:
        """
        统一调用入口，带重试和超时保护
        """
        start = time.time()
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                output = self.execute(**kwargs)
                return ToolResult(
                    success=True,
                    output=str(output),
                    tool_name=self.name,
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                    continue
                break

        return ToolResult(
            success=False,
            output="",
            tool_name=self.name,
            error=last_error,
            duration_ms=(time.time() - start) * 1000,
        )

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """子类实现具体逻辑"""
        ...

    def to_openai_function(self) -> dict:
        """转为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


class ToolRegistry:
    """工具注册中心：管理所有可用工具"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """生成工具清单描述文本，注入到 Agent 的 System Prompt 中"""
        lines = []
        for tool in self._tools.values():
            params_str = ", ".join(
                f"{k}: {v.get('description', '')}" for k, v in tool.parameters.items()
            )
            lines.append(f"- **{tool.name}**: {tool.description}")
            if params_str:
                lines.append(f"  参数: {params_str}")
        return "\n".join(lines)

    def get_openai_functions(self) -> list[dict]:
        return [t.to_openai_function() for t in self._tools.values()]
