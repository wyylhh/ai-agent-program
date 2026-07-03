from .base import ToolResult, BaseTool, ToolRegistry
from .search import SearchTool
from .calculator import CalculatorTool
from .code_interpreter import CodeInterpreterTool
from .weather import WeatherTool


def get_default_tools() -> ToolRegistry:
    """获取默认注册的工具集"""
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(CalculatorTool())
    registry.register(CodeInterpreterTool())
    registry.register(WeatherTool())
    return registry
