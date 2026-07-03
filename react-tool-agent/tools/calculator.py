"""
数学计算器工具：安全地执行数学表达式
"""
from tools.base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "执行数学计算。支持四则运算、乘方、三角函数、对数等。当需要进行精确数值计算时使用。"
    parameters = {
        "expression": {
            "type": "string",
            "description": "数学表达式，如 '2 + 3 * 4'、'sqrt(16)'、'sin(pi/2)'、'log(100,10)'",
        },
    }

    # 安全的内置函数白名单，防止代码注入
    _SAFE_NAMES = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sqrt": lambda x: x ** 0.5,
        "sin": __import__("math").sin, "cos": __import__("math").cos,
        "tan": __import__("math").tan,
        "log": __import__("math").log, "log10": __import__("math").log10,
        "pi": __import__("math").pi, "e": __import__("math").e,
        "ceil": __import__("math").ceil, "floor": __import__("math").floor,
        "radians": __import__("math").radians, "degrees": __import__("math").degrees,
    }

    def execute(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, self._SAFE_NAMES)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}\n表达式: {expression}"
