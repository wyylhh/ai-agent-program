"""
Python 代码解释器工具：在沙箱中执行 Python 代码
"""
import sys
import io
from tools.base import BaseTool


class CodeInterpreterTool(BaseTool):
    name = "code_interpreter"
    description = "执行 Python 代码进行数据计算、分析或验证。输入完整的 Python 代码片段，使用 print() 输出结果。"
    parameters = {
        "code": {
            "type": "string",
            "description": "要执行的 Python 代码，用 print() 输出结果。例如: 'print(sum(range(1, 101)))'",
        },
    }

    def execute(self, code: str) -> str:
        # 捕获标准输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # 安全的执行环境
        safe_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len,
                "int": int, "float": float, "str": str, "list": list,
                "dict": dict, "set": set, "tuple": tuple, "bool": bool,
                "sum": sum, "min": min, "max": max, "sorted": sorted,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "abs": abs, "round": round, "pow": pow,
                "type": type, "isinstance": isinstance,
                "Exception": Exception, "ValueError": ValueError,
                "True": True, "False": False, "None": None,
            },
        }

        try:
            exec(code, safe_globals, {})
            output = sys.stdout.getvalue()
            if not output.strip():
                output = "[代码执行完毕，无输出]"
            return output.strip()
        except Exception as e:
            return f"代码执行错误: {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
