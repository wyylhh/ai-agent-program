"""
搜索引擎工具：调用 DuckDuckGo 搜索网页
"""
from tools.base import BaseTool


class SearchTool(BaseTool):
    name = "web_search"
    description = "搜索互联网获取最新信息。当需要查找实时信息、事实、新闻或你不知道的内容时使用。"
    parameters = {
        "query": {
            "type": "string",
            "description": "搜索关键词，用中文或英文",
        },
        "max_results": {
            "type": "integer",
            "description": "返回结果数量，默认3",
        },
    }

    def execute(self, query: str, max_results: int = 3) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"未找到与 '{query}' 相关的结果"
            lines = [f"搜索 '{query}' 的结果："]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}\n   {r['body'][:300]}\n   URL: {r['href']}")
            return "\n\n".join(lines)
        except ImportError:
            # 回退：模拟搜索（离线演示用）
            return (
                f"[模拟搜索] 关键词 '{query}' 的搜索结果：\n"
                f"1. 由于未安装 duckduckgo_search 或网络不可用，这是一个模拟结果。\n"
                f"   请安装: pip install duckduckgo_search"
            )
