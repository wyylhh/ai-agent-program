"""
全局配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM 配置（兼容任何 OpenAI 兼容 API）
# ============================================================
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-e6ec7a09847d4ec8b4bf473ab2ab3e99")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ============================================================
# Agent 参数
# ============================================================
MAX_ITERATIONS = 10      # 最大推理步数
TEMPERATURE = 0.3        # 生成温度
TOOL_TIMEOUT = 30        # 工具调用超时（秒）
MAX_RETRIES = 2          # 工具失败最大重试次数
