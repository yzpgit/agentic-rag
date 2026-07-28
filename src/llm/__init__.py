"""LLM 模块：导入子模块以触发 @register 注册"""
from . import openai_provider  # noqa: F401
# 阶段2可扩展: qwen_provider / ollama_provider（均可复用 OpenAI 兼容客户端）
