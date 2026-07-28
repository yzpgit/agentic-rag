"""
插件聚合导入
============
集中 import 所有可插拔模块，触发各自的 @register 注册。
Orchestrator 在组装前先 import 本模块，确保 REGISTRY 填充完毕。
新增插件时只需在此追加一行 import。
"""
# 文档解析
from .document import pdf_parser, markdown_parser, html_parser, docx_parser  # noqa: F401

# 切分
from .chunking import recursive  # noqa: F401

# 检索
from .retriever import vector, bm25, hybrid, chroma  # noqa: F401

# LLM
from .llm import openai_provider  # noqa: F401

# 阶段2 将新增：
# from .reranker import bge_reranker
# from .agent import graph as agent_graph
