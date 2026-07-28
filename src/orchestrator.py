"""
Pipeline Orchestrator
=====================
配置驱动组装：读取 config.yaml → build 各可插拔模块 → 提供 ingest / stream_query。
- agent.enabled=False（阶段1）：朴素 RAG 流程（retrieve → rerank? → llm.stream）
- agent.enabled=True（阶段2）：交给 LangGraph Agent 工作流
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, AsyncIterator

from . import plugins  # noqa: F401  触发注册
from .base import BaseChunker, BaseRetriever, BaseReranker, BaseLLM, BaseAgent
from .config import get_config
from .registry import build, REGISTRY

# 扩展名 → parser 注册名
_EXT_TO_PARSER = {
    ".pdf": "pdf", ".md": "markdown", ".markdown": "markdown",
    ".html": "html", ".htm": "html", ".docx": "docx",
}


class Pipeline:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or get_config()
        self._build()

    # ---------- 组装 ----------
    def _build(self) -> None:
        dcfg = self.cfg["document"]
        ccfg = dcfg["chunking"]
        rcfg = self.cfg["retriever"]
        rercfg = self.cfg["reranker"]
        acfg = self.cfg["agent"]
        lcfg = self.cfg["llm"]

        # 解析器：按 config.document.parsers 列表构建，并保留扩展名映射
        self.parsers: dict[str, Any] = {}
        for name in dcfg["parsers"]:
            if name in REGISTRY["parser"]:
                self.parsers[name] = build("parser", name)

        # 切分器
        self.chunker: BaseChunker = build("chunker", ccfg["strategy"],
                                          chunk_size=ccfg["chunk_size"],
                                          overlap=ccfg["overlap"])

        # 检索器：按 mode 选择子配置
        mode = rcfg["mode"]
        if mode == "vector":
            vcfg = rcfg["vector"]
            # provider 决定向量库实现：numpy（默认）/ chroma
            vprovider = vcfg.get("provider", "numpy")
            if vprovider == "chroma":
                self.retriever: BaseRetriever = build("retriever", "chroma", **vcfg)
            else:
                self.retriever = build("retriever", "vector", **vcfg)
        elif mode == "bm25":
            self.retriever = build("retriever", "bm25", **rcfg["bm25"])
        else:  # hybrid
            self.retriever = build("retriever", "hybrid", **rcfg)

        # Reranker（可选）
        self.reranker: BaseReranker | None = None
        if rercfg.get("enabled") and rercfg["provider"] in REGISTRY["reranker"]:
            self.reranker = build("reranker", rercfg["provider"],
                                  top_k=rercfg["top_k"])

        # LLM
        self.llm: BaseLLM = build("llm", lcfg["provider"], **{
            k: v for k, v in lcfg.items() if k != "provider"
        })

        # Agent（阶段2，可选）
        self.agent: BaseAgent | None = None
        if acfg.get("enabled") and "langgraph" in REGISTRY["agent"]:
            self.agent = build("agent", "langgraph", **acfg)

    def rebuild(self, cfg: dict | None = None) -> None:
        """前端热重载：用新配置重建 Pipeline"""
        if cfg is not None:
            self.cfg = cfg
        self._build()

    # ---------- 索引构建 ----------
    def ingest(self, file_path: str) -> int:
        ext = Path(file_path).suffix.lower()
        parser_name = _EXT_TO_PARSER.get(ext)
        if not parser_name or parser_name not in self.parsers:
            raise ValueError(f"不支持的文件类型: {ext}（已启用解析器: {list(self.parsers)}）")

        doc = self.parsers[parser_name].parse(file_path)
        chunks = self.chunker.chunk(doc)
        self.retriever.add(chunks)
        return len(chunks)

    # ---------- 查询 ----------
    async def stream_query(self, query: str) -> AsyncIterator[dict]:
        """返回事件流，供 SSE 推送"""
        if self.agent is not None:
            # 阶段2：交给 Agent 工作流
            async for ev in self.agent.run(query, self.retriever,
                                           self.reranker, self.llm):
                yield ev
            return

        # 阶段1：朴素 RAG
        yield {"type": "trace", "node": "retrieve", "msg": "正在检索相关资料..."}
        results = self.retriever.retrieve(query)
        if self.reranker is not None and results:
            yield {"type": "trace", "node": "rerank", "msg": "精排中..."}
            results = self.reranker.rerank(query, results)

        for i, r in enumerate(results, 1):
            yield {
                "type": "source",
                "index": i,
                "source": r.source,
                "content": r.chunk.content,
                "score": round(r.score, 4),
            }

        if not results:
            yield {"type": "trace", "node": "generate",
                   "msg": "未检索到相关资料，直接回答..."}
            context = ""
        else:
            yield {"type": "trace", "node": "generate", "msg": "生成答案..."}
            context = "\n\n".join(
                f"[{i}] {r.chunk.content}" for i, r in enumerate(results, 1)
            )

        async for kind, text in self.llm.stream(query, context):
            if kind == "reasoning":
                # LongCat-2.0 等模型的思考过程（阶段2可在前端可视化）
                yield {"type": "reasoning", "text": text}
            else:
                yield {"type": "token", "text": text}
        yield {"type": "done"}


# 全局单例（API 层复用）
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def reset_pipeline(cfg: dict | None = None) -> Pipeline:
    """配置热重载后调用，重建全局 Pipeline"""
    global _pipeline
    _pipeline = Pipeline(cfg)
    return _pipeline
