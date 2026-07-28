"""
各可插拔模块的抽象基类 + 公共数据结构
每个基类约定最小接口，实现类只需继承 + @register 即可被 Orchestrator 装配。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


# ============================================================
# 公共数据结构
# ============================================================
@dataclass
class Document:
    """解析后的文档"""
    id: str
    content: str
    source: str               # 文件路径或 URL
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """切分后的检索单元"""
    id: str
    content: str
    document_id: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None          # Parent-Child 切分时使用
    parent_content: str | None = None
    embedding: list[float] | None = None
    score: float = 0.0                    # 检索/重排打分


@dataclass
class RetrievalResult:
    """单次检索结果（带引用信息）"""
    chunk: Chunk
    score: float
    source: str


# ============================================================
# 抽象基类
# ============================================================
class BaseParser(ABC):
    """文档解析器：原始文件 → Document"""

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        ...

    @staticmethod
    def supported_extensions() -> list[str]:
        return []


class BaseChunker(ABC):
    """切分器：Document → List[Chunk]"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50, **kw: Any):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        ...


class BaseRetriever(ABC):
    """检索器：query → List[RetrievalResult]"""

    def __init__(self, top_k: int = 5, **kw: Any):
        self.top_k = top_k

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """索引构建阶段：把 chunks 加入索引"""

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        ...


class BaseReranker(ABC):
    """精排器：query + 候选 → 重排后的候选"""

    def __init__(self, top_k: int = 5, **kw: Any):
        self.top_k = top_k

    @abstractmethod
    def rerank(self, query: str,
               results: list[RetrievalResult]) -> list[RetrievalResult]:
        ...


class BaseLLM(ABC):
    """LLM Provider"""

    def __init__(self, model: str = "gpt-4o-mini",
                 temperature: float = 0.2, streaming: bool = True, **kw: Any):
        self.model = model
        self.temperature = temperature
        self.streaming = streaming

    @abstractmethod
    def generate(self, prompt: str, context: str = "") -> str:
        """同步生成"""

    @abstractmethod
    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[tuple[str, str]]:
        """流式生成，yield 元组 (kind, text)。
        kind ∈ {'content'(正式回答), 'reasoning'(思考过程, 可选)}"""


class BaseAgent(ABC):
    """Agent 工作流（阶段2实现 LangGraph 版）"""

    def __init__(self, max_iterations: int = 3,
                 relevance_threshold: float = 0.7, **kw: Any):
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold

    @abstractmethod
    async def run(self, query: str,
                  retriever: BaseRetriever,
                  reranker: BaseReranker | None,
                  llm: BaseLLM) -> AsyncIterator[dict]:
        """返回事件流: {'type': 'trace'|'token'|'source'|'done', ...}"""
