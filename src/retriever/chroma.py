"""
向量检索器（Chroma 实现，可选插件）
==================================
使用 ChromaDB 作为后端，HNSW 索引，支持元数据过滤。
通过自定义 EmbeddingFunction 包装云 API（讯飞/通义/OpenAI），
避免下载本地 onnxruntime 模型。

启用方式：
  1. pip install chromadb
  2. config.yaml: retriever.vector.provider: chroma
"""
from __future__ import annotations
import os
from typing import Any

from ..base import BaseRetriever, Chunk, RetrievalResult
from ..registry import register
from ..indexing.embedder import Embedder


@register("retriever", "chroma")
class ChromaRetriever(BaseRetriever):
    def __init__(self, provider: str = "chroma",
                 embedding: str = "xfyun",
                 embedding_model: str = "xop3qwen8bembedding",
                 top_k: int = 5,
                 collection_name: str = "agentic_rag",
                 persist_path: str = "data/index/chroma",
                 **kw):
        super().__init__(top_k=top_k)
        # 延迟导入，未安装 chromadb 时不会影响其他检索器
        import chromadb
        from chromadb.api.types import EmbeddingFunction, Embeddings, Documents

        self.embedder = Embedder(embedding, embedding_model)
        self.collection_name = collection_name

        # 包装云 embedding 为 Chroma 认识的 EmbeddingFunction
        outer = self

        class _CloudEF(EmbeddingFunction):
            """把云 Embedder 适配为 Chroma 的 EmbeddingFunction"""

            def __call__(self, input: Documents) -> Embeddings:
                return outer.embedder.embed(list(input))

            def name(self) -> str:
                return "cloud_embedding"

        self._ef = _CloudEF()

        # 持久化客户端
        os.makedirs(persist_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def ntotal(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        docs = [c.content for c in chunks]
        metas = [{"document_id": c.document_id,
                  "source": c.metadata.get("source", ""),
                  **c.metadata} for c in chunks]
        # Chroma 的 upsert 会自动调用 _ef 做向量化
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        k = k or self.top_k
        if self.ntotal == 0:
            return []
        res = self._collection.query(
            query_texts=[query],
            n_results=min(k, self.ntotal),
        )
        results: list[RetrievalResult] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunk = Chunk(
                id=meta.get("id", ""),
                content=doc,
                document_id=meta.get("document_id", ""),
                metadata=meta,
            )
            # Chroma cosine 距离 → 相似度
            score = 1.0 - float(dist)
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                source=meta.get("source", ""),
            ))
        return results
