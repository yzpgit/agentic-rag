"""
向量检索器（纯 numpy 实现）
==========================
使用 L2 归一化后的内积（等价 cosine 相似度）。
支持 save/load，便于 ingest 与 serve 共享索引。
已移除 faiss 依赖，改用 numpy + 暴力检索（万级文档内性能足够）。
"""
from __future__ import annotations
import os
import pickle

from ..base import BaseRetriever, Chunk, RetrievalResult
from ..registry import register
from ..indexing.embedder import Embedder


@register("retriever", "vector")
class VectorRetriever(BaseRetriever):
    def __init__(self, provider: str = "xfyun",
                 embedding: str = "xfyun",
                 embedding_model: str = "xop3qwen8bembedding",
                 top_k: int = 5,
                 index_path: str = "data/index/vector",
                 **kw):
        super().__init__(top_k=top_k)
        self.index_path = index_path
        self.embedder = Embedder(embedding, embedding_model)
        self.dim = self.embedder.dim
        import numpy as np
        self._np = np
        # 矩阵初始化（ntotal=0）
        self.matrix: np.ndarray = np.zeros((0, self.dim), dtype="float32")
        self.chunks: list[Chunk] = []
        self._load()

    @property
    def ntotal(self) -> int:
        return self.matrix.shape[0]

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        np = self._np
        texts = [c.content for c in chunks]
        vecs = np.array(self.embedder.embed(texts), dtype="float32")
        # L2 归一化（等价 cosine）
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self.matrix = np.vstack([self.matrix, vecs])
        self.chunks.extend(chunks)
        self._save()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        np = self._np
        k = k or self.top_k
        if self.ntotal == 0:
            return []
        qv = np.array([self.embedder.embed([query])[0]], dtype="float32")
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm
        # 暴力内积检索（cosine）
        scores = (self.matrix @ qv.T).reshape(-1)
        k = min(k, self.ntotal)
        # argpartition 取 top-k，再排序
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        results = []
        for i in idx:
            chunk = self.chunks[i]
            results.append(RetrievalResult(
                chunk=chunk,
                score=float(scores[i]),
                source=chunk.metadata.get("source", ""),
            ))
        return results

    # ---------- 持久化 ----------
    def _save(self) -> None:
        if self.ntotal == 0:
            return
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        np = self._np
        np.save(self.index_path + ".npy", self.matrix)
        with open(self.index_path + ".pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def _load(self) -> None:
        npy_path = self.index_path + ".npy"
        pkl_path = self.index_path + ".pkl"
        if os.path.exists(npy_path) and os.path.exists(pkl_path):
            np = self._np
            self.matrix = np.load(npy_path)
            with open(pkl_path, "rb") as f:
                self.chunks = pickle.load(f)
