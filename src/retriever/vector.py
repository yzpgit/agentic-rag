"""
向量检索器（FAISS）
==================
默认使用 FAISS IndexFlatIP + L2 归一化后的内积（等价 cosine）。
支持 save/load，便于 ingest 与 serve 共享索引。
"""
from __future__ import annotations
import os
import pickle

from ..base import BaseRetriever, Chunk, RetrievalResult
from ..registry import register
from ..indexing.embedder import Embedder


@register("retriever", "vector")
class VectorRetriever(BaseRetriever):
    def __init__(self, provider: str = "faiss",
                 embedding: str = "sentence-transformers",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 top_k: int = 5,
                 index_path: str = "data/index/vector",
                 **kw):
        super().__init__(top_k=top_k)
        self.provider = provider
        self.index_path = index_path
        self.embedder = Embedder(embedding, embedding_model)
        self.dim = self.embedder.dim
        self._init_index()
        self.chunks: list[Chunk] = []
        self._load()

    def _init_index(self):
        import faiss
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(self.dim)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        import numpy as np
        texts = [c.content for c in chunks]
        vecs = np.array(self.embedder.embed(texts), dtype="float32")
        self.index.add(vecs)
        self.chunks.extend(chunks)
        self._save()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        import numpy as np
        k = k or self.top_k
        if self.index.ntotal == 0:
            return []
        qv = np.array([self.embedder.embed([query])[0]], dtype="float32")
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(qv, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk=chunk,
                score=float(score),
                source=chunk.metadata.get("source", ""),
            ))
        return results

    # ---------- 持久化 ----------
    def _save(self) -> None:
        if self.index.ntotal == 0:
            return
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        self._faiss.write_index(self.index, self.index_path + ".faiss")
        with open(self.index_path + ".pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def _load(self) -> None:
        faiss_path = self.index_path + ".faiss"
        pkl_path = self.index_path + ".pkl"
        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            self.index = self._faiss.read_index(faiss_path)
            with open(pkl_path, "rb") as f:
                self.chunks = pickle.load(f)
