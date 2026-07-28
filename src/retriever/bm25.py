"""BM25 关键词检索器（rank_bm25）"""
from __future__ import annotations
import re

from ..base import BaseRetriever, Chunk, RetrievalResult
from ..registry import register


@register("retriever", "bm25")
class BM25Retriever(BaseRetriever):
    def __init__(self, provider: str = "rank_bm25", top_k: int = 5, **kw):
        super().__init__(top_k=top_k)
        self.chunks: list[Chunk] = []
        self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # 简单分词：英文按词、中文按字（足够 BM25 关键词召回）
        tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fa5]", text.lower())
        return tokens

    def add(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi
        self.chunks.extend(chunks)
        corpus = [self._tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        k = k or self.top_k
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        # 按分数降序取 top_k（注意：小语料下 BM25 的 IDF 可能为负，
        # 负分仍代表相对排序，不应过滤，否则会丢弃全部结果）
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk=chunk, score=float(score),
                source=chunk.metadata.get("source", ""),
            ))
        return results
