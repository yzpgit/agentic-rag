"""
混合检索器（向量 + BM25 + RRF 融合）
===================================
双路召回后用 Reciprocal Rank Fusion 融合排序。
借鉴 mv33-4/Hybrid-Rag 与 paras-the-coder/Agentic-Hybrid-RAG 的 RRF 实现。
"""
from __future__ import annotations

from ..base import BaseRetriever, RetrievalResult
from ..registry import register, build


@register("retriever", "hybrid")
class HybridRetriever(BaseRetriever):
    def __init__(self, vector: dict | None = None,
                 bm25: dict | None = None,
                 fusion: dict | None = None,
                 mode: str | None = None,      # 吸收 config 顶层 key
                 top_k: int = 5,
                 **kw):
        super().__init__(top_k=top_k)
        vcfg = vector or {}
        bcfg = bm25 or {}
        fcfg = fusion or {}
        # 子检索器各自独立索引，build 时各自加载已持久化的索引
        self.vector = build("retriever", "vector", **vcfg)
        self.bm25 = build("retriever", "bm25", **bcfg)
        self.fusion_method = fcfg.get("method", "rrf")
        self.rrf_k = fcfg.get("rrf_k", 60)

    def add(self, chunks: list) -> None:
        # 两路索引同一批 chunks（共享 chunk.id 以便 RRF 对齐）
        self.vector.add(chunks)
        self.bm25.add(chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        k = k or self.top_k
        # 双路各自多召回一些，给融合更大候选池
        pool = max(k * 4, 20)
        vr = self.vector.retrieve(query, k=pool)
        br = self.bm25.retrieve(query, k=pool)
        if self.fusion_method == "rrf":
            return self._rrf(vr, br, k)
        # weighted: 简单按归一化分数加权
        return self._weighted(vr, br, k)

    def _rrf(self, vr: list[RetrievalResult], br: list[RetrievalResult],
             k: int) -> list[RetrievalResult]:
        scores: dict[str, dict] = {}
        for rank, r in enumerate(vr):
            cid = r.chunk.id
            scores.setdefault(cid, {"result": r, "s": 0.0})
            scores[cid]["s"] += 1.0 / (self.rrf_k + rank + 1)
        for rank, r in enumerate(br):
            cid = r.chunk.id
            scores.setdefault(cid, {"result": r, "s": 0.0})
            scores[cid]["s"] += 1.0 / (self.rrf_k + rank + 1)
        ordered = sorted(scores.values(), key=lambda x: x["s"], reverse=True)[:k]
        return [RetrievalResult(chunk=d["result"].chunk, score=d["s"],
                                source=d["result"].source) for d in ordered]

    def _weighted(self, vr: list[RetrievalResult], br: list[RetrievalResult],
                  k: int, w_v: float = 0.6, w_b: float = 0.4) -> list[RetrievalResult]:
        def _norm(rs):
            if not rs:
                return {}
            mx = max((r.score for r in rs), default=1.0) or 1.0
            return {r.chunk.id: (r, r.score / mx) for r in rs}
        vn, bn = _norm(vr), _norm(br)
        ids = set(vn) | set(bn)
        merged = []
        for cid in ids:
            r = (vn.get(cid) or bn.get(cid))[0]
            s = w_v * vn.get(cid, (None, 0.0))[1] + w_b * bn.get(cid, (None, 0.0))[1]
            merged.append(RetrievalResult(chunk=r.chunk, score=float(s), source=r.source))
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:k]
