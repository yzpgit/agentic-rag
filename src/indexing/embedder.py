"""
统一 Embedding 封装
===================
按 provider 切换：sentence-transformers / bge-m3 / openai
向量检索器复用此类，新增 embedding 只需在此扩展。
"""
from __future__ import annotations
import os
from typing import Optional


# 各 provider 的默认向量维度（避免启动即加载模型）
_DEFAULT_DIM = {
    "sentence-transformers": 384,   # all-MiniLM-L6-v2
    "bge-m3": 1024,
    "openai": 1536,                 # text-embedding-3-small
}


class Embedder:
    """Embedding 封装，懒加载：服务启动时不加载模型，首次 embed 才加载"""

    def __init__(self, provider: str = "sentence-transformers",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.provider = provider
        self.model_name = embedding_model
        self._model = None
        self._client = None
        self._dim: Optional[int] = _DEFAULT_DIM.get(provider)
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if self.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        elif self.provider == "bge-m3":
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_name or "BAAI/bge-m3",
                                         use_fp16=True)
            probe = self._model.encode(["dim"], batch_size=1)["dense_vecs"]
            self._dim = probe.shape[1]
        elif self.provider == "openai":
            from openai import OpenAI
            base_url = os.getenv("OPENAI_BASE_URL")
            self._client = OpenAI(base_url=base_url) if base_url else OpenAI()
            self._dim = 1536
        else:
            raise ValueError(f"未知 embedding provider: {self.provider}")
        self._loaded = True

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        if self.provider == "sentence-transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return vecs.tolist()
        if self.provider == "bge-m3":
            vecs = self._model.encode(texts, batch_size=12,
                                      max_length=8192)["dense_vecs"]
            return vecs.tolist()
        if self.provider == "openai":
            resp = self._client.embeddings.create(
                input=texts, model="text-embedding-3-small"
            )
            return [d.embedding for d in resp.data]
        raise ValueError
