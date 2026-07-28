"""
统一 Embedding 封装
===================
按 provider 切换：openai兼容 / xfyun（讯飞云）
向量检索器复用此类，新增 embedding 只需在此扩展。
"""
from __future__ import annotations
import os
from typing import Optional


# 各 provider 的默认向量维度（避免启动即调用API）
_DEFAULT_DIM = {
    "xfyun": 768,                  # 讯飞 MaaS xop3qwen8bembedding
    "openai": 1536,                # text-embedding-3-small
    "qwen": 1024,                  # 阿里 text-embedding-v3
}


class Embedder:
    """Embedding 封装，懒加载：服务启动时不初始化客户端，首次 embed 才初始化"""

    def __init__(self, provider: str = "xfyun",
                 embedding_model: str = "xop3qwen8bembedding",
                 base_url: str | None = None,
                 api_key: str | None = None):
        self.provider = provider
        self.model_name = embedding_model
        # 允许显式传入，否则从环境变量读取
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        self._dim: Optional[int] = _DEFAULT_DIM.get(provider)
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if self.provider in ("xfyun", "openai", "qwen"):
            # 均走 OpenAI 兼容接口
            from openai import OpenAI
            base_url = self.base_url or os.getenv("EMBEDDING_BASE_URL")
            api_key = self.api_key or os.getenv("EMBEDDING_API_KEY")
            if not base_url or not api_key:
                raise ValueError(
                    f"embedding provider={self.provider} 需配置 "
                    f"EMBEDDING_BASE_URL 和 EMBEDDING_API_KEY"
                )
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            # 维度探测（首次调用时确定，避免硬编码不准）
            if self._dim is None:
                probe = self._client.embeddings.create(
                    input=["dim"], model=self.model_name
                )
                self._dim = len(probe.data[0].embedding)
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
        # 兼容 OpenAI / 讯飞 / 通义 的 /v1/embeddings 接口
        resp = self._client.embeddings.create(
            input=texts, model=self.model_name
        )
        return [d.embedding for d in resp.data]
