"""
OpenAI LLM Provider
===================
通过 OPENAI_BASE_URL 环境变量可无缝切换到通义千问 / Ollama 等 OpenAI 兼容接口。
"""
from __future__ import annotations
import os
from typing import AsyncIterator

from ..base import BaseLLM
from ..registry import register

_SYSTEM_WITH_CONTEXT = (
    "你是一个严谨的研究助手。请 ONLY 基于以下检索到的资料回答问题。"
    "若资料不足或无法回答，请明确说明，不要编造。"
    "在引用资料处标注 [编号]，编号对应资料顺序。\n\n"
    "检索资料:\n{context}"
)


@register("llm", "openai")
class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini",
                 temperature: float = 0.2, streaming: bool = True,
                 thinking: bool = False, **kw):
        super().__init__(model, temperature, streaming)
        self.thinking = thinking
        self._client = None
        self._async_client = None

    def _ensure_clients(self):
        """懒加载：首次调用时才创建 client（服务启动不依赖凭证）"""
        if self._client is not None:
            return
        from openai import OpenAI, AsyncOpenAI
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY", "dummy")
        self._client = OpenAI(base_url=base_url, api_key=api_key) if base_url \
            else OpenAI()
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key) \
            if base_url else AsyncOpenAI()

    @property
    def _extra_body(self) -> dict:
        """非标准字段透传：LongCat-2.0 等模型用 thinking 控制思考过程"""
        # thinking=False → 关闭思考（阶段1，回答快速直接）
        # thinking=True  → 开启思考（阶段2，reasoning_content 可视化为 Agentic 卖点）
        return {"thinking": {"type": "enabled" if self.thinking else "disabled"}}

    @property
    def client(self):
        self._ensure_clients()
        return self._client

    @property
    def async_client(self):
        self._ensure_clients()
        return self._async_client

    def _messages(self, prompt: str, context: str) -> list[dict]:
        if context:
            return [
                {"role": "system",
                 "content": _SYSTEM_WITH_CONTEXT.format(context=context)},
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str, context: str = "") -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, context),
            temperature=self.temperature,
            stream=False,
            extra_body=self._extra_body,
        )
        return resp.choices[0].message.content or ""

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """流式生成。yield 元组 (kind, text)，kind ∈ {'reasoning','content'}。
        阶段1 thinking=False 时只有 content；阶段2 thinking=True 时先 reasoning 后 content。"""
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, context),
            temperature=self.temperature,
            stream=True,
            extra_body=self._extra_body,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            # LongCat-2.0 等模型把思考放在 reasoning_content，正式回答在 content
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                yield ("reasoning", rc)
            if delta.content:
                yield ("content", delta.content)
