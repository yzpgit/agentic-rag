"""
递归字符切分器
==============
按优先级分隔符递归切分，尽量在段落/句子边界断开。
借鉴 LangChain RecursiveCharacterTextSplitter 的思路，纯内置实现以减少依赖。
"""
from __future__ import annotations
import uuid

from ..base import BaseChunker, Document, Chunk
from ..registry import register

# 分隔符优先级：从结构化到扁平
_SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]


@register("chunker", "recursive")
class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 50, **kw):
        super().__init__(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self._split_text(doc.content, self.chunk_size, self._separators())
        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(Chunk(
                id=f"{doc.id}_{i}",
                content=piece,
                document_id=doc.id,
                metadata={**doc.metadata, "source": doc.source, "chunk_index": i},
            ))
        return chunks

    @staticmethod
    def _separators() -> list[str]:
        return list(_SEPARATORS)

    def _split_text(self, text: str, size: int, separators: list[str]) -> list[str]:
        if len(text) <= size:
            return [text]
        sep = separators[0] or ""
        if not sep:
            # 无分隔符可用，硬切
            return [text[i:i + size] for i in range(0, len(text), size - self.overlap)]

        parts = text.split(sep)
        merged: list[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + sep + p) if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    merged.append(buf)
                # 单段仍超长，递归用下一级分隔符
                if len(p) > size:
                    merged.extend(self._split_text(p, size, separators[1:]))
                else:
                    buf = p
        if buf:
            merged.append(buf)
        return merged
