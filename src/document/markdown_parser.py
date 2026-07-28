"""Markdown 解析器（纯文本读取，保留原文本）"""
from __future__ import annotations
import uuid
from pathlib import Path

from ..base import BaseParser, Document
from ..registry import register


@register("parser", "markdown")
class MarkdownParser(BaseParser):
    def parse(self, file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            source=str(file_path),
            metadata={"format": "markdown", "filename": Path(file_path).name},
        )

    @staticmethod
    def supported_extensions() -> list[str]:
        return [".md", ".markdown"]
