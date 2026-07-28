"""HTML 解析器（基于 BeautifulSoup，提取正文文本）"""
from __future__ import annotations
import uuid
from pathlib import Path

from ..base import BaseParser, Document
from ..registry import register


@register("parser", "html")
class HTMLParser(BaseParser):
    def parse(self, file_path: str) -> Document:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        # 移除脚本与样式
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        content = soup.get_text(separator="\n").strip()
        # 压缩多余空行
        content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            source=str(file_path),
            metadata={"format": "html", "filename": Path(file_path).name},
        )

    @staticmethod
    def supported_extensions() -> list[str]:
        return [".html", ".htm"]
