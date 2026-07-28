"""PDF 解析器（基于 pypdf）"""
from __future__ import annotations
import uuid
from pathlib import Path

from ..base import BaseParser, Document
from ..registry import register


@register("parser", "pdf")
class PDFParser(BaseParser):
    def parse(self, file_path: str) -> Document:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        content = "\n\n".join(pages).strip()
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            source=str(file_path),
            metadata={
                "format": "pdf",
                "pages": len(reader.pages),
                "filename": Path(file_path).name,
            },
        )

    @staticmethod
    def supported_extensions() -> list[str]:
        return [".pdf"]
