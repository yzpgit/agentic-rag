"""
批量入库脚本
============
用法:
  python scripts/ingest.py data/docs/            # 入库整个目录
  python scripts/ingest.py data/docs/paper.pdf   # 入库单个文件
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import get_pipeline  # noqa: E402

SUPPORTED = {".pdf", ".md", ".markdown", ".html", ".htm", ".docx"}


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/ingest.py <文件或目录>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = [p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED]
    else:
        files = [target]

    if not files:
        print(f"未找到支持的文件: {target}")
        sys.exit(1)

    pipeline = get_pipeline()
    total = 0
    for f in files:
        try:
            n = pipeline.ingest(str(f))
            print(f"✓ {f.name}: {n} chunks")
            total += n
        except Exception as e:
            print(f"✗ {f.name}: {e}")
    print(f"\n完成，共 {total} chunks 入库")


if __name__ == "__main__":
    main()
