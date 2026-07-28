"""
容器启动时自动入库预置文档
==========================
CloudBase 云托管实例无状态，重启后本地索引丢失。
把演示文档随镜像打包，启动时自动入库，保证冷启动后立即可用。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPPORTED = {".pdf", ".md", ".markdown", ".html", ".htm", ".docx"}
SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def main():
    if not SEED_DIR.exists():
        return
    files = [p for p in SEED_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        print("[seed] 无预置文档，跳过")
        return

    # 延迟导入，避免与 uvicorn worker 重复加载
    from src.orchestrator import get_pipeline
    pipeline = get_pipeline()
    total = 0
    for f in files:
        try:
            n = pipeline.ingest(str(f))
            print(f"[seed] ✓ {f.name}: {n} chunks")
            total += n
        except Exception as e:
            print(f"[seed] ✗ {f.name}: {e}")
    print(f"[seed] 完成，共 {total} chunks 入库")


if __name__ == "__main__":
    main()
