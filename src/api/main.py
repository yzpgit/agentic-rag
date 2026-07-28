"""
FastAPI 入口
============
- 挂载所有路由
- 提供静态前端（frontend/index.html）
- 启动: uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .routes import chat, upload, config, eval as eval_route

_FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"

app = FastAPI(title="Agentic RAG", version="0.1.0")


@app.on_event("startup")
async def _startup():
    # 预热 Pipeline（加载已持久化的索引）
    from ..orchestrator import get_pipeline
    get_pipeline()


@app.get("/health")
async def health():
    return {"status": "ok"}


# 路由
app.include_router(chat.router, tags=["chat"])
app.include_router(upload.router, tags=["upload"])
app.include_router(config.router, tags=["config"])
app.include_router(eval_route.router, tags=["eval"])


@app.get("/")
async def index():
    return FileResponse(_FRONTEND)
