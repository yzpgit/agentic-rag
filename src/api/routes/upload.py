"""文档上传路由：保存文件 → 调用 Pipeline.ingest 构建索引"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from ...orchestrator import get_pipeline

router = APIRouter()
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "uploaded"
    dest = UPLOAD_DIR / filename
    with open(dest, "wb") as f:
        f.write(await file.read())

    pipeline = get_pipeline()
    try:
        n = pipeline.ingest(str(dest))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"filename": filename, "chunks": n, "saved_to": str(dest)}
