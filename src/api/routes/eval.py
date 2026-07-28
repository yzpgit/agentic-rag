"""评测路由：RAGAS + CRUD-RAG 双轨评测"""
from __future__ import annotations
import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

router = APIRouter()

_EVAL_RESULT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval" / "result.json"


@router.get("/eval")
async def eval_status() -> dict:
    """查看评测状态和最近一次结果"""
    if _EVAL_RESULT.exists():
        with open(_EVAL_RESULT, "r", encoding="utf-8") as f:
            result = json.load(f)
        return {
            "status": "completed",
            "dataset": "CRUD-RAG (arXiv:2401.17043)",
            "judge_llm": "LongCat-2.0",
            "scores": result.get("scores", {}),
        }
    return {
        "status": "pending",
        "message": "尚未运行评测，请在服务器执行: python scripts/eval_ragas.py",
    }


@router.post("/eval/run")
async def eval_run(background_tasks: BackgroundTasks,
                   sample: int = 20,
                   mode: str | None = None) -> dict:
    """触发后台评测（异步执行，不阻塞响应）

    - sample: 抽样条数（默认20）
    - mode: 检索模式 vector/bm25/hybrid（默认用 config.yaml 配置）
    """
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/eval_ragas.py", "--sample", str(sample)]
    if mode:
        cmd += ["--mode", mode]

    background_tasks.add_task(
        subprocess.run, cmd,
        cwd=str(_EVAL_RESULT.parent.parent.parent),
    )
    return {
        "status": "started",
        "message": f"评测已在后台启动（sample={sample}, mode={mode or 'default'}）",
        "note": "评测约需3-5分钟，完成后 GET /eval 查看结果",
    }
