"""评测路由：RAGAS + CRUD-RAG 双轨评测"""
from __future__ import annotations
import json
import os
import threading
import time
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EVAL_RESULT = _ROOT / "data" / "eval" / "result.json"
_EVAL_PROGRESS = _ROOT / "data" / "eval" / "progress.json"

# 全局评测状态（进程内）
_eval_lock = threading.Lock()
_eval_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "sample": 0,
    "mode": None,
    "error": None,
}


def _write_progress(stage: str, msg: str, pct: int) -> None:
    _EVAL_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(_EVAL_PROGRESS, "w", encoding="utf-8") as f:
        json.dump({
            "stage": stage, "msg": msg, "pct": pct,
            "running": _eval_state["running"],
            "started_at": _eval_state["started_at"],
        }, f, ensure_ascii=False)


def _get_eval_python() -> str:
    """获取评测用的 python 解释器路径

    优先级：
    1. 环境变量 EVAL_PYTHON（显式指定）
    2. ~/eval-env/bin/python（uv 创建的虚拟环境）
    3. sys.executable（兜底，容器内）
    """
    # 1. 环境变量显式指定
    env_py = os.environ.get("EVAL_PYTHON")
    if env_py and Path(env_py).exists():
        return env_py
    # 2. uv venv 路径（容器内挂载宿主机 venv 时可用）
    candidates = [
        Path.home() / "eval-env" / "bin" / "python",
        Path("/home/ubuntu/eval-env/bin/python"),
        Path("/root/eval-env/bin/python"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # 3. 兜底
    return sys.executable


def _run_eval(sample: int, mode: str | None) -> None:
    """子线程跑评测脚本，写进度文件"""
    with _eval_lock:
        if _eval_state["running"]:
            return
        _eval_state.update({
            "running": True, "started_at": time.time(), "finished_at": None,
            "sample": sample, "mode": mode, "error": None,
        })
    _write_progress("init", "评测启动中…", 5)
    try:
        py = _get_eval_python()
        cmd = [py, "scripts/eval_ragas.py", "--sample", str(sample)]
        if mode:
            cmd += ["--mode", mode]
        # 实时读取输出，解析 [n/4] 进度
        proc = subprocess.Popen(
            cmd, cwd=str(_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        stage_pct = {"[1/4]": 15, "[2/4]": 35, "[3/4]": 55, "[4/4]": 80}
        for line in proc.stdout:  # type: ignore
            line = line.strip()
            if not line:
                continue
            for tag, pct in stage_pct.items():
                if tag in line:
                    _write_progress("running", line, pct)
                    break
        proc.wait()
        if proc.returncode != 0:
            _eval_state["error"] = f"评测脚本退出码 {proc.returncode}"
            _write_progress("error", _eval_state["error"], 0)
        else:
            _write_progress("done", "评测完成", 100)
    except Exception as e:
        _eval_state["error"] = str(e)
        _write_progress("error", str(e), 0)
    finally:
        _eval_state["running"] = False
        _eval_state["finished_at"] = time.time()


class EvalRequest(BaseModel):
    sample: int = 20
    mode: str | None = None


@router.get("/eval")
async def eval_status() -> dict:
    """查看评测状态和最近一次结果"""
    # 读取进度
    progress = {"running": False, "stage": "idle", "msg": "", "pct": 0}
    if _EVAL_PROGRESS.exists():
        try:
            with open(_EVAL_PROGRESS, "r", encoding="utf-8") as f:
                progress = json.load(f)
        except Exception:
            pass
    # 进程内状态更准确
    progress["running"] = _eval_state["running"]
    if _eval_state["error"]:
        progress["stage"] = "error"
        progress["msg"] = _eval_state["error"]

    # 读取结果
    result = None
    if _EVAL_RESULT.exists():
        try:
            with open(_EVAL_RESULT, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            pass

    return {
        "status": "completed" if result else "pending",
        "dataset": "CRUD-RAG (arXiv:2401.17043)",
        "judge_llm": "LongCat-2.0",
        "progress": progress,
        "scores": result.get("scores", {}) if result else {},
        "details": (result.get("details", []) if result else [])[:5],  # 只返回前5条明细
        "detail_count": len(result.get("details", [])) if result else 0,
    }


@router.post("/eval/run")
async def eval_run(req: EvalRequest, background_tasks: BackgroundTasks) -> dict:
    """触发后台评测（异步执行，不阻塞响应）"""
    if _eval_state["running"]:
        return {"status": "busy", "message": "评测正在进行中，请等待完成"}
    background_tasks.add_task(_run_eval, req.sample, req.mode)
    return {
        "status": "started",
        "message": f"评测已启动（sample={req.sample}, mode={req.mode or 'default'}）",
        "note": "评测约需3-5分钟，轮询 GET /eval 查看进度",
    }
