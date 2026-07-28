"""评测路由（阶段1占位，阶段4实现双轨评测）"""
from __future__ import annotations
from fastapi import APIRouter

router = APIRouter()


@router.get("/eval")
async def eval_status() -> dict:
    return {
        "status": "pending",
        "message": "评测模块将在阶段4实现：确定性指标(nDCG/MRR/Recall@K) + Ragas",
    }
