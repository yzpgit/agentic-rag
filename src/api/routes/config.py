"""配置路由：读取 / 热重载 / 列出已注册插件"""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter, Body

from ...config import get_config_manager
from ...orchestrator import reset_pipeline
from ...registry import list_plugins

router = APIRouter()


@router.get("/config")
async def get_cfg() -> dict:
    return get_config_manager().get()


@router.post("/config")
async def update_cfg(cfg: dict = Body(...)) -> dict:
    """整体替换配置并重建 Pipeline（前端热重载）"""
    new_cfg = get_config_manager().update(cfg)
    reset_pipeline(new_cfg)
    return new_cfg


@router.get("/plugins")
async def plugins() -> dict[str, list[str]]:
    """列出所有已注册的可插拔模块，供前端配置面板展示可选项"""
    return list_plugins()
