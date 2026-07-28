"""对话路由：SSE 流式推送 trace / source / token / done 事件"""
from __future__ import annotations
import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from ...orchestrator import get_pipeline

router = APIRouter()


@router.get("/chat/stream")
async def chat_stream(q: str):
    """SSE 流式问答。前端用 EventSource 监听各类事件。"""
    pipeline = get_pipeline()

    async def event_generator():
        async for ev in pipeline.stream_query(q):
            yield {
                "event": ev.get("type", "message"),
                "data": json.dumps(ev, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
