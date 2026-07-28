"""
插件注册中心
============
所有可插拔模块通过 @register 装饰器自注册，Orchestrator 通过 build() 按
config 动态实例化。新增模块只需：实现基类 + 加装饰器，零侵入主流程。

可用类别：parser / chunker / retriever / reranker / llm / agent
"""
from __future__ import annotations
from typing import Any, Callable, Type

# 类别 → {name → cls}
REGISTRY: dict[str, dict[str, Type]] = {
    "parser": {},
    "chunker": {},
    "retriever": {},
    "reranker": {},
    "llm": {},
    "agent": {},
}


def register(category: str, name: str) -> Callable[[Type], Type]:
    """装饰器：把实现类注册到 REGISTRY[category][name]"""
    if category not in REGISTRY:
        raise ValueError(f"未知插件类别: {category}，可用: {list(REGISTRY)}")

    def deco(cls: Type) -> Type:
        if name in REGISTRY[category]:
            raise ValueError(f"重复注册: {category}/{name}")
        REGISTRY[category][name] = cls
        return cls

    return deco


def build(category: str, name: str, **kwargs: Any) -> Any:
    """工厂方法：按 name 实例化已注册的插件"""
    available = REGISTRY[category]
    if name not in available:
        raise ValueError(
            f"未注册的 {category}: {name}，可用: {list(available)}"
        )
    return available[name](**kwargs)


def list_plugins(category: str | None = None) -> dict:
    """列出已注册插件，便于 /config 接口回显"""
    if category:
        return {category: list(REGISTRY[category])}
    return {c: list(v) for c, v in REGISTRY.items()}
