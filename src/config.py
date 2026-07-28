"""
配置加载：读取 config/config.yaml，提供全局访问与热重载
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import threading

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


class ConfigManager:
    """单例配置管理器，支持热重载"""

    def __init__(self, path: Path = _DEFAULT_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._cfg: dict[str, Any] = {}
        self.reload()

    def reload(self, path: Path | None = None) -> dict:
        if path is not None:
            self._path = path
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f) or {}
            return copy.deepcopy(self._cfg)

    def update(self, new_cfg: dict) -> dict:
        """前端热重载：整体替换配置"""
        with self._lock:
            self._cfg = copy.deepcopy(new_cfg)
            return copy.deepcopy(self._cfg)

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._cfg)

    @property
    def path(self) -> Path:
        return self._path


# 全局单例
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> dict:
    return get_config_manager().get()
