"""YAML 설정 로딩.

config/settings.yaml, config/themes.yaml 을 읽어 dict 로 돌려준다.
설정은 프로세스당 한 번만 읽고 캐시한다.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("STOCKMGR_CONFIG_DIR", PROJECT_ROOT / "config"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "settings.yaml")


@functools.lru_cache(maxsize=1)
def themes() -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "themes.yaml")


def get(path: str, default: Any = None) -> Any:
    """`get("risk.atr_stop_multiple")` 처럼 점 표기로 설정값을 읽는다."""
    node: Any = settings()
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def cache_dir() -> Path:
    path = Path(get("data.cache_dir", "data/cache"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset() -> None:
    """테스트에서 설정 디렉터리를 바꾼 뒤 캐시를 비울 때 사용."""
    settings.cache_clear()
    themes.cache_clear()
