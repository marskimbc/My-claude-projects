"""디스크 캐시.

시세 서버(KRX/야후)는 짧은 시간에 반복 호출하면 차단되기 쉬우므로
조회 결과를 CSV 로 저장하고 TTL 안에서는 재사용한다.
pyarrow 의존을 피하려고 parquet 대신 CSV 를 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .. import config


def _key_to_path(namespace: str, key: dict[str, Any]) -> Path:
    digest = hashlib.sha1(
        json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    directory = config.cache_dir() / namespace
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.csv"


def _is_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    if ttl_hours <= 0:
        return False
    return (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def frame(
    namespace: str,
    key: dict[str, Any],
    producer: Callable[[], pd.DataFrame],
    ttl_hours: float | None = None,
    force_refresh: bool = False,
    date_index: bool = True,
) -> pd.DataFrame:
    """`producer()` 결과를 캐시한다. 빈 결과는 캐시하지 않는다.

    date_index=False 는 종목코드처럼 앞자리 0 이 의미 있는 문자열 인덱스용이다
    (CSV 왕복에서 '069500' 이 69500 으로 바뀌는 것을 막는다).
    """
    if ttl_hours is None:
        ttl_hours = float(config.get("data.cache_ttl_hours", 12))
    path = _key_to_path(namespace, key)

    def _read() -> pd.DataFrame:
        if date_index:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return pd.read_csv(path, index_col=0, dtype={0: str}).rename(index=str)

    if not force_refresh and _is_fresh(path, ttl_hours):
        cached = _read()
        if not cached.empty:
            return cached

    produced = producer()
    if produced is not None and not produced.empty:
        produced.to_csv(path, encoding="utf-8")
        return produced

    # 새로 못 받았는데 오래된 캐시라도 있으면 그거라도 쓴다 (stale-if-error).
    if path.exists():
        return _read()
    return pd.DataFrame()


def clear(namespace: str | None = None) -> int:
    """캐시 파일을 지우고 삭제한 개수를 반환한다."""
    root = config.cache_dir() / namespace if namespace else config.cache_dir()
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob("*.csv"):
        path.unlink()
        removed += 1
    return removed
