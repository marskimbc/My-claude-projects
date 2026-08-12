"""해외 매크로/원자재 어댑터 (yfinance).

금·은·구리·원유·달러·금리·해외지수처럼 KRX 에 없는 시계열을 받아온다.
KRX 어댑터와 동일한 컬럼 규약(open/high/low/close/volume)을 지킨다.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from . import cache
from .krx import DataUnavailable

log = logging.getLogger(__name__)


def _yfinance():
    try:
        import yfinance as yf  # noqa: PLC0415 - 선택적 의존성
    except ImportError as exc:  # pragma: no cover - 설치 환경에 따라 다름
        raise DataUnavailable("yfinance 가 설치되어 있지 않습니다. pip install yfinance") from exc
    return yf


def _iso(value: dt.date | str) -> str:
    if isinstance(value, str):
        return value if "-" in value else f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value.isoformat()


def ohlcv(symbol: str, start: dt.date | str, end: dt.date | str,
          force_refresh: bool = False) -> pd.DataFrame:
    """야후 파이낸스 일봉. 실패하면 빈 DataFrame(캐시 없을 때)."""

    def produce() -> pd.DataFrame:
        yf = _yfinance()
        try:
            raw = yf.download(symbol, start=_iso(start), end=_iso(end),
                              progress=False, auto_adjust=True)
        except Exception as exc:  # pragma: no cover - 네트워크 의존
            raise DataUnavailable(f"야후 조회 실패({symbol}): {exc}") from exc
        if raw is None or raw.empty:
            raise DataUnavailable(f"야후 응답이 비었습니다({symbol})")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.index.name = "date"
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
        return raw[keep].sort_index()

    try:
        return cache.frame("macro", {"s": symbol, "from": _iso(start), "to": _iso(end)},
                           produce, force_refresh=force_refresh)
    except DataUnavailable as exc:
        log.warning("%s", exc)
        return pd.DataFrame()
