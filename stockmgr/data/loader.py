"""데이터 접근 파사드.

분석/전략 계층은 이 클래스만 쓴다. 실제 시세(KRX/야후)를 쓸지,
합성 데이터를 쓸지는 여기서 결정한다.

mode
  live    : 실시간만 사용. 실패하면 예외.
  offline : 합성 데이터만 사용 (네트워크 없음).
  auto    : 실시간 시도 후 실패하면 합성으로 자동 강등 (기본값).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass, field

import pandas as pd

from .. import config
from . import krx, macro, synthetic
from .krx import DataUnavailable

log = logging.getLogger(__name__)

Mode = str  # "live" | "offline" | "auto"


def _hash_unit(text: str) -> float:
    """문자열 → [0,1) 결정적 실수. 합성 데이터의 개성 부여용."""
    digest = hashlib.sha1(text.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


@dataclass
class MarketData:
    mode: Mode = "auto"
    end: dt.date = field(default_factory=dt.date.today)
    lookback_days: int | None = None
    force_refresh: bool = False
    degraded: bool = False  # auto 모드에서 합성으로 강등되었는지

    def __post_init__(self) -> None:
        if self.lookback_days is None:
            self.lookback_days = int(config.get("data.lookback_days", 500))
        if self.mode == "offline":
            self.degraded = True

    # -- 공통 --------------------------------------------------------------
    @property
    def start(self) -> dt.date:
        return self.end - dt.timedelta(days=int(self.lookback_days))

    @property
    def seed(self) -> int:
        return int(config.get("data.offline_seed", 20260812))

    @property
    def offline(self) -> bool:
        return self.mode == "offline" or self.degraded

    def _fallback(self, what: str, exc: Exception) -> bool:
        """auto 모드면 합성으로 강등하고 True, live 모드면 예외를 올린다."""
        if self.mode == "live":
            raise exc
        if not self.degraded:
            log.warning("실시간 데이터를 쓸 수 없어 오프라인(합성) 모드로 전환합니다: %s", exc)
        self.degraded = True
        return True

    # 오프라인 리포트가 터무니없는 숫자를 보여주지 않도록 대표 자산의 기준가를 고정한다.
    _MACRO_BASE = {
        "KRW=X": 1380.0, "DX-Y.NYB": 103.0, "^TNX": 4.2, "^VIX": 15.0,
        "^GSPC": 5600.0, "^IXIC": 18000.0, "^SOX": 5200.0,
        "GC=F": 2400.0, "SI=F": 29.0, "HG=F": 4.3, "CL=F": 78.0, "VNQ": 90.0,
    }

    def _synthetic_price(self, symbol: str, *, kind: str = "stock") -> pd.DataFrame:
        u = _hash_unit(symbol)
        profile = {
            "index": dict(beta=1.0, alpha=0.0, vol=0.10, start_price=2600.0),
            "etf": dict(beta=0.8 + u * 0.8, alpha=(u - 0.5) * 0.30, vol=0.20 + u * 0.15),
            "stock": dict(beta=0.7 + u * 1.0, alpha=(u - 0.5) * 0.45, vol=0.28 + u * 0.25),
            "macro": dict(beta=0.2 + u * 0.4, alpha=(u - 0.5) * 0.15, vol=0.15 + u * 0.10),
        }[kind]
        if kind == "macro":
            profile["start_price"] = self._MACRO_BASE.get(symbol.split(":", 1)[-1])
        return synthetic.ohlcv(symbol, self.start, self.end,
                               base_seed=self.seed, **profile)

    # -- 지수 --------------------------------------------------------------
    def benchmark(self, key: str = "kospi") -> pd.DataFrame:
        spec = (config.themes().get("benchmarks") or {}).get(key)
        if spec is None:
            raise KeyError(f"알 수 없는 벤치마크: {key}")
        code = str(spec["code"])
        if not self.offline:
            try:
                df = krx.index_ohlcv(code, self.start, self.end,
                                     force_refresh=self.force_refresh)
                if not df.empty:
                    return df
                self._fallback(key, DataUnavailable(f"{key} 지수 데이터가 비었습니다"))
            except DataUnavailable as exc:
                self._fallback(key, exc)
        return self._synthetic_price(f"index:{code}", kind="index")

    def benchmark_label(self, key: str) -> str:
        spec = (config.themes().get("benchmarks") or {}).get(key, {})
        return spec.get("label", key)

    # -- 가격 --------------------------------------------------------------
    def price(self, ticker: str, kind: str = "stock") -> pd.DataFrame:
        """ETF/개별종목 일봉. kind 는 'etf' 또는 'stock'."""
        if not self.offline:
            fetch = krx.etf_ohlcv if kind == "etf" else krx.stock_ohlcv
            try:
                df = fetch(ticker, self.start, self.end, force_refresh=self.force_refresh)
                if not df.empty:
                    return df
            except DataUnavailable as exc:
                self._fallback(ticker, exc)
        return self._synthetic_price(f"{kind}:{ticker}", kind=kind)

    def macro_series(self, symbol: str) -> pd.DataFrame:
        if not self.offline:
            df = macro.ohlcv(symbol, self.start, self.end,
                             force_refresh=self.force_refresh)
            if not df.empty:
                return df
            log.warning("매크로 %s 를 받지 못해 합성값으로 대체합니다", symbol)
        return self._synthetic_price(f"macro:{symbol}", kind="macro")

    # -- 목록 --------------------------------------------------------------
    def etf_list(self) -> pd.DataFrame:
        if not self.offline:
            try:
                df = krx.etf_universe(self.end, force_refresh=self.force_refresh)
                if not df.empty:
                    return df
                self._fallback("etf_list", DataUnavailable("ETF 목록이 비었습니다"))
            except DataUnavailable as exc:
                self._fallback("etf_list", exc)
        return synthetic.etf_universe(config.themes())

    def stock_list(self, market: str = "KOSPI") -> pd.DataFrame:
        if not self.offline:
            try:
                df = krx.stock_universe(market, self.end, force_refresh=self.force_refresh)
                if not df.empty:
                    return df
                self._fallback("stock_list", DataUnavailable(f"{market} 목록이 비었습니다"))
            except DataUnavailable as exc:
                self._fallback("stock_list", exc)
        rows = [{"ticker": f"{i:06d}", "name": f"{market}종목{i}", "market": market}
                for i in range(1, 61)]
        return pd.DataFrame(rows)

    def constituents(self, etf_ticker: str, limit: int = 40) -> pd.DataFrame:
        """ETF 구성종목. columns: ticker, weight."""
        if not self.offline:
            try:
                df = krx.etf_constituents(etf_ticker, self.end,
                                          force_refresh=self.force_refresh)
                if not df.empty:
                    return df.head(limit)
            except DataUnavailable as exc:
                self._fallback(etf_ticker, exc)
        seed = int(_hash_unit(etf_ticker) * 500)
        rows = [{"ticker": f"{(seed + i * 37) % 999999:06d}",
                 "weight": round(max(0.5, 12 - i * 0.7), 2)}
                for i in range(min(limit, 15))]
        return pd.DataFrame(rows)

    # -- 수급 --------------------------------------------------------------
    def flow(self, ticker: str) -> pd.DataFrame:
        if not self.offline:
            try:
                df = krx.investor_flow(ticker, self.start, self.end,
                                       force_refresh=self.force_refresh)
                if not df.empty:
                    return df
            except DataUnavailable as exc:
                self._fallback(ticker, exc)
        bias = _hash_unit(f"bias:{ticker}") - 0.5
        return synthetic.investor_flow(ticker, self.start, self.end,
                                       base_seed=self.seed, bias=bias)

    def name_of(self, ticker: str, kind: str = "stock") -> str:
        """종목코드 → 종목명. 실패하면 코드 그대로."""
        source = self.etf_list() if kind == "etf" else self.stock_list()
        hit = source[source["ticker"] == ticker]
        if not hit.empty:
            return str(hit.iloc[0]["name"])
        if kind == "stock" and not self.offline:
            other = self.stock_list("KOSDAQ")
            hit = other[other["ticker"] == ticker]
            if not hit.empty:
                return str(hit.iloc[0]["name"])
        return f"종목{ticker}" if self.offline else ticker
