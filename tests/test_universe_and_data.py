import datetime as dt

import pandas as pd
import pytest

from stockmgr import universe
from stockmgr.data import cache
from stockmgr.data.krx import _COLUMN_MAP, _normalize
from stockmgr.data.loader import MarketData


def test_keyword_match_respects_exclude():
    assert universe._matches("KODEX 방산", ["방산"], [])
    assert not universe._matches("KODEX 방산 인버스", ["방산"], ["인버스"])


def test_short_ascii_keyword_uses_word_boundary():
    assert universe._matches("TIGER AI반도체", ["AI"], [])
    # 'AI' 가 다른 단어에 묻힌 경우는 매칭하지 않는다.
    assert not universe._matches("KODEX CHAIN", ["AI"], [])


def test_keyword_match_ignores_spacing():
    assert universe._matches("KODEX 2 차전지", ["2차전지"], [])


def test_resolve_theme_finds_members(market):
    theme = universe.resolve_theme(market, "defense")
    assert not theme.empty
    assert all(isinstance(t, str) for t in theme.tickers)


def test_resolve_unknown_theme_raises(market):
    with pytest.raises(KeyError):
        universe.resolve_theme(market, "없는테마")


def test_all_themes_covers_config(market):
    resolved = universe.all_themes(market)
    assert set(resolved) == set(universe.theme_labels())


def test_column_normalization_translates_korean():
    raw = pd.DataFrame(
        {"시가": [1], "고가": [2], "저가": [0], "종가": [1], "거래량": [10]},
        index=pd.Index(["20260810"], name="날짜"),
    )
    normalized = _normalize(raw)
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.index.name == "date"
    assert "거래대금" in _COLUMN_MAP


def test_cache_preserves_leading_zero_tickers(tmp_path, monkeypatch):
    from stockmgr import config
    monkeypatch.setattr(config, "cache_dir", lambda: tmp_path)

    frame = pd.DataFrame({"weight": [3.0]}, index=pd.Index(["005930"], name="ticker"))
    calls = []

    def produce():
        calls.append(1)
        return frame

    first = cache.frame("t", {"k": 1}, produce, date_index=False)
    second = cache.frame("t", {"k": 1}, produce, date_index=False)
    assert len(calls) == 1                     # 두 번째는 캐시에서 읽는다
    assert list(second.index) == ["005930"]    # 앞자리 0 이 살아 있다
    assert first.equals(second)


def test_cache_does_not_store_empty_results(tmp_path, monkeypatch):
    from stockmgr import config
    monkeypatch.setattr(config, "cache_dir", lambda: tmp_path)

    calls = []

    def produce():
        calls.append(1)
        return pd.DataFrame()

    cache.frame("t", {"k": 2}, produce)
    cache.frame("t", {"k": 2}, produce)
    assert len(calls) == 2


def test_offline_market_is_deterministic():
    a = MarketData(mode="offline", end=dt.date(2026, 8, 12))
    b = MarketData(mode="offline", end=dt.date(2026, 8, 12))
    assert a.benchmark("kospi")["close"].equals(b.benchmark("kospi")["close"])


def test_offline_market_never_calls_network(market, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("오프라인 모드에서 네트워크를 호출했습니다")

    monkeypatch.setattr("stockmgr.data.krx._pykrx", explode)
    monkeypatch.setattr("stockmgr.data.macro._yfinance", explode)
    assert not market.benchmark("kospi").empty
    assert not market.price("005930").empty
    assert not market.macro_series("GC=F").empty


def test_auto_mode_degrades_when_source_fails(monkeypatch):
    from stockmgr.data import krx

    def explode(*args, **kwargs):
        raise krx.DataUnavailable("차단됨")

    monkeypatch.setattr(krx, "index_ohlcv", explode)
    data = MarketData(mode="auto", end=dt.date(2026, 8, 12))
    prices = data.benchmark("kospi")
    assert data.degraded is True
    assert not prices.empty


def test_live_mode_propagates_failure(monkeypatch):
    from stockmgr.data import krx

    def explode(*args, **kwargs):
        raise krx.DataUnavailable("차단됨")

    monkeypatch.setattr(krx, "index_ohlcv", explode)
    data = MarketData(mode="live", end=dt.date(2026, 8, 12))
    with pytest.raises(krx.DataUnavailable):
        data.benchmark("kospi")
