import numpy as np
import pandas as pd
import pytest

from stockmgr.analysis import indicators as ind


@pytest.fixture
def rising() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=300)
    close = pd.Series(np.linspace(100, 200, len(index)), index=index)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1_000_000.0,
    })


def test_sma_needs_full_window(rising):
    result = ind.sma(rising["close"], 20)
    assert result.iloc[:19].isna().all()
    assert result.iloc[19] == pytest.approx(rising["close"].iloc[:20].mean())


def test_rsi_stays_in_range_and_is_high_when_only_rising(rising):
    result = ind.rsi(rising["close"], 14).dropna()
    assert ((result >= 0) & (result <= 100)).all()
    assert result.iloc[-1] > 90


def test_atr_positive_and_matches_range(rising):
    result = ind.atr(rising, 14).dropna()
    assert (result > 0).all()


def test_max_drawdown_zero_for_monotonic_series(rising):
    assert ind.max_drawdown(rising["close"]) == pytest.approx(0.0)


def test_max_drawdown_detects_decline():
    values = pd.Series([100.0, 120.0, 60.0, 80.0])
    assert ind.max_drawdown(values) == pytest.approx(-50.0)


def test_returns_uses_window_offset(rising):
    close = rising["close"]
    expected = (close.iloc[-1] / close.iloc[-21] - 1) * 100
    assert ind.returns(close, 20) == pytest.approx(expected)


def test_returns_nan_when_history_too_short():
    assert np.isnan(ind.returns(pd.Series([1.0, 2.0]), 20))


def test_slope_positive_for_uptrend(rising):
    assert ind.slope_pct(rising["close"], 20) > 0


def test_bollinger_pct_b_between_bounds(rising):
    bands = ind.bollinger(rising["close"], 20, 2.0).dropna()
    assert (bands["upper"] > bands["mid"]).all()
    assert (bands["mid"] > bands["lower"]).all()


def test_adx_is_bounded(rising):
    result = ind.adx(rising, 14)["adx"].dropna()
    assert ((result >= 0) & (result <= 100)).all()


def test_enrich_adds_expected_columns(rising):
    enriched = ind.enrich(rising, {})
    for column in ("ma_short", "ma_mid", "rsi", "macd", "atr", "adx", "atr_pct"):
        assert column in enriched.columns
    assert len(enriched) == len(rising)
