import numpy as np
import pandas as pd
import pytest

from stockmgr.analysis import indicators as ind
from stockmgr.analysis.score import ScoreCard, evaluate, squash, trend_score
from stockmgr.analysis.trend import DOWN, STRONG_DOWN, STRONG_UP, UP, classify
from stockmgr.strategy import signals


def _frame(values: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(values))
    close = pd.Series(values, index=index)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1_000_000.0,
    })


UPTREND = _frame(np.linspace(100, 220, 320))
DOWNTREND = _frame(np.linspace(220, 100, 320))


def test_squash_is_centered_and_monotonic():
    assert squash(0, 1) == pytest.approx(50.0)
    assert squash(5, 1) > squash(1, 1) > squash(0, 1) > squash(-1, 1)
    assert 0 <= squash(-100, 1) <= 100


def test_squash_handles_nan():
    assert squash(float("nan"), 1) == 50.0


def test_trend_score_separates_up_and_down():
    up = trend_score(ind.enrich(UPTREND, {}))
    down = trend_score(ind.enrich(DOWNTREND, {}))
    assert up > 70 > down


def test_classify_labels_match_direction():
    assert classify(UPTREND).label in (STRONG_UP, UP)
    assert classify(DOWNTREND).label in (STRONG_DOWN, DOWN)


def test_classify_handles_short_history():
    state = classify(_frame(np.linspace(100, 110, 10)))
    assert state.reasons == ["데이터 부족"]


def test_evaluate_returns_bounded_score_and_metrics():
    card = evaluate("000000", UPTREND, name="테스트")
    assert 0 <= card.total <= 100
    assert card.metrics["close"] == pytest.approx(UPTREND["close"].iloc[-1])
    assert set(card.components) == {
        "trend", "momentum", "relative_strength", "flow", "risk"}


def test_evaluate_on_empty_frame_is_neutral():
    card = evaluate("000000", pd.DataFrame())
    assert card.total == 50.0


def test_relative_strength_rewards_outperformance():
    bench = DOWNTREND["close"]
    strong = evaluate("A", UPTREND, benchmark=bench)
    weak = evaluate("B", DOWNTREND, benchmark=bench)
    assert strong.components["relative_strength"] > weak.components["relative_strength"]


def test_signal_action_follows_score():
    high = signals.generate(ScoreCard("A", "A", total=85.0,
                                      components={"trend": 90.0}, metrics={}))
    low = signals.generate(ScoreCard("B", "B", total=20.0,
                                     components={"trend": 10.0}, metrics={}))
    assert high.action == signals.BUY
    assert low.action == signals.SELL


def test_risk_off_regime_blocks_new_buys():
    card = ScoreCard("A", "A", total=85.0, components={"trend": 90.0}, metrics={})
    signal = signals.generate(card, regime_label="risk_off")
    assert signal.action == signals.WATCH
    assert any("위험회피" in c for c in signal.cautions)


def test_overbought_downgrades_buy_to_watch():
    card = ScoreCard("A", "A", total=85.0, components={"trend": 90.0},
                     metrics={"rsi": 88.0})
    signal = signals.generate(card)
    assert signal.action == signals.WATCH
    assert any("과열" in c for c in signal.cautions)


def test_illiquid_candidate_is_downgraded():
    card = ScoreCard("A", "A", total=85.0, components={"trend": 90.0},
                     metrics={"avg_value_20d": 1_000.0})
    signal = signals.generate(card)
    assert signal.action == signals.WATCH
    assert any("유동성" in c for c in signal.cautions)
