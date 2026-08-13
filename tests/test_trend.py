"""추세 분석 검증 — 기울기, 변화점 검출, 잔여여유 예측."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_health.trend import detect_change_points, estimate_rul, ewma, theil_sen_slope

# generate_sample.py 가 rapid_plugging 에 막힘을 주입한 시점
INJECTED_ONSET = pd.Timestamp("2025-07-30")   # START(2025-01-06) + 205일


def _daily_series(values, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_theil_sen_recovers_known_slope():
    series = _daily_series(1.0 + 0.02 * np.arange(100))
    slope, lo, hi = theil_sen_slope(series)
    assert slope == pytest.approx(0.02, rel=1e-6)
    assert lo <= 0.02 <= hi


def test_theil_sen_is_robust_to_outliers():
    """최소자승과 달리 계측 스파이크에 끌려가지 않아야 한다."""
    values = 1.0 + 0.02 * np.arange(100)
    values[[10, 40, 77]] = 50.0   # 계측 스파이크

    series = _daily_series(values)
    robust, _, _ = theil_sen_slope(series)
    ols = np.polyfit(np.arange(100), values, 1)[0]

    assert robust == pytest.approx(0.02, abs=0.002)
    assert abs(ols - 0.02) > abs(robust - 0.02) * 10, "최소자승은 크게 끌려가야 대비가 성립"


def test_theil_sen_needs_enough_points():
    assert np.isnan(theil_sen_slope(_daily_series([1.0, 2.0]))[0])


def test_ewma_smooths_without_shifting_level():
    series = _daily_series(np.full(200, 5.0) + np.random.default_rng(0).normal(0, 1, 200))
    smoothed = ewma(series, 7)
    assert smoothed.std() < series.std() / 2
    assert smoothed.iloc[-50:].mean() == pytest.approx(5.0, abs=0.5)


# --- 변화점 탐지 ------------------------------------------------------------

def test_cusum_detects_injected_onset(rapid):
    """주입한 막힘 개시 시점을 2주 내에 검출해야 한다."""
    ups = [p for p in rapid.change_points if p.direction == "up"]
    assert ups, "급성 막힘인데 변화점을 하나도 못 찾았다"

    first = ups[0].date
    lag = (first - INJECTED_ONSET).days
    assert -3 <= lag <= 14, f"검출 시점 {first:%Y-%m-%d} (주입 대비 {lag:+d}일)"


def test_cusum_no_false_alarm_in_normal_operation(normal):
    """정상 운전에서는 악화 알람이 발생하지 않아야 한다."""
    ups = [p for p in normal.change_points if p.direction == "up"]
    assert len(ups) == 0, f"정상 운전에서 오알람 {len(ups)}건: {[str(p.date.date()) for p in ups]}"


def test_cusum_detects_bakeout_recovery(normal):
    """bake-out 으로 차압이 내려간 시점은 '개선' 방향으로 잡혀야 한다."""
    downs = [p for p in normal.change_points if p.direction == "down"]
    assert downs, "bake-out 효과가 검출되지 않았다"

    bakeout_dates = [pd.Timestamp("2025-05-19"), pd.Timestamp("2025-09-22")]
    for bd in bakeout_dates:
        assert any(abs((p.date - bd).days) <= 21 for p in downs), f"{bd:%Y-%m-%d} bake-out 미검출"


def test_cusum_alarms_persistently_during_ongoing_fouling(gradual):
    """계속 나빠지는 중이면 알람이 반복 발생해야 한다."""
    ups = [p for p in gradual.change_points if p.direction == "up"]
    assert len(ups) >= 3


def test_cusum_ignores_slow_normal_drift():
    """수준은 올랐지만 속도가 일정한 완만한 드리프트는 알람 대상이 아니다."""
    rng = np.random.default_rng(1)
    values = 1.0 + 0.0002 * np.arange(365) + rng.normal(0, 0.01, 365)
    _, points = detect_change_points(_daily_series(values), h_sigma=6.0)
    assert len([p for p in points if p.direction == "up"]) == 0


def test_cusum_catches_step_change():
    """뚜렷한 계단 변화는 반드시 잡아야 한다."""
    rng = np.random.default_rng(2)
    values = np.concatenate([
        1.0 + rng.normal(0, 0.01, 180),
        1.0 + 0.02 * np.arange(1, 186) + rng.normal(0, 0.01, 185),
    ])
    _, points = detect_change_points(_daily_series(values), h_sigma=6.0)
    ups = [p for p in points if p.direction == "up"]

    assert ups
    change_date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=180)
    assert abs((ups[0].date - change_date).days) <= 10


# --- 잔여여유 예측 ----------------------------------------------------------

def test_rul_projects_linear_trend_correctly():
    """한계까지 남은 일수를 선형 외삽으로 정확히 계산해야 한다."""
    series = _daily_series(1.0 + 0.01 * np.arange(100))   # 현재 1.99, 0.01/일
    rul = estimate_rul(series, limit_value=3.0, fit_window_days=100)

    assert rul.days_remaining is not None
    # 현재값은 마지막 7일 중앙값(≈1.96) → (3.0-1.96)/0.01 ≈ 104일
    assert rul.days_remaining == pytest.approx(104, abs=5)
    assert rul.slope_per_day == pytest.approx(0.01, rel=1e-6)


def test_rul_returns_none_when_flat():
    """추세가 없으면 도달 시점을 만들어내지 않아야 한다."""
    rng = np.random.default_rng(3)
    series = _daily_series(1.0 + rng.normal(0, 0.005, 200))
    rul = estimate_rul(series, limit_value=3.0)

    assert rul.days_remaining is None
    assert "산출 불가" in rul.note or "확인되지 않아" in rul.note


def test_rul_reports_zero_when_limit_already_exceeded():
    series = _daily_series(np.linspace(2.0, 3.5, 100))
    rul = estimate_rul(series, limit_value=3.0)
    assert rul.days_remaining == 0.0
    assert "한계 도달" in rul.note


def test_rapid_scenario_has_short_rul(rapid):
    """급성 막힘은 잔여여유가 매우 짧게 나와야 한다."""
    assert rapid.rul.days_remaining is not None
    assert rapid.rul.days_remaining < 60


def test_normal_scenario_has_long_rul(normal):
    """정상 운전은 잔여여유가 길거나 산출 불가여야 한다."""
    assert normal.rul.days_remaining is None or normal.rul.days_remaining > 730


def test_rul_ordering_across_scenarios(normal, gradual, rapid):
    """정상 > 완만 > 급성 순으로 잔여여유가 짧아져야 한다."""
    def days(analysis):
        return analysis.rul.days_remaining if analysis.rul.days_remaining is not None else 1e9

    assert days(normal) > days(gradual) > days(rapid)
