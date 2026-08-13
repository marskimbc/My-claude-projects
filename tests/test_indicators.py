"""지표 산출 검증 — 열화를 주입하면 해당 지표가 반응하는가."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_health.pipeline import analyze


def test_all_indicators_active_with_full_tag_set(normal):
    """태그가 모두 있으면 15개 지표가 전부 채점되어야 한다."""
    assert len(normal.indicators.active_ids) == 15
    assert normal.indicators.excluded == {}


def test_indicator_raw_metrics_have_expected_direction(normal, rapid):
    """모든 지표는 '높을수록 나쁨'으로 통일되어 있어야 한다."""
    n_raw = normal.indicators.raw.iloc[-1]
    r_raw = rapid.indicators.raw.iloc[-1]

    for ind_id in ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]:
        assert r_raw[ind_id] > n_raw[ind_id], f"{ind_id}: 막힘 시 값이 더 커야 한다"


def test_baseline_period_scores_near_perfect(gradual):
    """베이스라인 구간에서는 모든 막힘 지표가 사실상 0 이어야 한다."""
    base_end = gradual.baseline.end
    deg = gradual.indicators.degradation
    in_baseline = deg.loc[deg.index < base_end]

    for ind_id in ["A1", "A3", "B1", "B2", "C1"]:
        assert in_baseline[ind_id].max() < 0.15, f"{ind_id} 베이스라인 구간에서 감점 발생"


def test_dp_ratio_is_near_one_at_baseline(gradual):
    """정규화 차압 비율은 베이스라인 구간에서 1.0 근처여야 한다."""
    a1 = gradual.indicators.raw["A1"]
    baseline_values = a1.loc[a1.index < gradual.baseline.end]
    assert baseline_values.median() == pytest.approx(1.0, abs=0.03)


def test_cumulative_indicators_increase_monotonically(normal):
    """누적 지표(D군)는 단조 증가해야 한다."""
    for ind_id in ["D1", "D2"]:
        series = normal.indicators.raw[ind_id].dropna()
        assert (series.diff().dropna() >= -1e-9).all(), f"{ind_id} 가 감소했다"


def test_sector_temp_std_responds_to_channeling(normal, rapid):
    """채널링이 심한 시나리오에서 섹터 온도편차가 크게 벌어져야 한다."""
    assert normal.indicators.raw["B3"].iloc[-1] < 10.0
    assert rapid.indicators.raw["B3"].iloc[-1] > 25.0


def test_indicator_excluded_when_required_tag_missing(scenarios):
    """댐퍼 태그가 없으면 A4 가 제외되고 사유가 기록되어야 한다."""
    df = scenarios["normal"].drop(columns=["DAMPER_OPEN"])
    analysis = analyze(df)

    assert "A4" in analysis.indicators.excluded
    assert "태그 미보유" in analysis.indicators.excluded["A4"]
    assert "A4" not in analysis.indicators.degradation.columns


def test_fan_load_falls_back_to_current_when_inverter_missing(scenarios):
    """인버터 주파수가 없으면 전류로 대체 산출되어야 한다."""
    df = scenarios["gradual_fouling"].drop(columns=["FAN_INV_HZ"])
    analysis = analyze(df)

    assert "A3" in analysis.indicators.degradation.columns, "전류 대체 산출이 동작해야 한다"
    assert analysis.indicators.raw["A3"].iloc[-1] > 1.0


def test_injected_blockage_raises_dp_indicator(scenarios):
    """차압만 인위적으로 30% 올리면 A1 이 그만큼 반응해야 한다."""
    df = scenarios["normal"].copy()
    half = len(df) // 2
    df.loc[half:, "RTO_DP_BED"] = df.loc[half:, "RTO_DP_BED"] * 1.30

    analysis = analyze(df)
    a1 = analysis.indicators.raw["A1"]
    before = a1.iloc[: len(a1) // 2 - 10].median()
    after = a1.iloc[-30:].median()

    assert after / before == pytest.approx(1.30, rel=0.05)


def test_steady_state_filter_excludes_shutdowns(normal):
    """정지 구간이 정상운전 판정에서 빠졌는지 확인."""
    df = normal.dataset.df
    shutdown_rows = df[df["flow"] < 100]
    assert len(shutdown_rows) > 0, "샘플에 정지 구간이 있어야 한다"
    assert not shutdown_rows["is_steady"].any(), "정지 구간이 정상운전으로 잡혔다"


def test_daily_aggregation_drops_low_coverage_days(normal):
    """관측이 적은 날은 일 집계에서 빠져야 한다."""
    assert (normal.daily["runtime_hours"] >= 4.0).all()
