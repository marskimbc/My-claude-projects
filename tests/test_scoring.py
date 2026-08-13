"""채점 엔진 검증 — 배점 합계, 등급 판정, 시나리오별 기대 결과."""

from __future__ import annotations

import pandas as pd
import pytest

from rto_health import indicators as ind_mod
from rto_health import preprocess, scoring
from rto_health.io_loader import load_config


def test_configured_points_sum_to_100(config):
    """weights.yaml 의 배점 합계는 정확히 100이어야 한다."""
    specs = ind_mod.load_specs(config)
    total = sum(s.points for s in specs.values())
    assert total == pytest.approx(100.0)


def test_group_points_match_indicator_sums(config):
    """그룹 총점과 소속 지표 배점 합이 일치해야 한다."""
    specs = ind_mod.load_specs(config)
    for gid, meta in config.weights["groups"].items():
        members = sum(s.points for s in specs.values() if s.group == gid)
        assert members == pytest.approx(float(meta["points"])), f"그룹 {gid}"


def test_effective_points_sum_to_100_after_redistribution(normal):
    """지표가 제외되어도 유효 배점 합계는 100을 유지해야 한다."""
    total = sum(normal.indicators.effective_points.values())
    assert total == pytest.approx(100.0)


def test_points_redistributed_when_tags_missing(scenarios):
    """섹터 온도·VOC 태그가 없어도 배점 합계 100과 정상 채점이 유지되어야 한다."""
    df = scenarios["normal"].drop(
        columns=[c for c in scenarios["normal"].columns if c.startswith("TT_SECTOR_")]
        + ["VOC_INLET", "TMS_OUTLET"]
    )
    from rto_health.pipeline import analyze

    analysis = analyze(df)

    assert "B3" in analysis.indicators.excluded, "섹터 온도 지표는 제외되어야 한다"
    assert "C4" in analysis.indicators.excluded
    assert sum(analysis.indicators.effective_points.values()) == pytest.approx(100.0)
    # 남은 지표의 배점은 원래보다 커진다(재배분)
    assert analysis.indicators.effective_points["A1"] > 18.0


def test_degradation_mapping_bounds():
    """열화도는 항상 0~1 범위이고 onset/severe 경계에서 정확해야 한다."""
    metric = pd.Series([0.5, 1.0, 1.1, 1.55, 2.0, 3.0])
    deg = ind_mod.to_degradation(metric, onset=1.1, severe=2.0)

    assert deg.min() >= 0.0 and deg.max() <= 1.0
    assert deg.iloc[2] == pytest.approx(0.0)   # onset
    assert deg.iloc[4] == pytest.approx(1.0)   # severe
    assert deg.iloc[3] == pytest.approx(0.5)   # 중간
    assert deg.iloc[0] == 0.0                  # onset 미만은 무감점


def test_all_degradations_within_bounds(analyses):
    for name, analysis in analyses.items():
        deg = analysis.indicators.degradation
        assert deg.min().min() >= 0.0, name
        assert deg.max().max() <= 1.0, name


def test_score_equals_100_minus_deductions(rapid):
    """점수는 100에서 감점 합계를 뺀 값이어야 한다."""
    snap = rapid.snapshot()
    total_deduction = sum(snap.deductions.values())
    assert snap.score == pytest.approx(100.0 - total_deduction, abs=1e-6)


def test_grade_thresholds(config):
    """등급 경계가 weights.yaml 정의대로 동작해야 한다."""
    cases = [(95, "A"), (90, "A"), (89.9, "B"), (75, "B"), (74.9, "C"),
             (60, "C"), (59.9, "D"), (45, "D"), (44.9, "E"), (0, "E")]
    for score, expected in cases:
        assert scoring._grade_for(score, config)["grade"] == expected, f"{score}점"


# --- 시나리오별 기대 결과 --------------------------------------------------

def test_normal_scenario_stays_healthy(normal):
    """정상 운전은 A등급을 유지해야 한다 (오탐 없음)."""
    snap = normal.snapshot()
    assert snap.score >= 90.0, f"정상 시나리오가 {snap.score:.1f}점"
    assert snap.grade == "A"


def test_normal_scenario_has_no_blockage_signal(normal):
    """정상 운전에서 막힘 직접 지표(A군)는 거의 깎이지 않아야 한다.

    D군(누적 부하)은 세정 주기가 다가오면 정상적으로 감점되므로 제외한다.
    """
    snap = normal.snapshot()
    assert snap.group_deductions.get("A", 0.0) < 2.0
    assert snap.group_deductions.get("B", 0.0) < 2.0


def test_gradual_fouling_reaches_progressing_grade(gradual):
    """서서히 막히는 시나리오는 1년 후 '진행' 수준(C~D)에 도달해야 한다."""
    snap = gradual.snapshot()
    assert 45.0 <= snap.score < 75.0, f"{snap.score:.1f}점"
    assert snap.grade in ("C", "D")


def test_rapid_plugging_reaches_critical_grade(rapid):
    """급성 막힘은 E등급(위험)까지 떨어져야 한다."""
    snap = rapid.snapshot()
    assert snap.score < 45.0, f"{snap.score:.1f}점"
    assert snap.grade == "E"


def test_scores_are_ordered_across_scenarios(normal, gradual, rapid):
    """정상 > 완만한 막힘 > 급성 막힘 순서가 지켜져야 한다."""
    assert normal.snapshot().score > gradual.snapshot().score > rapid.snapshot().score


def test_score_declines_monotonically_in_rapid_scenario(rapid):
    """급성 막힘 시나리오에서 점수는 전반적으로 하락해야 한다."""
    monthly = rapid.score.resample("30D").median()
    first_half = monthly.iloc[: len(monthly) // 2].mean()
    second_half = monthly.iloc[len(monthly) // 2 :].mean()
    assert second_half < first_half - 30.0


def test_snapshot_at_earlier_date_scores_higher(rapid):
    """막힘 이전 시점을 조회하면 훨씬 높은 점수가 나와야 한다."""
    early = rapid.snapshot("2025-05-01")
    latest = rapid.snapshot()
    assert early.score > 85.0
    assert early.score > latest.score + 40.0


def test_score_table_covers_every_indicator(rapid, config):
    """지표 상세표에 15개 지표가 모두 나와야 한다 (제외된 것도 사유와 함께)."""
    table = scoring.score_table(rapid.snapshot(), rapid.indicators)
    assert len(table) == len(ind_mod.load_specs(config))
    assert table["감점"].sum() == pytest.approx(100.0 - rapid.snapshot().score, abs=0.5)
