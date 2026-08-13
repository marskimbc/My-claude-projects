"""고장모드 판정 검증.

가장 중요한 것은 **막힘이 아닌 것을 막힘이라고 하지 않는 것**이다.
차압 정상 + 열효율 저하 + 섹터 편차 큼 → 로터리밸브 씰 누설/축열재 침하이며,
이 경우 세정이나 bake-out 은 아무 효과가 없다. 오진의 대가가 크므로 별도로 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_health.pipeline import analyze

SECTOR_COLS = [f"TT_SECTOR_{i:02d}" for i in range(1, 13)]


def _inject_from(df: pd.DataFrame, start_day: int) -> np.ndarray:
    """start_day 이후 행을 가리키는 마스크."""
    elapsed = (df["TIMESTAMP"] - df["TIMESTAMP"].min()).dt.total_seconds() / 86400.0
    return (elapsed >= start_day).to_numpy()


def test_rapid_plugging_diagnosed_as_channeling(rapid):
    """급성 막힘 + 큰 섹터 편차 → 채널링으로 판정되어야 한다."""
    guide = rapid.guidance()
    mode_ids = [m.id for m in guide["all_modes"]]

    assert "channeling" in mode_ids
    assert guide["all_modes"][0].id == "channeling", "우선순위상 채널링이 1순위여야 한다"
    assert any("세정" in a or "교체" in a for a in guide["all_modes"][0].actions)


def test_gradual_fouling_diagnosed_as_uniform(gradual):
    """편차 없이 고르게 막히면 균일 오염 → bake-out 권고여야 한다."""
    guide = gradual.guidance()
    mode_ids = [m.id for m in guide["all_modes"]]

    assert "uniform_fouling" in mode_ids
    assert "channeling" not in mode_ids, "편차가 작은데 채널링으로 오진했다"
    primary = guide["all_modes"][0]
    assert any("bake-out" in a for a in primary.actions)


def test_normal_operation_has_no_failure_mode(normal):
    """정상 운전에서는 고장모드가 판정되지 않아야 한다."""
    guide = normal.guidance()
    assert guide["all_modes"] == []
    assert "미검출" in guide["headline"]


def test_seal_leak_not_misdiagnosed_as_blockage(scenarios):
    """차압 정상 + 효율 저하 + 섹터 편차 → 씰 누설/침하로 판정되어야 한다.

    이것을 막힘으로 오진하면 현장은 효과 없는 세정에 정지 시간을 쓰게 된다.
    """
    df = scenarios["normal"].copy()
    mask = _inject_from(df, 240)

    # 차압은 건드리지 않는다 — 유로는 정상
    # 축열층을 우회하는 만큼 배출온도가 오른다 (TER 약 6.6%p 저하)
    df.loc[mask, "TT_STACK"] = df.loc[mask, "TT_STACK"] + 50.0
    # 우회 경로가 생기면 섹터별 온도가 크게 갈린다
    for i, col in enumerate(SECTOR_COLS):
        offset = 30.0 if i % 2 == 0 else -30.0
        df.loc[mask, col] = df.loc[mask, col] + offset

    analysis = analyze(df)
    snap = analysis.snapshot()
    mode_ids = [m.id for m in analysis.guidance()["all_modes"]]

    assert snap.degradations["A1"] < 0.20, "차압은 정상이어야 한다"
    assert "media_settling_or_seal_leak" in mode_ids, f"판정된 모드: {mode_ids}"
    assert "uniform_fouling" not in mode_ids
    assert "channeling" not in mode_ids

    primary = analysis.guidance()["all_modes"][0]
    assert primary.id == "media_settling_or_seal_leak", "우선순위 5로 가장 먼저 판정되어야 한다"
    assert any("씰" in a for a in primary.actions)
    assert any("효과 없음" in a for a in primary.actions), "세정 금지 안내가 있어야 한다"


def test_media_loss_detected_when_dp_drops_below_baseline(scenarios):
    """차압이 오히려 내려가고 효율도 떨어지면 축열재 유실로 판정되어야 한다."""
    df = scenarios["normal"].copy()
    mask = _inject_from(df, 240)

    df.loc[mask, "RTO_DP_BED"] = df.loc[mask, "RTO_DP_BED"] * 0.80   # 유로가 뚫림
    df.loc[mask, "TT_STACK"] = df.loc[mask, "TT_STACK"] + 45.0       # 축열 성능 상실
    for i, col in enumerate(SECTOR_COLS):
        df.loc[mask, col] = df.loc[mask, col] + (28.0 if i % 3 == 0 else -14.0)

    analysis = analyze(df)
    mode_ids = [m.id for m in analysis.guidance()["all_modes"]]

    assert analysis.indicators.raw["A1"].iloc[-1] < 0.95, "차압이 베이스라인 아래여야 한다"
    assert "media_loss" in mode_ids, f"판정된 모드: {mode_ids}"


def test_burner_fault_separated_from_blockage(scenarios):
    """차압·효율 정상인데 연료만 늘면 버너/제어계 이상으로 판정되어야 한다."""
    df = scenarios["normal"].copy()
    mask = _inject_from(df, 240)
    df.loc[mask, "FUEL_LNG_FLOW"] = df.loc[mask, "FUEL_LNG_FLOW"] * 1.9

    analysis = analyze(df)
    mode_ids = [m.id for m in analysis.guidance()["all_modes"]]

    assert "burner_or_control" in mode_ids, f"판정된 모드: {mode_ids}"
    assert "uniform_fouling" not in mode_ids
    assert "channeling" not in mode_ids


def test_diagnosis_includes_verifiable_evidence(rapid):
    """판정 근거에 실제 측정값이 포함되어야 현장에서 검증할 수 있다."""
    primary = rapid.guidance()["all_modes"][0]

    assert primary.evidence, "판정 근거가 비어 있다"
    assert any("측정" in e for e in primary.evidence)
    assert 0.0 <= primary.confidence <= 1.0


def test_guidance_lists_top_contributors(gradual):
    """가이드에 감점 상위 지표가 함께 제시되어야 한다."""
    guide = gradual.guidance()
    assert guide["top_contributors"]
    assert all("감점" in item for item in guide["top_contributors"])


def test_rules_skipped_when_required_indicator_unavailable(scenarios):
    """섹터 온도가 없으면 이를 요구하는 규칙은 판정하지 않아야 한다(추측 금지)."""
    df = scenarios["rapid_plugging"].drop(columns=SECTOR_COLS)
    analysis = analyze(df)
    mode_ids = [m.id for m in analysis.guidance()["all_modes"]]

    assert "channeling" not in mode_ids, "B3 없이 채널링을 단정하면 안 된다"
    assert "B3" in analysis.indicators.excluded
