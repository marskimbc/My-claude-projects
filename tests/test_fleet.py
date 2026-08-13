"""20대 fleet 확장 검증.

이 확장의 핵심 주장은 두 가지다.

1. **동급기 비교가 자기 베이스라인과 다른 정보를 준다.** 같은 계열에서 한 대만
   나빠진 경우와 전부 함께 나빠진 경우는 조치가 완전히 다르다 —
   전자는 그 호기를 정비하고, 후자는 설비가 아니라 유입측을 조사해야 한다.
2. **단일 설비 경로를 깨지 않는다.** 기존 60건이 그대로 통과해야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_health import fleet as fleet_mod
from rto_health.io_loader import Config, load_config
from rto_health.pipeline import analyze

EXPECTED_FLEET_SIZE = 20


# --- 레지스트리 -------------------------------------------------------------

def test_fleet_expands_to_twenty_units(fleet_specs):
    """VRT 14대 + ORT 6대 = 20대로 전개되어야 한다."""
    assert len(fleet_specs) == EXPECTED_FLEET_SIZE
    assert sum(1 for s in fleet_specs if s.type_key == "VRT") == 14
    assert sum(1 for s in fleet_specs if s.type_key == "ORT") == 6


def test_equipment_ids_match_site_naming(fleet_specs):
    ids = {s.equipment_id for s in fleet_specs}
    for expected in ("VRT-A31101A", "VRT-A31103E", "VRT-A31105C", "ORT-A31103B"):
        assert expected in ids
    # A31104 는 존재하지 않는다
    assert not any(s.series_id == "VRT-A31104" for s in fleet_specs)


def test_series_unit_counts(fleet_specs):
    counts: dict[str, int] = {}
    for s in fleet_specs:
        counts[s.series_id] = counts.get(s.series_id, 0) + 1
    assert counts == {
        "VRT-A31101": 2, "VRT-A31102": 2, "VRT-A31103": 5, "VRT-A31105": 5,
        "ORT-A31101": 2, "ORT-A31102": 2, "ORT-A31103": 2,
    }


def test_type_defaults_are_inherited(fleet_specs):
    """VRT(유기배기)와 ORT(냄새배기)는 막힘 기전이 달라 기본값이 달라야 한다."""
    by_id = {s.equipment_id: s for s in fleet_specs}
    vrt = by_id["VRT-A31101A"]
    ort = by_id["ORT-A31101A"]

    assert vrt.label == "유기배기" and ort.label == "냄새배기"
    assert vrt.spec["design_voc_ppm"] > ort.spec["design_voc_ppm"], "유기배기가 고농도여야 한다"
    assert ort.spec["recommended_cleaning_interval_h"] > vrt.spec["recommended_cleaning_interval_h"]
    # defaults 에서만 오는 값
    assert vrt.spec["sector_count"] == 12


def test_inheritance_priority():
    """overrides > series > types > defaults 순으로 이겨야 한다."""
    specs = fleet_mod.parse_fleet({
        "defaults": {"sector_count": 12, "design_flow_cmm": 100, "media_type": "honeycomb"},
        "types": {"VRT": {"design_flow_cmm": 200, "design_voc_ppm": 400, "label": "유기배기"}},
        "series": [{"id": "VRT-X1", "units": ["A", "B"], "design_flow_cmm": 300}],
        "overrides": {"VRT-X1B": {"design_flow_cmm": 400}},
    })
    by_id = {s.equipment_id: s for s in specs}

    assert by_id["VRT-X1A"].design_flow == 300      # series 가 types 를 이긴다
    assert by_id["VRT-X1B"].design_flow == 400      # overrides 가 series 를 이긴다
    assert by_id["VRT-X1A"].spec["design_voc_ppm"] == 400   # types 에서 상속
    assert by_id["VRT-X1A"].spec["sector_count"] == 12      # defaults 에서 상속


def test_design_voc_load_derived_from_spec(fleet_specs):
    """누적 VOC 부하 기준값이 사양에서 자동 산출되어야 한다."""
    spec = next(s for s in fleet_specs if s.equipment_id == "VRT-A31103C")
    events = spec.to_events_block()
    expected = (
        spec.spec["design_voc_ppm"]
        * spec.design_flow
        * spec.spec["recommended_cleaning_interval_h"]
    )
    assert events["equipment"]["design_voc_load_ppm_cmm_h"] == pytest.approx(expected)


# --- 태그 분해 --------------------------------------------------------------

def test_split_extracts_single_unit(fleet_frame, fleet_specs):
    """와이드 프레임에서 설비 하나만 뽑아 단일모드 이름으로 되돌려야 한다."""
    cfg = load_config()
    spec = next(s for s in fleet_specs if s.equipment_id == "VRT-A31103C")
    unit = fleet_mod.split_by_equipment(fleet_frame, spec, cfg)

    assert "RTO_DP_BED" in unit.columns and "TIMESTAMP" in unit.columns
    assert "TT_SECTOR_12" in unit.columns
    assert len(unit) == len(fleet_frame)
    # 다른 설비의 컬럼이 섞여 들어오면 안 된다
    assert not any("VRT-A31103" in str(c) for c in unit.columns)


def test_split_gives_different_data_per_unit(fleet_frame, fleet_specs):
    cfg = load_config()
    by_id = {s.equipment_id: s for s in fleet_specs}
    a = fleet_mod.split_by_equipment(fleet_frame, by_id["VRT-A31103A"], cfg)
    c = fleet_mod.split_by_equipment(fleet_frame, by_id["VRT-A31103C"], cfg)

    assert not np.allclose(a["RTO_DP_BED"].to_numpy(), c["RTO_DP_BED"].to_numpy())
    # C 는 단독 막힘 설비이므로 후반 차압이 더 높아야 한다
    assert c["RTO_DP_BED"].iloc[-500:].median() > a["RTO_DP_BED"].iloc[-500:].median()


def test_detect_equipment_reports_full_coverage(fleet_frame, fleet_specs):
    cfg = load_config()
    report = fleet_mod.detect_equipment(fleet_frame, cfg, fleet_specs)

    assert len(report) == EXPECTED_FLEET_SIZE
    assert (report["상태"] == "✅ 정상").all()
    assert (report["확보 태그"] == report["전체 태그"]).all()


def test_detect_equipment_flags_missing_tags(fleet_frame, fleet_specs):
    """태그 매핑이 틀리면 조용히 지표가 빠진다 — 그 전에 잡아야 한다."""
    cfg = load_config()
    dropped = fleet_frame.drop(columns=[c for c in fleet_frame.columns if c.startswith("ORT-A31103B_")])
    report = fleet_mod.detect_equipment(dropped, cfg, fleet_specs).set_index("설비")

    assert report.loc["ORT-A31103B", "상태"] == "⛔ 데이터 없음"
    assert report.loc["VRT-A31101A", "상태"] == "✅ 정상"


# --- 전체 분석 --------------------------------------------------------------

def test_all_units_analyzed(fleet):
    assert len(fleet.units) == EXPECTED_FLEET_SIZE
    assert fleet.failed == {}


def test_hourly_data_produces_daily_aggregation(fleet):
    """1시간 간격 데이터에서도 일 집계와 정상운전 판정이 동작해야 한다."""
    analysis = fleet.units["VRT-A31102A"]
    assert len(analysis.daily) > 100
    assert (analysis.daily["runtime_hours"] >= 4.0).all()
    assert analysis.refs.exponent_source == "fitted", "시간 단위에서도 유량지수가 피팅되어야 한다"


def test_ranking_is_sorted_worst_first(fleet):
    ranking = fleet.ranking()
    assert len(ranking) == EXPECTED_FLEET_SIZE
    assert ranking["점수"].is_monotonic_increasing
    assert ranking.iloc[0]["설비"] in ("ORT-A31102A", "VRT-A31103C")


def test_healthy_units_stay_grade_a(fleet):
    """열화를 심지 않은 12대는 A등급을 유지해야 한다 (오탐 없음)."""
    planted = {
        "VRT-A31103C", "VRT-A31101B", "ORT-A31102A",
        *(f"VRT-A31105{u}" for u in "ABCDE"),
    }
    for eq_id, analysis in fleet.units.items():
        if eq_id in planted:
            continue
        snap = analysis.snapshot()
        assert snap.grade == "A", f"{eq_id} 가 {snap.score:.1f}점 ({snap.grade})"


# --- 동급기 비교 (A6) -------------------------------------------------------

def test_peer_indicator_active_for_multi_unit_series(fleet):
    analysis = fleet.units["VRT-A31103A"]
    assert "A6" in analysis.indicators.degradation.columns
    assert fleet.peer_count["VRT-A31103A"] == 4


def test_peer_points_redistributed_when_no_peers():
    """동급기가 없으면 A6 이 빠지고 배점 합계 100 이 유지되어야 한다."""
    specs = fleet_mod.parse_fleet({
        "defaults": {"sector_count": 12, "design_flow_cmm": 850},
        "series": [{"id": "VRT-SOLO", "units": ["A"]}],
    })
    assert len(specs) == 1

    analysis = analyze("data/sample/normal.csv")
    assert "A6" in analysis.indicators.excluded
    assert sum(analysis.indicators.effective_points.values()) == pytest.approx(100.0)


def test_lone_bad_unit_flagged_by_peer_comparison(fleet):
    """VRT-A31103C 만 막혔다 → 동급기 대비 지표가 크게 반응해야 한다."""
    bad = fleet.units["VRT-A31103C"]
    assert bad.indicators.raw["A6"].iloc[-1] > 1.4

    for sibling in ("A", "B", "D", "E"):
        peer = fleet.units[f"VRT-A31103{sibling}"]
        assert peer.indicators.raw["A6"].iloc[-1] < 1.10, f"{sibling} 호기가 잘못 지목됐다"


def test_lone_bad_unit_diagnosed_as_peer_outlier(fleet):
    modes = [m.id for m in fleet.units["VRT-A31103C"].guidance()["all_modes"]]
    assert "peer_outlier" in modes
    assert "fleet_wide_fouling" not in modes

    primary = fleet.units["VRT-A31103C"].guidance()["all_modes"][0]
    assert primary.id == "peer_outlier"
    assert any("우선순위" in a for a in primary.actions)


def test_series_wide_fouling_is_invisible_to_peer_comparison(fleet):
    """VRT-A31105 는 5대가 함께 나빠졌다 → 동급기 대비로는 안 보여야 한다."""
    for unit in "ABCDE":
        ratio = fleet.units[f"VRT-A31105{unit}"].indicators.raw["A6"].iloc[-1]
        assert 0.85 < ratio < 1.15, f"{unit} 호기 동급기 비 {ratio:.2f}"


def test_series_wide_fouling_diagnosed_as_common_cause(fleet):
    """계열 전체가 나빠졌으면 개별 정비가 아니라 유입측 조사로 유도해야 한다."""
    for unit in "ABCDE":
        analysis = fleet.units[f"VRT-A31105{unit}"]
        guide = analysis.guidance()
        modes = [m.id for m in guide["all_modes"]]

        assert "fleet_wide_fouling" in modes, f"{unit} 호기: {modes}"
        assert "peer_outlier" not in modes
        assert guide["all_modes"][0].id == "fleet_wide_fouling"
        assert any("공통 전처리" in a or "공정측" in a for a in guide["all_modes"][0].actions)


def test_a1_still_catches_series_wide_fouling(fleet):
    """동급기 비교가 조용해도 자기 베이스라인 대비 지표는 반응해야 한다.

    A1 과 A6 은 서로를 보완한다 — 둘 다 조용하면 진짜로 정상인 것이다.
    """
    for unit in "ABCDE":
        analysis = fleet.units[f"VRT-A31105{unit}"]
        assert analysis.snapshot().degradations["A1"] >= 0.30
        assert analysis.snapshot().score < 75.0


def test_seal_leak_still_separated_in_fleet_mode(fleet):
    """차압 정상 + 효율 저하 → 막힘이 아니라 씰 누설로 판정되어야 한다."""
    analysis = fleet.units["VRT-A31101B"]
    modes = [m.id for m in analysis.guidance()["all_modes"]]

    assert "media_settling_or_seal_leak" in modes
    assert "peer_outlier" not in modes, "차압이 정상인데 동급기 이상으로 잡으면 안 된다"
    assert analysis.snapshot().degradations["A1"] < 0.25


def test_rapid_plugging_unit_is_worst(fleet):
    analysis = fleet.units["ORT-A31102A"]
    snap = analysis.snapshot()
    assert snap.grade == "E"
    assert analysis.rul.days_remaining is not None and analysis.rul.days_remaining < 60
    assert "channeling" in [m.id for m in analysis.guidance()["all_modes"]]


# --- 베이스라인 지정 --------------------------------------------------------

def _config_with_baseline(baseline: dict) -> Config:
    cfg = load_config()
    return Config(
        tags=cfg.tags,
        weights=cfg.weights,
        events={
            "equipment": {"name": "TEST", "design_flow_cmm": 850, "design_max_dp_mmh2o": 250},
            "maintenance_events": [],
            "baseline": baseline,
        },
    )


def test_baseline_defaults_to_assumed_with_warning(scenarios):
    """정비 이력도 지정도 없으면 '가정'임을 분명히 알려야 한다."""
    analysis = analyze(scenarios["normal"], _config_with_baseline({}))
    assert analysis.baseline.source == "auto"
    assert analysis.baseline.is_assumed
    assert analysis.baseline.warning and "가정" in analysis.baseline.warning


def test_baseline_period_overrides_auto(scenarios):
    """구간을 직접 지정하면 그 구간이 기준이 되어야 한다."""
    analysis = analyze(
        scenarios["gradual_fouling"],
        _config_with_baseline({"period": {"start": "2025-06-01", "end": "2025-06-15"}}),
    )
    assert analysis.baseline.source == "period"
    assert analysis.baseline.warning is None
    assert analysis.baseline.start == pd.Timestamp("2025-06-01")
    assert analysis.baseline.n_samples > 0


def test_later_baseline_understates_earlier_fouling(scenarios):
    """이미 오염된 구간을 기준으로 잡으면 열화가 과소평가된다 — 경고가 필요한 이유."""
    clean = analyze(
        scenarios["gradual_fouling"],
        _config_with_baseline({"period": {"start": "2025-01-07", "end": "2025-01-21"}}),
    )
    contaminated = analyze(
        scenarios["gradual_fouling"],
        _config_with_baseline({"period": {"start": "2025-09-01", "end": "2025-09-15"}}),
    )
    assert contaminated.snapshot().score > clean.snapshot().score + 10


def test_manual_baseline_uses_supplied_values(scenarios):
    """실측 구간이 없어도 설계값으로 채점을 시작할 수 있어야 한다."""
    analysis = analyze(
        scenarios["normal"],
        _config_with_baseline({"manual": {"dp_bed": 70, "flow": 850, "t_in": 40, "ter": 0.95}}),
    )
    assert analysis.baseline.source == "manual"
    assert analysis.baseline.ref("dp_norm") == pytest.approx(70.0)
    assert analysis.baseline.ref("ter") == pytest.approx(0.95)

    # 조건부 기준모델을 학습할 구간이 없으므로 잔차 기반 지표는 빠진다
    for ind_id in ("B2", "C3"):
        assert ind_id in analysis.indicators.excluded
    assert sum(analysis.indicators.effective_points.values()) == pytest.approx(100.0)


# --- 견고성 -----------------------------------------------------------------

def test_one_broken_unit_does_not_stop_the_rest(fleet_frame, fleet_specs):
    """계측 고장 설비 하나 때문에 나머지 19대가 막히면 안 된다."""
    from rto_health.fleet import analyze_fleet

    broken = fleet_frame.copy()
    for col in [c for c in broken.columns if c.startswith("VRT-A31102B_")]:
        broken[col] = np.nan

    result = analyze_fleet(broken)
    assert "VRT-A31102B" in result.failed
    assert len(result.units) == EXPECTED_FLEET_SIZE - 1
    assert result.units["VRT-A31103C"].snapshot().score < 60


def test_accumulated_files_are_merged_and_deduplicated(tmp_path, fleet_frame):
    """1시간 단위로 쌓이는 월별 export 를 폴더째 읽을 수 있어야 한다."""
    from rto_health.io_loader import read_frames

    half = len(fleet_frame) // 2
    fleet_frame.iloc[:half].to_csv(tmp_path / "2026-01.csv", index=False)
    # 구간이 겹치게 두 번째 파일을 만든다 (재추출 상황)
    fleet_frame.iloc[half - 100:].to_csv(tmp_path / "2026-02.csv", index=False)

    merged = read_frames(tmp_path)
    assert len(merged) == len(fleet_frame), "중복 구간이 제거되어야 한다"
    assert merged["TIMESTAMP"].is_monotonic_increasing
    assert not merged["TIMESTAMP"].duplicated().any()
