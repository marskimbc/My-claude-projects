"""물리 보정 검증 — 이 테스트가 통과하지 않으면 나머지 전부가 무의미하다.

정규화의 목적은 단 하나: **생산량 변동을 막힘으로 오판하지 않는 것**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_health.normalize import (
    NormalizationRefs,
    fit_flow_exponent,
    normalize_dp,
    thermal_efficiency,
)

GENERATOR_EXPONENT = 1.15   # generate_sample.py 가 실제로 사용한 값


def test_flow_exponent_recovers_generator_value(scenarios):
    """회귀 피팅이 데이터 생성에 쓰인 유량 지수를 되찾아내야 한다."""
    df = scenarios["normal"].rename(
        columns={"RTO_DP_BED": "dp_bed", "RTO_FLOW": "flow", "TT_INLET": "t_in"}
    )
    # 막힘이 거의 없는 초반 60일만 사용
    mask = pd.Series(df.index < 60 * 144, index=df.index)

    n, r2 = fit_flow_exponent(df, mask)
    assert n == pytest.approx(GENERATOR_EXPONENT, abs=0.05), f"피팅된 n={n}"
    assert r2 > 0.95


def test_flow_exponent_uses_configured_bounds(scenarios):
    """지수는 물리적으로 타당한 범위를 벗어나지 않아야 한다."""
    df = scenarios["normal"].rename(
        columns={"RTO_DP_BED": "dp_bed", "RTO_FLOW": "flow", "TT_INLET": "t_in"}
    )
    mask = pd.Series(True, index=df.index)
    n, _ = fit_flow_exponent(df, mask, bounds=(1.5, 2.0))
    assert 1.5 <= n <= 2.0


def test_dp_norm_is_invariant_to_flow_changes():
    """풍량이 2배로 변해도 막힘이 없으면 정규화 차압은 일정해야 한다.

    이것이 실패하면 생산량이 늘어난 것을 막힘으로 오판하게 된다.
    """
    refs = NormalizationRefs(
        flow_exponent=GENERATOR_EXPONENT,
        ref_flow=850.0,
        ref_gas_temp_c=40.0,
        exponent_source="config",
    )
    flow = np.array([425.0, 600.0, 850.0, 1100.0, 1700.0])   # 4배 범위
    t_in = np.full(5, 40.0)
    # 막힘 없는 상태의 실제 차압 (생성기와 동일한 물리식)
    dp = 80.0 * (flow / 850.0) ** GENERATOR_EXPONENT

    dp_norm = normalize_dp(dp, flow, t_in, refs)

    assert np.allclose(dp_norm, 80.0, rtol=1e-9)
    assert dp.std() > 20.0, "원값은 크게 흔들려야 테스트가 의미 있다"


def test_dp_norm_is_invariant_to_gas_temperature():
    """유입가스 온도(계절 변동)도 정규화로 제거되어야 한다."""
    refs = NormalizationRefs(
        flow_exponent=GENERATOR_EXPONENT, ref_flow=850.0, ref_gas_temp_c=40.0,
        exponent_source="config",
    )
    t_in = np.array([5.0, 20.0, 40.0, 60.0, 80.0])
    flow = np.full(5, 850.0)
    dp = 80.0 * ((t_in + 273.15) / 313.15)

    dp_norm = normalize_dp(dp, flow, t_in, refs)
    assert np.allclose(dp_norm, 80.0, rtol=1e-9)


def test_dp_norm_still_detects_real_blockage():
    """정규화가 진짜 막힘까지 지워버리면 안 된다."""
    refs = NormalizationRefs(
        flow_exponent=GENERATOR_EXPONENT, ref_flow=850.0, ref_gas_temp_c=40.0,
        exponent_source="config",
    )
    flow = np.array([850.0, 850.0])
    t_in = np.array([40.0, 40.0])
    dp = np.array([80.0, 80.0 * 1.8])   # 두 번째는 1.8배 막힘

    dp_norm = normalize_dp(dp, flow, t_in, refs)
    assert dp_norm[1] / dp_norm[0] == pytest.approx(1.8, rel=1e-9)


def test_thermal_efficiency_matches_definition():
    t_comb = pd.Series([800.0, 800.0])
    t_in = pd.Series([40.0, 40.0])
    t_stack = pd.Series([78.0, 116.0])

    ter = thermal_efficiency(t_comb, t_in, t_stack)
    assert ter.iloc[0] == pytest.approx((800 - 78) / (800 - 40))
    assert ter.iloc[1] < ter.iloc[0], "배출온도가 높으면 효율은 낮아야 한다"


def test_thermal_efficiency_rejects_transient_states():
    """연소실이 식은 과도구간에서는 TER 을 계산하지 않는다."""
    t_comb = pd.Series([90.0])     # 정지 직후
    t_in = pd.Series([40.0])
    t_stack = pd.Series([60.0])

    ter = thermal_efficiency(t_comb, t_in, t_stack)
    assert ter.isna().all()


def test_pipeline_fits_exponent_from_data(normal):
    """파이프라인이 설정 고정값이 아니라 실측에서 지수를 피팅했는지."""
    assert normal.refs.exponent_source == "fitted"
    assert normal.refs.flow_exponent == pytest.approx(GENERATOR_EXPONENT, abs=0.05)
    assert normal.refs.fit_r2 is not None and normal.refs.fit_r2 > 0.95
