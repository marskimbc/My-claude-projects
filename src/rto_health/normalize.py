"""물리 보정 — 이 시스템의 정확도를 좌우하는 핵심 모듈.

차압 원값을 그대로 추세로 보면 **생산량이 늘어난 것을 막힘으로 오판**한다.
풍량이 20% 늘면 차압은 (형상에 따라) 23~44% 오르는데, 이는 막힘과 무관하다.
따라서 기준 조건으로 환산한 뒤에 비교한다.

    ΔP_norm = ΔP × (Q_ref/Q)^n × (T_ref/T)

  · n = 유량 지수. 축열재 형상에 따라 다르다.
        허니컴(층류 지배) ≈ 1.0~1.2,  랜덤 새들(난류) ≈ 1.8~2.0
        → 현장 실측값에 맞추기 위해 베이스라인 구간에서 회귀로 피팅한다.
  · T 는 절대온도(K). 가스 밀도·점도 보정 항.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io_loader import Config, Dataset

KELVIN = 273.15


@dataclass
class NormalizationRefs:
    """차압 정규화에 쓰는 기준 조건."""

    flow_exponent: float
    ref_flow: float
    ref_gas_temp_c: float
    exponent_source: str  # 'fitted' | 'config' | 'default'
    fit_r2: float | None = None


def fit_flow_exponent(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    bounds: tuple[float, float] = (0.8, 2.2),
) -> tuple[float, float]:
    """베이스라인 구간에서 차압-유량 지수 n 을 회귀로 추정한다.

    온도 보정을 먼저 하고 log-log 회귀를 하면 기울기가 곧 n 이다.

        log(ΔP × T_ref/T) = log(a) + n · log(Q)

    Returns:
        (n, R²). 유효 표본이 부족하면 (nan, nan).
    """
    sub = df.loc[mask, ["dp_bed", "flow", "t_in"]].dropna()
    sub = sub[(sub["dp_bed"] > 0) & (sub["flow"] > 0)]
    if len(sub) < 50:
        return float("nan"), float("nan")

    t_k = sub["t_in"].to_numpy() + KELVIN
    t_ref_k = float(np.median(t_k))
    dp_tc = sub["dp_bed"].to_numpy() * (t_ref_k / t_k)

    x = np.log(sub["flow"].to_numpy())
    y = np.log(dp_tc)
    # 유량 변동이 거의 없으면 기울기를 신뢰할 수 없다
    if x.std() < 0.02:
        return float("nan"), float("nan")

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return float(np.clip(slope, *bounds)), r2


def resolve_refs(
    ds: Dataset,
    config: Config,
    baseline_mask: pd.Series | np.ndarray,
) -> NormalizationRefs:
    """설정 우선순위에 따라 정규화 기준 조건을 확정한다.

    유량 지수: config 명시값 > 베이스라인 회귀 피팅 > 축열재 형상별 기본값
    """
    phys = config.weights.get("physical", {})
    bounds = tuple(phys.get("flow_exponent_bounds", (0.8, 2.2)))  # type: ignore[arg-type]

    configured = phys.get("flow_exponent")
    fitted, r2 = fit_flow_exponent(ds.df, baseline_mask, bounds=bounds)

    if configured is not None:
        n, source = float(configured), "config"
    elif np.isfinite(fitted):
        n, source = fitted, "fitted"
    else:
        defaults = phys.get("flow_exponent_default", {}) or {}
        n = float(defaults.get(config.media_type, 1.5))
        source = "default"

    base = ds.df.loc[baseline_mask]
    ref_flow = phys.get("ref_flow_cmm")
    if ref_flow is None:
        ref_flow = float(base["flow"].median())
    ref_temp = phys.get("ref_gas_temp_c")
    if ref_temp is None:
        ref_temp = float(base["t_in"].median())

    return NormalizationRefs(
        flow_exponent=n,
        ref_flow=float(ref_flow),
        ref_gas_temp_c=float(ref_temp),
        exponent_source=source,
        fit_r2=r2 if np.isfinite(r2) else None,
    )


def normalize_dp(
    dp: pd.Series | np.ndarray,
    flow: pd.Series | np.ndarray,
    gas_temp_c: pd.Series | np.ndarray,
    refs: NormalizationRefs,
) -> np.ndarray:
    """차압을 기준 조건(기준 풍량·기준 온도)으로 환산한다."""
    dp = np.asarray(dp, dtype=float)
    flow = np.asarray(flow, dtype=float)
    t_k = np.asarray(gas_temp_c, dtype=float) + KELVIN
    ref_t_k = refs.ref_gas_temp_c + KELVIN

    with np.errstate(divide="ignore", invalid="ignore"):
        flow_term = np.where(flow > 0, (refs.ref_flow / flow) ** refs.flow_exponent, np.nan)
        temp_term = np.where(t_k > 0, ref_t_k / t_k, np.nan)
    return dp * flow_term * temp_term


def thermal_efficiency(
    t_comb: pd.Series, t_in: pd.Series, t_stack: pd.Series, *, min_span_c: float = 100.0
) -> pd.Series:
    """열회수효율 TER = (T_comb − T_stack) / (T_comb − T_in).

    분모가 작으면(연소실이 식은 과도구간) 값이 발산하므로 NaN 처리한다.
    """
    span = t_comb - t_in
    ter = (t_comb - t_stack) / span
    ter = ter.where(span >= min_span_c)
    return ter.where((ter > 0.3) & (ter < 1.0))


def add_derived(ds: Dataset, refs: NormalizationRefs) -> Dataset:
    """정규화 차압·TER·연료 원단위·섹터 온도편차를 원데이터에 추가한다."""
    df = ds.df.copy()

    df["dp_norm"] = normalize_dp(df["dp_bed"], df["flow"], df["t_in"], refs)
    df["ter"] = thermal_efficiency(df["t_comb"], df["t_in"], df["t_stack"])

    if "fuel_flow" in df.columns:
        # 단위 처리풍량당 연료 사용량 (Nm3 / CMM·h)
        df["fuel_intensity"] = df["fuel_flow"] / df["flow"].where(df["flow"] > 0)

    if ds.sector_cols:
        sector = df[ds.sector_cols]
        df["sector_temp_std"] = sector.std(axis=1)
        df["sector_temp_range"] = sector.max(axis=1) - sector.min(axis=1)

    return Dataset(
        df=df,
        available=set(ds.available),
        sector_cols=list(ds.sector_cols),
        missing_tags=list(ds.missing_tags),
    )
