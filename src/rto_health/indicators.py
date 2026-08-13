"""15개 막힘 지표의 raw metric 산출과 0~1 열화도 변환.

각 지표는 "높을수록 나쁜" 방향으로 통일해 정의하고, config/weights.yaml 의
onset~severe 구간에서 0~1 로 선형 사상한다.

    metric ≤ onset   → 열화도 0   (정상)
    metric ≥ severe  → 열화도 1   (배점 전액 감점)

보유하지 않은 태그를 요구하는 지표는 자동으로 제외되고, 남은 지표들끼리
100점을 재배분한다(= 없는 항목 때문에 점수가 부당하게 깎이지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import Baseline
from .io_loader import Config, Dataset
from .trend import rolling_slope


@dataclass
class IndicatorSpec:
    """weights.yaml 의 지표 정의 한 건."""

    id: str
    group: str
    name: str
    points: float
    metric: str
    unit: str
    description: str
    onset: float
    severe: float
    requires: list[str] = field(default_factory=list)       # 전부 있어야 함
    requires_any: list[str] = field(default_factory=list)   # 하나만 있으면 됨(대체 태그)


@dataclass
class IndicatorSet:
    """전체 지표 계산 결과."""

    raw: pd.DataFrame            # index=date, columns=지표ID → raw metric
    degradation: pd.DataFrame    # index=date, columns=지표ID → 0~1 열화도
    specs: dict[str, IndicatorSpec]
    effective_points: dict[str, float]   # 재배분 후 배점
    excluded: dict[str, str]             # 지표ID → 제외 사유
    group_names: dict[str, str]

    @property
    def active_ids(self) -> list[str]:
        return list(self.degradation.columns)


def load_specs(config: Config) -> dict[str, IndicatorSpec]:
    specs: dict[str, IndicatorSpec] = {}
    for ind_id, raw in (config.weights.get("indicators") or {}).items():
        specs[ind_id] = IndicatorSpec(
            id=ind_id,
            group=raw.get("group", "?"),
            name=raw.get("name", ind_id),
            points=float(raw.get("points", 0)),
            metric=raw.get("metric", ind_id),
            unit=raw.get("unit", ""),
            description=raw.get("description", ""),
            onset=float(raw.get("onset", 0)),
            severe=float(raw.get("severe", 1)),
            requires=list(raw.get("requires") or []),
            requires_any=list(raw.get("requires_any") or []),
        )
    return specs


def to_degradation(metric: pd.Series, onset: float, severe: float) -> pd.Series:
    """raw metric 을 0~1 열화도로 선형 사상한다."""
    span = severe - onset
    if span == 0:
        return (metric > onset).astype(float)
    return ((metric - onset) / span).clip(lower=0.0, upper=1.0)


def _safe_ratio(numerator: pd.Series, denominator: float) -> pd.Series:
    if not np.isfinite(denominator) or denominator == 0:
        return pd.Series(np.nan, index=numerator.index)
    return numerator / denominator


def compute_raw_metrics(
    daily: pd.DataFrame,
    baseline: Baseline,
    config: Config,
    ds: Dataset,
    *,
    peer_dp_ratio: pd.Series | None = None,
) -> pd.DataFrame:
    """일 단위 집계에서 지표의 raw metric 을 계산한다.

    Args:
        peer_dp_ratio: 동일 계열 동급기의 정규화 차압 비율 중앙값(일별).
            fleet 분석에서만 주어지며, 없으면 A6 을 산출하지 않는다.
    """
    out = pd.DataFrame(index=daily.index)
    equip = config.equipment
    trend_cfg = config.weights.get("trend", {})

    # ===== A군. 유동저항 ====================================================
    if "dp_norm" in daily.columns:
        ratio = _safe_ratio(daily["dp_norm"], baseline.ref("dp_norm"))
        out["A1"] = ratio
        # 30일 창의 robust 기울기 (비율/일)
        out["A2"] = rolling_slope(ratio, 30, stride=int(trend_cfg.get("slope_stride_days", 3)))

    if "fan_hz_ratio" in daily.columns:
        out["A3"] = daily["fan_hz_ratio"]
    elif "fan_amp" in daily.columns:
        # 인버터 신호가 없으면 전류로 대체. 전류는 회전수의 약 2제곱에 비례하므로
        # 주파수 비율로 환산해 동일 척도로 맞춘다.
        out["A3"] = _safe_ratio(daily["fan_amp"], baseline.ref("fan_amp")) ** (1 / 2.1)

    if "damper_pct" in daily.columns:
        out["A4"] = daily["damper_pct"]

    if "dp_cv" in daily.columns:
        out["A5"] = daily["dp_cv"]

    # A6 동급기 대비 — 자기 베이스라인이 오염돼 있어도 유효한 유일한 유동저항 지표
    if peer_dp_ratio is not None and "A1" in out.columns:
        peer = peer_dp_ratio.reindex(out.index)
        out["A6"] = (out["A1"] / peer.where(peer > 1e-6)).replace([np.inf, -np.inf], np.nan)

    # ===== B군. 열교환 성능 ==================================================
    if "ter" in daily.columns:
        out["B1"] = (baseline.ref("ter") - daily["ter"]) * 100.0   # %p

    if "stack_temp_residual" in daily.columns:
        out["B2"] = daily["stack_temp_residual"]

    if "sector_temp_std" in daily.columns:
        out["B3"] = daily["sector_temp_std"]

    if "comb_instability" in daily.columns:
        out["B4"] = daily["comb_instability"]

    # ===== C군. 에너지·처리 성능 =============================================
    if "fuel_intensity" in daily.columns:
        out["C1"] = _safe_ratio(daily["fuel_intensity"], baseline.ref("fuel_intensity"))

    if "burner_duty_ratio" in daily.columns:
        out["C2"] = daily["burner_duty_ratio"]
    elif "burner_duty" in daily.columns:
        out["C2"] = _safe_ratio(daily["burner_duty"], baseline.ref("burner_duty"))

    if "fuel_residual" in daily.columns:
        # 기대보다 연료를 '더' 쓰는 쪽만 이상으로 본다
        out["C3"] = daily["fuel_residual"].clip(lower=0.0)

    if "voc_out" in daily.columns:
        out["C4"] = _safe_ratio(daily["voc_out"], baseline.ref("voc_out"))

    # ===== D군. 누적 오염 부하 ===============================================
    # 마지막 세정 이후로만 적산한다
    since = baseline.reset_date if baseline.reset_date is not None else daily.index[0]
    after_reset = daily.index >= since

    if "voc_load" in daily.columns:
        design_load = float(equip.get("design_voc_load_ppm_cmm_h", 6.0e7))
        cum = daily["voc_load"].where(after_reset, 0.0).cumsum()
        out["D1"] = cum / design_load if design_load > 0 else np.nan

    if "runtime_hours" in daily.columns:
        interval = float(equip.get("recommended_cleaning_interval_h", 8760))
        cum_h = daily["runtime_hours"].where(after_reset, 0.0).cumsum()
        out["D2"] = cum_h / interval if interval > 0 else np.nan

    _ = trend_cfg  # 창 길이는 A2 에서 고정 30일 사용
    return out


def compute(
    daily: pd.DataFrame,
    baseline: Baseline,
    config: Config,
    ds: Dataset,
    *,
    min_valid_ratio: float = 0.3,
    raw: pd.DataFrame | None = None,
    peer_dp_ratio: pd.Series | None = None,
) -> IndicatorSet:
    """raw metric → 열화도 → 배점 재배분까지 수행한다.

    Args:
        raw: 이미 계산해 둔 raw metric. 배점·임계치만 바꿔 다시 채점할 때
            넘기면 무거운 재계산(30일 창 robust 기울기 등)을 건너뛴다.
    """
    specs = load_specs(config)
    if raw is None:
        raw = compute_raw_metrics(daily, baseline, config, ds, peer_dp_ratio=peer_dp_ratio)

    degradation = pd.DataFrame(index=daily.index)
    excluded: dict[str, str] = {}

    for ind_id, spec in specs.items():
        missing = [t for t in spec.requires if t not in ds.available]
        if missing:
            excluded[ind_id] = f"태그 미보유: {', '.join(missing)}"
            continue
        if spec.requires_any and not any(t in ds.available for t in spec.requires_any):
            excluded[ind_id] = f"태그 미보유: {' 또는 '.join(spec.requires_any)} 중 하나 필요"
            continue
        if ind_id not in raw.columns:
            excluded[ind_id] = "산출에 필요한 파생값이 없음"
            continue

        series = raw[ind_id]
        valid_ratio = float(series.notna().mean()) if len(series) else 0.0
        if valid_ratio < min_valid_ratio:
            excluded[ind_id] = f"유효 데이터 부족 ({valid_ratio:.0%})"
            continue

        degradation[ind_id] = to_degradation(series, spec.onset, spec.severe)

    # --- 배점 재배분: 살아있는 지표들끼리 100점을 나눈다 -----------------------
    total_points = sum(specs[i].points for i in degradation.columns)
    effective_points: dict[str, float] = {}
    if total_points > 0:
        scale = 100.0 / total_points
        effective_points = {i: specs[i].points * scale for i in degradation.columns}

    groups = config.weights.get("groups") or {}
    group_names = {k: v.get("name", k) for k, v in groups.items()}

    return IndicatorSet(
        raw=raw,
        degradation=degradation,
        specs=specs,
        effective_points=effective_points,
        excluded=excluded,
        group_names=group_names,
    )
