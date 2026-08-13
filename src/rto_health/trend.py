"""변화추이 분석 — 평활, 추세 기울기, 변화점 탐지, 잔여여유 예측.

"지금 얼마나 나쁜가"만큼 "얼마나 빨리 나빠지고 있는가"가 중요하다. 차압이
같은 120 mmH2O 라도 1년에 걸쳐 올라온 것과 2주 만에 올라온 것은 전혀 다른
조치를 요구한다.

이상치에 강한 Theil-Sen 회귀를 쓰는 이유: 운전 데이터에는 계측 스파이크와
일시적 공정 이상이 섞여 있어 최소자승 기울기가 쉽게 끌려간다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ChangePoint:
    """CUSUM 이 검출한 변화 시작 시점."""

    date: pd.Timestamp
    statistic: float
    direction: str  # 'up' | 'down'


@dataclass
class RULEstimate:
    """잔여여유(Remaining Useful Life) 추정 결과."""

    days_remaining: float | None
    predicted_date: pd.Timestamp | None
    limit_value: float
    current_value: float
    slope_per_day: float
    fit_quality_r2: float | None
    ci_days: tuple[float, float] | None = None
    note: str = ""


def ewma(series: pd.Series, span_days: float) -> pd.Series:
    """지수가중이동평균. 일 단위 시계열 기준."""
    return series.ewm(span=max(span_days, 1.0), adjust=False).mean()


def theil_sen_slope(series: pd.Series) -> tuple[float, float, float]:
    """일 단위 시계열의 robust 기울기(값/일)와 95% 신뢰구간.

    Returns:
        (slope, lo, hi). 표본이 3개 미만이면 (nan, nan, nan).
    """
    s = series.dropna()
    if len(s) < 3:
        return float("nan"), float("nan"), float("nan")
    x = (s.index - s.index[0]).days.to_numpy(dtype=float)
    if np.ptp(x) < 1:
        return float("nan"), float("nan"), float("nan")
    slope, _intercept, lo, hi = stats.theilslopes(s.to_numpy(dtype=float), x, alpha=0.95)
    return float(slope), float(lo), float(hi)


def rolling_slope(series: pd.Series, window_days: int) -> pd.Series:
    """직전 window_days 구간의 Theil-Sen 기울기를 매일 산출한다."""
    s = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if s.empty:
        return out

    for ts in s.index:
        window = s.loc[ts - pd.Timedelta(days=window_days) : ts]
        if len(window) >= max(5, window_days // 4):
            out.loc[ts] = theil_sen_slope(window)[0]
    return out


def cusum(
    series: pd.Series,
    reference_mask: pd.Series | np.ndarray | None = None,
    *,
    k_sigma: float = 0.5,
    h_sigma: float = 5.0,
    min_sigma_ratio: float = 0.015,
    use_increments: bool = True,
) -> tuple[pd.Series, list[ChangePoint]]:
    """CUSUM 변화점 탐지 — 열화 속도가 달라진 시점을 찾는다.

    표준화된 편차의 누적합이 임계 h 를 넘는 시점을 변화 시작으로 본다.
    단일 임계치 방식보다 훨씬 빨리 반응한다.

    Args:
        reference_mask: 정상 기준으로 삼을 구간. 생략 시 앞쪽 20% 사용.
        min_sigma_ratio: sigma 하한 비율 (0 나눗셈 및 과민 알람 방지).
        use_increments: True 면 값이 아닌 일간 증분을 감시한다(권장).
    """
    s = series.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float), []

    if use_increments:
        # 값 자체가 아니라 '하루당 변화량'을 감시한다.
        # 축열재는 정상 상태에서도 아주 느리게 오염되므로 수준(level) CUSUM 은
        # 그 정상 드리프트마저 계속 누적해 결국 반드시 알람을 낸다. 우리가 찾는 것은
        # "값이 높아졌다"가 아니라 "나빠지는 속도가 달라졌다"이므로 증분을 본다.
        work = s.diff().dropna()
    else:
        work = s
    if len(work) < 10:
        return pd.Series(dtype=float), []

    ref = pd.Series(dtype=float)
    if reference_mask is not None:
        aligned = pd.Series(reference_mask).reindex(work.index).fillna(False).astype(bool)
        ref = work[aligned]
    if len(ref) < 5:
        # 기준 구간이 지정되지 않았거나 너무 짧으면 앞쪽 20% 를 정상으로 간주
        ref = work.iloc[: max(5, len(work) // 5)]

    mu = float(ref.median())
    # 산포는 전 구간 MAD 로 추정한다. 열화 구간이 섞여 있어도 중앙값 기반이라
    # 조용한 구간의 노이즈 수준을 그대로 잡아내므로, 짧은 베이스라인에서
    # sigma 가 과소평가되어 알람이 남발되는 문제를 피할 수 있다.
    values = work.to_numpy(dtype=float)
    mad = float(np.median(np.abs(values - np.median(values))))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(work.std(ddof=1)) or 1.0
    sigma = max(sigma, abs(float(np.median(values))) * min_sigma_ratio, 1e-12)

    s = work
    z = (values - mu) / sigma
    hi = np.zeros(len(z))
    lo = np.zeros(len(z))
    points: list[ChangePoint] = []
    acc_hi = acc_lo = 0.0

    for i, zi in enumerate(z):
        acc_hi = max(0.0, acc_hi + zi - k_sigma)
        acc_lo = min(0.0, acc_lo + zi + k_sigma)
        hi[i], lo[i] = acc_hi, acc_lo

        if acc_hi > h_sigma:
            points.append(ChangePoint(date=s.index[i], statistic=acc_hi, direction="up"))
            acc_hi = 0.0
        if acc_lo < -h_sigma:
            points.append(ChangePoint(date=s.index[i], statistic=abs(acc_lo), direction="down"))
            acc_lo = 0.0

    stat = pd.Series(hi, index=s.index, name="cusum_hi").reindex(series.index)
    return stat, points


def _first_alarm_per_run(points: list[ChangePoint], min_gap_days: int = 20) -> list[ChangePoint]:
    """연속으로 터진 알람을 하나로 묶어 '변화 시작 시점'만 남긴다."""
    kept: list[ChangePoint] = []
    for p in points:
        if not kept or (p.date - kept[-1].date).days > min_gap_days or p.direction != kept[-1].direction:
            kept.append(p)
    return kept


def detect_change_points(
    series: pd.Series,
    reference_mask: pd.Series | None = None,
    *,
    k_sigma: float = 0.5,
    h_sigma: float = 5.0,
    min_gap_days: int = 20,
    use_increments: bool = True,
) -> tuple[pd.Series, list[ChangePoint]]:
    stat, points = cusum(
        series, reference_mask, k_sigma=k_sigma, h_sigma=h_sigma, use_increments=use_increments
    )
    return stat, _first_alarm_per_run(points, min_gap_days=min_gap_days)


def estimate_rul(
    series: pd.Series,
    limit_value: float,
    *,
    fit_window_days: int = 60,
    min_slope: float = 1e-5,
) -> RULEstimate:
    """지표가 운전 한계에 도달하는 시점을 외삽으로 추정한다.

    최근 fit_window_days 구간의 robust 기울기를 선형 외삽한다. 기울기 신뢰구간을
    함께 외삽해 낙관/비관 범위를 제시한다.
    """
    s = series.dropna()
    if s.empty:
        return RULEstimate(None, None, limit_value, float("nan"), float("nan"), None, note="데이터 없음")

    last_date = s.index[-1]
    window = s.loc[last_date - pd.Timedelta(days=fit_window_days) :]
    current = float(window.iloc[-min(7, len(window)) :].median())

    if current >= limit_value:
        return RULEstimate(0.0, last_date, limit_value, current, float("nan"), None,
                           note="이미 운전 한계 도달")

    slope, lo, hi = theil_sen_slope(window)
    if not np.isfinite(slope) or slope <= min_slope:
        return RULEstimate(None, None, limit_value, current, slope if np.isfinite(slope) else 0.0,
                           None, note="상승 추세가 확인되지 않아 도달 시점 산출 불가")

    # 기울기 신뢰구간이 0 을 포함하면 '노이즈일 뿐'이라는 뜻이다.
    # 이때 외삽하면 잡음에서 수만 일짜리 그럴듯한 예측을 만들어내게 된다.
    if np.isfinite(lo) and lo <= 0:
        return RULEstimate(None, None, limit_value, current, float(slope), None,
                           note="상승 추세가 통계적으로 유의하지 않아 도달 시점 산출 불가")

    remaining = (limit_value - current) / slope

    ci: tuple[float, float] | None = None
    if np.isfinite(lo) and np.isfinite(hi) and lo > min_slope:
        # 기울기가 클수록 빨리 도달 → hi 가 비관(짧은 쪽)
        ci = (float((limit_value - current) / hi), float((limit_value - current) / lo))

    # 회귀 적합도
    x = (window.index - window.index[0]).days.to_numpy(dtype=float)
    y = window.to_numpy(dtype=float)
    r2: float | None = None
    if len(y) > 3 and np.ptp(x) > 0:
        pred = y[0] + slope * x
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0:
            r2 = float(1.0 - np.sum((y - pred) ** 2) / ss_tot)

    return RULEstimate(
        days_remaining=float(remaining),
        predicted_date=last_date + pd.Timedelta(days=float(remaining)),
        limit_value=limit_value,
        current_value=current,
        slope_per_day=float(slope),
        fit_quality_r2=r2,
        ci_days=ci,
    )
