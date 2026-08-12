"""점수화.

각 축(추세/모멘텀/상대강도/수급/리스크)을 0~100 으로 정규화한 뒤
settings.yaml 의 가중치로 합산한다. 점수는 '순위를 매기기 위한 도구'이지
수익률 예측값이 아니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from . import indicators as ind


def squash(value: float, scale: float) -> float:
    """실수를 0~100 으로. value=0 이면 50, value=scale 이면 약 73점."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 50.0
    if scale <= 0:
        return 50.0
    return float(100 / (1 + math.exp(-value / scale)))


def _last(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


# --------------------------------------------------------------------------
# 개별 축
# --------------------------------------------------------------------------

def trend_score(enriched: pd.DataFrame) -> float:
    """이동평균 정배열 정도 + 중기 이평 기울기 + 추세 강도(ADX/DI)."""
    if enriched.empty:
        return 50.0
    row = enriched.iloc[-1]
    close = row.get("close", float("nan"))

    points, possible = 0.0, 0.0
    ladder = ["ma_short", "ma_mid", "ma_long", "ma_trend"]
    levels = [close] + [row.get(name, float("nan")) for name in ladder]
    for upper, lower in zip(levels, levels[1:]):
        if pd.isna(upper) or pd.isna(lower):
            continue
        possible += 1
        if upper > lower:
            points += 1
    alignment = (points / possible * 100) if possible else 50.0

    slope = ind.slope_pct(enriched["ma_mid"], 20)
    slope_pts = squash(slope, 0.12)

    adx_value = row.get("adx", float("nan"))
    directional = row.get("plus_di", float("nan")) - row.get("minus_di", float("nan"))
    if pd.isna(adx_value) or pd.isna(directional):
        strength = 50.0
    else:
        # ADX 가 클수록 방향(DI 차이)에 확신을 더 준다.
        strength = squash(directional * min(adx_value, 40) / 20, 10)

    return float(0.5 * alignment + 0.3 * slope_pts + 0.2 * strength)


def momentum_score(close: pd.Series, windows: list[int] | None = None) -> float:
    windows = windows or config.get("scoring.momentum_windows", [20, 60, 120])
    weights = [0.5, 0.3, 0.2][: len(windows)]
    parts, used = 0.0, 0.0
    for window, weight in zip(windows, weights):
        value = ind.returns(close, window)
        if math.isnan(value):
            continue
        # 기간이 길수록 같은 % 라도 덜 놀랍다 → 기간 비례로 스케일 조정
        parts += weight * squash(value, 4 * math.sqrt(window / 20))
        used += weight
    base = parts / used if used else 50.0

    rsi_now = _last(ind.rsi(close, 14))
    if not math.isnan(rsi_now):
        # 55~70 이 가장 건강한 구간. 과열(>80)은 감점.
        penalty = 0.0
        if rsi_now > 80:
            penalty = (rsi_now - 80) * 1.5
        elif rsi_now < 35:
            penalty = (35 - rsi_now) * 0.8
        base -= penalty
    return float(np.clip(base, 0, 100))


def relative_strength_score(close: pd.Series, benchmark: pd.Series,
                            windows: list[int] | None = None) -> float:
    """벤치마크 대비 초과수익. 지수보다 잘 가는가만 본다."""
    windows = windows or [20, 60, 120]
    aligned = pd.concat([close.rename("a"), benchmark.rename("b")], axis=1).dropna()
    if len(aligned) < 25:
        return 50.0
    parts, used = 0.0, 0.0
    for window, weight in zip(windows, [0.5, 0.3, 0.2]):
        a = ind.returns(aligned["a"], window)
        b = ind.returns(aligned["b"], window)
        if math.isnan(a) or math.isnan(b):
            continue
        parts += weight * squash(a - b, 3 * math.sqrt(window / 20))
        used += weight
    return float(parts / used) if used else 50.0


def flow_score(flow: pd.DataFrame, window: int = 20) -> float:
    """외국인+기관 순매수의 최근 강도. 자기 이력 대비 z-score 로 본다."""
    if flow is None or flow.empty:
        return 50.0
    columns = [c for c in ("foreign", "inst") if c in flow.columns]
    if not columns:
        return 50.0
    net = flow[columns].sum(axis=1)
    recent = net.tail(window).sum()
    rolling = net.rolling(window).sum().dropna()
    if len(rolling) < 10 or rolling.std(ddof=0) == 0:
        return 50.0
    z = (recent - rolling.mean()) / rolling.std(ddof=0)
    return squash(float(z), 1.0)


def risk_score(enriched: pd.DataFrame) -> float:
    """변동성과 낙폭이 작을수록 높은 점수."""
    if enriched.empty:
        return 50.0
    close = enriched["close"]
    vol = ind.annualized_vol(close, 60)
    mdd = ind.max_drawdown(close, 120)
    dd_high = ind.drawdown_from_high(close, 252)

    vol_pts = squash(-(vol - 30) / 1.0, 12) if not math.isnan(vol) else 50.0
    mdd_pts = squash((mdd + 20) / 1.0, 10) if not math.isnan(mdd) else 50.0
    dd_pts = squash((dd_high + 15) / 1.0, 8) if not math.isnan(dd_high) else 50.0
    return float(0.4 * vol_pts + 0.35 * mdd_pts + 0.25 * dd_pts)


# --------------------------------------------------------------------------
# 종합
# --------------------------------------------------------------------------

@dataclass
class ScoreCard:
    ticker: str
    name: str = ""
    total: float = 50.0
    components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        row = {"ticker": self.ticker, "name": self.name, "score": round(self.total, 1)}
        row.update({f"s_{k}": round(v, 1) for k, v in self.components.items()})
        row.update({k: (round(v, 2) if isinstance(v, float) and not math.isnan(v) else v)
                    for k, v in self.metrics.items()})
        return row


def evaluate(ticker: str, prices: pd.DataFrame, *, name: str = "",
             benchmark: pd.Series | None = None,
             flow: pd.DataFrame | None = None) -> ScoreCard:
    """한 종목/ETF/지수의 종합 점수표를 만든다."""
    params = config.get("indicators", {}) or {}
    weights = config.get("scoring.weights", {}) or {}

    if prices is None or prices.empty or "close" not in prices.columns:
        return ScoreCard(ticker=ticker, name=name)

    enriched = ind.enrich(prices, params)
    close = enriched["close"]

    components = {
        "trend": trend_score(enriched),
        "momentum": momentum_score(close),
        "relative_strength": (relative_strength_score(close, benchmark)
                              if benchmark is not None else 50.0),
        "flow": flow_score(flow) if flow is not None else 50.0,
        "risk": risk_score(enriched),
    }

    total, weight_sum = 0.0, 0.0
    for key, value in components.items():
        weight = float(weights.get(key, 0.0))
        total += weight * value
        weight_sum += weight
    total = total / weight_sum if weight_sum else float(np.mean(list(components.values())))

    row = enriched.iloc[-1]
    metrics = {
        "close": float(row["close"]),
        "ret_20d": ind.returns(close, 20),
        "ret_60d": ind.returns(close, 60),
        "ret_120d": ind.returns(close, 120),
        "rsi": float(row.get("rsi", float("nan"))),
        "adx": float(row.get("adx", float("nan"))),
        "atr": float(row.get("atr", float("nan"))),
        "atr_pct": float(row.get("atr_pct", float("nan"))),
        "vol_60d": ind.annualized_vol(close, 60),
        "mdd_120d": ind.max_drawdown(close, 120),
        "from_52w_high": ind.drawdown_from_high(close, 252),
    }
    if "value" in enriched.columns:
        metrics["avg_value_20d"] = float(enriched["value"].tail(20).mean())
    elif "volume" in enriched.columns:
        metrics["avg_value_20d"] = float(
            (enriched["volume"] * enriched["close"]).tail(20).mean())

    return ScoreCard(ticker=ticker, name=name, total=float(np.clip(total, 0, 100)),
                     components=components, metrics=metrics)


def to_frame(cards: list[ScoreCard]) -> pd.DataFrame:
    if not cards:
        return pd.DataFrame()
    return pd.DataFrame([c.as_dict() for c in cards]).sort_values(
        "score", ascending=False).reset_index(drop=True)
