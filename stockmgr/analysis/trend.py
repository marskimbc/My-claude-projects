"""추세 판정.

가격 시계열 하나를 받아 사람이 읽을 수 있는 추세 상태로 요약한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .. import config
from . import indicators as ind
from .score import trend_score

STRONG_UP = "강한상승"
UP = "상승"
SIDEWAYS = "횡보"
DOWN = "하락"
STRONG_DOWN = "강한하락"


@dataclass
class TrendState:
    label: str
    score: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def bullish(self) -> bool:
        return self.label in (STRONG_UP, UP)


def _fmt(value: float, unit: str = "%") -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:+.1f}{unit}"


def classify(prices: pd.DataFrame) -> TrendState:
    if prices is None or prices.empty or len(prices) < 30:
        return TrendState(label=SIDEWAYS, score=50.0, reasons=["데이터 부족"])

    params = config.get("indicators", {}) or {}
    enriched = ind.enrich(prices, params)
    row = enriched.iloc[-1]
    close = enriched["close"]
    score = trend_score(enriched)

    reasons: list[str] = []
    trend_ma = row.get("ma_trend")
    if pd.notna(trend_ma):
        above = row["close"] > trend_ma
        reasons.append(f"200일선 {'상회' if above else '하회'}"
                       f" ({row['close'] / trend_ma - 1:+.1%})")
    if pd.notna(row.get("ma_short")) and pd.notna(row.get("ma_mid")):
        reasons.append("20>60일선 정배열" if row["ma_short"] > row["ma_mid"]
                       else "20<60일선 역배열")

    adx_value = row.get("adx", float("nan"))
    if pd.notna(adx_value):
        if adx_value >= 25:
            reasons.append(f"추세 강함(ADX {adx_value:.0f})")
        elif adx_value < 20:
            reasons.append(f"방향성 약함(ADX {adx_value:.0f})")

    ret_60 = ind.returns(close, 60)
    reasons.append(f"3개월 {_fmt(ret_60)}")

    if score >= 72:
        label = STRONG_UP
    elif score >= 58:
        label = UP
    elif score >= 42:
        label = SIDEWAYS
    elif score >= 28:
        label = DOWN
    else:
        label = STRONG_DOWN

    metrics = {
        "close": float(row["close"]),
        "ret_20d": ind.returns(close, 20),
        "ret_60d": ret_60,
        "ret_120d": ind.returns(close, 120),
        "rsi": float(row.get("rsi", float("nan"))),
        "adx": float(adx_value) if pd.notna(adx_value) else float("nan"),
        "vol_60d": ind.annualized_vol(close, 60),
        "from_52w_high": ind.drawdown_from_high(close, 252),
        "above_ma200_pct": (float(row["close"] / trend_ma - 1) * 100
                            if pd.notna(trend_ma) else float("nan")),
    }
    return TrendState(label=label, score=score, reasons=reasons, metrics=metrics)
