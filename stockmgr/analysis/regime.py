"""시장 국면(regime) 판정.

코스피/코스닥 추세에 매크로(환율·금리·VIX·해외지수)를 얹어
risk_on / neutral / risk_off 중 하나로 결론 내고, 그에 맞는 총 주식 노출 한도를 준다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .. import config
from ..data.loader import MarketData
from . import indicators as ind
from .trend import TrendState, classify

RISK_ON = "risk_on"
NEUTRAL = "neutral"
RISK_OFF = "risk_off"

LABELS = {RISK_ON: "위험선호", NEUTRAL: "중립", RISK_OFF: "위험회피"}


@dataclass
class Regime:
    label: str
    score: float
    exposure: float
    index_states: dict[str, TrendState] = field(default_factory=dict)
    macro: dict[str, dict] = field(default_factory=dict)
    drivers: list[str] = field(default_factory=list)

    @property
    def label_ko(self) -> str:
        return LABELS.get(self.label, self.label)


def _macro_snapshot(market: MarketData) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, spec in (config.themes().get("macro") or {}).items():
        series = market.macro_series(spec["symbol"])
        if series.empty:
            continue
        close = series["close"]
        out[key] = {
            "label": spec.get("label", key),
            "symbol": spec["symbol"],
            "invert": bool(spec.get("invert", False)),
            "last": float(close.iloc[-1]),
            "ret_5d": ind.returns(close, 5),
            "ret_20d": ind.returns(close, 20),
            "ret_60d": ind.returns(close, 60),
        }
    return out


def _macro_score(snapshot: dict[str, dict]) -> tuple[float, list[str]]:
    """매크로 변화량을 0~100 점수와 설명으로 바꾼다."""
    if not snapshot:
        return 50.0, []
    from .score import squash  # 순환 임포트 회피

    scores: list[float] = []
    drivers: list[str] = []
    for key, item in snapshot.items():
        change = item.get("ret_20d", float("nan"))
        if change is None or math.isnan(change):
            continue
        signed = -change if item["invert"] else change
        scores.append(squash(signed, 3.0))
        if abs(change) >= 3:
            direction = "상승" if change > 0 else "하락"
            effect = "부담" if signed < 0 else "우호"
            drivers.append(f"{item['label']} 1개월 {change:+.1f}% {direction} → 증시에 {effect}")
    if not scores:
        return 50.0, drivers
    return float(sum(scores) / len(scores)), drivers


def detect(market: MarketData) -> Regime:
    index_states: dict[str, TrendState] = {}
    for key in ("kospi", "kosdaq"):
        prices = market.benchmark(key)
        index_states[key] = classify(prices)

    kospi = index_states["kospi"]
    kosdaq = index_states["kosdaq"]
    macro_snapshot = _macro_snapshot(market)
    macro_points, drivers = _macro_score(macro_snapshot)

    # 코스피 60%, 코스닥 15%, 매크로 25%
    score = 0.60 * kospi.score + 0.15 * kosdaq.score + 0.25 * macro_points

    bull = float(config.get("regime.bull_score", 60))
    bear = float(config.get("regime.bear_score", 40))
    if score >= bull:
        label = RISK_ON
    elif score < bear:
        label = RISK_OFF
    else:
        label = NEUTRAL

    exposure = float(config.get(f"regime.exposure.{label}", 0.6))

    drivers = [f"코스피 {kospi.label}({kospi.score:.0f}점)",
               f"코스닥 {kosdaq.label}({kosdaq.score:.0f}점)"] + drivers

    return Regime(label=label, score=score, exposure=exposure,
                  index_states=index_states, macro=macro_snapshot, drivers=drivers)


def macro_frame(regime: Regime) -> pd.DataFrame:
    if not regime.macro:
        return pd.DataFrame()
    rows = [{"지표": v["label"], "심볼": v["symbol"], "현재": round(v["last"], 2),
             "1주": v["ret_5d"], "1개월": v["ret_20d"], "3개월": v["ret_60d"]}
            for v in regime.macro.values()]
    return pd.DataFrame(rows).round(2)
