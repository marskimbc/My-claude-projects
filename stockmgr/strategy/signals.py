"""매매 신호.

점수(ScoreCard)와 지표 상태를 조합해 5단계 액션을 낸다.
임계값은 settings.yaml 의 signals 항목에서 조정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .. import config
from ..analysis import indicators as ind
from ..analysis.score import ScoreCard

BUY = "매수"
WATCH = "관심"
HOLD = "보유"
REDUCE = "비중축소"
SELL = "매도"

ORDER = {BUY: 0, WATCH: 1, HOLD: 2, REDUCE: 3, SELL: 4}


@dataclass
class Signal:
    ticker: str
    name: str
    action: str
    score: float
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        return self.action in (BUY, WATCH)


def _liquid_enough(card: ScoreCard) -> bool:
    floor = float(config.get("signals.min_avg_trading_value", 0))
    value = card.metrics.get("avg_value_20d")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True  # 거래대금을 모르면 유동성 필터를 적용하지 않는다
    return value >= floor


def generate(card: ScoreCard, prices: pd.DataFrame | None = None,
             regime_label: str = "neutral") -> Signal:
    cfg = config.get("signals", {}) or {}
    buy_at = float(cfg.get("buy_score", 70))
    watch_at = float(cfg.get("watch_score", 55))
    reduce_at = float(cfg.get("reduce_score", 45))
    sell_at = float(cfg.get("sell_score", 35))
    overbought = float(cfg.get("rsi_overbought", 75))
    oversold = float(cfg.get("rsi_oversold", 30))

    score = card.total
    reasons: list[str] = []
    cautions: list[str] = []

    if score >= buy_at:
        action = BUY
    elif score >= watch_at:
        action = WATCH
    elif score >= reduce_at:
        action = HOLD
    elif score >= sell_at:
        action = REDUCE
    else:
        action = SELL

    components = card.components
    top = sorted(components.items(), key=lambda kv: -kv[1])[:2]
    labels = {"trend": "추세", "momentum": "모멘텀", "relative_strength": "상대강도",
              "flow": "수급", "risk": "안정성"}
    reasons += [f"{labels.get(k, k)} {v:.0f}점" for k, v in top]

    weak = [k for k, v in components.items() if v < 40]
    cautions += [f"{labels.get(k, k)} 약함({components[k]:.0f}점)" for k in weak]

    rsi_value = card.metrics.get("rsi", float("nan"))
    if isinstance(rsi_value, float) and not math.isnan(rsi_value):
        if rsi_value >= overbought:
            cautions.append(f"단기 과열(RSI {rsi_value:.0f}) — 분할 진입 권장")
            if action == BUY:
                action = WATCH
        elif rsi_value <= oversold:
            cautions.append(f"과매도(RSI {rsi_value:.0f}) — 반등 확인 후 진입")

    if not _liquid_enough(card):
        cautions.append("20일 평균 거래대금 기준 미달 — 유동성 주의")
        if action == BUY:
            action = WATCH

    # 위험회피 국면에서는 신규 매수를 관심으로 낮추고 청산 신호를 강화한다.
    if regime_label == "risk_off":
        if action == BUY:
            action = WATCH
            cautions.append("시장 국면 위험회피 — 신규 매수 보류")
        elif action == REDUCE:
            action = SELL

    if prices is not None and not prices.empty and len(prices) > 60:
        close = prices["close"]
        ma20 = ind.sma(close, 20)
        if pd.notna(ma20.iloc[-1]) and close.iloc[-1] < ma20.iloc[-1]:
            cautions.append("20일선 하회")
        gap = ind.drawdown_from_high(close, 252)
        if not math.isnan(gap) and gap > -3:
            reasons.append("52주 신고가 부근")

    return Signal(ticker=card.ticker, name=card.name, action=action,
                  score=score, reasons=reasons, cautions=cautions)
