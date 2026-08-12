"""리스크 관리 — 손절/목표가와 포지션 크기.

포지션 크기는 '얼마나 확신하는가'가 아니라 '틀렸을 때 얼마를 잃는가'로 정한다.
계좌 대비 허용 손실(account_risk_per_trade)을 손절폭으로 나눠 수량을 구한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import config


@dataclass
class TradePlan:
    entry: float
    stop: float
    target: float
    shares: int
    weight: float          # 계좌 대비 비중 (0~1)
    risk_amount: float     # 손절 시 예상 손실액
    reward_risk: float     # 손익비

    @property
    def stop_pct(self) -> float:
        return (self.stop / self.entry - 1) * 100 if self.entry else float("nan")

    @property
    def target_pct(self) -> float:
        return (self.target / self.entry - 1) * 100 if self.entry else float("nan")


def build_plan(entry: float, atr: float, capital: float, *,
               exposure: float = 1.0,
               conviction: float = 1.0) -> TradePlan:
    """ATR 기반 손절/목표와 리스크 균등 포지션 사이징.

    exposure   : 시장 국면에 따른 전체 노출 배수 (0~1)
    conviction : 점수 기반 확신 배수 (0.5~1.0 정도)
    """
    cfg = config.get("risk", {}) or {}
    risk_per_trade = float(cfg.get("account_risk_per_trade", 0.01))
    stop_mult = float(cfg.get("atr_stop_multiple", 2.0))
    target_mult = float(cfg.get("atr_target_multiple", 4.0))
    max_weight = float(cfg.get("max_position_weight", 0.15))

    if entry <= 0 or capital <= 0:
        return TradePlan(entry, entry, entry, 0, 0.0, 0.0, 0.0)

    # ATR 을 못 구하면 진입가의 5% 를 손절폭으로 가정한다.
    if atr is None or (isinstance(atr, float) and math.isnan(atr)) or atr <= 0:
        atr = entry * 0.05 / stop_mult

    stop_distance = stop_mult * atr
    stop = max(entry - stop_distance, 0.0)
    target = entry + target_mult * atr

    budget = capital * risk_per_trade * exposure * conviction
    shares = int(budget // stop_distance) if stop_distance > 0 else 0

    cap_shares = int((capital * max_weight * exposure) // entry)
    shares = max(0, min(shares, cap_shares))

    position_value = shares * entry
    return TradePlan(
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        shares=shares,
        weight=position_value / capital if capital else 0.0,
        risk_amount=round(shares * stop_distance, 0),
        reward_risk=round(target_mult / stop_mult, 2) if stop_mult else 0.0,
    )


def conviction_from_score(score: float) -> float:
    """점수 50~90 을 확신 배수 0.5~1.0 으로 선형 매핑."""
    if score is None or math.isnan(score):
        return 0.5
    return float(min(1.0, max(0.5, 0.5 + (score - 50) / 80)))


def _scale(plan: TradePlan, factor: float) -> TradePlan:
    return TradePlan(
        entry=plan.entry, stop=plan.stop, target=plan.target,
        shares=int(plan.shares * factor),
        weight=plan.weight * factor,
        risk_amount=round(plan.risk_amount * factor, 0),
        reward_risk=plan.reward_risk,
    )


def cap_total(items: list, exposure: float, key, setter) -> list:
    """전체 주식 비중이 국면별 노출 한도를 넘지 않도록 비례 축소한다.

    items 는 아무 객체나 될 수 있고, key(item)->TradePlan / setter(item, plan) 로
    계획을 읽고 쓴다.
    """
    limit = min(float(config.get("risk.max_total_weight", 0.95)), max(exposure, 0.0))
    total = sum(key(item).weight for item in items)
    if total <= limit or total <= 0:
        return items
    factor = limit / total
    for item in items:
        setter(item, _scale(key(item), factor))
    return items


def cap_by_theme(plans: list[tuple[str, TradePlan]]) -> list[tuple[str, TradePlan]]:
    """테마별 비중 상한(max_theme_weight)을 넘으면 비례 축소한다."""
    limit = float(config.get("risk.max_theme_weight", 0.30))
    totals: dict[str, float] = {}
    for theme, plan in plans:
        totals[theme] = totals.get(theme, 0.0) + plan.weight

    adjusted: list[tuple[str, TradePlan]] = []
    for theme, plan in plans:
        total = totals.get(theme, 0.0)
        if total > limit > 0:
            plan = _scale(plan, limit / total)
        adjusted.append((theme, plan))
    return adjusted
