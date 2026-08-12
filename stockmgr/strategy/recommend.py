"""전체 분석 파이프라인 → 최종 추천.

1) 시장 국면 판정 (지수 + 매크로)
2) 테마 상대강도 / 로테이션
3) 원자재·대체자산 상관
4) 주도 테마의 ETF·개별종목 스크리닝
5) 국면에 맞춘 포지션 사이징과 손절/목표가
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import pandas as pd

from .. import config, universe
from ..analysis import correlation, regime as regime_mod, relative
from ..analysis.regime import Regime
from ..analysis.score import ScoreCard
from ..analysis.trend import TrendState, classify
from ..data.loader import MarketData
from . import risk, signals
from .screener import screen_theme

log = logging.getLogger(__name__)

DISCLAIMER = (
    "본 리포트는 공개 시세를 기계적으로 계산한 참고 자료입니다. "
    "투자 자문이 아니며 최종 판단과 책임은 투자자 본인에게 있습니다."
)


@dataclass
class Idea:
    theme: str
    theme_label: str
    kind: str            # "ETF" | "STOCK"
    ticker: str
    name: str
    score: float
    action: str
    plan: risk.TradePlan
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "테마": self.theme_label,
            "구분": self.kind,
            "종목코드": self.ticker,
            "종목명": self.name,
            "점수": round(self.score, 1),
            "액션": self.action,
            "현재가": self.plan.entry,
            "손절": self.plan.stop,
            "목표": self.plan.target,
            "수량": self.plan.shares,
            "비중%": round(self.plan.weight * 100, 1),
            "손익비": self.plan.reward_risk,
            "근거": " · ".join(self.reasons[:2]),
            "유의": " · ".join(self.cautions[:2]),
        }


@dataclass
class Report:
    generated_at: dt.datetime
    capital: float
    offline: bool
    regime: Regime
    index_states: dict[str, TrendState] = field(default_factory=dict)
    index_series: dict[str, pd.Series] = field(default_factory=dict)
    theme_snapshots: dict[str, relative.ThemeSnapshot] = field(default_factory=dict)
    theme_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_notes: list[str] = field(default_factory=list)
    ideas: list[Idea] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @property
    def ideas_table(self) -> pd.DataFrame:
        if not self.ideas:
            return pd.DataFrame()
        return pd.DataFrame([i.as_row() for i in self.ideas])

    @property
    def buy_ideas(self) -> list[Idea]:
        return [i for i in self.ideas if i.action == signals.BUY]


def _index_states(market: MarketData) -> tuple[dict[str, TrendState], dict[str, pd.Series]]:
    states: dict[str, TrendState] = {}
    series: dict[str, pd.Series] = {}
    for key in (config.themes().get("benchmarks") or {}):
        prices = market.benchmark(key)
        if prices.empty:
            continue
        label = market.benchmark_label(key)
        states[label] = classify(prices)
        series[label] = prices["close"]
    return states, series


def _ideas_from_theme(result: dict, benchmark_regime: str, capital: float,
                      exposure: float, max_per_theme: int) -> list[Idea]:
    ideas: list[Idea] = []
    for kind, cards in (("ETF", result["etfs"]), ("STOCK", result["stocks"])):
        picked = 0
        for card in cards:
            if picked >= max_per_theme:
                break
            signal = signals.generate(card, regime_label=benchmark_regime)
            if signal.action not in (signals.BUY, signals.WATCH):
                continue
            plan = risk.build_plan(
                entry=float(card.metrics.get("close", 0.0)),
                atr=float(card.metrics.get("atr", float("nan"))),
                capital=capital,
                exposure=exposure,
                conviction=risk.conviction_from_score(card.total),
            )
            if plan.shares <= 0:
                continue
            ideas.append(Idea(
                theme=result["theme"], theme_label=result["label"], kind=kind,
                ticker=card.ticker, name=card.name, score=card.total,
                action=signal.action, plan=plan,
                reasons=signal.reasons, cautions=signal.cautions,
            ))
            picked += 1
    return ideas


def build(market: MarketData, capital: float | None = None,
          themes: list[str] | None = None,
          max_ideas: int | None = None) -> Report:
    if capital is None:
        capital = float(config.get("backtest.initial_capital", 10_000_000))
    if max_ideas is None:
        max_ideas = int(config.get("risk.max_positions", 10))

    log.info("시장 국면 판정 중…")
    market_regime = regime_mod.detect(market)
    benchmark = market.benchmark("kospi")["close"]

    log.info("테마 유니버스 해석 중…")
    resolved = universe.all_themes(market)
    if themes:
        resolved = {k: v for k, v in resolved.items() if k in themes}

    log.info("테마 상대강도 계산 중…")
    snapshots = relative.snapshot(market, resolved, benchmark)
    theme_table = relative.to_frame(snapshots)

    log.info("원자재/매크로 상관 계산 중…")
    commodities = correlation.commodity_series(market)
    corr = correlation.matrix(snapshots, commodities)
    notes = correlation.highlights(corr)

    # 상대강도 상위 테마부터 스크리닝한다.
    ranked = sorted((s for s in snapshots.values() if not s.empty),
                    key=lambda s: -s.rs_score)
    max_per_theme = max(1, int(config.get("screener.top_n_per_theme", 5)) // 2)

    ideas: list[Idea] = []
    for snap in ranked:
        if len(ideas) >= max_ideas:
            break
        log.info("스크리닝: %s", snap.label)
        result = screen_theme(market, snap, benchmark)
        ideas.extend(_ideas_from_theme(result, market_regime.label, capital,
                                       market_regime.exposure, max_per_theme))

    ideas.sort(key=lambda i: (signals.ORDER.get(i.action, 9), -i.score))

    # 한 종목이 여러 테마에 편입돼 있으면 점수가 가장 높은 테마 한 곳에만 남긴다.
    seen: set[str] = set()
    deduped: list[Idea] = []
    for idea in ideas:
        if idea.ticker in seen:
            continue
        seen.add(idea.ticker)
        deduped.append(idea)
    ideas = deduped[:max_ideas]

    capped = risk.cap_by_theme([(i.theme, i.plan) for i in ideas])
    for idea, (_, plan) in zip(ideas, capped):
        idea.plan = plan
    ideas = risk.cap_total(ideas, market_regime.exposure, key=lambda i: i.plan,
                           setter=lambda i, p: setattr(i, "plan", p))
    ideas = [i for i in ideas if i.plan.shares > 0]

    index_states, index_series = _index_states(market)
    return Report(
        generated_at=dt.datetime.now(),
        capital=capital,
        offline=market.offline,
        regime=market_regime,
        index_states=index_states,
        index_series=index_series,
        theme_snapshots=snapshots,
        theme_table=theme_table,
        macro_table=regime_mod.macro_frame(market_regime),
        correlation=corr,
        correlation_notes=notes,
        ideas=ideas,
    )
