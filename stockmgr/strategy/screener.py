"""종목 스크리닝.

테마 → 대표 ETF → 구성종목(PDF) 순으로 후보를 좁힌 뒤 점수를 매긴다.
'테마에 실제로 편입된 종목'만 보기 때문에 테마명만 비슷한 종목이 섞이지 않는다.
"""

from __future__ import annotations

import logging

import pandas as pd

from .. import config
from ..analysis.relative import ThemeSnapshot
from ..analysis.score import ScoreCard, evaluate
from ..data.loader import MarketData

log = logging.getLogger(__name__)


def candidate_tickers(market: MarketData, snapshot: ThemeSnapshot,
                      limit: int | None = None) -> pd.DataFrame:
    """테마 ETF 들의 구성종목을 비중 합산해 후보 종목 목록을 만든다."""
    if limit is None:
        limit = int(config.get("screener.universe_size_per_theme", 40))
    if snapshot.members.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    pooled: dict[str, float] = {}
    # 최근 성과 상위 ETF 3개의 구성종목만 본다 (전부 뒤지면 느리고 노이즈가 크다).
    for ticker in snapshot.members["ticker"].head(3):
        pdf = market.constituents(ticker, limit=limit)
        for _, row in pdf.iterrows():
            code = str(row["ticker"]).zfill(6)
            pooled[code] = pooled.get(code, 0.0) + float(row.get("weight", 0.0) or 0.0)

    if not pooled:
        return pd.DataFrame(columns=["ticker", "weight"])

    frame = pd.DataFrame({"ticker": list(pooled), "weight": list(pooled.values())})
    return frame.sort_values("weight", ascending=False).head(limit).reset_index(drop=True)


def score_stocks(market: MarketData, tickers: list[str],
                 benchmark: pd.Series, with_flow: bool = True) -> list[ScoreCard]:
    cards: list[ScoreCard] = []
    for ticker in tickers:
        prices = market.price(ticker, kind="stock")
        if prices.empty or len(prices) < 60:
            continue
        flow = market.flow(ticker) if with_flow else None
        card = evaluate(ticker, prices, name=market.name_of(ticker, "stock"),
                        benchmark=benchmark, flow=flow)
        cards.append(card)
    return sorted(cards, key=lambda c: -c.total)


def score_etfs(market: MarketData, snapshot: ThemeSnapshot,
               benchmark: pd.Series) -> list[ScoreCard]:
    cards: list[ScoreCard] = []
    for _, member in snapshot.members.iterrows():
        prices = market.price(member["ticker"], kind="etf")
        if prices.empty or len(prices) < 60:
            continue
        cards.append(evaluate(member["ticker"], prices, name=member["name"],
                              benchmark=benchmark))
    return sorted(cards, key=lambda c: -c.total)


def screen_theme(market: MarketData, snapshot: ThemeSnapshot,
                 benchmark: pd.Series, top_n: int | None = None) -> dict:
    """테마 하나에 대해 ETF 순위 + 개별종목 순위를 만든다."""
    if top_n is None:
        top_n = int(config.get("screener.top_n_per_theme", 5))

    etf_cards = score_etfs(market, snapshot, benchmark)
    candidates = candidate_tickers(market, snapshot)
    stock_cards = score_stocks(market, candidates["ticker"].tolist(), benchmark)

    return {
        "theme": snapshot.key,
        "label": snapshot.label,
        "stage": snapshot.stage,
        "etfs": etf_cards[:top_n],
        "stocks": stock_cards[:top_n],
    }
