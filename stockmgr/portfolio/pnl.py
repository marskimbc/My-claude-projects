"""보유 종목 평가와 리스크 점검."""

from __future__ import annotations

import pandas as pd

from ..analysis.score import evaluate
from ..data.loader import MarketData
from ..strategy import signals
from .holdings import Portfolio


def valuate(market: MarketData, portfolio: Portfolio) -> pd.DataFrame:
    """현재가 기준 평가손익 + 재평가 점수/액션."""
    if portfolio.is_empty():
        return pd.DataFrame()

    benchmark = market.benchmark("kospi")["close"]
    rows = []
    for position in portfolio.positions.values():
        prices = market.price(position.ticker, kind=position.kind)
        if prices.empty:
            continue
        last = float(prices["close"].iloc[-1])
        card = evaluate(position.ticker, prices, name=position.name,
                        benchmark=benchmark)
        signal = signals.generate(card, prices)

        market_value = last * position.shares
        pnl = market_value - position.cost
        stop_hit = position.stop is not None and last <= position.stop
        target_hit = position.target is not None and last >= position.target

        rows.append({
            "종목코드": position.ticker,
            "종목명": position.name,
            "수량": position.shares,
            "평균단가": round(position.avg_price, 2),
            "현재가": round(last, 2),
            "평가금액": round(market_value, 0),
            "평가손익": round(pnl, 0),
            "수익률%": round((last / position.avg_price - 1) * 100, 2)
            if position.avg_price else float("nan"),
            "점수": round(card.total, 1),
            "액션": "손절도달" if stop_hit else ("목표도달" if target_hit else signal.action),
            "손절": position.stop,
            "목표": position.target,
        })

    frame = pd.DataFrame(rows)
    return frame.sort_values("평가손익", ascending=False).reset_index(drop=True)


def summary(frame: pd.DataFrame, cash: float = 0.0) -> dict:
    """현금은 별도 항목으로 보여준다(주식 평가액과 섞으면 손익률이 왜곡된다)."""
    if frame.empty:
        return {"주식평가액": 0.0, "매입금액": 0.0, "평가손익": 0.0,
                "수익률%": 0.0, "현금": round(cash, 0),
                "총자산": round(cash, 0), "종목수": 0}
    market_value = float(frame["평가금액"].sum())
    cost = float((frame["평균단가"] * frame["수량"]).sum())
    pnl = market_value - cost
    return {
        "주식평가액": round(market_value, 0),
        "매입금액": round(cost, 0),
        "평가손익": round(pnl, 0),
        "수익률%": round(pnl / cost * 100, 2) if cost else 0.0,
        "현금": round(cash, 0),
        "총자산": round(market_value + cash, 0),
        "종목수": int(len(frame)),
    }


def alerts(frame: pd.DataFrame) -> list[str]:
    """즉시 확인이 필요한 항목만 뽑는다."""
    if frame.empty:
        return []
    out: list[str] = []
    for _, row in frame.iterrows():
        if row["액션"] == "손절도달":
            out.append(f"⚠ {row['종목명']}({row['종목코드']}) 손절가 도달 — 현재 {row['현재가']:,.0f}")
        elif row["액션"] == "목표도달":
            out.append(f"◎ {row['종목명']}({row['종목코드']}) 목표가 도달 — 일부 익절 검토")
        elif row["액션"] in (signals.SELL, signals.REDUCE):
            out.append(f"↓ {row['종목명']}({row['종목코드']}) 점수 {row['점수']} — {row['액션']} 검토")
    return out
