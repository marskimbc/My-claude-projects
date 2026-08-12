"""점수 기반 리밸런싱 백테스트.

목적은 '이 점수 체계가 벤치마크보다 나은가'를 눈으로 확인하는 것이다.
리밸런싱 시점마다 그 시점까지의 데이터만 써서 점수를 매기므로 미래 정보가 새지 않는다.
슬리피지·거래정지·상장폐지는 반영하지 않으니 결과를 과신하면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..analysis import indicators as ind
from ..analysis.score import momentum_score, trend_score

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.equity.empty


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    if len(index) == 0:
        return []
    series = pd.Series(index, index=index)
    rule = "W-MON" if frequency == "weekly" else "MS"
    grouped = series.resample(rule).first().dropna()
    return [d for d in grouped.tolist() if d in index]


def _point_in_time_score(prices: pd.DataFrame) -> float:
    """리밸런싱 시점까지의 데이터만으로 매기는 축약 점수 (추세 60% + 모멘텀 40%)."""
    if len(prices) < 130:
        return float("nan")
    enriched = ind.enrich(prices, config.get("indicators", {}) or {})
    return 0.6 * trend_score(enriched) + 0.4 * momentum_score(enriched["close"])


def _stats(equity: pd.Series, benchmark: pd.Series | None) -> dict:
    if equity.empty:
        return {}
    daily = equity.pct_change().dropna()
    years = max(len(equity) / TRADING_DAYS, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1
    vol = daily.std(ddof=0) * np.sqrt(TRADING_DAYS)
    out = {
        "총수익률%": round(total * 100, 2),
        "CAGR%": round(cagr * 100, 2),
        "변동성%": round(vol * 100, 2),
        "샤프": round(cagr / vol, 2) if vol > 0 else float("nan"),
        "MDD%": round(ind.max_drawdown(equity), 2),
        "승률%": round((daily > 0).mean() * 100, 1),
    }
    if benchmark is not None and not benchmark.empty:
        bench_total = benchmark.iloc[-1] / benchmark.iloc[0] - 1
        out["벤치마크수익률%"] = round(bench_total * 100, 2)
        out["초과수익률%"] = round((total - bench_total) * 100, 2)
    return out


def run(price_map: dict[str, pd.DataFrame], benchmark: pd.Series | None = None, *,
        top_n: int = 5, initial_capital: float | None = None,
        fee_bps: float | None = None, frequency: str | None = None) -> BacktestResult:
    """price_map: {ticker: OHLCV DataFrame}. 동일가중 롱 온리."""
    if not price_map:
        return BacktestResult()

    if initial_capital is None:
        initial_capital = float(config.get("backtest.initial_capital", 10_000_000))
    if fee_bps is None:
        fee_bps = float(config.get("backtest.fee_bps", 15))
    if frequency is None:
        frequency = str(config.get("backtest.rebalance", "monthly"))

    closes = pd.DataFrame({t: df["close"] for t, df in price_map.items()}).sort_index()
    closes = closes.ffill().dropna(how="all")
    if closes.empty or len(closes) < 150:
        return BacktestResult()

    dates = closes.index
    rebalance_days = _rebalance_dates(dates, frequency)
    if not rebalance_days:
        return BacktestResult()

    fee_rate = fee_bps / 10_000
    equity = pd.Series(index=dates, dtype=float)
    weights = pd.Series(0.0, index=closes.columns)
    capital = float(initial_capital)
    trade_rows: list[dict] = []

    previous_holdings: set[str] = set()
    for i, day in enumerate(dates):
        if i > 0:
            daily_returns = closes.iloc[i] / closes.iloc[i - 1] - 1
            portfolio_return = float((weights * daily_returns.fillna(0)).sum())
            capital *= 1 + portfolio_return
        equity.iloc[i] = capital

        if day in rebalance_days:
            scores: dict[str, float] = {}
            history = closes.loc[:day]
            for ticker in closes.columns:
                frame = price_map[ticker].loc[:day]
                if frame.empty or frame["close"].isna().all():
                    continue
                score = _point_in_time_score(frame)
                if not np.isnan(score):
                    scores[ticker] = score

            picks = sorted(scores, key=lambda t: -scores[t])[:top_n]
            new_weights = pd.Series(0.0, index=closes.columns)
            if picks:
                new_weights[picks] = 1.0 / len(picks)

            turnover = float((new_weights - weights).abs().sum())
            capital *= 1 - turnover * fee_rate / 2
            weights = new_weights

            holdings = set(picks)
            for ticker in holdings - previous_holdings:
                trade_rows.append({"date": day, "ticker": ticker, "side": "BUY",
                                   "price": float(history[ticker].iloc[-1]),
                                   "score": round(scores.get(ticker, float("nan")), 1)})
            for ticker in previous_holdings - holdings:
                trade_rows.append({"date": day, "ticker": ticker, "side": "SELL",
                                   "price": float(history[ticker].iloc[-1]),
                                   "score": round(scores.get(ticker, float("nan")), 1)})
            previous_holdings = holdings

    equity = equity.dropna()
    bench = None
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.reindex(equity.index).ffill().dropna()
        if not bench.empty:
            bench = bench / bench.iloc[0] * initial_capital

    return BacktestResult(
        equity=equity,
        benchmark=bench if bench is not None else pd.Series(dtype=float),
        trades=pd.DataFrame(trade_rows),
        stats=_stats(equity, bench),
    )
