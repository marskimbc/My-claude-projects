"""테마 간 상대강도 / 로테이션.

각 테마를 대표 ETF 바스켓(동일가중)으로 만들고, 코스피 대비 초과성과와
그 초과성과의 '가속 여부'로 로테이션 국면을 나눈다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..data.loader import MarketData
from ..universe import ThemeUniverse
from . import indicators as ind

LEADING = "주도"
IMPROVING = "개선"
WEAKENING = "둔화"
LAGGING = "소외"


@dataclass
class ThemeSnapshot:
    key: str
    label: str
    basket: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    members: pd.DataFrame = field(default_factory=pd.DataFrame)
    excess: dict[str, float] = field(default_factory=dict)
    stage: str = LAGGING
    rs_score: float = 50.0

    @property
    def empty(self) -> bool:
        return self.basket.empty


def basket_series(market: MarketData, theme: ThemeUniverse,
                  max_members: int = 8) -> tuple[pd.Series, pd.DataFrame]:
    """테마 구성 ETF 들의 동일가중 지수(기준=100)와 멤버별 수익률 표."""
    frames: dict[str, pd.Series] = {}
    rows: list[dict] = []
    for _, member in theme.members.head(max_members).iterrows():
        prices = market.price(member["ticker"], kind="etf")
        if prices.empty or len(prices) < 30:
            continue
        frames[member["ticker"]] = prices["close"]
        rows.append({
            "ticker": member["ticker"],
            "name": member["name"],
            "close": float(prices["close"].iloc[-1]),
            "ret_20d": ind.returns(prices["close"], 20),
            "ret_60d": ind.returns(prices["close"], 60),
        })
    if not frames:
        return pd.Series(dtype=float), pd.DataFrame()

    combined = pd.DataFrame(frames).dropna(how="all").ffill()
    normalized = combined.divide(combined.bfill().iloc[0]).mean(axis=1) * 100
    members = pd.DataFrame(rows).sort_values("ret_60d", ascending=False)
    return normalized.rename(theme.key), members.reset_index(drop=True)


def _stage(excess_short: float, excess_long: float) -> str:
    """단기/장기 초과수익 조합으로 로테이션 사분면을 정한다."""
    if math.isnan(excess_short) or math.isnan(excess_long):
        return LAGGING
    if excess_long > 0 and excess_short > 0:
        return LEADING
    if excess_long <= 0 and excess_short > 0:
        return IMPROVING
    if excess_long > 0 and excess_short <= 0:
        return WEAKENING
    return LAGGING


def snapshot(market: MarketData, themes: dict[str, ThemeUniverse],
             benchmark: pd.Series) -> dict[str, ThemeSnapshot]:
    from .score import relative_strength_score  # 순환 임포트 회피

    out: dict[str, ThemeSnapshot] = {}
    for key, theme in themes.items():
        basket, members = basket_series(market, theme)
        snap = ThemeSnapshot(key=key, label=theme.label, basket=basket, members=members)
        if basket.empty:
            out[key] = snap
            continue

        aligned = pd.concat([basket.rename("t"), benchmark.rename("b")], axis=1).dropna()
        excess = {}
        for window, name in ((20, "1개월"), (60, "3개월"), (120, "6개월")):
            theme_ret = ind.returns(aligned["t"], window)
            bench_ret = ind.returns(aligned["b"], window)
            excess[name] = (theme_ret - bench_ret
                            if not (math.isnan(theme_ret) or math.isnan(bench_ret))
                            else float("nan"))
        snap.excess = excess
        snap.stage = _stage(excess.get("1개월", float("nan")),
                            excess.get("3개월", float("nan")))
        snap.rs_score = relative_strength_score(aligned["t"], aligned["b"])
        out[key] = snap
    return out


def to_frame(snapshots: dict[str, ThemeSnapshot]) -> pd.DataFrame:
    rows = []
    for snap in snapshots.values():
        if snap.empty:
            continue
        rows.append({
            "테마": snap.label,
            "국면": snap.stage,
            "RS점수": round(snap.rs_score, 1),
            "초과1M": snap.excess.get("1개월", float("nan")),
            "초과3M": snap.excess.get("3개월", float("nan")),
            "초과6M": snap.excess.get("6개월", float("nan")),
            "종목수": len(snap.members),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("RS점수", ascending=False).round(2).reset_index(drop=True)
