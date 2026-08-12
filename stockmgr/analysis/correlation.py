"""테마 ↔ 원자재/매크로 상관 분석.

금·은·구리·부동산·환율 같은 변수가 어떤 테마와 같이 움직이는지 본다.
상관은 인과가 아니므로 '해석의 힌트'로만 쓴다.
"""

from __future__ import annotations

import pandas as pd

from ..data.loader import MarketData
from .. import config, universe
from .relative import ThemeSnapshot, basket_series


def commodity_series(market: MarketData) -> dict[str, pd.Series]:
    """원자재/대체자산 시계열. 국내 ETF 를 우선 쓰고 없으면 해외 심볼로 대체."""
    out: dict[str, pd.Series] = {}
    for key, spec in (config.themes().get("commodities") or {}).items():
        label = spec.get("label", key)
        theme = universe.resolve_theme(market, key, group="commodities")
        series = pd.Series(dtype=float)
        if not theme.empty:
            series, _ = basket_series(market, theme, max_members=4)
        if series.empty and spec.get("macro"):
            frame = market.macro_series(spec["macro"])
            if not frame.empty:
                series = frame["close"]
        if not series.empty:
            out[label] = series.rename(label)
    return out


def matrix(theme_snapshots: dict[str, ThemeSnapshot],
           commodities: dict[str, pd.Series],
           window: int = 120) -> pd.DataFrame:
    """일간 수익률 기준 상관계수 행렬 (행: 테마, 열: 원자재)."""
    columns: dict[str, pd.Series] = {}
    for snap in theme_snapshots.values():
        if not snap.empty:
            columns[snap.label] = snap.basket
    columns.update(commodities)
    if len(columns) < 2:
        return pd.DataFrame()

    prices = pd.DataFrame(columns).ffill().tail(window + 1)
    daily = prices.pct_change().dropna(how="all")
    if len(daily) < 20:
        return pd.DataFrame()

    corr = daily.corr()
    theme_labels = [s.label for s in theme_snapshots.values() if not s.empty]
    commodity_labels = [c for c in commodities if c in corr.columns]
    if not theme_labels or not commodity_labels:
        return pd.DataFrame()
    return corr.loc[theme_labels, commodity_labels].round(2)


def highlights(corr: pd.DataFrame, threshold: float = 0.4) -> list[str]:
    """눈에 띄는 상관만 문장으로 뽑는다."""
    if corr.empty:
        return []
    notes: list[str] = []
    for theme in corr.index:
        for asset in corr.columns:
            value = corr.loc[theme, asset]
            if pd.isna(value) or abs(value) < threshold:
                continue
            direction = "같은 방향" if value > 0 else "반대 방향"
            notes.append(f"{theme} ↔ {asset}: {value:+.2f} ({direction})")
    return sorted(notes, key=lambda s: -abs(float(s.split(": ")[1].split(" ")[0])))[:10]
