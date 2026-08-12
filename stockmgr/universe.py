"""테마 → 실제 ETF 종목 해석.

종목코드를 고정해두면 상장/폐지 때마다 깨지므로, themes.yaml 의 이름 패턴을
KRX 전체 ETF 목록에 매칭해 실행 시점에 유니버스를 만든다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .data.loader import MarketData


@dataclass
class ThemeUniverse:
    key: str
    label: str
    members: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ticker", "name"]))

    @property
    def empty(self) -> bool:
        return self.members.empty

    @property
    def tickers(self) -> list[str]:
        return self.members["ticker"].tolist()


def _matches(name: str, include: list[str], exclude: list[str]) -> bool:
    spaced = name.upper()                 # 단어 경계 판정용 (공백 유지)
    packed = spaced.replace(" ", "")      # 'AI 반도체' == 'AI반도체' 판정용
    for word in exclude:
        if word and word.upper().replace(" ", "") in packed:
            return False
    for word in include:
        needle = word.upper().replace(" ", "")
        if not needle:
            continue
        # 'AI' 처럼 짧은 영문 키워드는 단어 경계를 지켜 오탐(CHAIN 등)을 줄인다.
        if needle.isascii() and len(needle) <= 3:
            if re.search(rf"(?<![A-Z0-9]){re.escape(needle)}(?![A-Z0-9])", spaced):
                return True
        elif needle in packed:
            return True
    return False


def resolve_theme(market: MarketData, key: str, group: str = "themes") -> ThemeUniverse:
    spec = (config.themes().get(group) or {}).get(key)
    if spec is None:
        raise KeyError(f"알 수 없는 {group}: {key}")

    label = spec.get("label", key)
    include = list(spec.get("include") or [])
    exclude = list(spec.get("exclude") or []) + list(config.themes().get("global_exclude") or [])
    pinned = [str(t).zfill(6) for t in (spec.get("pinned") or [])]

    etfs = market.etf_list()
    if etfs.empty:
        return ThemeUniverse(key=key, label=label)

    mask = etfs["name"].astype(str).apply(lambda n: _matches(n, include, exclude))
    members = etfs[mask | etfs["ticker"].isin(pinned)].copy()
    members = members.drop_duplicates(subset="ticker").reset_index(drop=True)
    return ThemeUniverse(key=key, label=label, members=members[["ticker", "name"]])


def all_themes(market: MarketData) -> dict[str, ThemeUniverse]:
    return {key: resolve_theme(market, key) for key in (config.themes().get("themes") or {})}


def all_commodities(market: MarketData) -> dict[str, ThemeUniverse]:
    return {key: resolve_theme(market, key, group="commodities")
            for key in (config.themes().get("commodities") or {})}


def theme_labels() -> dict[str, str]:
    return {k: v.get("label", k) for k, v in (config.themes().get("themes") or {}).items()}
