"""보유 종목 저장소 (JSON).

증권사 연동 없이 손으로 관리하는 단순 원장. 같은 종목을 여러 번 사면
평균단가를 갱신하고, 팔면 수량을 줄인다.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config

DEFAULT_PATH = config.PROJECT_ROOT / "data" / "holdings.json"


@dataclass
class Position:
    ticker: str
    name: str = ""
    shares: int = 0
    avg_price: float = 0.0
    kind: str = "stock"          # stock | etf
    theme: str = ""
    stop: float | None = None
    target: float | None = None
    opened_at: str = field(default_factory=lambda: dt.date.today().isoformat())
    note: str = ""

    @property
    def cost(self) -> float:
        return self.shares * self.avg_price


class Portfolio:
    def __init__(self, path: Path | str | None = None, cash: float = 0.0):
        self.path = Path(path or DEFAULT_PATH)
        self.cash = cash
        self.positions: dict[str, Position] = {}
        if self.path.exists():
            self.load()

    # -- 저장/로드 ---------------------------------------------------------
    def load(self) -> None:
        with self.path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        self.cash = float(payload.get("cash", 0.0))
        self.positions = {
            item["ticker"]: Position(**item) for item in payload.get("positions", [])
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "positions": [asdict(p) for p in self.positions.values()],
        }
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    # -- 매매 --------------------------------------------------------------
    def buy(self, ticker: str, shares: int, price: float, *, name: str = "",
            kind: str = "stock", theme: str = "", stop: float | None = None,
            target: float | None = None) -> Position:
        if shares <= 0 or price <= 0:
            raise ValueError("수량과 가격은 0보다 커야 합니다")
        ticker = str(ticker).zfill(6)
        existing = self.positions.get(ticker)
        if existing is None:
            position = Position(ticker=ticker, name=name or ticker, shares=shares,
                                avg_price=float(price), kind=kind, theme=theme,
                                stop=stop, target=target)
            self.positions[ticker] = position
        else:
            total_cost = existing.cost + shares * price
            existing.shares += shares
            existing.avg_price = total_cost / existing.shares
            if name:
                existing.name = name
            if stop is not None:
                existing.stop = stop
            if target is not None:
                existing.target = target
            position = existing
        self.cash -= shares * price
        return position

    def sell(self, ticker: str, shares: int, price: float) -> float:
        """매도하고 실현손익을 돌려준다."""
        ticker = str(ticker).zfill(6)
        position = self.positions.get(ticker)
        if position is None:
            raise KeyError(f"보유하지 않은 종목입니다: {ticker}")
        if shares <= 0 or shares > position.shares:
            raise ValueError(f"매도 수량이 올바르지 않습니다 (보유 {position.shares}주)")

        realized = (price - position.avg_price) * shares
        position.shares -= shares
        self.cash += shares * price
        if position.shares == 0:
            del self.positions[ticker]
        return realized

    def is_empty(self) -> bool:
        return not self.positions
