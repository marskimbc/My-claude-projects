"""오프라인(합성) 시세 생성기.

용도는 두 가지다.
  1. 사내망/프록시 등으로 시세 서버에 접근할 수 없을 때 파이프라인 전체를 돌려보기
  2. 테스트에서 네트워크 없이 결정적인 입력을 만들기

시장 공통 요인 + 테마 요인 + 개별 잡음의 3단 구조로 만들어서
상관관계·상대강도 분석이 의미 있는 값을 내도록 했다. 실제 시세가 아니므로
매매 판단에 쓰면 안 된다.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _seed_for(symbol: str, base_seed: int) -> int:
    digest = hashlib.sha1(f"{base_seed}:{symbol}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _sessions(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    """주말을 제외한 영업일 인덱스 (공휴일은 무시)."""
    return pd.bdate_range(start=start, end=end, name="date")


def _market_factor(index: pd.DatetimeIndex, base_seed: int) -> np.ndarray:
    """모든 자산이 공유하는 시장 요인 수익률."""
    rng = np.random.default_rng(_seed_for("__market__", base_seed))
    drift = 0.06 / TRADING_DAYS
    vol = 0.16 / np.sqrt(TRADING_DAYS)
    shocks = rng.normal(drift, vol, len(index))
    # 완만한 국면 전환(강세/약세 사이클)을 얹는다.
    cycle = 0.00035 * np.sin(np.linspace(0, 3 * np.pi, len(index)))
    return shocks + cycle


def ohlcv(symbol: str, start: dt.date | str, end: dt.date | str, *,
          base_seed: int = 20260812, beta: float = 1.0, alpha: float = 0.0,
          vol: float = 0.28, start_price: float | None = None) -> pd.DataFrame:
    """심볼 하나의 합성 일봉을 만든다. 같은 인자면 항상 같은 결과."""
    start = pd.Timestamp(start).date() if not isinstance(start, dt.date) else start
    end = pd.Timestamp(end).date() if not isinstance(end, dt.date) else end
    index = _sessions(start, end)
    if len(index) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    rng = np.random.default_rng(_seed_for(symbol, base_seed))
    market = _market_factor(index, base_seed)
    idio = rng.normal(0.0, vol / np.sqrt(TRADING_DAYS), len(index))
    returns = alpha / TRADING_DAYS + beta * market + idio

    if start_price is None:
        start_price = float(rng.uniform(8_000, 90_000))
    close = start_price * np.exp(np.cumsum(returns))

    intraday = np.abs(rng.normal(0, 0.008, len(index))) + 0.002
    open_ = close * (1 + rng.normal(0, 0.004, len(index)))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = rng.lognormal(mean=12.5, sigma=0.6, size=len(index)).round()

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).round(2)


def investor_flow(symbol: str, start: dt.date | str, end: dt.date | str, *,
                  base_seed: int = 20260812, bias: float = 0.0) -> pd.DataFrame:
    """합성 투자자별 순매수 대금(원)."""
    index = _sessions(
        pd.Timestamp(start).date() if not isinstance(start, dt.date) else start,
        pd.Timestamp(end).date() if not isinstance(end, dt.date) else end,
    )
    rng = np.random.default_rng(_seed_for(f"flow:{symbol}", base_seed))
    scale = 3e9
    foreign = rng.normal(bias * scale * 0.3, scale, len(index))
    inst = rng.normal(bias * scale * 0.2, scale * 0.8, len(index))
    return pd.DataFrame(
        {"foreign": foreign.round(), "inst": inst.round(),
         "retail": (-(foreign + inst)).round()},
        index=index,
    )


def etf_universe(themes_cfg: dict) -> pd.DataFrame:
    """테마 설정으로부터 그럴듯한 ETF 목록을 만들어 낸다(오프라인용)."""
    brands = ["KODEX", "TIGER", "KBSTAR", "ARIRANG", "HANARO"]
    rows: list[dict[str, str]] = []
    counter = 1
    for group in ("themes", "commodities"):
        for _, spec in (themes_cfg.get(group) or {}).items():
            keywords = spec.get("include") or []
            for keyword in keywords[:3]:
                for brand in brands[:2]:
                    rows.append({
                        "ticker": f"9{counter:05d}",
                        "name": f"{brand} {keyword}",
                    })
                    counter += 1
    return pd.DataFrame(rows)
