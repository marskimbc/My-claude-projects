"""KRX 데이터 어댑터 (pykrx).

pykrx 는 한글 컬럼(시가/고가/저가/종가/거래량/거래대금)을 돌려주므로
여기서 전부 영문 표준 컬럼(open/high/low/close/volume/value)으로 바꾼다.
이 계층 밖에서는 한글 컬럼을 다루지 않는다.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from . import cache

log = logging.getLogger(__name__)

_COLUMN_MAP = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "거래대금": "value",
    "등락률": "change_pct",
    "NAV": "nav",
    "기초지수": "underlying",
    "상장시가총액": "market_cap",
    "시가총액": "market_cap",
    "기관합계": "inst",
    "외국인합계": "foreign",
    "개인": "retail",
    "기타법인": "corp",
    "전체": "total",
    "비중": "weight",
    "금액": "amount",
    "계약수": "shares",
}

_OHLCV = ["open", "high", "low", "close", "volume"]


class DataUnavailable(RuntimeError):
    """시세 서버에 접근할 수 없거나 응답이 비어 있을 때."""


def _pykrx():
    try:
        from pykrx import stock  # noqa: PLC0415 - 선택적 의존성
    except ImportError as exc:  # pragma: no cover - 설치 환경에 따라 다름
        raise DataUnavailable("pykrx 가 설치되어 있지 않습니다. pip install pykrx") from exc
    return stock


def _ymd(value: dt.date | str) -> str:
    if isinstance(value, str):
        return value.replace("-", "")
    return value.strftime("%Y%m%d")


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.rename(columns=_COLUMN_MAP).copy()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """일별 시계열용: 컬럼명을 바꾸고 인덱스를 날짜로 맞춘다."""
    out = _rename(df)
    if out.empty:
        return out
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out.sort_index()


def _call(fn, *args, normalize=_normalize, **kwargs) -> pd.DataFrame:
    """pykrx 호출을 감싸 네트워크/파싱 오류를 DataUnavailable 로 통일한다."""
    try:
        return normalize(fn(*args, **kwargs))
    except DataUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - 네트워크 의존
        raise DataUnavailable(f"KRX 조회 실패: {fn.__name__}({args}): {exc}") from exc


# --------------------------------------------------------------------------
# 시세
# --------------------------------------------------------------------------

def index_ohlcv(code: str, start: dt.date | str, end: dt.date | str,
                force_refresh: bool = False) -> pd.DataFrame:
    """지수 일봉. code 는 pykrx 지수코드(예: 코스피 '1001')."""
    def produce() -> pd.DataFrame:
        stock = _pykrx()
        return _call(stock.get_index_ohlcv, _ymd(start), _ymd(end), code)

    df = cache.frame("krx_index", {"code": code, "s": _ymd(start), "e": _ymd(end)},
                     produce, force_refresh=force_refresh)
    return df[[c for c in _OHLCV if c in df.columns]] if not df.empty else df


def etf_ohlcv(ticker: str, start: dt.date | str, end: dt.date | str,
              force_refresh: bool = False) -> pd.DataFrame:
    def produce() -> pd.DataFrame:
        stock = _pykrx()
        return _call(stock.get_etf_ohlcv_by_date, _ymd(start), _ymd(end), ticker)

    df = cache.frame("krx_etf", {"t": ticker, "s": _ymd(start), "e": _ymd(end)},
                     produce, force_refresh=force_refresh)
    if df.empty:
        return df
    keep = [c for c in (*_OHLCV, "value", "nav") if c in df.columns]
    return df[keep]


def stock_ohlcv(ticker: str, start: dt.date | str, end: dt.date | str,
                force_refresh: bool = False) -> pd.DataFrame:
    def produce() -> pd.DataFrame:
        stock = _pykrx()
        return _call(stock.get_market_ohlcv, _ymd(start), _ymd(end), ticker)

    df = cache.frame("krx_stock", {"t": ticker, "s": _ymd(start), "e": _ymd(end)},
                     produce, force_refresh=force_refresh)
    if df.empty:
        return df
    keep = [c for c in (*_OHLCV, "value") if c in df.columns]
    return df[keep]


# --------------------------------------------------------------------------
# 종목 목록 / 메타
# --------------------------------------------------------------------------

def etf_universe(date: dt.date | str | None = None,
                 force_refresh: bool = False) -> pd.DataFrame:
    """상장 ETF 전체 목록. columns: ticker, name."""
    day = _ymd(date or dt.date.today())

    def produce() -> pd.DataFrame:
        stock = _pykrx()
        try:
            tickers = stock.get_etf_ticker_list(day)
            rows = [{"ticker": t, "name": stock.get_etf_ticker_name(t)} for t in tickers]
        except Exception as exc:  # pragma: no cover - 네트워크 의존
            raise DataUnavailable(f"ETF 목록 조회 실패: {exc}") from exc
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.set_index("ticker")

    df = cache.frame("krx_etf_list", {"d": day}, produce, date_index=False,
                     ttl_hours=24 * 7, force_refresh=force_refresh)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "name"])
    return df.reset_index().rename(columns={"index": "ticker"})


def stock_universe(market: str = "KOSPI", date: dt.date | str | None = None,
                   force_refresh: bool = False) -> pd.DataFrame:
    """시장별 상장 종목 목록. columns: ticker, name, market."""
    day = _ymd(date or dt.date.today())

    def produce() -> pd.DataFrame:
        stock = _pykrx()
        try:
            tickers = stock.get_market_ticker_list(day, market=market)
            rows = [{"ticker": t, "name": stock.get_market_ticker_name(t),
                     "market": market} for t in tickers]
        except Exception as exc:  # pragma: no cover - 네트워크 의존
            raise DataUnavailable(f"{market} 종목 목록 조회 실패: {exc}") from exc
        frame = pd.DataFrame(rows)
        return frame.set_index("ticker") if not frame.empty else frame

    df = cache.frame("krx_stock_list", {"m": market, "d": day}, produce, date_index=False,
                     ttl_hours=24 * 7, force_refresh=force_refresh)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "name", "market"])
    return df.reset_index().rename(columns={"index": "ticker"})


def etf_constituents(ticker: str, date: dt.date | str | None = None,
                     force_refresh: bool = False) -> pd.DataFrame:
    """ETF 의 PDF(구성종목). columns: ticker, weight."""
    day = _ymd(date or dt.date.today())

    def produce() -> pd.DataFrame:
        stock = _pykrx()
        df = _call(stock.get_etf_portfolio_deposit_file, ticker, day, normalize=_rename)
        if not df.empty:
            df.index = df.index.astype(str).str.zfill(6)
            df.index.name = "ticker"
        return df

    df = cache.frame("krx_etf_pdf", {"t": ticker, "d": day}, produce, date_index=False,
                     ttl_hours=24, force_refresh=force_refresh)
    if df.empty or "weight" not in df.columns:
        return pd.DataFrame(columns=["ticker", "weight"])
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "ticker"})
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    out = out[out["ticker"].str.fullmatch(r"\d{6}")]
    return out[["ticker", "weight"]].sort_values("weight", ascending=False)


def index_constituents(code: str, force_refresh: bool = False) -> list[str]:
    """지수 구성종목 코드 목록."""
    def produce() -> pd.DataFrame:
        stock = _pykrx()
        try:
            tickers = stock.get_index_portfolio_deposit_file(code)
        except Exception as exc:  # pragma: no cover - 네트워크 의존
            raise DataUnavailable(f"지수 구성종목 조회 실패({code}): {exc}") from exc
        frame = pd.DataFrame({"ticker": [str(t).zfill(6) for t in tickers]})
        frame["listed"] = 1
        return frame.set_index("ticker")

    df = cache.frame("krx_index_pdf", {"c": code}, produce, date_index=False,
                     ttl_hours=24 * 7, force_refresh=force_refresh)
    return [] if df.empty else [str(t).zfill(6) for t in df.index]


# --------------------------------------------------------------------------
# 수급
# --------------------------------------------------------------------------

def investor_flow(ticker: str, start: dt.date | str, end: dt.date | str,
                  force_refresh: bool = False) -> pd.DataFrame:
    """투자자별 순매수 대금(일별). columns: foreign, inst, retail 등."""
    def produce() -> pd.DataFrame:
        stock = _pykrx()
        return _call(stock.get_market_trading_value_by_date,
                     _ymd(start), _ymd(end), ticker)

    df = cache.frame("krx_flow", {"t": ticker, "s": _ymd(start), "e": _ymd(end)},
                     produce, force_refresh=force_refresh)
    if df.empty:
        return df
    keep = [c for c in ("foreign", "inst", "retail", "corp") if c in df.columns]
    return df[keep]


def fundamentals(market: str = "KOSPI", date: dt.date | str | None = None,
                 force_refresh: bool = False) -> pd.DataFrame:
    """시장 전체 밸류에이션 스냅샷 (PER/PBR/DIV 등). index: ticker."""
    day = _ymd(date or dt.date.today())

    def produce() -> pd.DataFrame:
        stock = _pykrx()
        try:
            df = stock.get_market_fundamental_by_ticker(day, market=market)
        except Exception as exc:  # pragma: no cover - 네트워크 의존
            raise DataUnavailable(f"펀더멘털 조회 실패({market}): {exc}") from exc
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = df.index.astype(str).str.zfill(6)
        df.index.name = "ticker"
        return df

    return cache.frame("krx_fundamental", {"m": market, "d": day}, produce,
                       date_index=False, ttl_hours=24, force_refresh=force_refresh)
