"""실시간(pykrx) 경로 검증.

CI/개발 환경에서 KRX 에 접속할 수 없는 경우가 많으므로, pykrx 가 돌려주는
한글 컬럼 응답을 흉내 낸 가짜 모듈로 어댑터 변환 로직을 검증한다.
네트워크 자체를 검증하는 테스트는 아니다.
"""

import datetime as dt

import pandas as pd
import pytest

from stockmgr.data import krx
from stockmgr.data.loader import MarketData


class FakePykrx:
    """pykrx.stock 의 응답 형태(한글 컬럼)를 그대로 흉내 낸다."""

    def __init__(self):
        self.calls: list[str] = []

    def _index(self, days: int = 5) -> pd.Index:
        return pd.Index(
            [(dt.date(2026, 8, 1) + dt.timedelta(days=i)).strftime("%Y%m%d")
             for i in range(days)],
            name="날짜",
        )

    def get_index_ohlcv(self, start, end, ticker):
        self.calls.append("index")
        index = self._index()
        return pd.DataFrame({
            "시가": range(100, 105), "고가": range(101, 106),
            "저가": range(99, 104), "종가": range(100, 105),
            "거래량": [1_000] * 5, "거래대금": [1_000_000] * 5,
            "상장시가총액": [1e12] * 5,
        }, index=index)

    def get_market_ohlcv(self, start, end, ticker):
        self.calls.append("stock")
        index = self._index()
        return pd.DataFrame({
            "시가": range(70_000, 70_005), "고가": range(70_100, 70_105),
            "저가": range(69_900, 69_905), "종가": range(70_000, 70_005),
            "거래량": [12_345] * 5, "거래대금": [8_640_000_000] * 5,
            "등락률": [0.1] * 5,
        }, index=index)

    def get_etf_ohlcv_by_date(self, start, end, ticker):
        self.calls.append("etf")
        index = self._index()
        return pd.DataFrame({
            "NAV": [10_000.0] * 5, "시가": range(10_000, 10_005),
            "고가": range(10_010, 10_015), "저가": range(9_990, 9_995),
            "종가": range(10_000, 10_005), "거래량": [500] * 5,
            "거래대금": [5_000_000] * 5, "기초지수": [2_500.0] * 5,
        }, index=index)

    def get_etf_ticker_list(self, date):
        return ["069500", "091160"]

    def get_etf_ticker_name(self, ticker):
        return {"069500": "KODEX 200", "091160": "KODEX 반도체"}[ticker]

    def get_market_ticker_list(self, date, market="KOSPI"):
        return ["005930", "000660"]

    def get_market_ticker_name(self, ticker):
        return {"005930": "삼성전자", "000660": "SK하이닉스"}[ticker]

    def get_etf_portfolio_deposit_file(self, ticker, date):
        # 종목코드가 앞자리 0 을 포함한 문자열 인덱스로 온다.
        return pd.DataFrame(
            {"계약수": [10, 20], "금액": [1e8, 2e8], "비중": [25.5, 12.3]},
            index=pd.Index(["005930", "000660"], name="종목코드"),
        )

    def get_index_portfolio_deposit_file(self, ticker):
        return ["005930", "000660"]

    def get_market_trading_value_by_date(self, start, end, ticker):
        index = self._index()
        return pd.DataFrame({
            "기관합계": [1e9] * 5, "기타법인": [1e7] * 5,
            "개인": [-2e9] * 5, "외국인합계": [1e9] * 5, "전체": [0] * 5,
        }, index=index)

    def get_market_fundamental_by_ticker(self, date, market="KOSPI"):
        return pd.DataFrame(
            {"BPS": [50_000, 40_000], "PER": [12.3, 9.8], "PBR": [1.4, 1.1],
             "EPS": [5_600, 4_100], "DIV": [2.1, 1.5], "DPS": [1_400, 1_200]},
            index=pd.Index(["005930", "000660"], name="티커"),
        )


@pytest.fixture
def fake(monkeypatch) -> FakePykrx:
    stub = FakePykrx()
    monkeypatch.setattr(krx, "_pykrx", lambda: stub)
    return stub


def test_index_ohlcv_is_translated(fake):
    df = krx.index_ohlcv("1001", "20260801", "20260805")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_stock_ohlcv_keeps_trading_value(fake):
    df = krx.stock_ohlcv("005930", "20260801", "20260805")
    assert "value" in df.columns
    assert df["close"].iloc[-1] == 70_004


def test_etf_ohlcv_keeps_nav(fake):
    df = krx.etf_ohlcv("069500", "20260801", "20260805")
    assert {"close", "nav"} <= set(df.columns)


def test_etf_universe_returns_names(fake):
    df = krx.etf_universe(dt.date(2026, 8, 12))
    assert list(df.columns) == ["ticker", "name"]
    assert df.loc[df["ticker"] == "069500", "name"].iloc[0] == "KODEX 200"


def test_stock_universe_labels_market(fake):
    df = krx.stock_universe("KOSPI", dt.date(2026, 8, 12))
    assert set(df["market"]) == {"KOSPI"}
    assert "005930" in set(df["ticker"])


def test_constituents_keep_leading_zeros_and_sort_by_weight(fake):
    df = krx.etf_constituents("069500", dt.date(2026, 8, 12))
    assert list(df.columns) == ["ticker", "weight"]
    assert df["ticker"].iloc[0] == "005930"          # 비중 내림차순
    assert all(len(t) == 6 for t in df["ticker"])    # '000660' 이 660 으로 깨지지 않는다


def test_index_constituents_returns_padded_codes(fake):
    assert krx.index_constituents("1028") == ["005930", "000660"]


def test_investor_flow_translates_investor_columns(fake):
    df = krx.investor_flow("005930", "20260801", "20260805")
    assert {"foreign", "inst", "retail"} <= set(df.columns)


def test_fundamentals_index_is_ticker(fake):
    df = krx.fundamentals("KOSPI", dt.date(2026, 8, 12))
    assert "PER" in df.columns
    assert "005930" in df.index


def test_live_market_uses_krx_and_stays_online(fake):
    data = MarketData(mode="live", end=dt.date(2026, 8, 12))
    assert not data.benchmark("kospi").empty
    assert not data.price("005930", kind="stock").empty
    assert not data.price("069500", kind="etf").empty
    assert data.name_of("005930", "stock") == "삼성전자"
    assert data.offline is False
    assert "index" in fake.calls


def test_live_market_scores_a_real_shaped_response(fake):
    from stockmgr.analysis.score import evaluate

    data = MarketData(mode="live", end=dt.date(2026, 8, 12))
    card = evaluate("005930", data.price("005930"), name="삼성전자")
    assert card.name == "삼성전자"
    assert 0 <= card.total <= 100
