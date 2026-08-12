import datetime as dt

import pytest

from stockmgr import config
from stockmgr.data.loader import MarketData


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """테스트가 서로의 캐시를 오염시키지 않도록 임시 디렉터리를 쓴다."""
    monkeypatch.setattr(config, "cache_dir", lambda: tmp_path / "cache")
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def market() -> MarketData:
    """네트워크를 타지 않는 결정적 시장 데이터."""
    return MarketData(mode="offline", end=dt.date(2026, 8, 12), lookback_days=500)
