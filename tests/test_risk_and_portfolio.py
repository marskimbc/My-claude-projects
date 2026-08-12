import pytest

from stockmgr.portfolio.holdings import Portfolio
from stockmgr.strategy import risk


def test_plan_places_stop_and_target_by_atr():
    plan = risk.build_plan(entry=10_000, atr=200, capital=10_000_000)
    assert plan.stop == pytest.approx(10_000 - 2 * 200)
    assert plan.target == pytest.approx(10_000 + 4 * 200)
    assert plan.reward_risk == pytest.approx(2.0)


def test_position_size_respects_account_risk():
    capital = 10_000_000
    plan = risk.build_plan(entry=10_000, atr=200, capital=capital)
    # 손절 시 손실이 계좌의 1% 를 크게 넘지 않아야 한다.
    assert plan.risk_amount <= capital * 0.011


def test_position_size_capped_by_max_weight():
    plan = risk.build_plan(entry=1_000, atr=1, capital=10_000_000)
    assert plan.weight <= 0.15 + 1e-9


def test_missing_atr_falls_back_to_percentage_stop():
    plan = risk.build_plan(entry=10_000, atr=float("nan"), capital=10_000_000)
    assert plan.stop < plan.entry
    assert plan.shares > 0


def test_zero_capital_returns_empty_plan():
    plan = risk.build_plan(entry=10_000, atr=200, capital=0)
    assert plan.shares == 0


def test_exposure_scales_position_down():
    full = risk.build_plan(entry=10_000, atr=200, capital=10_000_000, exposure=1.0)
    half = risk.build_plan(entry=10_000, atr=200, capital=10_000_000, exposure=0.5)
    assert half.shares < full.shares


def test_theme_cap_scales_overweight_theme():
    plans = [("ai", risk.build_plan(10_000, 200, 10_000_000)) for _ in range(4)]
    capped = risk.cap_by_theme(plans)
    assert sum(p.weight for _, p in capped) <= 0.30 + 1e-9


def test_conviction_mapping_is_bounded():
    assert risk.conviction_from_score(100) == 1.0
    assert risk.conviction_from_score(0) == 0.5
    assert 0.5 < risk.conviction_from_score(70) < 1.0


def test_portfolio_averages_price_on_additional_buy(tmp_path):
    book = Portfolio(tmp_path / "holdings.json")
    book.buy("069500", 10, 10_000, name="KODEX 200", kind="etf")
    book.buy("069500", 10, 12_000)
    position = book.positions["069500"]
    assert position.shares == 20
    assert position.avg_price == pytest.approx(11_000)
    assert position.name == "KODEX 200"


def test_portfolio_sell_returns_realized_pnl(tmp_path):
    book = Portfolio(tmp_path / "holdings.json")
    book.buy("069500", 10, 10_000)
    realized = book.sell("069500", 5, 12_000)
    assert realized == pytest.approx(10_000)
    assert book.positions["069500"].shares == 5


def test_portfolio_sell_all_removes_position(tmp_path):
    book = Portfolio(tmp_path / "holdings.json")
    book.buy("069500", 10, 10_000)
    book.sell("069500", 10, 11_000)
    assert book.is_empty()


def test_portfolio_rejects_oversell(tmp_path):
    book = Portfolio(tmp_path / "holdings.json")
    book.buy("069500", 10, 10_000)
    with pytest.raises(ValueError):
        book.sell("069500", 11, 10_000)


def test_portfolio_round_trips_through_disk(tmp_path):
    path = tmp_path / "holdings.json"
    book = Portfolio(path)
    book.cash = 1_000_000
    book.buy("069500", 10, 10_000, name="KODEX 200", kind="etf", stop=9_000)
    book.save()

    reloaded = Portfolio(path)
    assert reloaded.cash == book.cash
    assert reloaded.positions["069500"].stop == 9_000
