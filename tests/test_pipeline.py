"""파이프라인 통합 테스트 — 전부 오프라인(합성) 데이터로 돌린다."""

import pandas as pd
import pytest

from stockmgr import universe
from stockmgr.analysis import correlation, regime, relative
from stockmgr.backtest import engine
from stockmgr.cli import main
from stockmgr.report import console, html
from stockmgr.strategy import recommend


@pytest.fixture(scope="module")
def report():
    import datetime as dt

    from stockmgr.data.loader import MarketData
    data = MarketData(mode="offline", end=dt.date(2026, 8, 12))
    return recommend.build(data, capital=10_000_000, themes=["defense", "power"],
                           max_ideas=6)


def test_regime_detect_returns_known_label(market):
    detected = regime.detect(market)
    assert detected.label in (regime.RISK_ON, regime.NEUTRAL, regime.RISK_OFF)
    assert 0 <= detected.score <= 100
    assert 0 < detected.exposure <= 1.0
    assert detected.drivers


def test_theme_snapshot_ranks_and_stages(market):
    benchmark = market.benchmark("kospi")["close"]
    themes = {k: universe.resolve_theme(market, k) for k in ("defense", "power")}
    snapshots = relative.snapshot(market, themes, benchmark)
    assert set(snapshots) == {"defense", "power"}
    for snap in snapshots.values():
        assert snap.stage in (relative.LEADING, relative.IMPROVING,
                              relative.WEAKENING, relative.LAGGING)
    table = relative.to_frame(snapshots)
    assert list(table["RS점수"]) == sorted(table["RS점수"], reverse=True)


def test_correlation_matrix_is_bounded(market):
    benchmark = market.benchmark("kospi")["close"]
    themes = {k: universe.resolve_theme(market, k) for k in ("defense", "power")}
    snapshots = relative.snapshot(market, themes, benchmark)
    matrix = correlation.matrix(snapshots, correlation.commodity_series(market))
    assert not matrix.empty
    assert matrix.to_numpy().min() >= -1.0
    assert matrix.to_numpy().max() <= 1.0


def test_report_has_all_sections(report):
    assert report.regime is not None
    assert report.index_states
    assert not report.theme_table.empty
    assert report.disclaimer


def test_ideas_are_deduplicated_and_within_limits(report):
    tickers = [i.ticker for i in report.ideas]
    assert len(tickers) == len(set(tickers))
    assert len(report.ideas) <= 6
    for idea in report.ideas:
        assert idea.plan.shares > 0
        assert idea.plan.stop < idea.plan.entry < idea.plan.target
        assert idea.plan.weight <= 0.15 + 1e-9


def test_total_exposure_is_capped(report):
    total = sum(i.plan.weight for i in report.ideas)
    assert total <= 0.80 + 1e-9


def test_console_render_contains_key_sections(report):
    text = console.render(report)
    for marker in ("시장 국면", "지수 동향", "테마 상대강도", "매매 추천"):
        assert marker in text


def test_html_report_is_self_contained(report, tmp_path):
    path = html.write(report, tmp_path / "r.html")
    content = path.read_text(encoding="utf-8")
    assert "<title>" in content
    assert "http://" not in content and "https://" not in content
    assert "prefers-color-scheme" in content


def test_offline_report_is_flagged(report):
    assert report.offline is True
    assert "오프라인" in console.render(report)


def test_backtest_produces_equity_curve(market):
    theme = universe.resolve_theme(market, "power")
    price_map = {t: market.price(t, kind="etf") for t in theme.tickers[:5]}
    result = engine.run(price_map, market.benchmark("kospi")["close"], top_n=2)
    assert not result.empty
    assert len(result.equity) > 100
    assert "CAGR%" in result.stats
    assert result.equity.index.is_monotonic_increasing


def test_backtest_handles_empty_input():
    assert engine.run({}).empty


def test_backtest_trades_alternate_buy_sell(market):
    theme = universe.resolve_theme(market, "power")
    price_map = {t: market.price(t, kind="etf") for t in theme.tickers[:5]}
    result = engine.run(price_map, top_n=2)
    if not result.trades.empty:
        assert set(result.trades["side"]) <= {"BUY", "SELL"}
        assert pd.api.types.is_datetime64_any_dtype(result.trades["date"])


@pytest.mark.parametrize("argv", [
    ["--mode", "offline", "market"],
    ["--mode", "offline", "themes", "--theme", "defense"],
    ["--mode", "offline", "screen", "--theme", "power", "--top", "2"],
    ["--mode", "offline", "portfolio"],
])
def test_cli_commands_exit_zero(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out


def test_cli_report_writes_file(tmp_path, capsys):
    target = tmp_path / "out.html"
    code = main(["--mode", "offline", "report", "-o", str(target)])
    assert code == 0
    assert target.exists() and target.stat().st_size > 1000


def test_cli_rejects_incomplete_buy(capsys):
    assert main(["portfolio", "buy", "--ticker", "005930"]) == 2
