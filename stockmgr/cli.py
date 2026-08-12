"""명령줄 진입점.

    python -m stockmgr <명령> [옵션]

공통 옵션
    --mode {auto,live,offline}  데이터 소스 (기본 auto: 실패 시 합성으로 강등)
    --date YYYY-MM-DD           기준일 (기본 오늘)
    --refresh                   캐시 무시하고 새로 조회
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

import pandas as pd

from . import config, universe
from .analysis import correlation as corr_mod, regime as regime_mod, relative
from .backtest import engine as backtest_engine
from .data import cache as cache_mod
from .data.loader import MarketData
from .portfolio import pnl
from .portfolio.holdings import Portfolio
from .report import console, html
from .strategy import recommend
from .strategy.screener import screen_theme


def _market(args: argparse.Namespace) -> MarketData:
    end = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    return MarketData(mode=args.mode, end=end, force_refresh=args.refresh)


def _warn_if_offline(market: MarketData) -> None:
    if market.offline:
        print(f"{console.YELLOW}※ 오프라인(합성) 데이터입니다. 실제 시세가 아닙니다."
              f"{console.RESET}\n", file=sys.stderr)


# --------------------------------------------------------------------------
# 명령
# --------------------------------------------------------------------------

def cmd_market(args: argparse.Namespace) -> int:
    market = _market(args)
    detected = regime_mod.detect(market)
    _warn_if_offline(market)

    print(console.render_regime(detected))

    rows = []
    for key in (config.themes().get("benchmarks") or {}):
        prices = market.benchmark(key)
        if prices.empty:
            continue
        from .analysis.trend import classify
        state = classify(prices)
        rows.append({
            "지수": market.benchmark_label(key),
            "추세": state.label,
            "점수": round(state.score, 1),
            "종가": state.metrics.get("close"),
            "1개월%": state.metrics.get("ret_20d"),
            "3개월%": state.metrics.get("ret_60d"),
            "6개월%": state.metrics.get("ret_120d"),
            "RSI": state.metrics.get("rsi"),
        })
    print(console.heading("지수 동향"))
    print(console.table(pd.DataFrame(rows)))

    print(console.heading("매크로 · 원자재"))
    print(console.table(regime_mod.macro_frame(detected)))
    return 0


def cmd_themes(args: argparse.Namespace) -> int:
    market = _market(args)
    benchmark = market.benchmark("kospi")["close"]
    resolved = universe.all_themes(market)
    if args.theme:
        resolved = {k: v for k, v in resolved.items() if k in args.theme}
    snapshots = relative.snapshot(market, resolved, benchmark)
    _warn_if_offline(market)

    print(console.heading("테마 상대강도 (코스피 대비)"))
    print(console.table(relative.to_frame(snapshots)))

    if args.members:
        for snap in snapshots.values():
            if snap.members.empty:
                continue
            print(console.heading(f"{snap.label} 구성 ETF"))
            print(console.table(snap.members.round(2)))

    commodities = corr_mod.commodity_series(market)
    matrix = corr_mod.matrix(snapshots, commodities)
    print(console.heading("테마 ↔ 원자재/대체자산 상관"))
    if matrix.empty:
        print(f"{console.DIM}(데이터 부족){console.RESET}")
    else:
        print(console.table(matrix.reset_index().rename(columns={"index": "테마"})))
        for note in corr_mod.highlights(matrix)[:6]:
            print(f"  · {note}")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    market = _market(args)
    benchmark = market.benchmark("kospi")["close"]
    resolved = universe.all_themes(market)
    keys = args.theme or list(resolved)
    snapshots = relative.snapshot(market, {k: resolved[k] for k in keys if k in resolved},
                                  benchmark)
    _warn_if_offline(market)

    from .analysis.score import to_frame
    for key, snap in snapshots.items():
        if snap.empty:
            print(f"{console.DIM}{snap.label}: 해당하는 ETF 를 찾지 못했습니다{console.RESET}")
            continue
        result = screen_theme(market, snap, benchmark, top_n=args.top)
        print(console.heading(f"{snap.label} — 상위 ETF"))
        print(console.table(to_frame(result["etfs"])[
            ["ticker", "name", "score", "s_trend", "s_momentum",
             "s_relative_strength", "s_risk", "ret_20d", "ret_60d", "rsi"]]))
        print(console.heading(f"{snap.label} — 상위 개별종목"))
        stocks = to_frame(result["stocks"])
        if stocks.empty:
            print(f"{console.DIM}(구성종목 데이터 없음){console.RESET}")
        else:
            print(console.table(stocks[
                ["ticker", "name", "score", "s_trend", "s_momentum",
                 "s_relative_strength", "s_flow", "ret_20d", "ret_60d", "rsi"]]))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    logging.getLogger("stockmgr").setLevel(logging.INFO if args.verbose else logging.WARNING)
    market = _market(args)
    report = recommend.build(market, capital=args.capital, themes=args.theme,
                             max_ideas=args.max)
    print(console.render(report))
    if args.html:
        path = html.write(report, args.html)
        print(f"\nHTML 리포트: {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    market = _market(args)
    report = recommend.build(market, capital=args.capital)
    path = html.write(report, args.output)
    print(f"HTML 리포트를 생성했습니다: {path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    market = _market(args)
    benchmark = market.benchmark("kospi")["close"]
    resolved = universe.all_themes(market)
    keys = args.theme or list(resolved)

    price_map: dict[str, pd.DataFrame] = {}
    labels: dict[str, str] = {}
    for key in keys:
        theme = resolved.get(key)
        if theme is None or theme.empty:
            continue
        for _, member in theme.members.head(6).iterrows():
            prices = market.price(member["ticker"], kind="etf")
            if len(prices) >= 150:
                price_map[member["ticker"]] = prices
                labels[member["ticker"]] = member["name"]
    _warn_if_offline(market)

    if not price_map:
        print("백테스트할 종목이 없습니다.", file=sys.stderr)
        return 1

    result = backtest_engine.run(price_map, benchmark, top_n=args.top)
    if result.empty:
        print("데이터가 부족해 백테스트를 수행하지 못했습니다.", file=sys.stderr)
        return 1

    print(console.heading(f"백테스트 ({len(price_map)}종목 중 상위 {args.top} 동일가중)"))
    print(console.table(pd.DataFrame([result.stats])))
    if not result.trades.empty:
        trades = result.trades.copy()
        trades["date"] = pd.to_datetime(trades["date"]).dt.strftime("%Y-%m-%d")
        trades["name"] = trades["ticker"].map(labels)
        print(console.heading("최근 매매 내역"))
        print(console.table(trades.tail(15).reset_index(drop=True)))
    print(f"\n{console.DIM}수수료·세금만 반영했고 슬리피지/거래정지/상폐는 "
          f"반영하지 않았습니다.{console.RESET}")
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    portfolio = Portfolio(args.file)

    if args.action == "buy":
        position = portfolio.buy(args.ticker, args.shares, args.price,
                                 name=args.name or "", kind=args.kind,
                                 theme=args.theme_name or "",
                                 stop=args.stop, target=args.target)
        portfolio.save()
        print(f"매수 기록: {position.name}({position.ticker}) "
              f"{position.shares:,}주 @ {position.avg_price:,.0f}")
        return 0

    if args.action == "cash":
        if args.amount is None:
            print(f"현금 잔고: {portfolio.cash:,.0f}원")
            return 0
        portfolio.cash = args.amount
        portfolio.save()
        print(f"현금 잔고를 {portfolio.cash:,.0f}원으로 설정했습니다.")
        return 0

    if args.action == "sell":
        realized = portfolio.sell(args.ticker, args.shares, args.price)
        portfolio.save()
        print(f"매도 기록: {args.ticker} {args.shares:,}주 @ {args.price:,.0f} "
              f"· 실현손익 {realized:+,.0f}원")
        return 0

    market = _market(args)
    frame = pnl.valuate(market, portfolio)
    _warn_if_offline(market)
    print(console.heading("보유 종목"))
    if frame.empty:
        print(f"{console.DIM}보유 종목이 없습니다. "
              f"'portfolio buy' 로 추가하세요.{console.RESET}")
        return 0
    print(console.table(frame))
    print(console.heading("요약"))
    print(console.table(pd.DataFrame([pnl.summary(frame, portfolio.cash)])))
    warnings = pnl.alerts(frame)
    if warnings:
        print(console.heading("점검 필요"))
        for line in warnings:
            print(f"  {line}")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    removed = cache_mod.clear(args.namespace)
    print(f"캐시 파일 {removed}개를 삭제했습니다.")
    return 0


# --------------------------------------------------------------------------
# 파서
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockmgr",
        description="국내 주식(코스피/코스닥) 지수·테마 ETF·원자재 분석 및 매매 추천",
    )
    parser.add_argument("--mode", choices=["auto", "live", "offline"], default="auto",
                        help="데이터 소스 (기본 auto)")
    parser.add_argument("--date", help="기준일 YYYY-MM-DD (기본 오늘)")
    parser.add_argument("--refresh", action="store_true", help="캐시 무시하고 재조회")
    parser.add_argument("--verbose", "-v", action="store_true", help="진행 로그 출력")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("market", help="지수·매크로 동향과 시장 국면")
    p.set_defaults(func=cmd_market)

    p = sub.add_parser("themes", help="테마 상대강도와 원자재 상관")
    p.add_argument("--theme", action="append", help="특정 테마만 (여러 번 지정 가능)")
    p.add_argument("--members", action="store_true", help="테마별 구성 ETF 도 출력")
    p.set_defaults(func=cmd_themes)

    p = sub.add_parser("screen", help="테마별 상위 ETF/종목 스크리닝")
    p.add_argument("--theme", action="append", help="특정 테마만")
    p.add_argument("--top", type=int, default=5, help="테마당 표시 개수")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("recommend", help="종합 리포트 + 매매 추천")
    p.add_argument("--capital", type=float, help="기준 자본금(원)")
    p.add_argument("--theme", action="append", help="특정 테마만")
    p.add_argument("--max", type=int, help="최대 추천 종목 수")
    p.add_argument("--html", help="HTML 리포트도 저장할 경로")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("report", help="HTML 대시보드 생성")
    p.add_argument("--output", "-o", default="reports/report.html")
    p.add_argument("--capital", type=float)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("backtest", help="점수 기반 리밸런싱 백테스트")
    p.add_argument("--theme", action="append")
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("portfolio", help="보유 종목 관리/평가")
    p.add_argument("action", nargs="?", default="status",
                   choices=["status", "buy", "sell", "cash"])
    p.add_argument("--file", help="원장 JSON 경로")
    p.add_argument("--amount", type=float, help="cash 액션에서 설정할 현금 잔고")
    p.add_argument("--ticker")
    p.add_argument("--shares", type=int)
    p.add_argument("--price", type=float)
    p.add_argument("--name")
    p.add_argument("--kind", choices=["stock", "etf"], default="stock")
    p.add_argument("--theme-name", dest="theme_name")
    p.add_argument("--stop", type=float)
    p.add_argument("--target", type=float)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("cache", help="캐시 삭제")
    p.add_argument("--namespace", help="특정 네임스페이스만 삭제")
    p.set_defaults(func=cmd_cache)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.WARNING)
    args = build_parser().parse_args(argv)

    if args.command == "portfolio" and args.action in ("buy", "sell"):
        missing = [f"--{n}" for n in ("ticker", "shares", "price")
                   if getattr(args, n) is None]
        if missing:
            print(f"{args.action} 에는 {', '.join(missing)} 가 필요합니다.", file=sys.stderr)
            return 2

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 최상단 방어
        if args.verbose:
            raise
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
