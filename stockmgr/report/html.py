"""HTML 대시보드 생성.

외부 CDN 없이 단일 파일로 떨어지도록 CSS/SVG 를 전부 인라인으로 넣는다.
차트는 의존성 없이 직접 그린 SVG 스파크라인이다.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from ..strategy import signals
from ..strategy.recommend import Report

_CSS = """
:root{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#16191d; --muted:#5d6570; --line:#e3e6ea;
  --up:#c0392b; --down:#1f6fb2; --accent:#3d5afe; --warn:#b8860b;
  --chip:#eef1f5;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14161a; --panel:#1c1f24; --ink:#e8eaed; --muted:#9aa2ad; --line:#2c3138;
    --up:#ff6b5b; --down:#5aa9e6; --accent:#8c9eff; --warn:#e0b25c; --chip:#262a31;
  }
}
:root[data-theme="dark"]{
  --bg:#14161a; --panel:#1c1f24; --ink:#e8eaed; --muted:#9aa2ad; --line:#2c3138;
  --up:#ff6b5b; --down:#5aa9e6; --accent:#8c9eff; --warn:#e0b25c; --chip:#262a31;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",
  "Noto Sans KR",system-ui,sans-serif;line-height:1.55;font-size:15px}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 72px}
header h1{font-size:1.5rem;margin:0 0 4px}
header .meta{color:var(--muted);font-size:.85rem}
.notice{margin:16px 0;padding:12px 14px;border-radius:10px;
  background:color-mix(in srgb,var(--warn) 14%,transparent);
  border:1px solid color-mix(in srgb,var(--warn) 40%,transparent);font-size:.9rem}
section{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:20px;margin:18px 0}
section h2{font-size:1.05rem;margin:0 0 14px;display:flex;align-items:center;gap:8px}
section h2 .num{display:inline-flex;width:22px;height:22px;border-radius:6px;
  background:var(--accent);color:#fff;font-size:.75rem;align-items:center;
  justify-content:center;flex:none}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.tile{background:var(--chip);border-radius:10px;padding:12px 14px}
.tile .k{font-size:.78rem;color:var(--muted)}
.tile .v{font-size:1.28rem;font-weight:650;margin-top:2px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.87rem;min-width:520px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;
  white-space:nowrap}
th:first-child,td:first-child,th.l,td.l{text-align:left}
thead th{color:var(--muted);font-weight:600;border-bottom:1.5px solid var(--line)}
tbody tr:hover{background:var(--chip)}
.pos{color:var(--up)} .neg{color:var(--down)}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;
  font-weight:600}
.b-buy{background:color-mix(in srgb,var(--up) 18%,transparent);color:var(--up)}
.b-watch{background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--accent)}
.b-hold{background:var(--chip);color:var(--muted)}
.b-reduce{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)}
.b-sell{background:color-mix(in srgb,var(--down) 18%,transparent);color:var(--down)}
.notes{margin:12px 0 0;padding-left:18px;color:var(--muted);font-size:.86rem}
.idea{border:1px solid var(--line);border-radius:11px;padding:14px;margin-top:12px}
.idea .top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  flex-wrap:wrap}
.idea .name{font-weight:650}
.idea .lv{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;font-size:.87rem}
.idea .lv b{font-weight:600}
.idea .why{margin-top:8px;font-size:.85rem;color:var(--muted)}
.spark{display:block}
footer{color:var(--muted);font-size:.82rem;margin-top:26px;text-align:center}
"""


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        if abs(value) >= 10000:
            return f"{value:,.0f}"
        return f"{value:,.{digits}f}"
    return html.escape(str(value))


def _signed(value, digits: int = 2, suffix: str = "%") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<span class='muted'>-</span>"
    cls = "pos" if value > 0 else ("neg" if value < 0 else "")
    return f"<span class='{cls}'>{value:+,.{digits}f}{suffix}</span>"


def _sparkline(series: pd.Series, width: int = 220, height: int = 46) -> str:
    clean = series.dropna().tail(180)
    if len(clean) < 3:
        return ""
    values = clean.to_numpy(dtype=float)
    low, high = float(values.min()), float(values.max())
    span = (high - low) or 1.0
    step = width / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - (v - low) / span * (height - 6) - 3:.1f}"
        for i, v in enumerate(values)
    )
    rising = values[-1] >= values[0]
    color = "var(--up)" if rising else "var(--down)"
    return (
        f"<svg class='spark' viewBox='0 0 {width} {height}' width='{width}' "
        f"height='{height}' role='img' aria-label='최근 추이'>"
        f"<polyline points='{points}' fill='none' stroke='{color}' "
        f"stroke-width='1.8' stroke-linejoin='round' stroke-linecap='round'/></svg>"
    )


def _table(frame: pd.DataFrame, signed_columns: tuple[str, ...] = ()) -> str:
    if frame is None or frame.empty:
        return "<p class='notes'>데이터 없음</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if column in signed_columns and isinstance(value, (int, float)):
                cells.append(f"<td>{_signed(float(value))}</td>")
            elif isinstance(value, str):
                cells.append(f"<td class='l'>{html.escape(value)}</td>")
            else:
                cells.append(f"<td>{_fmt(value)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _badge(action: str) -> str:
    key = {signals.BUY: "buy", signals.WATCH: "watch", signals.HOLD: "hold",
           signals.REDUCE: "reduce", signals.SELL: "sell"}.get(action, "hold")
    return f"<span class='badge b-{key}'>{html.escape(action)}</span>"


def _correlation_table(corr: pd.DataFrame) -> str:
    if corr.empty:
        return "<p class='notes'>상관 계산에 필요한 데이터가 부족합니다.</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in corr.columns)
    rows = []
    for theme, row in corr.iterrows():
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("<td>-</td>")
                continue
            strength = min(abs(float(value)), 1.0) * 26
            tone = "var(--up)" if value > 0 else "var(--down)"
            cells.append(
                f"<td style='background:color-mix(in srgb,{tone} {strength:.0f}%,"
                f"transparent)'>{value:+.2f}</td>")
        rows.append(f"<tr><td class='l'>{html.escape(str(theme))}</td>"
                    f"{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table><thead><tr><th class='l'>테마</th>{head}</tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table></div>")


def build(report: Report) -> str:
    parts: list[str] = []
    parts.append(f"<title>국내주식 지수·테마 리포트</title><style>{_CSS}</style>")
    parts.append("<div class='wrap'>")
    parts.append(
        f"<header><h1>국내주식 지수·테마 분석 리포트</h1>"
        f"<div class='meta'>생성 {report.generated_at:%Y-%m-%d %H:%M} · "
        f"기준자본 {report.capital:,.0f}원</div></header>")

    if report.offline:
        parts.append("<div class='notice'>⚠ 오프라인(합성) 데이터로 만든 리포트입니다. "
                     "실제 시세가 아니므로 매매 판단에 사용하지 마세요.</div>")

    regime = report.regime
    parts.append("<section><h2><span class='num'>1</span>시장 국면</h2>")
    parts.append(
        "<div class='grid'>"
        f"<div class='tile'><div class='k'>판정</div>"
        f"<div class='v'>{html.escape(regime.label_ko)}</div></div>"
        f"<div class='tile'><div class='k'>종합 점수</div>"
        f"<div class='v'>{regime.score:.1f}</div></div>"
        f"<div class='tile'><div class='k'>권장 주식 노출</div>"
        f"<div class='v'>{regime.exposure:.0%}</div></div>"
        f"<div class='tile'><div class='k'>추천 종목수</div>"
        f"<div class='v'>{len(report.ideas)}</div></div>"
        "</div>")
    if regime.drivers:
        drivers = "".join(f"<li>{html.escape(d)}</li>" for d in regime.drivers[:8])
        parts.append(f"<ul class='notes'>{drivers}</ul>")
    parts.append("</section>")

    parts.append("<section><h2><span class='num'>2</span>지수 동향</h2><div class='grid'>")
    for label, state in report.index_states.items():
        series = report.index_series.get(label, pd.Series(dtype=float))
        parts.append(
            f"<div class='tile'><div class='k'>{html.escape(label)} · "
            f"{html.escape(state.label)}</div>"
            f"<div class='v'>{_fmt(state.metrics.get('close'))}</div>"
            f"<div class='k'>1M {_signed(state.metrics.get('ret_20d'))} · "
            f"3M {_signed(state.metrics.get('ret_60d'))}</div>"
            f"{_sparkline(series)}</div>")
    parts.append("</div></section>")

    parts.append("<section><h2><span class='num'>3</span>매크로 · 원자재</h2>")
    parts.append(_table(report.macro_table, signed_columns=("1주", "1개월", "3개월")))
    parts.append("</section>")

    parts.append("<section><h2><span class='num'>4</span>테마 상대강도</h2>")
    parts.append(_table(report.theme_table,
                        signed_columns=("초과1M", "초과3M", "초과6M")))
    parts.append("<p class='notes'>주도=단기·중기 모두 코스피 우위 / 개선=단기 반등 / "
                 "둔화=단기 약화 / 소외=둘 다 열위</p></section>")

    parts.append("<section><h2><span class='num'>5</span>테마 ↔ 원자재 상관</h2>")
    parts.append(_correlation_table(report.correlation))
    if report.correlation_notes:
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in report.correlation_notes[:8])
        parts.append(f"<ul class='notes'>{notes}</ul>")
    parts.append("</section>")

    parts.append("<section><h2><span class='num'>6</span>매매 추천</h2>")
    if not report.ideas:
        parts.append("<p class='notes'>조건을 만족하는 추천 종목이 없습니다.</p>")
    else:
        for idea in report.ideas:
            plan = idea.plan
            parts.append(
                f"<div class='idea'><div class='top'>"
                f"<div><span class='name'>{html.escape(idea.name)}</span> "
                f"<span class='meta'>({html.escape(idea.ticker)}) · "
                f"{html.escape(idea.theme_label)} · {idea.kind}</span></div>"
                f"<div>{_badge(idea.action)} <b>{idea.score:.1f}점</b></div></div>"
                f"<div class='lv'>"
                f"<span>진입 <b>{plan.entry:,.0f}</b></span>"
                f"<span>손절 <b>{plan.stop:,.0f}</b> ({plan.stop_pct:+.1f}%)</span>"
                f"<span>목표 <b>{plan.target:,.0f}</b> ({plan.target_pct:+.1f}%)</span>"
                f"<span>수량 <b>{plan.shares:,}</b>주</span>"
                f"<span>비중 <b>{plan.weight:.1%}</b></span>"
                f"<span>손익비 <b>1:{plan.reward_risk:g}</b></span></div>"
                f"<div class='why'>근거: {html.escape(' · '.join(idea.reasons[:3]))}"
                + (f"<br>유의: {html.escape(' · '.join(idea.cautions[:3]))}"
                   if idea.cautions else "")
                + "</div></div>")
    parts.append("</section>")

    parts.append(f"<footer>{html.escape(report.disclaimer)}</footer></div>")
    return "\n".join(parts)


def write(report: Report, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(report), encoding="utf-8")
    return target
