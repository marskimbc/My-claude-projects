"""터미널 리포트.

한글은 폭이 2인 문자라서 len() 으로 정렬하면 표가 어긋난다.
unicodedata.east_asian_width 로 실제 표시 폭을 계산해 맞춘다.
"""

from __future__ import annotations

import math
import unicodedata

import pandas as pd

from ..analysis.regime import Regime
from ..strategy import signals
from ..strategy.recommend import Report

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

ACTION_COLOR = {
    signals.BUY: GREEN,
    signals.WATCH: CYAN,
    signals.HOLD: "",
    signals.REDUCE: YELLOW,
    signals.SELL: RED,
}


def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text: str, size: int, align: str = "left") -> str:
    text = str(text)
    gap = max(0, size - width(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        if abs(value) >= 10000:
            return f"{value:,.0f}"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return f"{DIM}(데이터 없음){RESET}"

    view = frame.head(max_rows)
    columns = [str(c) for c in view.columns]
    cells = [[_cell(v) for v in row] for row in view.itertuples(index=False)]
    widths = [
        max(width(col), *(width(row[i]) for row in cells)) if cells else width(col)
        for i, col in enumerate(columns)
    ]

    numeric = [pd.api.types.is_numeric_dtype(view[c]) for c in view.columns]
    lines = [
        "  ".join(pad(col, widths[i], "right" if numeric[i] else "left")
                  for i, col in enumerate(columns)),
        "─" * (sum(widths) + 2 * (len(widths) - 1)),
    ]
    for row in cells:
        lines.append("  ".join(
            pad(value, widths[i], "right" if numeric[i] else "left")
            for i, value in enumerate(row)))
    if len(frame) > max_rows:
        lines.append(f"{DIM}… 외 {len(frame) - max_rows}건{RESET}")
    return "\n".join(lines)


def heading(text: str) -> str:
    return f"\n{BOLD}{text}{RESET}\n{'═' * width(text)}"


def render_regime(regime: Regime) -> str:
    color = {"risk_on": GREEN, "neutral": YELLOW, "risk_off": RED}.get(regime.label, "")
    out = [heading("① 시장 국면")]
    out.append(f"판정: {color}{BOLD}{regime.label_ko}{RESET} "
               f"(종합 {regime.score:.1f}점 / 권장 주식노출 {regime.exposure:.0%})")
    for driver in regime.drivers[:8]:
        out.append(f"  · {driver}")
    return "\n".join(out)


def render_indices(report: Report) -> str:
    rows = []
    for label, state in report.index_states.items():
        rows.append({
            "지수": label,
            "추세": state.label,
            "점수": round(state.score, 1),
            "종가": state.metrics.get("close"),
            "1개월%": state.metrics.get("ret_20d"),
            "3개월%": state.metrics.get("ret_60d"),
            "6개월%": state.metrics.get("ret_120d"),
            "RSI": state.metrics.get("rsi"),
            "52주고점대비%": state.metrics.get("from_52w_high"),
        })
    return heading("② 지수 동향") + "\n" + table(pd.DataFrame(rows))


def render_macro(report: Report) -> str:
    return heading("③ 매크로 · 원자재") + "\n" + table(report.macro_table)


def render_themes(report: Report) -> str:
    out = [heading("④ 테마 상대강도 (코스피 대비 초과수익)")]
    out.append(table(report.theme_table))
    out.append(f"{DIM}국면: 주도=단기·중기 모두 우위, 개선=단기 반등, "
               f"둔화=단기 약화, 소외=둘 다 열위{RESET}")
    return "\n".join(out)


def render_correlation(report: Report) -> str:
    out = [heading("⑤ 테마 ↔ 원자재/대체자산 상관 (최근 120일 일간수익률)")]
    if report.correlation.empty:
        out.append(f"{DIM}(상관 계산에 필요한 데이터가 부족합니다){RESET}")
        return "\n".join(out)
    out.append(table(report.correlation.reset_index().rename(columns={"index": "테마"})))
    if report.correlation_notes:
        out.append("")
        out += [f"  · {note}" for note in report.correlation_notes[:6]]
    return "\n".join(out)


def render_ideas(report: Report) -> str:
    out = [heading("⑥ 매매 추천")]
    if not report.ideas:
        out.append(f"{DIM}조건을 만족하는 추천 종목이 없습니다. "
                   f"관망하거나 임계값(settings.yaml)을 조정하세요.{RESET}")
        return "\n".join(out)

    out.append(f"기준 자본 {report.capital:,.0f}원 · 국면 {report.regime.label_ko} "
               f"· 노출한도 {report.regime.exposure:.0%}")
    out.append(table(report.ideas_table))
    out.append("")
    for idea in report.ideas[:6]:
        color = ACTION_COLOR.get(idea.action, "")
        plan = idea.plan
        out.append(f"{color}[{idea.action}]{RESET} {BOLD}{idea.name}{RESET}"
                   f"({idea.ticker}) · {idea.theme_label} · {idea.score:.1f}점")
        out.append(f"    진입 {plan.entry:,.0f} / 손절 {plan.stop:,.0f}"
                   f"({plan.stop_pct:+.1f}%) / 목표 {plan.target:,.0f}"
                   f"({plan.target_pct:+.1f}%) · {plan.shares:,}주"
                   f" · 비중 {plan.weight:.1%} · 손익비 1:{plan.reward_risk:g}")
        if idea.reasons:
            out.append(f"    근거: {' · '.join(idea.reasons[:3])}")
        if idea.cautions:
            out.append(f"    {YELLOW}유의: {' · '.join(idea.cautions[:3])}{RESET}")
    return "\n".join(out)


def render(report: Report) -> str:
    parts = [
        f"{BOLD}국내주식 지수·테마 분석 리포트{RESET}  "
        f"{DIM}{report.generated_at:%Y-%m-%d %H:%M}{RESET}",
    ]
    if report.offline:
        parts.append(f"{YELLOW}※ 오프라인(합성) 데이터로 생성된 리포트입니다. "
                     f"실제 시세가 아니므로 매매 판단에 사용하지 마세요.{RESET}")
    parts += [
        render_regime(report.regime),
        render_indices(report),
        render_macro(report),
        render_themes(report),
        render_correlation(report),
        render_ideas(report),
        f"\n{DIM}{report.disclaimer}{RESET}",
    ]
    return "\n".join(parts)
