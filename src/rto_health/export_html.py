"""20대 현황을 자체 완결 HTML 한 장으로 내보낸다.

Streamlit 대시보드는 서버가 필요하지만, 이 결과물은 파일 하나로 끝나므로
휴대폰·메신저·메일 어디로든 그냥 보낼 수 있다. 정비 회의 자료나 일일 보고에 쓴다.

외부 리소스를 전혀 참조하지 않는다(CSS·SVG 전부 인라인). 한글 표시를 위해
시스템 폰트 스택을 쓴다 — CJK 웹폰트를 끼워 넣으면 파일이 수 MB 로 불어난다.

사용법:
    python -m rto_health.export_html -i data/sample/fleet.csv -o fleet.html
"""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .fleet import FleetAnalysis, analyze_fleet

GRADE_LABEL = {"A": "정상", "B": "초기 열화", "C": "진행", "D": "심각", "E": "위험"}
GRADE_ORDER = ["A", "B", "C", "D", "E"]

# 상세 카드로 펼쳐 보여줄 상위 설비 수
DETAIL_COUNT = 6
# 스파크라인 표본 수 (경로 길이를 억제해 파일 크기를 유지)
SPARK_POINTS = 72

CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  color-scheme: light;
  --ground:        #eef1f3;
  --surface:       #ffffff;
  --surface-sunk:  #e4e9ec;
  --ink:           #101619;
  --ink-2:         #3d4d56;
  --ink-3:         #6b7c85;
  --rule:          #d3dbdf;
  --rule-soft:     #e6ebee;
  --accent:        #1f6b8f;
  --grade-a:       #1a7f37;
  --grade-b:       #a16207;
  --grade-c:       #c2410c;
  --grade-d:       #b91c1c;
  --grade-e:       #7f1d1d;
  --wash:          rgba(31,107,143,0.06);
  --shadow:        0 1px 2px rgba(16,22,25,0.06), 0 1px 8px rgba(16,22,25,0.04);
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:       #0d1316;
    --surface:      #161f24;
    --surface-sunk: #1e292f;
    --ink:          #eef3f5;
    --ink-2:        #b3c3cb;
    --ink-3:        #7f9099;
    --rule:         #2a373e;
    --rule-soft:    #212c32;
    --accent:       #58a8cd;
    --grade-a:      #3fb765;
    --grade-b:      #d9a01f;
    --grade-c:      #f08a4b;
    --grade-d:      #ef6b6b;
    --grade-e:      #c93a3a;
    --wash:         rgba(88,168,205,0.09);
    --shadow:       0 1px 2px rgba(0,0,0,0.4);
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:       #0d1316;
  --surface:      #161f24;
  --surface-sunk: #1e292f;
  --ink:          #eef3f5;
  --ink-2:        #b3c3cb;
  --ink-3:        #7f9099;
  --rule:         #2a373e;
  --rule-soft:    #212c32;
  --accent:       #58a8cd;
  --grade-a:      #3fb765;
  --grade-b:      #d9a01f;
  --grade-c:      #f08a4b;
  --grade-d:      #ef6b6b;
  --grade-e:      #c93a3a;
  --wash:         rgba(88,168,205,0.09);
  --shadow:       0 1px 2px rgba(0,0,0,0.4);
}

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR",
               "Malgun Gothic", sans-serif;
  font-size: 15px;
  line-height: 1.6;
  -webkit-text-size-adjust: 100%;
}

.wrap { max-width: 1120px; margin: 0 auto; padding: 20px 16px 72px; }

.mono {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
  letter-spacing: -0.01em;
}
.num { font-variant-numeric: tabular-nums; }

/* --- 머리말 ------------------------------------------------------------ */
.masthead { border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 22px; }
.eyebrow {
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--accent); font-weight: 700; margin: 0 0 6px;
}
h1 { font-size: clamp(22px, 5vw, 30px); line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.02em; text-wrap: balance; }
.meta { color: var(--ink-3); font-size: 13px; margin: 0; }

h2 {
  font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 700;
  margin: 34px 0 12px; padding-bottom: 7px; border-bottom: 1px solid var(--rule);
}
h2:first-of-type { margin-top: 26px; }

/* --- 요약 --------------------------------------------------------------- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 10px; }
.kpi {
  background: var(--surface); border: 1px solid var(--rule-soft);
  border-radius: 8px; padding: 12px 14px; box-shadow: var(--shadow);
}
.kpi .k { font-size: 12px; color: var(--ink-3); margin-bottom: 3px; }
.kpi .v { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; }
.kpi .s { font-size: 12px; color: var(--ink-3); }

.tally { display: flex; height: 30px; border-radius: 6px; overflow: hidden; margin-top: 12px; border: 1px solid var(--rule-soft); }
.tally span {
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; color: #fff; min-width: 26px;
}

/* --- 계열 그리드 -------------------------------------------------------- */
.series { margin-bottom: 14px; }
.series-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 7px; flex-wrap: wrap; }
.series-id { font-weight: 700; font-size: 15px; letter-spacing: -0.01em; }
.series-sub { font-size: 12px; color: var(--ink-3); }
.units { display: grid; grid-template-columns: repeat(auto-fill, minmax(94px, 1fr)); gap: 8px; }

.unit {
  position: relative; background: var(--surface); border: 1px solid var(--rule-soft);
  border-radius: 7px; padding: 9px 10px 9px 15px; box-shadow: var(--shadow); overflow: hidden;
}
.unit::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: var(--g);
}
/* E등급은 색 외에 빗금으로도 구분된다 — 색만으로 의미를 전달하지 않는다 */
.unit.crit::before {
  background: repeating-linear-gradient(135deg, var(--g) 0 3px, var(--surface) 3px 6px);
}
.unit .u { font-size: 11px; color: var(--ink-3); }
.unit .sc { font-size: 21px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; }
.unit .gd { font-size: 11px; font-weight: 700; color: var(--g); }
.unit .dl { font-size: 11px; color: var(--ink-3); }

/* --- 상세 카드 ---------------------------------------------------------- */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.card {
  background: var(--surface); border: 1px solid var(--rule-soft); border-left: 5px solid var(--g);
  border-radius: 8px; padding: 14px 16px; box-shadow: var(--shadow);
}
.card-top { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 2px; }
.card-id { font-weight: 700; font-size: 15px; }
.card-score { font-size: 24px; font-weight: 700; color: var(--g); letter-spacing: -0.02em; }
.card-sub { font-size: 12px; color: var(--ink-3); margin-bottom: 10px; }
.verdict { font-size: 14px; font-weight: 600; line-height: 1.45; margin: 10px 0 8px; }
.facts { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.fact {
  background: var(--surface-sunk); border-radius: 5px; padding: 3px 8px;
  font-size: 11.5px; color: var(--ink-2);
}
.fact b { color: var(--ink); font-weight: 700; }
.actions { margin: 0; padding-left: 17px; font-size: 13px; color: var(--ink-2); line-height: 1.55; }
.actions li { margin-bottom: 3px; }
.actions li::marker { color: var(--accent); }

.spark { display: block; width: 100%; height: 46px; margin: 4px 0 10px; }

/* --- 표 ----------------------------------------------------------------- */
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch;
          border: 1px solid var(--rule-soft); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; min-width: 560px; font-size: 13px; }
th, td { padding: 8px 11px; text-align: left; border-bottom: 1px solid var(--rule-soft); white-space: nowrap; }
th { font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3);
     font-weight: 700; background: var(--surface-sunk); position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: none; }
td.r, th.r { text-align: right; }
.pill {
  display: inline-block; min-width: 20px; text-align: center; padding: 1px 7px;
  border-radius: 4px; font-size: 11px; font-weight: 700; color: #fff; background: var(--g);
}
.wide { white-space: normal; min-width: 190px; }

/* --- 주석 --------------------------------------------------------------- */
.note {
  background: var(--wash); border-left: 3px solid var(--accent);
  border-radius: 0 7px 7px 0; padding: 12px 15px; font-size: 13.5px;
  color: var(--ink-2); line-height: 1.6; margin-bottom: 10px;
}
.note b, .note strong { color: var(--ink); }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--rule);
         font-size: 12px; color: var(--ink-3); line-height: 1.6; }

@media (max-width: 560px) {
  .wrap { padding: 16px 12px 56px; }
  .units { grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 6px; }
  .unit { padding: 8px 8px 8px 13px; }
  .unit .sc { font-size: 19px; }
}
"""


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _sparkline(series: pd.Series, color_var: str = "var(--g)") -> str:
    """정규화 차압 추이를 인라인 SVG 스파크라인으로 그린다.

    면적 채움 + 기준선(1.0) + 끝점 강조. 숫자 하나보다 '어떻게 여기까지 왔는가'를 보여준다.
    """
    s = series.dropna()
    if len(s) < 4:
        return ""

    if len(s) > SPARK_POINTS:
        s = s.iloc[:: max(1, len(s) // SPARK_POINTS)]
    y = s.to_numpy(dtype=float)

    w, h, pad = 300.0, 46.0, 4.0
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    lo = min(lo, 1.0)
    span = max(hi - lo, 1e-6)

    xs = np.linspace(0, w, len(y))
    ys = h - pad - (y - lo) / span * (h - 2 * pad)

    line = " ".join(f"{x:.1f},{v:.1f}" for x, v in zip(xs, ys))
    area = f"M0,{h:.1f} L" + line.replace(" ", " L") + f" L{w:.1f},{h:.1f} Z"
    base_y = h - pad - (1.0 - lo) / span * (h - 2 * pad)

    return (
        f'<svg class="spark" viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none" '
        f'role="img" aria-label="정규화 차압 추이"><path d="{area}" fill="{color_var}" '
        f'opacity="0.14"/><line x1="0" y1="{base_y:.1f}" x2="{w:.0f}" y2="{base_y:.1f}" '
        f'stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity="0.28"/>'
        f'<polyline points="{line}" fill="none" stroke="{color_var}" stroke-width="2" '
        f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3.2" fill="{color_var}"/></svg>'
    )


def _fmt(value, digits: int = 1, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return f"{value:,.{digits}f}"


def build_html(fleet: FleetAnalysis, *, title: str = "RTO 축열재 현황판") -> str:
    ranking = fleet.ranking()
    if ranking.empty:
        raise ValueError("분석 결과가 비어 있습니다.")

    info = ranking.set_index("설비").to_dict("index")
    counts = fleet.grade_counts()
    total = len(fleet.units)
    urgent = sum(counts.get(g, 0) for g in ("D", "E"))
    soon = int((ranking["잔여일"].dropna() <= 30).sum()) if "잔여일" in ranking else 0

    any_analysis = next(iter(fleet.units.values()))
    daily = any_analysis.daily
    period = f"{daily.index[0]:%Y-%m-%d} ~ {daily.index[-1]:%Y-%m-%d}"

    out: list[str] = []
    add = out.append

    add(f"<title>{_e(title)}</title>")
    add(f"<style>{CSS}</style>")
    add('<div class="wrap">')

    # --- 머리말 ---------------------------------------------------------
    add('<header class="masthead">')
    add('<p class="eyebrow">Can Type RTO · 축열재 막힘 사전예측</p>')
    add(f"<h1>{_e(title)}</h1>")
    add(f'<p class="meta">설비 {total}대 · 데이터 {_e(period)} · '
        f'생성 {datetime.now():%Y-%m-%d %H:%M}</p>')
    add("</header>")

    # --- 요약 -----------------------------------------------------------
    add("<h2>요약</h2>")
    add('<div class="kpis">')
    add(f'<div class="kpi"><div class="k">분석 설비</div><div class="v num">{total}</div>'
        f'<div class="s">{"전량 정상 분석" if not fleet.failed else f"실패 {len(fleet.failed)}대"}</div></div>')
    add(f'<div class="kpi"><div class="k">긴급·심각 (D·E)</div><div class="v num">{urgent}</div>'
        f'<div class="s">즉시 조치 검토</div></div>')
    add(f'<div class="kpi"><div class="k">30일 내 한계 도달</div><div class="v num">{soon}</div>'
        f'<div class="s">차압 추세 외삽</div></div>')

    worst = ranking.iloc[0]
    add(f'<div class="kpi" style="--g:var(--grade-{worst["등급"].lower()})">'
        f'<div class="k">최우선 정비 대상</div>'
        f'<div class="v mono" style="font-size:17px">{_e(worst["설비"])}</div>'
        f'<div class="s">{_fmt(worst["점수"])}점 · {_e(worst["등급"])}등급</div></div>')
    add("</div>")

    # 등급 분포를 폭으로 — 개수를 읽기 전에 비율이 먼저 보인다
    add('<div class="tally">')
    for grade in GRADE_ORDER:
        n = counts.get(grade, 0)
        if n == 0:
            continue
        pct = n / total * 100
        add(f'<span style="background:var(--grade-{grade.lower()});flex:{n}" '
            f'title="{grade}등급 {n}대">{grade} {n}</span>')
    add("</div>")
    add(f'<p class="meta" style="margin-top:6px">A 정상 · B 초기 열화 · C 진행 · D 심각 · E 위험</p>')

    # --- 계열별 현황 -----------------------------------------------------
    add("<h2>계열별 현황</h2>")
    for series_id in fleet.series_ids:
        units = sorted(fleet.series_units(series_id), key=lambda e: fleet.specs[e].unit)
        if not units:
            continue
        label = fleet.specs[units[0]].label

        add('<div class="series">')
        add('<div class="series-head">'
            f'<span class="series-id mono">{_e(series_id)}</span>'
            f'<span class="series-sub">{_e(label)} · {len(units)}대</span></div>')
        add('<div class="units">')
        for eq_id in units:
            spec = fleet.specs[eq_id]
            row = info.get(eq_id)
            if row is None:
                add(f'<div class="unit" style="--g:var(--ink-3)"><div class="u">{_e(spec.unit)} 호기</div>'
                    f'<div class="sc" style="font-size:14px">분석 불가</div></div>')
                continue
            grade = row["등급"]
            delta = row.get("30일 변화")
            delta_txt = "—" if delta is None or pd.isna(delta) else f"{delta:+.1f}"
            crit = " crit" if grade == "E" else ""
            add(f'<div class="unit{crit}" style="--g:var(--grade-{grade.lower()})">'
                f'<div class="u">{_e(spec.unit)} 호기</div>'
                f'<div class="sc num">{_fmt(row["점수"])}</div>'
                f'<div class="gd">{_e(grade)}등급</div>'
                f'<div class="dl num">30일 {delta_txt}</div></div>')
        add("</div></div>")

    # --- 상위 설비 상세 --------------------------------------------------
    add("<h2>정비 우선순위 — 상위 {}대</h2>".format(min(DETAIL_COUNT, len(ranking))))
    add('<div class="cards">')
    for _, row in ranking.head(DETAIL_COUNT).iterrows():
        eq_id = row["설비"]
        analysis = fleet.units[eq_id]
        grade = row["등급"]
        guide = analysis.guidance()
        primary = guide["primary"]

        add(f'<article class="card" style="--g:var(--grade-{grade.lower()})">')
        add('<div class="card-top">'
            f'<span class="card-id mono">{_e(eq_id)}</span>'
            f'<span class="card-score num">{_fmt(row["점수"])}</span></div>')
        add(f'<div class="card-sub">{_e(row["종류"])} · {_e(grade)}등급 '
            f'{_e(GRADE_LABEL.get(grade, ""))}</div>')

        if "A1" in analysis.indicators.raw.columns:
            add(_sparkline(analysis.indicators.raw["A1"]))

        facts = []
        a1 = analysis.indicators.raw["A1"].iloc[-1] if "A1" in analysis.indicators.raw.columns else None
        if a1 is not None and np.isfinite(a1):
            facts.append(f"정규화 차압 <b>{a1:.2f}배</b>")
        peer = row.get("동급 대비")
        if peer is not None and not pd.isna(peer):
            facts.append(f"동급 대비 <b>{peer:.2f}배</b> ({int(row.get('비교대수') or 0)}대)")
        rul = row.get("잔여일")
        if rul is not None and not pd.isna(rul):
            facts.append(f"한계까지 <b>{int(rul)}일</b>")
        delta = row.get("30일 변화")
        if delta is not None and not pd.isna(delta):
            facts.append(f"30일 <b>{delta:+.1f}점</b>")
        if facts:
            add('<div class="facts">' + "".join(f'<span class="fact">{f}</span>' for f in facts) + "</div>")

        add(f'<p class="verdict">{_e(guide["headline"])}</p>')
        if primary is not None and primary.actions:
            add('<ul class="actions">')
            for action in primary.actions[:4]:
                add(f"<li>{_e(action)}</li>")
            add("</ul>")
        add("</article>")
    add("</div>")

    # --- 전체 표 --------------------------------------------------------
    add("<h2>전체 설비</h2>")
    add('<div class="scroll"><table><thead><tr>'
        "<th>설비</th><th>종류</th><th class='r'>점수</th><th>등급</th>"
        "<th class='r'>30일</th><th class='r'>동급 대비</th><th class='r'>잔여일</th>"
        "<th class='wide'>판정</th><th>베이스라인</th>"
        "</tr></thead><tbody>")
    for _, row in ranking.iterrows():
        grade = row["등급"]
        delta = row.get("30일 변화")
        peer = row.get("동급 대비")
        rul = row.get("잔여일")
        add("<tr>"
            f'<td class="mono">{_e(row["설비"])}</td>'
            f'<td>{_e(row["종류"])}</td>'
            f'<td class="r num"><b>{_fmt(row["점수"])}</b></td>'
            f'<td><span class="pill" style="--g:var(--grade-{grade.lower()})">{_e(grade)}</span></td>'
            f'<td class="r num">{"—" if delta is None or pd.isna(delta) else f"{delta:+.1f}"}</td>'
            f'<td class="r num">{"—" if peer is None or pd.isna(peer) else f"{peer:.2f}"}</td>'
            f'<td class="r num">{"—" if rul is None or pd.isna(rul) else f"{int(rul)}"}</td>'
            f'<td class="wide">{_e(row["판정"])}</td>'
            f'<td>{_e(row["베이스라인"])}</td>'
            "</tr>")
    add("</tbody></table></div>")

    # --- 주석 -----------------------------------------------------------
    add("<h2>읽기 전 확인</h2>")

    assumed = [e for e, a in fleet.units.items() if a.baseline.is_assumed]
    if assumed:
        add(f'<div class="note"><b>베이스라인 자동 추정 {len(assumed)}대</b> — 정비 이력도 지정 구간도 '
            "없어 데이터 앞부분을 정상으로 가정했습니다. 이미 오염된 상태에서 수집을 시작했다면 "
            "막힘이 <b>과소평가</b>됩니다. <code>config/fleet.yaml</code> 의 <code>baselines</code> 에 "
            "정상 구간이나 설계값을 지정하십시오. 그 전까지는 <b>동급 대비</b> 값이 더 믿을 만합니다.</div>")

    if fleet.failed:
        add('<div class="note"><b>분석 실패 설비</b> — '
            + _e(", ".join(fleet.failed)) + " · 계측 데이터를 확인하십시오.</div>")

    add('<div class="note"><b>동급 대비</b>는 같은 계열 병렬 호기의 정규화 차압 중앙값과 비교한 값입니다. '
        "1.0 이면 형제 호기와 같은 수준입니다. 이 값이 높으면 <b>그 호기</b>를 정비하고, "
        "점수는 낮은데 이 값이 1.0 근처면 계열 전체가 함께 나빠진 것이므로 설비가 아니라 "
        "<b>유입측</b>을 조사해야 합니다.</div>")

    add('<div class="note">배점 가중치는 RTO 물리·정비 문헌에 근거한 <b>공학적 추정치</b>이며 '
        "통계 검증값이 아닙니다. 설비 사양(풍량·설계차압)도 현재 자리표시자이므로 실제 값으로 "
        "교체해야 잔여일 예측이 맞습니다.</div>")

    add("<footer>Can Type (Rotary 1-Can) RTO 축열재 막힘 사전예측 시스템 · "
        "정규화 차압 · 열회수효율 · 연료 원단위 · 동급기 비교를 100점으로 가중 합산합니다.<br>"
        "이 문서는 생성 시점의 스냅샷입니다. 대화형 분석은 Streamlit 대시보드를 사용하십시오.</footer>")
    add("</div>")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="20대 현황을 자체 완결 HTML 로 내보내기")
    parser.add_argument("--input", "-i", required=True, help="20대 통합 데이터 (파일 또는 폴더)")
    parser.add_argument("--output", "-o", default="fleet_report.html")
    parser.add_argument("--title", default="RTO 축열재 현황판")
    parser.add_argument("--config-dir", help="config 디렉토리 경로")
    args = parser.parse_args()

    fleet = analyze_fleet(args.input, config_dir=args.config_dir)
    page = build_html(fleet, title=args.title)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"생성 완료: {out} ({out.stat().st_size / 1024:.0f} KB · 설비 {len(fleet.units)}대)")


if __name__ == "__main__":
    main()
