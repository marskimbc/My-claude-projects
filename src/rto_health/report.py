"""한글 점검 리포트 생성 (Markdown / CLI).

현장 담당자와 정비 계획 회의에 그대로 올릴 수 있는 형태를 목표로 한다.
숫자만 나열하지 않고 항상 "그래서 무엇을 점검하라"까지 적는다.

사용법:
    python -m rto_health.report --input data/sample/rapid_plugging.csv
    python -m rto_health.report --input data/sample/gradual_fouling.csv -o report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import Analysis, analyze
from .scoring import score_table

GRADE_MARK = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "E": "⛔"}


def _md_table(df: pd.DataFrame) -> str:
    """DataFrame 을 Markdown 표로 변환한다 (tabulate 의존 없이)."""
    def cell(v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "–"
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def _fmt_rul(analysis: Analysis) -> str:
    rul = analysis.rul
    if rul.days_remaining is None:
        return f"산출 불가 — {rul.note or '상승 추세 미확인'}"
    if rul.days_remaining <= 0:
        return "**이미 운전 한계 도달**"
    if rul.days_remaining > 1095:
        return f"3년 이상 (현재 추세 유지 시). 현재 {rul.current_value:.2f}배 / 한계 {rul.limit_value:.2f}배"

    text = f"**약 {rul.days_remaining:,.0f}일 후** ({rul.predicted_date:%Y-%m-%d} 경)"
    if rul.ci_days:
        lo, hi = rul.ci_days
        if np.isfinite(lo) and np.isfinite(hi):
            text += f" · 범위 {lo:,.0f}~{hi:,.0f}일"
    text += f"\n  - 현재 정규화 차압 {rul.current_value:.2f}배 → 운전 한계 {rul.limit_value:.2f}배"
    text += f"\n  - 상승 속도 {rul.slope_per_day * 100:.3f} %p/일"
    return text


def build_markdown(analysis: Analysis, date=None) -> str:
    """분석 결과를 Markdown 점검 리포트로 만든다."""
    snap = analysis.snapshot(date)
    guide = analysis.guidance(date)
    equip = analysis.config.equipment
    ind = analysis.indicators

    lines: list[str] = []
    add = lines.append

    add(f"# RTO 축열재 막힘 점검 리포트")
    add("")
    add(f"- **설비**: {equip.get('name', 'RTO')} ({equip.get('type', '')})")
    add(f"- **분석 기준일**: {snap.date:%Y-%m-%d}")
    add(f"- **데이터 구간**: {analysis.daily.index[0]:%Y-%m-%d} ~ {analysis.daily.index[-1]:%Y-%m-%d} "
        f"({len(analysis.daily):,}일 · 정상운전 {analysis.daily['runtime_hours'].sum():,.0f}시간)")
    add("")

    # --- 1. 종합 판정 -------------------------------------------------------
    mark = GRADE_MARK.get(snap.grade, "")
    add("## 1. 종합 판정")
    add("")
    add(f"### {mark} 건전도 **{snap.score:.1f}점** / 100점 — {snap.grade}등급 ({snap.label})")
    add("")
    add(f"> **조치 방향**: {snap.action}")
    add("")
    add(f"- **판정 고장모드**: {guide['headline']}")
    add(f"- **운전 한계 도달 예상**: {_fmt_rul(analysis)}")
    add("")

    # 최근 추세
    score = analysis.score
    for label, days in (("30일", 30), ("90일", 90)):
        past = score.loc[score.index <= score.index[-1] - pd.Timedelta(days=days)]
        if not past.empty:
            delta = snap.score - float(past.iloc[-1])
            arrow = "▼" if delta < 0 else "▲"
            add(f"- **{label} 전 대비**: {arrow} {abs(delta):.1f}점 ({past.iloc[-1]:.1f} → {snap.score:.1f})")
    add("")

    # --- 2. 감점 분해 -------------------------------------------------------
    add("## 2. 감점 분해 — 어디서 점수가 깎였는가")
    add("")
    add("| 그룹 | 배점 | 감점 | 잔여 |")
    add("|---|---:|---:|---:|")
    groups = analysis.config.weights.get("groups") or {}
    for gid in sorted(groups):
        allotted = sum(ind.effective_points.get(i, 0.0) for i in ind.active_ids if ind.specs[i].group == gid)
        lost = snap.group_deductions.get(gid, 0.0)
        if allotted <= 0:
            continue
        add(f"| {gid}. {groups[gid].get('name', gid)} | {allotted:.1f} | **{lost:.1f}** | {allotted - lost:.1f} |")
    add(f"| **합계** | **100.0** | **{100 - snap.score:.1f}** | **{snap.score:.1f}** |")
    add("")

    add("### 감점 상위 지표")
    add("")
    if guide["top_contributors"]:
        for item in guide["top_contributors"]:
            add(f"- {item}")
    else:
        add("- 유의미한 감점 지표 없음")
    add("")

    # --- 3. 지표 상세 -------------------------------------------------------
    add("## 3. 지표 상세")
    add("")
    add(_md_table(score_table(snap, ind)))
    add("")

    # --- 4. 점검 가이드 -----------------------------------------------------
    add("## 4. 점검 가이드")
    add("")
    if guide["all_modes"]:
        for i, mode in enumerate(guide["all_modes"], start=1):
            add(f"### 4.{i} {mode.name}  (확신도 {mode.confidence:.0%})")
            add("")
            add(f"**추정 원인**: {mode.cause}")
            add("")
            add("**판정 근거**")
            for ev in mode.evidence:
                add(f"- {ev}")
            add("")
            add("**권고 조치**")
            for act in mode.actions:
                add(f"- [ ] {act}")
            add("")
    else:
        add(f"판정된 특이 고장모드가 없습니다. 등급별 기본 조치를 적용하세요: **{snap.action}**")
        add("")

    # --- 5. 변화점 ----------------------------------------------------------
    add("## 5. 변화 시점")
    add("")
    if analysis.change_points:
        add("CUSUM 이 검출한 열화 속도 변화 시점입니다. 해당 일자 전후의 공정 이력"
            "(원료 변경, 배치 이상, 전처리 필터 교체 등)을 확인하세요.")
        add("")
        add("| 일자 | 방향 | 통계량 |")
        add("|---|---|---:|")
        for p in analysis.change_points[:12]:
            direction = "악화 시작" if p.direction == "up" else "개선(정비 효과)"
            add(f"| {p.date:%Y-%m-%d} | {direction} | {p.statistic:.1f} |")
    else:
        add("검출된 변화 시점이 없습니다 — 열화 속도가 일정하게 유지되고 있습니다.")
    add("")

    # --- 6. 분석 조건 -------------------------------------------------------
    add("## 6. 분석 조건")
    add("")
    refs = analysis.refs
    base = analysis.baseline
    add(f"- **베이스라인 구간**: {base.start:%Y-%m-%d} ~ {base.end:%Y-%m-%d} "
        f"({base.n_samples:,} 샘플" + (f", 기준 정비: {base.reset_type}" if base.reset_type else "") + ")")
    add(f"- **차압 정규화**: ΔP_norm = ΔP × (Q_ref/Q)^n × (T_ref/T)")
    r2_txt = f", R²={refs.fit_r2:.4f}" if refs.fit_r2 is not None else ""
    source_txt = {"fitted": "실측 회귀 피팅", "config": "설정 고정값", "default": "형상별 기본값"}[refs.exponent_source]
    add(f"  - 유량 지수 n = **{refs.flow_exponent:.3f}** ({source_txt}{r2_txt})")
    add(f"  - 기준 풍량 {refs.ref_flow:,.0f} CMM · 기준 가스온도 {refs.ref_gas_temp_c:.1f} ℃")
    add(f"- **베이스라인 정규화 차압**: {base.ref('dp_norm'):.1f} mmH2O · **열회수효율**: {base.ref('ter') * 100:.2f} %")

    if ind.excluded:
        add("- **채점 제외 지표** (해당 배점은 잔여 지표로 재배분됨)")
        for ind_id, reason in ind.excluded.items():
            add(f"  - {ind_id} {ind.specs[ind_id].name}: {reason}")
    if analysis.dataset.missing_tags:
        add(f"- **미보유 태그**: {', '.join(analysis.dataset.missing_tags)}")
    add("")
    add("---")
    add("")
    add("> ⚠️ 배점 가중치는 RTO 물리·정비 문헌에 근거한 **공학적 추정치**이며 통계적으로 "
        "검증된 값이 아닙니다. 실제 정비 이력이 누적되면 `calibrate.py` 로 재보정하십시오.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RTO 축열재 막힘 점검 리포트 생성")
    parser.add_argument("--input", "-i", required=True, help="운전 데이터 CSV/Excel 경로")
    parser.add_argument("--output", "-o", help="저장 경로(.md). 생략 시 화면 출력")
    parser.add_argument("--date", "-d", help="기준일 YYYY-MM-DD. 생략 시 최신일")
    parser.add_argument("--config-dir", help="config 디렉토리 경로")
    args = parser.parse_args()

    analysis = analyze(args.input, config_dir=args.config_dir)
    text = build_markdown(analysis, args.date)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"리포트 저장 완료: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
