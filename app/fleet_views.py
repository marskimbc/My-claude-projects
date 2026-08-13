"""Fleet 전용 화면 구성 요소 — 계열별 그리드, 정비 우선순위, 계열 비교.

20대를 볼 때 가장 먼저 필요한 정보는 개별 설비 상세가 아니라
"어느 호기부터 손대야 하는가"다. 이 모듈은 그 질문에만 답한다.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import GRADE_COLOR, GRADE_ICON, apply_layout

# 계열 최대 호기 수 (A~E). 2호기 계열도 같은 폭으로 그려 열을 맞춘다.
GRID_COLUMNS = 5


def _tint(hex_color: str, alpha_hex: str = "20") -> str:
    return f"{hex_color}{alpha_hex}"


def render_kpi_row(fleet, pal) -> None:
    """상단 요약 — 등급 분포와 즉시 조치가 필요한 대수."""
    counts = fleet.grade_counts()
    ranking = fleet.ranking()

    total = len(fleet.units)
    urgent = sum(counts.get(g, 0) for g in ("D", "E"))
    soon = 0
    if not ranking.empty and "잔여일" in ranking:
        soon = int((ranking["잔여일"].dropna() <= 30).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 설비", f"{total} 대", f"실패 {len(fleet.failed)}대" if fleet.failed else "전량 정상 분석")
    c2.metric("긴급·심각 (D·E)", f"{urgent} 대")
    c3.metric("30일 내 한계 도달", f"{soon} 대")

    if not ranking.empty:
        worst = ranking.iloc[0]
        c4.markdown(
            f"<div style='font-size:13px;color:{pal.text_secondary}'>최우선 정비 대상</div>"
            f"<div style='font-size:20px;font-weight:700;color:{GRADE_COLOR.get(worst['등급'], pal.muted)}'>"
            f"{worst['설비']}</div>"
            f"<div style='font-size:13px;color:{pal.text_secondary}'>"
            f"{worst['점수']:.1f}점 · {worst['등급']}등급</div>",
            unsafe_allow_html=True,
        )

    # 등급 분포를 한 줄로
    chips = []
    for grade in "ABCDE":
        n = counts.get(grade, 0)
        if n == 0:
            continue
        chips.append(
            f"<span style='display:inline-block;margin-right:8px;padding:2px 10px;border-radius:12px;"
            f"background:{_tint(GRADE_COLOR[grade])};color:{pal.text_primary};font-size:13px'>"
            f"{GRADE_ICON[grade]} {grade}등급 <b>{n}</b>대</span>"
        )
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)


def render_series_grid(fleet, pal) -> None:
    """계열별 그리드 — 행=계열, 열=호기.

    칸마다 등급 색·호기 문자·점수를 함께 표기한다(색만으로 의미를 전달하지 않음).
    """
    ranking = fleet.ranking()
    info = ranking.set_index("설비").to_dict("index") if not ranking.empty else {}

    for series_id in fleet.series_ids:
        units = sorted(fleet.series_units(series_id), key=lambda e: fleet.specs[e].unit)
        label = fleet.specs[units[0]].label if units else ""

        cols = st.columns([1.25] + [1] * GRID_COLUMNS)
        cols[0].markdown(
            f"<div style='padding-top:14px'>"
            f"<div style='font-weight:600;font-size:14px;color:{pal.text_primary}'>{series_id}</div>"
            f"<div style='font-size:12px;color:{pal.text_secondary}'>{label} · {len(units)}대</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for slot in range(GRID_COLUMNS):
            if slot >= len(units):
                continue
            eq_id = units[slot]
            spec = fleet.specs[eq_id]

            if eq_id in fleet.failed:
                cols[slot + 1].markdown(
                    f"<div style='border-left:4px solid {pal.muted};background:{pal.grid};"
                    f"border-radius:6px;padding:8px 10px;margin:4px 0'>"
                    f"<div style='font-size:12px;color:{pal.text_secondary}'>{spec.unit} 호기</div>"
                    f"<div style='font-size:15px;font-weight:600;color:{pal.text_secondary}'>분석 불가</div>"
                    f"<div style='font-size:11px;color:{pal.muted}'>데이터 확인 필요</div></div>",
                    unsafe_allow_html=True,
                )
                continue

            row = info.get(eq_id, {})
            grade = row.get("등급", "?")
            color = GRADE_COLOR.get(grade, pal.muted)
            delta = row.get("30일 변화")
            delta_txt = "—" if delta is None or pd.isna(delta) else f"{delta:+.1f}"

            cols[slot + 1].markdown(
                f"<div style='border-left:4px solid {color};background:{_tint(color)};"
                f"border-radius:6px;padding:8px 10px;margin:4px 0'>"
                f"<div style='font-size:12px;color:{pal.text_secondary}'>{spec.unit} 호기</div>"
                f"<div style='font-size:22px;font-weight:700;color:{pal.text_primary};line-height:1.2'>"
                f"{row.get('점수', float('nan')):.1f}</div>"
                f"<div style='font-size:11px;color:{pal.text_secondary}'>"
                f"{GRADE_ICON.get(grade, '')} {grade}등급 · 30일 {delta_txt}</div></div>",
                unsafe_allow_html=True,
            )


def render_priority_table(fleet) -> None:
    """정비 우선순위 — 위험도 낮은 점수 순."""
    ranking = fleet.ranking()
    if ranking.empty:
        st.info("표시할 분석 결과가 없습니다.")
        return

    display = ranking.copy()
    display["등급"] = display["등급"].map(lambda g: f"{GRADE_ICON.get(g, '')} {g}")
    st.dataframe(
        display[["설비", "계열", "종류", "점수", "등급", "30일 변화", "잔여일",
                 "동급 대비", "비교대수", "판정", "베이스라인"]],
        hide_index=True,
        width="stretch",
        column_config={
            "점수": st.column_config.ProgressColumn(
                "건전도", min_value=0, max_value=100, format="%.1f"
            ),
            "잔여일": st.column_config.NumberColumn("한계까지", format="%d 일"),
            "동급 대비": st.column_config.NumberColumn(
                "동급 대비", format="%.2f배", help="동일 계열 정규화 차압 중앙값 대비. 1.0 이면 형제 호기와 같은 수준."
            ),
            "비교대수": st.column_config.NumberColumn("비교대수", format="%d 대"),
        },
    )


def chart_series_overlay(fleet, series_id: str, pal):
    """계열 내 호기별 정규화 차압 추이 겹쳐보기 — 단독 이상 호기를 찾는 화면."""
    units = sorted(fleet.series_units(series_id), key=lambda e: fleet.specs[e].unit)
    fig = go.Figure()

    for i, eq_id in enumerate(units):
        analysis = fleet.units.get(eq_id)
        if analysis is None or "A1" not in analysis.indicators.raw.columns:
            continue
        series = analysis.indicators.raw["A1"].dropna()
        fig.add_trace(go.Scatter(
            x=series.index, y=series, mode="lines",
            name=f"{fleet.specs[eq_id].unit} 호기",
            line=dict(color=pal.series[i % len(pal.series)], width=2),
            hovertemplate=f"{eq_id}<br>%{{x|%Y-%m-%d}}<br><b>%{{y:.2f}}배</b><extra></extra>",
        ))

    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color=pal.muted,
                  annotation_text="베이스라인", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=pal.muted)

    return apply_layout(fig, pal, height=360, show_legend=True).update_layout(
        title=f"{series_id} 호기별 정규화 베드 차압 (베이스라인 대비 배수)"
    )


def chart_peer_deviation(fleet, series_id: str, pal):
    """호기별 동급기 대비 편차 — 어느 호기가 튀는지 한눈에."""
    units = sorted(fleet.series_units(series_id), key=lambda e: fleet.specs[e].unit)
    labels, values, colors = [], [], []

    for eq_id in units:
        analysis = fleet.units.get(eq_id)
        if analysis is None or "A6" not in analysis.indicators.raw.columns:
            continue
        raw = analysis.indicators.raw["A6"].dropna()
        if raw.empty:
            continue
        value = float(raw.iloc[-1])
        labels.append(f"{fleet.specs[eq_id].unit} 호기")
        values.append(value)
        deg = analysis.snapshot().degradations.get("A6", 0.0)
        colors.append(GRADE_COLOR["D"] if deg > 0.35 else (GRADE_COLOR["B"] if deg > 0 else pal.series[2]))

    if not labels:
        return None

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color=pal.surface, width=2)),
        text=[f"{v:.2f}" for v in values], textposition="outside",
        textfont=dict(color=pal.text_secondary, size=12),
        hovertemplate="%{x}<br>동급기 대비 %{y:.2f}배<extra></extra>",
    ))
    fig.add_hline(y=1.10, line_width=1, line_dash="dash", line_color=GRADE_COLOR["B"],
                  annotation_text="정상 한계", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=GRADE_COLOR["B"])
    fig.add_hline(y=1.60, line_width=1, line_dash="dash", line_color=GRADE_COLOR["D"],
                  annotation_text="심각", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=GRADE_COLOR["D"])

    return apply_layout(fig, pal, height=300).update_layout(
        title="동급기 대비 차압 편차 (현재)", hovermode="closest", bargap=0.45
    )
