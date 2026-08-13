"""Can Type RTO 축열재 막힘 사전예측 대시보드.

실행:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from theme import DARK, GRADE_COLOR, GRADE_ICON, LIGHT, apply_layout, group_color  # noqa: E402

from rto_health import indicators as ind_mod  # noqa: E402
from rto_health import scoring  # noqa: E402
from rto_health.io_loader import load_config  # noqa: E402
from rto_health.pipeline import analyze  # noqa: E402
from rto_health.report import build_markdown  # noqa: E402
from rto_health.trend import ewma  # noqa: E402

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_LABELS = {
    "normal": "정상 운전",
    "gradual_fouling": "완만한 막힘 진행",
    "rapid_plugging": "급성 막힘 (VOC 스파이크 후)",
}

st.set_page_config(page_title="RTO 축열재 막힘 예측", page_icon="🏭", layout="wide")


def palette():
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = "light"
    return DARK if base == "dark" else LIGHT


# ---------------------------------------------------------------------------
# 데이터 로딩 · 분석
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="운전 데이터 분석 중…", ttl=3600)
def _analyze_bytes(payload: bytes, name: str):
    import io

    reader = pd.read_excel if name.lower().endswith((".xlsx", ".xls", ".xlsm")) else pd.read_csv
    return analyze(reader(io.BytesIO(payload)), config_dir=PROJECT_ROOT / "config")


@st.cache_data(show_spinner="운전 데이터 분석 중…", ttl=3600)
def _analyze_path(path: str):
    return analyze(path, config_dir=PROJECT_ROOT / "config")


def _ensure_samples() -> bool:
    return any(SAMPLE_DIR.glob("*.csv"))


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------
def chart_score_trend(analysis, pal, grades):
    """건전도 점수 추이 + 등급 밴드."""
    score = analysis.score
    smooth = ewma(score, 7)

    fig = go.Figure()
    # 등급 밴드를 배경으로 깔아 현재 위치를 한눈에 알 수 있게 한다
    bounds = sorted(((g["grade"], float(g["min_score"])) for g in grades), key=lambda x: x[1])
    for i, (grade, low) in enumerate(bounds):
        high = bounds[i + 1][1] if i + 1 < len(bounds) else 100.0
        fig.add_hrect(y0=low, y1=high, fillcolor=GRADE_COLOR.get(grade, "#888"),
                      opacity=0.08, line_width=0, layer="below",
                      annotation_text=grade, annotation_position="right",
                      annotation_font_size=10, annotation_font_color=pal.muted)

    fig.add_trace(go.Scatter(
        x=score.index, y=score, name="일별",
        mode="lines", line=dict(color=pal.muted, width=1), opacity=0.45,
        hovertemplate="%{x|%Y-%m-%d}<br>일별 %{y:.1f}점<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=smooth.index, y=smooth, name="7일 평활",
        mode="lines", line=dict(color=pal.series[0], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.1f}점</b><extra></extra>",
    ))

    for cp in analysis.change_points:
        fig.add_vline(x=cp.date, line_width=1, line_dash="dot",
                      line_color=GRADE_COLOR["D"] if cp.direction == "up" else GRADE_COLOR["A"])

    fig.update_yaxes(range=[0, 100])
    return apply_layout(fig, pal, height=340, show_legend=True).update_layout(
        title="건전도 점수 추이 (점)"
    )


def chart_deductions(snapshot, ind, pal, groups):
    """지표별 감점 — 그룹 색으로 묶어 어디서 깎였는지 보이게 한다."""
    rows = [
        (f"{i}. {ind.specs[i].name}", snapshot.deductions.get(i, 0.0), ind.specs[i].group)
        for i in ind.active_ids
    ]
    rows.sort(key=lambda r: r[1])

    fig = go.Figure()
    seen: set[str] = set()
    for label, value, gid in rows:
        fig.add_trace(go.Bar(
            x=[value], y=[label], orientation="h",
            marker=dict(color=group_color(gid, pal), line=dict(color=pal.surface, width=2)),
            name=groups.get(gid, {}).get("name", gid),
            legendgroup=gid, showlegend=gid not in seen,
            text=f"{value:.1f}", textposition="outside",
            textfont=dict(color=pal.text_secondary, size=11),
            hovertemplate=f"{label}<br>감점 %{{x:.1f}}점<extra></extra>",
        ))
        seen.add(gid)

    fig.update_xaxes(title_text="감점 (점)", showgrid=True, gridcolor=pal.grid)
    fig.update_yaxes(showgrid=False)
    return apply_layout(fig, pal, height=460, show_legend=True).update_layout(
        title="지표별 감점 기여도", bargap=0.35, hovermode="closest"
    )


def chart_group_summary(snapshot, ind, pal, groups):
    """그룹별 배점 대비 감점."""
    gids = sorted(groups)
    allotted = [sum(ind.effective_points.get(i, 0.0) for i in ind.active_ids
                    if ind.specs[i].group == g) for g in gids]
    lost = [snapshot.group_deductions.get(g, 0.0) for g in gids]
    labels = [f"{g}. {groups[g].get('name', g)}" for g in gids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[a - l for a, l in zip(allotted, lost)], name="잔여",
        marker=dict(color=pal.grid, line=dict(color=pal.surface, width=2)),
        hovertemplate="%{x}<br>잔여 %{y:.1f}점<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=lost, name="감점",
        marker=dict(color=[group_color(g, pal) for g in gids],
                    line=dict(color=pal.surface, width=2)),
        text=[f"−{v:.1f}" for v in lost], textposition="inside",
        textfont=dict(color="#ffffff", size=12),
        hovertemplate="%{x}<br>감점 %{y:.1f}점<extra></extra>",
    ))
    return apply_layout(fig, pal, height=300, show_legend=True).update_layout(
        title="그룹별 배점 대비 감점 (점)", barmode="stack", hovermode="closest"
    )


def chart_metric(series, title, pal, *, onset=None, severe=None, unit="", color_slot=0,
                 change_points=(), invert_bands=False):
    """단일 지표 추이. 축은 하나만 쓴다 — 이중축은 절대 만들지 않는다."""
    smooth = ewma(series.dropna(), 7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series, mode="lines",
        line=dict(color=pal.muted, width=1), opacity=0.4, name="일별",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3g}" + unit + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=smooth.index, y=smooth, mode="lines",
        line=dict(color=pal.series[color_slot], width=2), name="7일 평활",
        hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.3g}" + unit + "</b><extra></extra>",
    ))

    for value, label, color in (
        (onset, "정상 한계", GRADE_COLOR["B"]),
        (severe, "심각", GRADE_COLOR["D"]),
    ):
        if value is not None:
            fig.add_hline(y=value, line_width=1, line_dash="dash", line_color=color,
                          annotation_text=label, annotation_position="top left",
                          annotation_font_size=10, annotation_font_color=color)

    # 변화점이 많으면 선이 빽빽해져 그래프를 못 읽는다. 최초 3건만 표시한다.
    for cp in [c for c in change_points if c.direction == "up"][:3]:
        fig.add_vline(x=cp.date, line_width=1, line_dash="dot", line_color=GRADE_COLOR["D"])

    return apply_layout(fig, pal, height=260).update_layout(title=title)


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------
def main() -> None:
    pal = palette()

    st.title("🏭 RTO 축열재 막힘 사전예측")
    st.caption("Can Type (Rotary 1-Can) RTO · 세라믹 축열재 막힘 진행도를 100점으로 정량화합니다")

    # --- 사이드바: 데이터 선택 --------------------------------------------
    with st.sidebar:
        st.header("데이터")
        uploaded = st.file_uploader("운전 데이터 (CSV / Excel)", type=["csv", "xlsx", "xls"])

        analysis = None
        if uploaded is not None:
            analysis = _analyze_bytes(uploaded.getvalue(), uploaded.name)
        elif _ensure_samples():
            choice = st.selectbox(
                "샘플 시나리오", list(SAMPLE_LABELS),
                format_func=lambda k: SAMPLE_LABELS[k], index=1,
            )
            analysis = _analyze_path(str(SAMPLE_DIR / f"{choice}.csv"))
            st.caption("합성 샘플 데이터입니다. 실제 운전 데이터를 업로드하세요.")
        else:
            st.warning("샘플이 없습니다. 먼저 실행하세요:\n\n`python data/sample/generate_sample.py`")

    if analysis is None:
        st.info("좌측에서 운전 데이터를 업로드하거나 샘플 시나리오를 선택하세요.")
        return

    cfg = analysis.config
    groups = cfg.weights.get("groups") or {}
    grades = cfg.weights.get("grades") or []

    with st.sidebar:
        st.header("기준일")
        dates = analysis.daily.index
        as_of = st.slider(
            "분석 기준일", min_value=dates[0].to_pydatetime(), max_value=dates[-1].to_pydatetime(),
            value=dates[-1].to_pydatetime(), format="YYYY-MM-DD",
        )

    snap = analysis.snapshot(pd.Timestamp(as_of).normalize())
    guide = analysis.guidance(snap.date)
    ind = analysis.indicators

    tabs = st.tabs(["종합", "감점 기여도", "변화추이", "점검 가이드", "설정"])

    # ===== 1. 종합 =========================================================
    with tabs[0]:
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.4])
        color = GRADE_COLOR.get(snap.grade, pal.muted)
        c1.markdown(
            f"<div style='font-size:13px;color:{pal.text_secondary}'>설비 건전도</div>"
            f"<div style='font-size:56px;font-weight:700;line-height:1.1;color:{color}'>"
            f"{snap.score:.1f}<span style='font-size:22px;color:{pal.muted}'> / 100</span></div>"
            f"<div style='font-size:15px;color:{pal.text_primary}'>"
            f"{GRADE_ICON.get(snap.grade,'')} {snap.grade}등급 · {snap.label}</div>",
            unsafe_allow_html=True,
        )

        prev = analysis.score.loc[analysis.score.index <= snap.date - pd.Timedelta(days=30)]
        delta = f"{snap.score - float(prev.iloc[-1]):+.1f}점" if not prev.empty else "—"
        c2.metric("30일 전 대비", f"{snap.score:.1f}점", delta, delta_color="normal")

        rul = analysis.rul
        if rul.days_remaining is None:
            rul_text, rul_help = "산출 불가", rul.note
        elif rul.days_remaining > 1095:
            rul_text, rul_help = "3년 이상", "현재 추세 유지 시"
        else:
            rul_text = f"약 {rul.days_remaining:,.0f}일"
            rul_help = f"{rul.predicted_date:%Y-%m-%d} 경 · 상승 {rul.slope_per_day*100:.3f}%p/일"
        c3.metric("운전 한계 도달", rul_text, help="정규화 차압이 설계 한계에 도달하는 예상 시점")
        c3.caption(rul_help)

        c4.markdown(
            f"<div style='font-size:13px;color:{pal.text_secondary}'>판정 고장모드</div>"
            f"<div style='font-size:17px;font-weight:600;color:{pal.text_primary};margin:4px 0'>"
            f"{guide['headline']}</div>"
            f"<div style='font-size:13px;color:{pal.text_secondary}'>{snap.action}</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.plotly_chart(chart_score_trend(analysis, pal, grades), width='stretch')
        st.caption("배경 밴드는 등급 구간, 세로 점선은 CUSUM 이 검출한 변화점입니다 "
                   "(붉은색 = 악화 시작, 녹색 = 개선 · 정비 효과).")

        st.markdown("##### 감점 상위 지표")
        for item in guide["top_contributors"] or ["유의미한 감점 지표 없음"]:
            st.markdown(f"- {item}")

    # ===== 2. 감점 기여도 ===================================================
    with tabs[1]:
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(chart_group_summary(snap, ind, pal, groups), width='stretch')
        with right:
            st.markdown("##### 그룹별 요약")
            st.dataframe(
                pd.DataFrame([{
                    "그룹": f"{g}. {groups[g].get('name', g)}",
                    "배점": round(sum(ind.effective_points.get(i, 0.0) for i in ind.active_ids
                                    if ind.specs[i].group == g), 1),
                    "감점": round(snap.group_deductions.get(g, 0.0), 1),
                } for g in sorted(groups)]),
                hide_index=True, width='stretch',
            )

        st.plotly_chart(chart_deductions(snap, ind, pal, groups), width='stretch')

        st.markdown("##### 지표 상세 (표)")
        st.dataframe(scoring.score_table(snap, ind), hide_index=True, width='stretch')

    # ===== 3. 변화추이 =====================================================
    with tabs[2]:
        st.caption(
            "지표는 모두 **정규화 후**의 값입니다. 차압은 풍량·온도로 보정했으므로 "
            "생산량 변동이 아니라 막힘만 반영합니다. 붉은 점선은 CUSUM 이 검출한 악화 시작 시점입니다."
        )
        specs = ind.specs
        panels = [
            ("A1", "정규화 베드 차압 (베이스라인 대비 배수)", "배", 0),
            ("B1", "열회수효율 저하 (%p)", "%p", 1),
            ("C1", "연료 원단위 (베이스라인 대비 배수)", "배", 2),
            ("B3", "섹터 간 출구온도 편차 (℃)", "℃", 3),
        ]
        cols = st.columns(2)
        for i, (ind_id, title, unit, slot) in enumerate(panels):
            if ind_id not in ind.raw.columns:
                continue
            spec = specs[ind_id]
            with cols[i % 2]:
                st.plotly_chart(
                    chart_metric(ind.raw[ind_id], title, pal, onset=spec.onset, severe=spec.severe,
                                 unit=unit, color_slot=slot, change_points=analysis.change_points),
                    width='stretch',
                )

        st.markdown("##### 검출된 변화 시점")
        if analysis.change_points:
            st.dataframe(
                pd.DataFrame([{
                    "일자": f"{p.date:%Y-%m-%d}",
                    "방향": "악화 시작" if p.direction == "up" else "개선 (정비 효과)",
                    "통계량": round(p.statistic, 1),
                } for p in analysis.change_points]),
                hide_index=True, width='stretch',
            )
            st.caption("해당 일자 전후의 공정 이력(원료 변경, 배치 이상, 전처리 필터 교체)을 확인하세요.")
        else:
            st.success("검출된 변화 시점이 없습니다 — 열화 속도가 일정하게 유지되고 있습니다.")

    # ===== 4. 점검 가이드 ===================================================
    with tabs[3]:
        if not guide["all_modes"]:
            st.success(f"판정된 특이 고장모드가 없습니다. 등급별 기본 조치: **{snap.action}**")
        for i, mode in enumerate(guide["all_modes"]):
            with st.expander(f"**{mode.name}** — 확신도 {mode.confidence:.0%}", expanded=(i == 0)):
                st.markdown(f"**추정 원인**  \n{mode.cause}")
                st.markdown("**판정 근거**")
                for ev in mode.evidence:
                    st.markdown(f"- {ev}")
                st.markdown("**권고 조치**")
                for j, act in enumerate(mode.actions):
                    st.checkbox(act, key=f"{mode.id}_{j}_{snap.date:%Y%m%d}")

        st.divider()
        report_md = build_markdown(analysis, snap.date)
        st.download_button(
            "📄 점검 리포트 내려받기 (Markdown)", report_md,
            file_name=f"RTO_점검리포트_{snap.date:%Y%m%d}.md", mime="text/markdown",
        )
        with st.expander("리포트 미리보기"):
            st.markdown(report_md)

    # ===== 5. 설정 =========================================================
    with tabs[4]:
        st.markdown("##### 분석 조건")
        refs, base = analysis.refs, analysis.baseline
        source_label = {"fitted": "실측 회귀 피팅", "config": "설정 고정값",
                        "default": "축열재 형상별 기본값"}[refs.exponent_source]
        # 값에 증감 의미가 없으므로 metric 의 delta(초록 화살표)를 쓰지 않는다
        c1, c2, c3 = st.columns(3)
        c1.metric("유량 지수 n", f"{refs.flow_exponent:.3f}")
        c1.caption(source_label + (f" · 회귀 R² = {refs.fit_r2:.4f}" if refs.fit_r2 is not None else ""))
        c2.metric("기준 풍량", f"{refs.ref_flow:,.0f} CMM")
        c2.caption(f"기준 가스온도 {refs.ref_gas_temp_c:.1f} ℃")
        c3.metric("베이스라인 열회수효율", f"{base.ref('ter')*100:.2f} %")
        c3.caption(f"베이스라인 정규화 차압 {base.ref('dp_norm'):.1f} mmH2O")
        st.caption(
            f"베이스라인 구간: {base.start:%Y-%m-%d} ~ {base.end:%Y-%m-%d} "
            f"({base.n_samples:,} 샘플" + (f", 기준 정비: {base.reset_type}" if base.reset_type else "") + ")"
        )

        if ind.excluded:
            st.warning("**채점 제외 지표** (해당 배점은 잔여 지표로 재배분됨)\n\n" + "\n".join(
                f"- {k} {ind.specs[k].name}: {v}" for k, v in ind.excluded.items()))

        st.divider()
        st.markdown("##### 배점 · 임계치 조정")
        st.caption(
            "현장 설비 특성에 맞게 조정하면 아래에서 즉시 재채점됩니다. "
            "영구 반영은 `config/weights.yaml` 을 수정하세요."
        )

        editable = pd.DataFrame([{
            "지표": i, "명칭": ind.specs[i].name, "그룹": ind.specs[i].group,
            "배점": ind.specs[i].points, "정상한계(onset)": ind.specs[i].onset,
            "심각(severe)": ind.specs[i].severe,
        } for i in sorted(ind.specs)])

        edited = st.data_editor(
            editable, hide_index=True, width='stretch',
            disabled=["지표", "명칭", "그룹"], key="weight_editor",
        )

        if not edited.equals(editable):
            tuned = load_config(PROJECT_ROOT / "config")
            for _, row in edited.iterrows():
                entry = tuned.weights["indicators"].get(row["지표"])
                if entry:
                    entry["points"] = float(row["배점"])
                    entry["onset"] = float(row["정상한계(onset)"])
                    entry["severe"] = float(row["심각(severe)"])

            # raw metric 은 재사용하고 열화도·배점만 다시 계산한다
            retuned = ind_mod.compute(analysis.daily, base, tuned, analysis.dataset, raw=ind.raw)
            new_snap = scoring.score(retuned, tuned).snapshot(snap.date)

            total = edited["배점"].sum()
            st.info(f"조정 후 배점 합계: **{total:.1f}점** "
                    + ("" if abs(total - 100) < 0.05 else "(100점이 아니면 비율대로 환산됩니다)"))
            d1, d2 = st.columns(2)
            d1.metric("조정 후 건전도", f"{new_snap.score:.1f}점",
                      f"{new_snap.score - snap.score:+.1f}점")
            d2.metric("조정 후 등급", f"{GRADE_ICON.get(new_snap.grade,'')} {new_snap.grade}등급",
                      new_snap.label)

        st.divider()
        st.markdown("##### 태그 매핑")
        st.caption("현장 DCS 태그명이 다르면 `config/tags.yaml` 의 `source` 값을 수정하세요.")
        st.dataframe(
            pd.DataFrame([
                {"표준 변수": k, "현장 태그": (v or {}).get("source") or (v or {}).get("source_prefix", "—"),
                 "단위": (v or {}).get("unit", ""), "설명": (v or {}).get("description", ""),
                 "상태": "✅ 사용 중" if k in analysis.dataset.available else "— 미보유"}
                for k, v in cfg.tags.items()
            ]),
            hide_index=True, width='stretch',
        )

        st.warning(
            "⚠️ 배점 가중치는 RTO 물리·정비 문헌에 근거한 **공학적 추정치**이며 통계적으로 검증된 "
            "값이 아닙니다. 실제 정비 이력이 2회 이상 누적되면 `python -m rto_health.calibrate` 로 "
            "재보정하십시오."
        )


if __name__ == "__main__":
    main()
