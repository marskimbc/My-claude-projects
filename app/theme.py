"""대시보드 색상·차트 공통 설정.

색은 역할로만 지정한다. 시리즈 색은 고정 순서로 배정하고 절대 순환시키지 않으며,
등급은 상태색(status) 팔레트를 쓰되 아이콘·라벨과 항상 함께 표시한다
(색만으로 의미를 전달하지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    surface: str
    page: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]


LIGHT = Palette(
    surface="#fcfcfb",
    page="#f9f9f7",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
)

DARK = Palette(
    surface="#1a1a19",
    page="#0d0d0d",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
)

# 상태색은 테마와 무관하게 고정한다. D/E 는 같은 계열의 더 어두운 단계로
# '더 나쁨'을 나타내되, 아이콘과 라벨이 항상 함께 붙는다.
GRADE_COLOR = {
    "A": "#0ca30c",   # good
    "B": "#fab219",   # warning
    "C": "#ec835a",   # serious
    "D": "#d03b3b",   # critical
    "E": "#8f2020",   # critical (darker step)
}

GRADE_ICON = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "E": "⛔"}

# 지표 그룹 → 시리즈 슬롯 (고정 배정)
GROUP_COLOR_SLOT = {"A": 0, "B": 1, "C": 2, "D": 3}


def group_color(group_id: str, palette: Palette) -> str:
    return palette.series[GROUP_COLOR_SLOT.get(group_id, 0)]


def apply_layout(fig, palette: Palette, *, height: int = 300, show_legend: bool = False):
    """모든 차트에 동일한 크롬을 적용한다 — 눈금·격자는 뒤로 물린다."""
    fig.update_layout(
        height=height,
        # 우측 여백은 임계선 주석과 등급 밴드 라벨이 잘리지 않도록 넉넉히 둔다
        margin=dict(l=8, r=52, t=40, b=8),
        paper_bgcolor=palette.surface,
        plot_bgcolor=palette.surface,
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=palette.text_secondary,
            size=12,
        ),
        title=dict(font=dict(color=palette.text_primary, size=14), x=0, xanchor="left"),
        showlegend=show_legend,
        # 제목은 왼쪽, 범례는 오른쪽 끝에 붙여 같은 줄에서 겹치지 않게 한다
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=palette.surface, font_size=12,
                        font_family='system-ui, -apple-system, sans-serif'),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor=palette.axis, tickcolor=palette.axis, tickfont=dict(color=palette.muted),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=palette.grid, gridwidth=1, zeroline=False,
        linecolor=palette.axis, tickcolor=palette.axis, tickfont=dict(color=palette.muted),
    )
    return fig
