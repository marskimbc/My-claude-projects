"""가중 합산 채점 — 100점 만점 설비 건전도.

    건전도 = 100 − Σ( 지표별 열화도(0~1) × 배점 )

점수만 보면 "얼마나 나쁜지"는 알아도 "무엇 때문인지"는 모른다. 그래서 점수와
함께 **지표별·그룹별 감점 기여도**를 항상 같이 반환한다. 현장에서 쓰이려면
"78점"이 아니라 "78점, 그중 14점이 차압에서 깎였음"이어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import IndicatorSet
from .io_loader import Config


@dataclass
class ScoreSnapshot:
    """특정 시점의 채점 결과 한 건."""

    date: pd.Timestamp
    score: float
    grade: str
    label: str
    action: str
    deductions: dict[str, float] = field(default_factory=dict)      # 지표ID → 감점
    degradations: dict[str, float] = field(default_factory=dict)    # 지표ID → 0~1
    raw_metrics: dict[str, float] = field(default_factory=dict)     # 지표ID → raw
    group_deductions: dict[str, float] = field(default_factory=dict)

    def top_contributors(self, n: int = 5) -> list[tuple[str, float]]:
        """감점이 큰 지표 순으로 반환."""
        items = [(k, v) for k, v in self.deductions.items() if v > 0.05]
        return sorted(items, key=lambda kv: kv[1], reverse=True)[:n]


@dataclass
class ScoreResult:
    daily: pd.DataFrame           # score, grade 및 지표별 감점 컬럼
    deductions: pd.DataFrame      # index=date, columns=지표ID → 감점
    indicators: IndicatorSet
    config: Config

    @property
    def score(self) -> pd.Series:
        return self.daily["score"]

    def snapshot(self, date: pd.Timestamp | str | None = None) -> ScoreSnapshot:
        """지정일(생략 시 최신일)의 채점 스냅샷."""
        if self.daily.empty:
            raise ValueError("채점 결과가 비어 있습니다. 유효한 운전 데이터가 있는지 확인하세요.")

        if date is None:
            ts = self.daily.index[-1]
        else:
            ts = pd.Timestamp(date)
            if ts not in self.daily.index:
                idx = self.daily.index[self.daily.index <= ts]
                if len(idx) == 0:
                    raise ValueError(f"{ts:%Y-%m-%d} 이전의 채점 결과가 없습니다.")
                ts = idx[-1]

        row = self.daily.loc[ts]
        ded = self.deductions.loc[ts]
        specs = self.indicators.specs

        group_ded: dict[str, float] = {}
        for ind_id, value in ded.items():
            g = specs[ind_id].group
            group_ded[g] = group_ded.get(g, 0.0) + float(value)

        grade_info = _grade_for(float(row["score"]), self.config)
        return ScoreSnapshot(
            date=ts,
            score=float(row["score"]),
            grade=grade_info["grade"],
            label=grade_info["label"],
            action=grade_info["action"],
            deductions={k: float(v) for k, v in ded.items()},
            degradations={
                k: float(self.indicators.degradation.loc[ts, k])
                for k in self.indicators.degradation.columns
                if pd.notna(self.indicators.degradation.loc[ts, k])
            },
            raw_metrics={
                k: float(self.indicators.raw.loc[ts, k])
                for k in self.indicators.raw.columns
                if ts in self.indicators.raw.index and pd.notna(self.indicators.raw.loc[ts, k])
            },
            group_deductions=group_ded,
        )


def _grade_for(score: float, config: Config) -> dict[str, str]:
    """점수 → 등급. weights.yaml 의 grades 를 min_score 내림차순으로 평가."""
    grades = sorted(
        config.weights.get("grades") or [],
        key=lambda g: float(g.get("min_score", 0)),
        reverse=True,
    )
    for g in grades:
        if score >= float(g.get("min_score", 0)):
            return {
                "grade": str(g.get("grade", "?")),
                "label": str(g.get("label", "")),
                "action": str(g.get("action", "")),
            }
    return {"grade": "?", "label": "등급 미정의", "action": ""}


def score(indicators: IndicatorSet, config: Config, *, ffill_days: int = 7) -> ScoreResult:
    """지표 열화도를 가중 합산해 일별 건전도 점수를 만든다.

    결측 처리: 일시적 계측 결측은 직전 값을 최대 ffill_days 일까지 이어 쓰고,
    그래도 남는 결측(예: 추세 지표의 초기 구간)은 감점하지 않는다 — 증거가
    없는데 깎는 것은 오탐이다.
    """
    deg = indicators.degradation
    if deg.empty:
        empty = pd.DataFrame(columns=["score", "grade"])
        return ScoreResult(daily=empty, deductions=pd.DataFrame(), indicators=indicators, config=config)

    filled = deg.ffill(limit=ffill_days).fillna(0.0)

    deductions = pd.DataFrame(index=filled.index)
    for ind_id in filled.columns:
        deductions[ind_id] = filled[ind_id] * indicators.effective_points.get(ind_id, 0.0)

    total = deductions.sum(axis=1)
    scores = (100.0 - total).clip(lower=0.0, upper=100.0)

    daily = pd.DataFrame({"score": scores})
    daily["grade"] = [_grade_for(s, config)["grade"] for s in scores]
    for g in sorted({indicators.specs[i].group for i in filled.columns}):
        members = [i for i in filled.columns if indicators.specs[i].group == g]
        daily[f"deduction_{g}"] = deductions[members].sum(axis=1)

    return ScoreResult(daily=daily, deductions=deductions, indicators=indicators, config=config)


def score_table(snapshot: ScoreSnapshot, indicators: IndicatorSet) -> pd.DataFrame:
    """스냅샷을 사람이 읽는 표로 변환한다 (대시보드·리포트 공용)."""
    rows = []
    for ind_id, spec in indicators.specs.items():
        if ind_id in indicators.excluded:
            rows.append({
                "지표": f"{ind_id}. {spec.name}",
                "그룹": indicators.group_names.get(spec.group, spec.group),
                "배점": 0.0,
                "측정값": None,
                "단위": spec.unit,
                "열화도": None,
                "감점": 0.0,
                "비고": indicators.excluded[ind_id],
            })
            continue
        rows.append({
            "지표": f"{ind_id}. {spec.name}",
            "그룹": indicators.group_names.get(spec.group, spec.group),
            "배점": round(indicators.effective_points.get(ind_id, 0.0), 1),
            "측정값": _round(snapshot.raw_metrics.get(ind_id)),
            "단위": spec.unit,
            "열화도": _round(snapshot.degradations.get(ind_id), 2),
            "감점": round(snapshot.deductions.get(ind_id, 0.0), 1),
            "비고": f"정상 ≤{spec.onset:g} / 심각 ≥{spec.severe:g}",
        })

    df = pd.DataFrame(rows)
    return df.sort_values("감점", ascending=False).reset_index(drop=True)


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)
