"""고장모드 판정 — 점수를 '무엇을 점검할지'로 바꾸는 단계.

건전도 68점이라는 숫자만으로는 현장에서 할 일이 정해지지 않는다. 같은 68점이라도
원인이 균일 분진이면 bake-out 으로 끝나지만, 실리카 고착이면 bake-out 은 시간
낭비이고 세정/교체가 필요하다.

핵심은 **지표의 조합**이다. 특히 아래 패턴은 반드시 구분해야 한다.

    차압 정상 + 열효율 저하 + 섹터 편차 큼  →  막힘이 아니라 로터리밸브 씰 누설
                                             (세정해도 효과 없음)

config/weights.yaml 의 failure_modes 규칙을 그대로 평가하므로, 현장에서 새로운
고장 패턴을 발견하면 코드 수정 없이 규칙만 추가하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import IndicatorSet
from .io_loader import Config
from .scoring import ScoreSnapshot

# 지표 열화도가 아닌 별도 신호로 평가하는 특수 조건 키
SPECIAL_CONDITIONS = {"dp_below_baseline"}


@dataclass
class Diagnosis:
    """판정된 고장모드 한 건."""

    id: str
    name: str
    priority: int
    cause: str
    actions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0   # 조건 충족 여유도 기반 0~1


def _special_signal(key: str, indicators: IndicatorSet, date: pd.Timestamp) -> bool | None:
    """지표 열화도로 표현되지 않는 보조 신호를 평가한다."""
    if key == "dp_below_baseline":
        # 정규화 차압이 베이스라인보다 낮다 = 축열재 유실 / 우회 경로 의심
        if "A1" not in indicators.raw.columns or date not in indicators.raw.index:
            return None
        value = indicators.raw.loc[date, "A1"]
        return bool(pd.notna(value) and float(value) < 0.95)
    return None


def evaluate(
    snapshot: ScoreSnapshot,
    indicators: IndicatorSet,
    config: Config,
) -> list[Diagnosis]:
    """스냅샷 시점에 성립하는 고장모드를 모두 찾아 우선순위순으로 반환한다."""
    rules = sorted(
        config.weights.get("failure_modes") or [],
        key=lambda r: int(r.get("priority", 100)),
    )
    results: list[Diagnosis] = []

    for rule in rules:
        conditions: dict = rule.get("conditions") or {}
        matched = True
        evidence: list[str] = []
        margins: list[float] = []

        for key, cond in conditions.items():
            if key in SPECIAL_CONDITIONS:
                signal = _special_signal(key, indicators, snapshot.date)
                if signal is None or signal != bool(cond):
                    matched = False
                    break
                evidence.append("정규화 차압이 베이스라인 미만 (축열재 유실 의심)")
                margins.append(0.5)
                continue

            if key not in snapshot.degradations:
                # 해당 지표를 못 쓰는 설비면 이 규칙은 판정 불가
                matched = False
                break

            deg = snapshot.degradations[key]
            spec = indicators.specs[key]
            raw = snapshot.raw_metrics.get(key)

            lo = cond.get("min")
            hi = cond.get("max")
            if lo is not None and deg < float(lo):
                matched = False
                break
            if hi is not None and deg > float(hi):
                matched = False
                break

            if lo is not None:
                margins.append(min(1.0, (deg - float(lo)) / max(1e-6, 1.0 - float(lo))))
                evidence.append(
                    f"{key} {spec.name}: 열화도 {deg:.2f} (≥{float(lo):.2f} 조건 충족)"
                    + (f", 측정 {raw:.3g}{spec.unit}" if raw is not None and np.isfinite(raw) else "")
                )
            else:
                margins.append(min(1.0, (float(hi) - deg) / max(1e-6, float(hi)) if hi else 0.5))
                evidence.append(
                    f"{key} {spec.name}: 열화도 {deg:.2f} (≤{float(hi):.2f} 조건 충족)"
                    + (f", 측정 {raw:.3g}{spec.unit}" if raw is not None and np.isfinite(raw) else "")
                )

        if matched and conditions:
            results.append(
                Diagnosis(
                    id=str(rule.get("id", "unknown")),
                    name=str(rule.get("name", "")),
                    priority=int(rule.get("priority", 100)),
                    cause=str(rule.get("cause", "")),
                    actions=list(rule.get("actions") or []),
                    evidence=evidence,
                    confidence=float(np.mean(margins)) if margins else 0.0,
                )
            )

    return results


def guidance(
    snapshot: ScoreSnapshot,
    indicators: IndicatorSet,
    config: Config,
) -> dict:
    """점검 가이드 전체를 구성한다 — 판정된 고장모드 + 등급별 기본 조치."""
    modes = evaluate(snapshot, indicators, config)

    if modes:
        primary = modes[0]
        headline = primary.name
    else:
        primary = None
        headline = "특이 고장모드 미검출 — 등급별 기본 조치 적용"

    # 감점 상위 지표는 고장모드와 무관하게 항상 보여준다
    watch: list[str] = []
    for ind_id, ded in snapshot.top_contributors(4):
        spec = indicators.specs[ind_id]
        raw = snapshot.raw_metrics.get(ind_id)
        raw_txt = f" (측정 {raw:.3g}{spec.unit})" if raw is not None and np.isfinite(raw) else ""
        watch.append(f"{ind_id} {spec.name} — {ded:.1f}점 감점{raw_txt}")

    return {
        "date": snapshot.date,
        "score": snapshot.score,
        "grade": snapshot.grade,
        "grade_label": snapshot.label,
        "grade_action": snapshot.action,
        "headline": headline,
        "primary": primary,
        "all_modes": modes,
        "top_contributors": watch,
    }
