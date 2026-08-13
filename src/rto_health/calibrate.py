"""가중치 재보정 — 실제 정비 이력이 쌓인 뒤에 쓰는 모듈.

현재 config/weights.yaml 의 배점은 **공학적 추정치**다. 실제 운전·정비 이력이
확보되면 "어떤 지표가 실제로 막힘을 예고했는가"를 데이터로 확인해 배점을
현장에 맞게 다시 나눌 수 있다.

방법: 각 지표의 열화도와 '정비까지 남은 시간' 사이의 상관을 구하고, 상관이
높은 지표에 더 큰 배점을 준다. 실제로 막힘을 예고한 지표에 무게가 실린다.

    python -m rto_health.calibrate --input <운전데이터> --apply
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .pipeline import analyze


@dataclass
class CalibrationResult:
    """지표별 상관분석 및 제안 배점."""

    table: pd.DataFrame          # 지표별 상관계수·p값·현재/제안 배점
    n_events: int
    method: str
    note: str = ""


def _time_to_event(index: pd.DatetimeIndex, event_dates: list[pd.Timestamp]) -> pd.Series:
    """각 날짜에서 '다음 정비까지 남은 일수'를 계산한다."""
    future = pd.Series(np.nan, index=index, dtype=float)
    for i, ts in enumerate(index):
        upcoming = [d for d in event_dates if d >= ts]
        if upcoming:
            future.iloc[i] = (min(upcoming) - ts).days
    return future


def calibrate(
    source: str | Path | pd.DataFrame,
    *,
    config_dir: str | Path | None = None,
    min_points: float = 1.0,
) -> CalibrationResult:
    """지표 열화도와 '정비까지 남은 시간'의 상관으로 배점을 재산출한다.

    상관이 강한 지표(= 막힘을 실제로 예고한 지표)에 더 큰 배점을 배분한다.
    그룹별 총점(A 45 / B 25 / C 20 / D 10)은 유지하고 그룹 **안에서만**
    재배분한다 — 인과 근접도에 따른 그룹 구조는 물리적 근거이므로 데이터
    한 사이클로 뒤집지 않는다.
    """
    analysis = analyze(source, config_dir=config_dir)
    ind = analysis.indicators
    cfg = analysis.config

    # 정비 이력을 이벤트로 사용 (세정·교체 = 막힘이 한계에 이른 시점)
    events = cfg.maintenance_events()
    reset_types = set(cfg.weights.get("baseline", {}).get("reset_event_types", ["full_clean", "media_replace"]))
    event_dates = [
        pd.Timestamp(d) for d, t in zip(events["date"], events["type"]) if t in reset_types
    ] if not events.empty else []

    deg = ind.degradation
    rows = []

    if len(event_dates) >= 2:
        target = -_time_to_event(deg.index, event_dates)   # 정비가 가까울수록 큰 값
        method = "정비까지 남은 시간과의 Spearman 상관"
        note = ""
    else:
        # 정비 이력이 1건 이하면 이벤트 기반 상관을 구할 수 없다.
        # 차선책: 정규화 차압(A1)을 막힘의 대리 지표로 삼는다.
        target = ind.raw["A1"] if "A1" in ind.raw.columns else pd.Series(dtype=float)
        method = "정규화 차압(A1)과의 Spearman 상관 — 대리 지표"
        note = (
            "정비 이력이 2건 미만이라 이벤트 기반 보정을 할 수 없어 A1 을 대리 지표로 "
            "사용했습니다. 세정/교체 이력이 2회 이상 쌓인 뒤 다시 실행하십시오."
        )

    for ind_id in deg.columns:
        pair = pd.concat([deg[ind_id], target], axis=1).dropna()
        pair.columns = ["deg", "target"]
        if len(pair) < 30 or pair["deg"].nunique() < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = stats.spearmanr(pair["deg"], pair["target"])

        spec = ind.specs[ind_id]
        rows.append({
            "지표": ind_id,
            "명칭": spec.name,
            "그룹": spec.group,
            "상관계수": rho,
            "p값": pval,
            "현재배점": spec.points,
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return CalibrationResult(table=table, n_events=len(event_dates), method=method,
                                 note="채점 가능한 지표가 없습니다.")

    # --- 그룹 내 재배분 -----------------------------------------------------
    # 통계적으로 유의하지 않거나(p>0.05) 음의 상관인 지표는 최소 배점만 남긴다.
    weight = table["상관계수"].fillna(0.0).clip(lower=0.0)
    weight = weight.where(table["p값"].fillna(1.0) <= 0.05, 0.0)
    table["_w"] = weight

    proposed = []
    for _, group in table.groupby("그룹"):
        pool = float(group["현재배점"].sum())
        floor = min_points * len(group)
        w_sum = float(group["_w"].sum())
        if w_sum <= 0 or pool <= floor:
            proposed.extend([(i, group.loc[i, "현재배점"]) for i in group.index])
            continue
        share = (pool - floor) * group["_w"] / w_sum + min_points
        proposed.extend([(i, float(share.loc[i])) for i in group.index])

    table["제안배점"] = pd.Series(dict(proposed))
    table["변화"] = table["제안배점"] - table["현재배점"]
    table = table.drop(columns=["_w"]).sort_values(["그룹", "제안배점"], ascending=[True, False])

    return CalibrationResult(
        table=table.round(3),
        n_events=len(event_dates),
        method=method,
        note=note,
    )


def apply_to_config(result: CalibrationResult, config_path: str | Path) -> None:
    """제안 배점을 weights.yaml 에 반영한다 (덮어쓰기)."""
    path = Path(config_path)
    with open(path, encoding="utf-8") as fh:
        weights = yaml.safe_load(fh)

    for _, row in result.table.iterrows():
        ind_id = row["지표"]
        if ind_id in weights.get("indicators", {}) and np.isfinite(row["제안배점"]):
            weights["indicators"][ind_id]["points"] = round(float(row["제안배점"]), 1)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(weights, fh, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="지표 배점 재보정")
    parser.add_argument("--input", "-i", required=True, help="운전 데이터 경로")
    parser.add_argument("--config-dir", help="config 디렉토리")
    parser.add_argument("--apply", action="store_true", help="제안 배점을 weights.yaml 에 실제로 반영")
    args = parser.parse_args()

    result = calibrate(args.input, config_dir=args.config_dir)
    print(f"보정 방법: {result.method}")
    print(f"정비 이벤트 수: {result.n_events}")
    if result.note:
        print(f"⚠️  {result.note}")
    print()
    print(result.table.to_string(index=False))

    if args.apply:
        from .io_loader import CONFIG_DIR
        target = Path(args.config_dir or CONFIG_DIR) / "weights.yaml"
        apply_to_config(result, target)
        print(f"\n반영 완료: {target}")
    else:
        print("\n(실제 반영하려면 --apply 옵션을 추가하세요)")


if __name__ == "__main__":
    main()
