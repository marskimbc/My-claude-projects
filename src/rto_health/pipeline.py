"""전체 분석 파이프라인 — 원데이터 한 개를 넣으면 결과 한 묶음이 나온다.

    로딩 → 정제 → 정상구간 판정 → 베이스라인 선정 → 물리 보정 → 일 집계
         → 지표 산출 → 채점 → 추세/변화점/RUL → 고장모드 판정

베이스라인 선정과 물리 보정은 서로를 필요로 하므로(정규화 기준값이 베이스라인
구간에서 나온다) 2단계로 나눠 푼다: 시간 기준으로 먼저 구간을 잡아 지수·기준값을
구하고, 정규화가 끝난 뒤 그 구간에서 기준값과 조건부 모델을 확정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import baseline as baseline_mod
from . import diagnose, indicators, normalize, preprocess, scoring
from .io_loader import Config, Dataset, load_config, load_operating_data
from .normalize import NormalizationRefs
from .trend import ChangePoint, RULEstimate, detect_change_points, estimate_rul, ewma


@dataclass
class Prepared:
    """채점 직전까지 끝난 중간 상태 (패스 1의 결과).

    동급기 비교는 전 설비의 일 집계가 나와야 계산할 수 있으므로, 파이프라인을
    여기서 한 번 끊는다. 단일 설비 분석은 두 단계를 연달아 호출할 뿐이다.
    """

    config: Config
    dataset: Dataset
    baseline: baseline_mod.Baseline
    refs: NormalizationRefs
    daily: pd.DataFrame

    @property
    def dp_ratio(self) -> pd.Series:
        """정규화 차압의 베이스라인 대비 비율 — 동급기 비교의 기준량."""
        base = self.baseline.ref("dp_norm")
        if "dp_norm" not in self.daily.columns or not base or not pd.notna(base) or base <= 0:
            return pd.Series(dtype=float)
        return self.daily["dp_norm"] / base


@dataclass
class Analysis:
    """분석 결과 전체."""

    config: Config
    dataset: Dataset
    baseline: baseline_mod.Baseline
    refs: NormalizationRefs
    daily: pd.DataFrame
    indicators: indicators.IndicatorSet
    result: scoring.ScoreResult
    change_points: list[ChangePoint]
    cusum_stat: pd.Series
    rul: RULEstimate
    dp_limit_ratio: float

    def snapshot(self, date=None) -> scoring.ScoreSnapshot:
        return self.result.snapshot(date)

    def guidance(self, date=None) -> dict:
        return diagnose.guidance(self.snapshot(date), self.indicators, self.config)

    @property
    def score(self) -> pd.Series:
        return self.result.score

    def score_smoothed(self, span_days: float = 7) -> pd.Series:
        return ewma(self.result.score, span_days)


def analyze(
    source: str | Path | pd.DataFrame,
    config: Config | None = None,
    *,
    config_dir: str | Path | None = None,
    resample: str | None = None,
) -> Analysis:
    """운전 데이터 한 건을 끝까지 분석한다 (전처리 + 채점)."""
    return score_prepared(prepare(source, config, config_dir=config_dir, resample=resample))


def prepare(
    source: str | Path | pd.DataFrame,
    config: Config | None = None,
    *,
    config_dir: str | Path | None = None,
    resample: str | None = None,
) -> Prepared:
    """패스 1 — 로딩부터 일 집계까지. 채점 직전에서 멈춘다."""
    cfg = config or load_config(config_dir)

    ds = load_operating_data(source, cfg, resample=resample)
    ds = preprocess.clean(ds)
    ds = preprocess.mark_steady_state(ds, cfg)

    # 1단계: 시간 기준 베이스라인 구간으로 정규화 기준값을 구한다
    provisional = baseline_mod.select_baseline(ds, cfg)
    refs = normalize.resolve_refs(ds, cfg, provisional.mask)

    # 2단계: 정규화 후 같은 구간에서 기준값·조건부 모델을 확정한다
    ds = normalize.add_derived(ds, refs)
    base = baseline_mod.select_baseline(ds, cfg)
    ds = baseline_mod.apply_reference_models(ds, base)
    base = baseline_mod.select_baseline(ds, cfg)   # 잔차 컬럼 포함해 기준값 재산출

    daily = preprocess.aggregate_daily(ds)
    if daily.empty:
        raise ValueError(
            "정상 운전 구간이 없습니다. config/weights.yaml 의 steady_state 임계값 "
            "(min_flow_ratio, min_comb_temp_c)이 설비 실제 운전범위와 맞는지 확인하세요."
        )

    return Prepared(config=cfg, dataset=ds, baseline=base, refs=refs, daily=daily)


def score_prepared(
    prep: Prepared,
    *,
    peer_dp_ratio: pd.Series | None = None,
) -> Analysis:
    """패스 3 — 지표 산출부터 고장모드 판정까지.

    Args:
        peer_dp_ratio: 동일 계열 동급기의 정규화 차압 비율 중앙값. 있으면 A6 이 채점된다.
    """
    cfg, ds, base, refs, daily = prep.config, prep.dataset, prep.baseline, prep.refs, prep.daily

    ind = indicators.compute(daily, base, cfg, ds, peer_dp_ratio=peer_dp_ratio)
    result = scoring.score(ind, cfg)

    # --- 추세: 정규화 차압 비율 위에서 변화점과 잔여여유를 본다 -----------------
    trend_cfg = cfg.weights.get("trend", {})
    dp_ratio = ind.raw["A1"] if "A1" in ind.raw.columns else pd.Series(dtype=float)

    baseline_days = pd.Series(
        (daily.index >= base.start) & (daily.index < base.end), index=daily.index
    )
    # 일간 노이즈를 먼저 걷어내고 증분을 본다. 평활 없이 차분하면 계측 노이즈가
    # 증분을 지배해 변화 신호가 묻힌다.
    cusum_stat, points = detect_change_points(
        ewma(dp_ratio, float(trend_cfg.get("ewma_span_hours", 24)) / 24.0 * 7.0),
        baseline_days,
        k_sigma=float(trend_cfg.get("cusum_k_sigma", 0.5)),
        h_sigma=float(trend_cfg.get("cusum_h_sigma", 5.0)),
    )

    # 운전 한계 차압을 베이스라인 대비 비율로 환산해 RUL 을 구한다
    base_dp = base.ref("dp_bed", float("nan"))
    dp_limit_ratio = cfg.design_max_dp / base_dp if base_dp and base_dp > 0 else float("nan")
    rul = estimate_rul(
        dp_ratio,
        dp_limit_ratio,
        fit_window_days=int(trend_cfg.get("rul_fit_window_days", 60)),
        min_slope=float(trend_cfg.get("rul_min_slope_ratio_per_day", 1e-5)),
    )

    return Analysis(
        config=cfg,
        dataset=ds,
        baseline=base,
        refs=refs,
        daily=daily,
        indicators=ind,
        result=result,
        change_points=points,
        cusum_stat=cusum_stat,
        rul=rul,
        dp_limit_ratio=dp_limit_ratio,
    )
