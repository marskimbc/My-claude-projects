"""베이스라인(정상 기준) 구간 선정과 조건부 기준모델 학습.

"차압이 100 mmH2O 다"는 그 자체로 아무 의미가 없다. **세정 직후 같은 조건에서
얼마였는가**와 비교해야 막힘 정도를 알 수 있다. 따라서 마지막 세정/교체 직후
구간을 정상 기준으로 잡고, 이후 데이터를 이 기준과 비교한다.

또한 배출온도·팬 주파수·연료량처럼 운전조건(풍량·유입온도·VOC농도)에 강하게
의존하는 값은 단순 절대비교 대신 **조건부 기준모델의 잔차**를 쓴다. 같은 조건에서
얼마나 벗어났는지를 보는 편이 오탐이 훨씬 적다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io_loader import Config, Dataset


@dataclass
class LinearModel:
    """절편 포함 최소자승 선형모델. 조건부 기대값 산출용."""

    features: list[str]
    coef: np.ndarray
    sigma: float
    n_obs: int

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = np.column_stack([np.ones(len(df))] + [df[f].to_numpy(dtype=float) for f in self.features])
        return x @ self.coef


@dataclass
class Baseline:
    """정상 기준 구간과 그로부터 뽑은 기준값들."""

    start: pd.Timestamp
    end: pd.Timestamp
    mask: pd.Series
    refs: dict[str, float] = field(default_factory=dict)
    models: dict[str, LinearModel] = field(default_factory=dict)
    reset_date: pd.Timestamp | None = None
    reset_type: str | None = None
    n_samples: int = 0
    source: str = "auto"        # 'period' | 'manual' | 'maintenance' | 'auto'
    warning: str | None = None

    #: 베이스라인 출처가 신뢰할 만한지 — 'auto' 는 데이터 앞부분을 정상으로 **가정**한 것이다
    @property
    def is_assumed(self) -> bool:
        return self.source == "auto"

    @property
    def source_label(self) -> str:
        return {
            "period": "구간 지정",
            "manual": "수동 입력",
            "maintenance": "정비 이력 기준",
            "auto": "⚠️ 자동 추정",
        }.get(self.source, self.source)

    def ref(self, key: str, default: float = float("nan")) -> float:
        value = self.refs.get(key, default)
        return default if value is None or not np.isfinite(value) else float(value)


def _fit(df: pd.DataFrame, target: str, features: list[str], min_obs: int = 100) -> LinearModel | None:
    """최소자승 회귀. 표본이 모자라거나 컬럼이 없으면 None."""
    cols = [target] + features
    if any(c not in df.columns for c in cols):
        return None
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < min_obs:
        return None

    x = np.column_stack([np.ones(len(sub))] + [sub[f].to_numpy(dtype=float) for f in features])
    y = sub[target].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    sigma = float(np.std(resid, ddof=len(features) + 1))
    return LinearModel(features=features, coef=coef, sigma=max(sigma, 1e-9), n_obs=len(sub))


def select_baseline(ds: Dataset, config: Config) -> Baseline:
    """마지막 세정/교체 이후 N일을 정상 기준 구간으로 잡는다."""
    cfg = config.weights.get("baseline", {})
    days = float(cfg.get("days_after_maintenance", 14))
    min_samples = int(cfg.get("min_samples", 300))
    reset_types = set(cfg.get("reset_event_types", ["full_clean", "media_replace"]))

    df = ds.df
    steady = df.get("is_steady", pd.Series(True, index=df.index)).fillna(False)
    data_start, data_end = df["timestamp"].min(), df["timestamp"].max()

    # --- ② 수동 입력이 있으면 데이터와 무관하게 그 값을 기준으로 쓴다 -----------
    # 데이터에 깨끗한 구간이 아예 없는 설비(누적관리를 오염 상태에서 시작한 경우)를 위한 경로.
    manual = (config.baseline_spec or {}).get("manual")
    if manual:
        return _manual_baseline(ds, manual, data_start)

    # --- ① 구간 지정이 있으면 그 구간을 정상으로 본다 --------------------------
    period = (config.baseline_spec or {}).get("period")
    source = "auto"
    reset_date: pd.Timestamp | None = None
    reset_type: str | None = None

    if period and period.get("start"):
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"]) if period.get("end") else start + pd.Timedelta(days=days)
        source = "period"
    else:
        # --- ③ 정비 이력 기준, 없으면 데이터 앞부분(가정) ----------------------
        events = config.maintenance_events()
        if not events.empty:
            resets = events[events["type"].isin(reset_types)]
            resets = resets[
                (resets["date"] >= data_start - pd.Timedelta(days=1)) & (resets["date"] <= data_end)
            ]
            if not resets.empty:
                reset_date = resets["date"].iloc[-1]
                reset_type = str(resets["type"].iloc[-1])
                source = "maintenance"

        start = max(reset_date, data_start) if reset_date is not None else data_start
        end = start + pd.Timedelta(days=days)

    mask = steady & (df["timestamp"] >= start) & (df["timestamp"] < end)

    # 정지·저부하가 겹쳐 표본이 모자라면 창을 넓혀 확보한다
    if int(mask.sum()) < min_samples:
        after = steady & (df["timestamp"] >= start)
        idx = df.index[after][:min_samples]
        if len(idx) > 0:
            mask = pd.Series(False, index=df.index)
            mask.loc[idx] = True
            end = df.loc[idx[-1], "timestamp"]

    base_df = df.loc[mask]
    refs = {
        key: float(base_df[key].median())
        for key in (
            "dp_norm", "dp_bed", "flow", "t_in", "t_stack", "t_comb", "ter",
            "fuel_intensity", "fuel_flow", "voc_in", "voc_out", "burner_duty",
            "fan_hz", "fan_amp", "damper_pct", "sector_temp_std",
        )
        if key in base_df.columns and base_df[key].notna().any()
    }

    # 조건부 기준모델 — 같은 운전조건에서의 기대값을 학습한다
    models: dict[str, LinearModel | None] = {
        "stack_temp": _fit(base_df, "t_stack", ["flow", "t_in", "t_comb"]),
        "fan_hz": _fit(base_df, "fan_hz", ["flow"]),
        "burner_duty": _fit(base_df, "burner_duty", ["flow", "voc_in", "t_in"]),
        "fuel": _fit(base_df, "fuel_flow", ["flow", "voc_in", "t_in"]),
    }

    warning = None
    if source == "auto":
        warning = (
            "정비 이력도 지정 구간도 없어 데이터 앞부분을 정상으로 가정했습니다. "
            "이미 오염된 상태에서 수집을 시작했다면 막힘이 과소평가됩니다 — "
            "config/fleet.yaml 의 baselines 에 정상 구간(period) 또는 설계값(manual)을 지정하고, "
            "그 전까지는 동급기 대비 지표(A6)를 함께 보십시오."
        )

    return Baseline(
        start=start,
        end=end,
        mask=mask,
        refs=refs,
        models={k: v for k, v in models.items() if v is not None},
        reset_date=reset_date,
        reset_type=reset_type,
        n_samples=int(mask.sum()),
        source=source,
        warning=warning,
    )


def _manual_baseline(ds: Dataset, manual: dict, data_start: pd.Timestamp) -> Baseline:
    """설계값·시운전값을 기준으로 삼는다.

    조건부 기준모델은 학습할 데이터가 없으므로 만들지 않는다. 해당 잔차 기반 지표
    (B2 배출온도, A3 팬부하, C2 버너듀티, C3 VOC-연료)는 자동으로 채점에서 빠지고
    남은 지표가 100점을 나눠 갖는다.
    """
    refs = {k: float(v) for k, v in manual.items() if isinstance(v, (int, float))}
    if "ter" not in refs and {"t_comb", "t_in", "t_stack"} <= set(refs):
        span = refs["t_comb"] - refs["t_in"]
        if span > 0:
            refs["ter"] = (refs["t_comb"] - refs["t_stack"]) / span
    if "dp_norm" not in refs and "dp_bed" in refs:
        # 기준 조건 자체가 이 값이므로 정규화 차압도 동일하게 둔다
        refs["dp_norm"] = refs["dp_bed"]
    if "fuel_intensity" not in refs and {"fuel_flow", "flow"} <= set(refs) and refs["flow"] > 0:
        refs["fuel_intensity"] = refs["fuel_flow"] / refs["flow"]

    return Baseline(
        start=data_start,
        end=data_start,
        mask=pd.Series(False, index=ds.df.index),
        refs=refs,
        models={},
        n_samples=0,
        source="manual",
        warning=(
            "설계값/시운전값을 기준으로 채점 중입니다. 실측 구간이 없어 조건부 기준모델을 "
            "학습하지 못했으므로 잔차 기반 지표(A3·B2·C2·C3)는 채점에서 제외됩니다."
        ),
    )


def apply_reference_models(ds: Dataset, baseline: Baseline) -> Dataset:
    """조건부 기준모델의 잔차/비율을 원데이터에 추가한다.

    추가 컬럼:
        stack_temp_residual — 같은 조건 대비 배출온도 상승분 (degC)
        fan_hz_ratio        — 같은 풍량 대비 인버터 주파수 비율
        burner_duty_ratio   — 같은 부하 대비 버너 가동률 비율
        fuel_residual       — 연료 사용량 잔차의 표준화값 (sigma)
    """
    df = ds.df.copy()

    model = baseline.models.get("stack_temp")
    if model is not None and all(c in df.columns for c in model.features):
        df["stack_temp_residual"] = df["t_stack"] - model.predict(df)

    model = baseline.models.get("fan_hz")
    if model is not None and all(c in df.columns for c in model.features):
        expected = model.predict(df)
        df["fan_hz_ratio"] = np.where(expected > 1e-6, df["fan_hz"] / expected, np.nan)

    model = baseline.models.get("burner_duty")
    if model is not None and all(c in df.columns for c in model.features):
        expected = model.predict(df)
        df["burner_duty_ratio"] = np.where(expected > 1.0, df["burner_duty"] / expected, np.nan)

    model = baseline.models.get("fuel")
    if model is not None and all(c in df.columns for c in model.features):
        df["fuel_residual"] = (df["fuel_flow"] - model.predict(df)) / model.sigma

    return Dataset(
        df=df,
        available=set(ds.available),
        sector_cols=list(ds.sector_cols),
        missing_tags=list(ds.missing_tags),
    )
