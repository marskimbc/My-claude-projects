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

    # 데이터 구간 안에 있는 가장 최근의 리셋성 정비를 찾는다
    events = config.maintenance_events()
    reset_date: pd.Timestamp | None = None
    reset_type: str | None = None
    if not events.empty:
        resets = events[events["type"].isin(reset_types)]
        resets = resets[(resets["date"] >= data_start - pd.Timedelta(days=1)) & (resets["date"] <= data_end)]
        if not resets.empty:
            reset_date = resets["date"].iloc[-1]
            reset_type = str(resets["type"].iloc[-1])

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

    return Baseline(
        start=start,
        end=end,
        mask=mask,
        refs=refs,
        models={k: v for k, v in models.items() if v is not None},
        reset_date=reset_date,
        reset_type=reset_type,
        n_samples=int(mask.sum()),
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
