"""결측·이상치 처리, 정상 운전 구간 판정, 일 단위 집계.

막힘 지표는 반드시 **정상 운전 구간에서만** 계산해야 한다. 기동/정지/저부하
구간의 차압이나 온도를 섞으면 막힘과 무관한 변동이 지표를 흔들어 오탐이 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io_loader import Config, Dataset

# 계측기 고장·통신 오류로 나오는 명백한 비물리값을 걸러내기 위한 범위
PHYSICAL_RANGE: dict[str, tuple[float, float]] = {
    "dp_bed": (0.0, 1000.0),
    "dp_total": (0.0, 1500.0),
    "flow": (0.0, 5000.0),
    "fan_hz": (0.0, 70.0),
    "fan_amp": (0.0, 1000.0),
    "damper_pct": (0.0, 100.0),
    "t_comb": (-20.0, 1300.0),
    "t_comb_sp": (0.0, 1300.0),
    "t_in": (-40.0, 400.0),
    "t_stack": (-40.0, 900.0),
    "fuel_flow": (0.0, 2000.0),
    "burner_duty": (0.0, 100.0),
    "voc_in": (0.0, 50000.0),
    "voc_out": (0.0, 5000.0),
    "ambient_temp": (-40.0, 60.0),
}

# 하루를 유효한 관측일로 인정하기 위한 최소 정상운전 샘플 수(시간 기준)
MIN_STEADY_HOURS_PER_DAY = 4.0


def clean(ds: Dataset, *, max_gap_minutes: int = 30) -> Dataset:
    """비물리값을 NaN 으로 만들고 짧은 결측만 보간한다.

    긴 결측은 채우지 않는다 — 정지 구간을 억지로 메우면 없는 운전을 만들어낸다.
    """
    df = ds.df.copy()
    for col, (lo, hi) in PHYSICAL_RANGE.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    for col in ds.sector_cols:
        df.loc[(df[col] < -40) | (df[col] > 900), col] = np.nan

    step = _sample_step_minutes(df)
    limit = max(1, int(round(max_gap_minutes / step)))
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].interpolate(limit=limit, limit_area="inside")

    return Dataset(
        df=df,
        available=set(ds.available),
        sector_cols=list(ds.sector_cols),
        missing_tags=list(ds.missing_tags),
    )


def _sample_step_minutes(df: pd.DataFrame) -> float:
    """샘플링 주기(분)를 데이터에서 추정한다."""
    if len(df) < 2:
        return 10.0
    delta = df["timestamp"].diff().dropna().median()
    minutes = delta.total_seconds() / 60.0
    return minutes if minutes > 0 else 10.0


def mark_steady_state(ds: Dataset, config: Config) -> Dataset:
    """정상 운전 구간 여부를 `is_steady` 컬럼으로 표시한다."""
    cfg = config.weights.get("steady_state", {})
    df = ds.df

    min_flow = float(cfg.get("min_flow_ratio", 0.35)) * config.design_flow
    min_comb = float(cfg.get("min_comb_temp_c", 700.0))
    max_dev = float(cfg.get("max_comb_temp_dev_c", 80.0))
    warmup_min = float(cfg.get("warmup_minutes", 60.0))

    running = (df["flow"] >= min_flow) & (df["t_comb"] >= min_comb)
    if "dp_bed" in df.columns:
        running &= df["dp_bed"] > 0
    running = running.fillna(False)

    # 승온 과도구간 제외: 가동 블록의 앞부분 warmup_minutes 를 버린다
    step = _sample_step_minutes(df)
    warmup_samples = int(np.ceil(warmup_min / step))
    steady = running.to_numpy().copy()
    if warmup_samples > 0:
        block_id = (running != running.shift(1)).cumsum()
        pos_in_block = df.groupby(block_id).cumcount().to_numpy()
        steady &= pos_in_block >= warmup_samples

    # 설정치 대비 편차가 크면 제어 과도상태로 보고 제외
    if "t_comb_sp" in df.columns:
        dev_ok = (df["t_comb"] - df["t_comb_sp"]).abs() <= max_dev
        steady &= dev_ok.fillna(False).to_numpy()

    out = df.copy()
    out["is_steady"] = steady
    return Dataset(
        df=out,
        available=set(ds.available),
        sector_cols=list(ds.sector_cols),
        missing_tags=list(ds.missing_tags),
    )


def aggregate_daily(ds: Dataset) -> pd.DataFrame:
    """정상 운전 샘플만 모아 일 단위 특성으로 집계한다.

    지표는 일 단위 시계열 위에서 계산한다. 10분 원데이터로 바로 채점하면
    운전 노이즈가 그대로 점수에 실린다.

    반환 컬럼:
        중앙값 계열 — dp_norm, ter, fuel_intensity 등
        변동성 계열 — dp_cv(차압 변동계수), comb_instability(연소실 제어 편차)
        적산 계열   — runtime_hours, voc_load(∫ VOC×풍량 dt)
    """
    df = ds.df
    steady = df[df.get("is_steady", pd.Series(True, index=df.index))].copy()
    if steady.empty:
        return pd.DataFrame()

    step_h = _sample_step_minutes(df) / 60.0
    steady["date"] = steady["timestamp"].dt.normalize()

    median_cols = [
        c
        for c in (
            "dp_bed", "dp_total", "dp_norm", "flow", "fan_hz", "fan_amp", "damper_pct",
            "t_comb", "t_in", "t_stack", "fuel_flow", "burner_duty", "voc_in", "voc_out",
            "ambient_temp", "ter", "fuel_intensity", "sector_temp_std",
            "stack_temp_residual", "fan_hz_ratio", "burner_duty_ratio", "fuel_residual",
        )
        if c in steady.columns
    ]

    grouped = steady.groupby("date")
    daily = grouped[median_cols].median()
    daily["n_samples"] = grouped.size()
    daily["runtime_hours"] = daily["n_samples"] * step_h

    # 일중 변동성 — 국부 폐쇄·채널링은 평균이 아니라 흔들림으로 먼저 나타난다.
    # 반드시 정규화 차압으로 계산한다. 원값은 하루 안에서도 생산량·온도를 따라
    # 오르내리므로, 원값 변동계수는 막힘이 아니라 생산 패턴을 측정하게 된다.
    dp_var_col = "dp_norm" if "dp_norm" in steady.columns else "dp_bed"
    if dp_var_col in steady.columns:
        dp_stats = grouped[dp_var_col].agg(["mean", "std"])
        daily["dp_cv"] = (dp_stats["std"] / dp_stats["mean"]).replace([np.inf, -np.inf], np.nan)
    if "t_comb" in steady.columns:
        if "t_comb_sp" in steady.columns:
            steady["_comb_dev"] = steady["t_comb"] - steady["t_comb_sp"]
        else:
            steady["_comb_dev"] = steady["t_comb"] - steady["t_comb"].median()
        daily["comb_instability"] = steady.groupby("date")["_comb_dev"].std()

    # 누적 오염 부하 (일별 증분)
    if {"voc_in", "flow"} <= set(steady.columns):
        steady["_voc_load"] = steady["voc_in"] * steady["flow"] * step_h
        daily["voc_load"] = steady.groupby("date")["_voc_load"].sum()

    # 관측이 너무 적은 날은 통계가 불안정하므로 제외
    daily = daily[daily["runtime_hours"] >= MIN_STEADY_HOURS_PER_DAY]
    return daily.sort_index()
