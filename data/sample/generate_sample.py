"""물리 기반 RTO 합성 운전 데이터 생성기.

실제 운전 이력이 확보되기 전까지 개발·검증에 사용한다. 단순한 난수가 아니라
아래 물리 관계를 실제로 구현하므로, 지표 계산 로직이 올바른지 검증할 수 있다.

  · ΔP = ΔP_ref × (Q/Q_ref)^n × (T/T_ref) × 막힘계수
  · T_stack = T_comb − TER × (T_comb − T_in)
  · 보조연료 = 열손실(1−TER) − VOC 자체 발열
  · 채널링 시 섹터별 출구온도 편차 확대, 체류시간 단축으로 배출농도 상승

출력 컬럼명은 config/tags.yaml 의 `source` 태그명을 그대로 사용하므로
태그 매핑 경로까지 함께 검증된다.

사용법:
    python data/sample/generate_sample.py            # 3개 시나리오 전부 생성
    python data/sample/generate_sample.py --scenario rapid_plugging
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- 설비 기준 제원 (config/events.yaml 의 equipment 와 일치) -----------------
START = "2025-01-06"
DAYS = 365
FREQ_MIN = 10
SECTOR_COUNT = 12

Q_REF = 850.0          # 기준 풍량 CMM
T_REF_K = 313.15       # 기준 유입가스 온도 40 degC
DP_REF = 80.0          # 세정 직후 기준 차압 mmH2O
FLOW_EXPONENT = 1.15   # 허니컴 축열재 (층류 지배)
TER_CLEAN = 0.95       # 세정 직후 열회수효율
T_COMB_SP = 800.0
HZ_REF = 42.0
AMP_REF = 96.0

# bake-out 실시일 (config/events.yaml 과 동일). 막힘을 부분적으로 회복시킨다.
BAKEOUT_DAYS = [133, 259]
BAKEOUT_RECOVERY = 0.45   # 실시 시 fouling 의 45% 제거

# 계획 정지 구간 (일, 길이 시간)
SHUTDOWNS = [(90, 10), (210, 14), (300, 8)]


def _smooth_noise(rng: np.random.Generator, n: int, scale: float, span: int) -> np.ndarray:
    """저주파 랜덤워크. 공정 변동처럼 천천히 흔들리는 성분을 만든다."""
    raw = rng.normal(0.0, scale, n)
    kernel = np.ones(span) / span
    return np.convolve(raw, kernel, mode="same") * np.sqrt(span)


def _fouling_profile(scenario: str, day: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """시나리오별 막힘 진행도 f 와 채널링 정도 c 를 만든다.

    f: 0 = 세정 직후, 1.0 ≈ 설계 한계 수준의 균일 막힘
    c: 0 = 균일, 1.0 = 다수 섹터 폐쇄로 심한 편류
    """
    n = len(day)

    if scenario == "normal":
        # 1년간 완만한 자연 오염. bake-out 으로 대부분 회복.
        f = 0.10 * (day / DAYS)
        c = np.full(n, 0.01)

    elif scenario == "gradual_fouling":
        # 미세 분진이 꾸준히 누적. 후반으로 갈수록 가속(유로 축소 → 유속 증가 → 포집 증가).
        x = day / DAYS
        f = 0.40 * (x ** 1.40)
        c = 0.12 * (x ** 2.0)

    elif scenario == "rapid_plugging":
        # 200일 근처 공정 이상으로 고농도 VOC 유입 → 폴리머 응축 → 급격한 국부 폐쇄.
        onset = 205.0
        x = np.clip((day - onset) / (DAYS - onset), 0.0, None)
        f = 0.09 * (day / DAYS) + 1.25 * (x ** 1.25)
        c = 0.02 + 0.80 * (x ** 1.1)

    else:
        raise ValueError(f"알 수 없는 시나리오: {scenario}")

    # bake-out 회복 반영 (누적 감소분을 이후 구간에 적용)
    f = f.copy()
    for bd in BAKEOUT_DAYS:
        mask = day >= bd
        if not mask.any():
            continue
        level = f[mask][0]
        f[mask] -= level * BAKEOUT_RECOVERY

    f = np.clip(f + _smooth_noise(rng, n, 0.004, 288), 0.0, None)
    c = np.clip(c + _smooth_noise(rng, n, 0.003, 288), 0.0, None)
    return f, c


def build_scenario(scenario: str, seed: int = 42) -> pd.DataFrame:
    """한 시나리오의 전체 운전 데이터를 생성한다."""
    rng = np.random.default_rng(seed)

    ts = pd.date_range(START, periods=DAYS * 24 * 60 // FREQ_MIN, freq=f"{FREQ_MIN}min")
    n = len(ts)
    day = np.arange(n) * FREQ_MIN / 1440.0
    hour_of_day = ts.hour + ts.minute / 60.0
    is_weekend = ts.dayofweek >= 5

    # --- 외기온: 계절 + 일교차 ---------------------------------------------
    ambient = (
        13.0
        + 12.0 * np.sin(2 * np.pi * (day - 110) / 365.0)
        + 5.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
        + rng.normal(0, 1.0, n)
    )

    # --- 생산 부하(풍량): 주야/주말 패턴 + 완만한 변동 ------------------------
    load = 1.0 - 0.18 * np.sin(2 * np.pi * (hour_of_day - 4) / 24.0)
    load = np.where(is_weekend, load * 0.72, load)
    load = load * (1.0 + _smooth_noise(rng, n, 0.02, 144))
    flow = Q_REF * np.clip(load, 0.3, 1.35)

    # --- 정지 구간 ----------------------------------------------------------
    running = np.ones(n, dtype=bool)
    for start_day, dur_h in SHUTDOWNS:
        s = int(start_day * 1440 / FREQ_MIN)
        e = s + int(dur_h * 60 / FREQ_MIN)
        running[s:e] = False
        # 정지 후 1시간은 승온 과도구간
        running_warm_end = e + int(60 / FREQ_MIN)
        flow[s:running_warm_end] *= 0.15

    flow = np.where(running, flow, rng.uniform(0, 40, n))

    # --- 유입가스 온도 / VOC 농도 --------------------------------------------
    t_in = ambient + 22.0 + _smooth_noise(rng, n, 1.5, 72)
    voc_in = np.clip(300.0 + 90.0 * (load - 1.0) * 3 + _smooth_noise(rng, n, 25.0, 216), 20, None)

    f, c = _fouling_profile(scenario, day, rng)

    # rapid_plugging: 막힘을 유발한 VOC 스파이크를 실제로 데이터에 넣는다
    if scenario == "rapid_plugging":
        for spike_day in (198, 201, 204):
            s = int(spike_day * 1440 / FREQ_MIN)
            e = s + int(14 * 60 / FREQ_MIN)
            voc_in[s:e] *= rng.uniform(3.2, 4.4)

    # --- 차압: 물리식 그대로 ------------------------------------------------
    # 막힘계수: 균일 막힘은 선형, 채널링은 유효 단면 축소로 제곱 효과
    plug_factor = 1.0 + 1.55 * f + 1.30 * c**2
    dp_clean = DP_REF * (flow / Q_REF) ** FLOW_EXPONENT * ((t_in + 273.15) / T_REF_K)
    dp_bed = dp_clean * plug_factor
    dp_bed = np.where(running, dp_bed, dp_bed * 0.05)
    # 채널링 시 맥동 증가
    dp_bed = dp_bed * (1.0 + rng.normal(0, 0.006 + 0.085 * c, n))
    dp_total = dp_bed + 35.0 * (flow / Q_REF) ** 1.8 + rng.normal(0, 1.2, n)

    # --- 송풍기: 높아진 저항을 이겨내기 위해 회전수/개도 상승 -------------------
    fan_hz = np.clip(HZ_REF * (flow / Q_REF) * plug_factor**0.32 + rng.normal(0, 0.15, n), 0.0, 60.0)
    fan_amp = np.clip(AMP_REF * (fan_hz / HZ_REF) ** 2.1 + rng.normal(0, 0.8, n), 0.0, None)
    damper_pct = np.clip(45.0 + 42.0 * (plug_factor - 1.0) + rng.normal(0, 0.8, n), 5, 100)

    # --- 열회수효율과 온도 프로파일 ------------------------------------------
    ter = TER_CLEAN - 0.055 * f - 0.16 * c
    ter = np.clip(ter + rng.normal(0, 0.0018, n), 0.55, 0.99)

    t_comb_sp = np.full(n, T_COMB_SP)
    instability = 1.6 + 22.0 * c + 4.0 * f
    t_comb = t_comb_sp + rng.normal(0, instability, n)
    t_comb = np.where(running, t_comb, np.clip(t_comb - 600, 60, None))

    t_stack = t_comb - ter * (t_comb - t_in)

    # 섹터별 출구온도: 채널링 시 특정 섹터군이 냉/온으로 갈린다
    sector_bias = np.array([-1.0, -0.9, -1.15, 0.35, 0.5, 0.62, 0.7, 0.45, 0.3, -0.35, -0.6, 0.28])
    sectors = {}
    for i in range(SECTOR_COUNT):
        offset = sector_bias[i] * (58.0 * c) + rng.normal(0, 2.2, n)
        sectors[f"TT_SECTOR_{i + 1:02d}"] = t_stack + 26.0 + offset

    # --- 연료: 열수지 (손실 보충 − VOC 자체 발열) -----------------------------
    heat_loss = flow * (1.0 - ter) * (t_comb - t_in) / 1000.0
    voc_heat = 0.042 * flow * voc_in / 1000.0
    fuel_flow = np.clip(heat_loss - voc_heat + rng.normal(0, 0.6, n), 0.0, None)
    fuel_flow = np.where(running, fuel_flow, 0.0)
    burner_duty = np.clip(fuel_flow / 55.0 * 100.0 + rng.normal(0, 0.9, n), 0, 100)

    # --- 배출농도: 편류로 체류시간이 짧아지면 상승 ----------------------------
    voc_out = np.clip(7.5 * (1.0 + 2.8 * c) * (1.0 + 0.25 * f) + rng.normal(0, 0.6, n), 0.2, None)

    df = pd.DataFrame(
        {
            "TIMESTAMP": ts,
            "RTO_DP_BED": dp_bed.round(2),
            "RTO_DP_TOTAL": dp_total.round(2),
            "RTO_FLOW": flow.round(1),
            "FAN_INV_HZ": fan_hz.round(2),
            "FAN_CURRENT": fan_amp.round(1),
            "DAMPER_OPEN": damper_pct.round(1),
            "TIC_COMB_PV": t_comb.round(1),
            "TIC_COMB_SP": t_comb_sp.round(1),
            "TT_INLET": t_in.round(1),
            "TT_STACK": t_stack.round(1),
            "FUEL_LNG_FLOW": fuel_flow.round(2),
            "BURNER_DUTY": burner_duty.round(1),
            "VOC_INLET": voc_in.round(1),
            "TMS_OUTLET": voc_out.round(2),
            "AMBIENT_TEMP": ambient.round(1),
        }
    )
    for name, values in sectors.items():
        df[name] = values.round(1)

    return df


SCENARIOS = ("normal", "gradual_fouling", "rapid_plugging")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTO 합성 운전 데이터 생성")
    parser.add_argument("--scenario", choices=SCENARIOS, help="생략 시 전체 생성")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=str(Path(__file__).parent))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    targets = [args.scenario] if args.scenario else list(SCENARIOS)

    for name in targets:
        df = build_scenario(name, seed=args.seed)
        path = outdir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"생성 완료: {path}  ({len(df):,} 행 × {len(df.columns)} 열)")


if __name__ == "__main__":
    main()
