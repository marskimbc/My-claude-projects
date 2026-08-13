"""RTO 20대 통합 가상 운전 데이터 생성기.

`generate_sample.py` 의 물리 모델을 그대로 재사용한다(물리식이 두 벌로 갈라지지
않게 하기 위함). 설비마다 풍량·기준차압·VOC 농도만 사양에서 가져다 쓰고,
컬럼명은 **config/tags.yaml 의 fleet 규칙으로 직접 생성**한다 — 생성기와 로더가
같은 매핑을 쓰므로 둘이 어긋날 수가 없다.

  · 1시간 간격 (실제 운영에서 시간 단위로 누적할 예정)
  · VRT 유기배기: 고농도 VOC → 오염 빠름
  · ORT 냄새배기: 저농도·대풍량 → 오염 느림

시연·검증을 위해 각 상태를 심어 둔다.

    VRT-A31103C   단독 막힘 (형제 4대 정상)  → A6 반응 → peer_outlier
    VRT-A31105 전체  계열 동반 열화            → A6 조용, A1 반응 → fleet_wide_fouling
    ORT-A31102A   급성 막힘                   → E등급 · 짧은 RUL
    VRT-A31101B   씰 누설 (차압 정상·효율 저하) → 막힘 아님으로 구분
    나머지 12대    정상

사용법:
    python data/sample/generate_fleet_sample.py
    python data/sample/generate_fleet_sample.py --days 180 --out /tmp/fleet.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rto_health import fleet as fleet_mod  # noqa: E402
from rto_health.io_loader import load_config  # noqa: E402

# 정상이 아닌 설비만 적는다. {설비: (시나리오, 심각도)}
FLEET_PLAN: dict[str, tuple[str, float]] = {
    # 단독 이상 — 같은 계열 A/B/D/E 는 정상이라 동급기 비교로 즉시 드러난다
    "VRT-A31103C": ("gradual_fouling", 1.9),
    # 계열 전체 동반 열화 — 서로 비슷해서 동급기 비교로는 안 보이고 A1 으로만 보인다
    "VRT-A31105A": ("gradual_fouling", 0.95),
    "VRT-A31105B": ("gradual_fouling", 1.05),
    "VRT-A31105C": ("gradual_fouling", 1.00),
    "VRT-A31105D": ("gradual_fouling", 0.90),
    "VRT-A31105E": ("gradual_fouling", 1.10),
    # 급성 막힘
    "ORT-A31102A": ("rapid_plugging", 1.0),
    # 씰 누설 — 차압은 정상인데 효율만 떨어진다
    "VRT-A31101B": ("seal_leak", 1.0),
}

DEFAULT_SCENARIO = ("normal", 1.0)

# 운전 한계 차압 대비 세정 직후 기준 차압의 비 (limit_ratio ≈ 3.1 이 되도록)
DP_REF_DIVISOR = 3.1


def _load_single_generator():
    """generate_sample.py 를 모듈로 불러온다 (패키지가 아니므로 직접 로드)."""
    path = Path(__file__).parent / "generate_sample.py"
    spec = importlib.util.spec_from_file_location("generate_sample", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_fleet(
    *,
    days: int = 365,
    freq_min: int = 60,
    seed: int = 42,
    plan: dict[str, tuple[str, float]] | None = None,
    config_dir: str | Path | None = None,
) -> pd.DataFrame:
    """20대 전체를 한 프레임으로 생성한다."""
    gen = _load_single_generator()
    cfg = load_config(config_dir or PROJECT_ROOT / "config")
    specs = fleet_mod.load_fleet(config_dir or PROJECT_ROOT / "config")
    if not specs:
        raise ValueError("config/fleet.yaml 에서 설비를 찾지 못했습니다.")

    plan = FLEET_PLAN if plan is None else plan
    frames: list[pd.DataFrame] = []
    timestamp: pd.Series | None = None

    for i, spec in enumerate(specs):
        scenario, severity = plan.get(spec.equipment_id, DEFAULT_SCENARIO)
        design_dp = float(spec.spec.get("design_max_dp_mmh2o", 250.0))

        unit = gen.build_scenario(
            scenario,
            seed=seed + i * 101,           # 설비마다 독립적인 노이즈
            freq_min=freq_min,
            days=days,
            q_ref=spec.design_flow,
            dp_ref=design_dp / DP_REF_DIVISOR,
            voc_base=float(spec.spec.get("design_voc_ppm", 300.0)),
            severity=severity,
        )

        if timestamp is None:
            timestamp = unit["TIMESTAMP"]

        # {단일모드 이름 → 실제 fleet 컬럼명} 으로 뒤집어 쓴다.
        # 로더가 쓰는 것과 같은 매핑이므로 생성기와 로더가 어긋날 수 없다.
        rename = {
            single: actual
            for actual, single in fleet_mod.expected_columns(spec, cfg).items()
        }
        unit = unit.drop(columns=["TIMESTAMP"]).rename(columns=rename)
        # 매핑되지 않은 컬럼은 버린다 (로더가 어차피 읽지 않는다)
        frames.append(unit[[c for c in unit.columns if c in rename.values()]])

    shared_ts = (cfg.tags.get("fleet") or {}).get("shared", {}).get("timestamp", "TIMESTAMP")
    out = pd.concat([timestamp.rename(shared_ts)] + frames, axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RTO 20대 통합 가상 운전 데이터 생성")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--freq-min", type=int, default=60, help="샘플링 주기(분). 기본 1시간")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(Path(__file__).parent / "fleet.csv"))
    args = parser.parse_args()

    df = build_fleet(days=args.days, freq_min=args.freq_min, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"생성 완료: {out}")
    print(f"  {len(df):,} 행 × {len(df.columns):,} 열 · {size_mb:.1f} MB")
    print(f"  {args.days}일 · {args.freq_min}분 간격 · 설비 20대")
    print("\n심어둔 상태:")
    for eq_id, (scenario, sev) in sorted(FLEET_PLAN.items()):
        print(f"  {eq_id:14s} {scenario:16s} (심각도 {sev})")


if __name__ == "__main__":
    main()
