"""설정 파일 및 운전 데이터 로딩.

현장 DCS 태그명은 사이트마다 다르므로 config/tags.yaml 의 매핑을 거쳐
시스템 내부 표준 변수명으로 바꾼 뒤 사용한다. 보유하지 않은 태그는
`Dataset.available` 에서 빠지고, 해당 태그를 요구하는 지표는 채점에서
자동 제외된 뒤 잔여 지표로 100점이 재배분된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# 프로젝트 루트 (src/rto_health/io_loader.py → 2단계 상위)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# 태그 매핑에서 단일 컬럼으로 다루는 표준 변수 (t_sector 는 별도 처리)
SCALAR_VARS = (
    "timestamp",
    "dp_bed",
    "dp_total",
    "flow",
    "fan_hz",
    "fan_amp",
    "damper_pct",
    "t_comb",
    "t_comb_sp",
    "t_in",
    "t_stack",
    "fuel_flow",
    "burner_duty",
    "voc_in",
    "voc_out",
    "ambient_temp",
)


@dataclass
class Config:
    """3종 설정 파일을 묶어 들고 다니는 컨테이너."""

    tags: dict[str, Any]
    weights: dict[str, Any]
    events: dict[str, Any]

    @property
    def equipment(self) -> dict[str, Any]:
        return self.events.get("equipment", {})

    @property
    def design_flow(self) -> float:
        return float(self.equipment.get("design_flow_cmm", 1000.0))

    @property
    def design_max_dp(self) -> float:
        return float(self.equipment.get("design_max_dp_mmh2o", 250.0))

    @property
    def media_type(self) -> str:
        return str(self.equipment.get("media_type", "honeycomb"))

    def maintenance_events(self) -> pd.DataFrame:
        """정비 이력을 날짜순 DataFrame 으로 반환."""
        rows = self.events.get("maintenance_events") or []
        if not rows:
            return pd.DataFrame(columns=["date", "type", "note"])
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        if "note" not in df.columns:
            df["note"] = ""
        return df.sort_values("date").reset_index(drop=True)


@dataclass
class Dataset:
    """표준 변수명으로 정리된 운전 데이터."""

    df: pd.DataFrame
    available: set[str] = field(default_factory=set)
    sector_cols: list[str] = field(default_factory=list)
    missing_tags: list[str] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        return all(n in self.available for n in names)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    config_dir: str | Path | None = None,
    *,
    tags: str | Path | None = None,
    weights: str | Path | None = None,
    events: str | Path | None = None,
) -> Config:
    """config 디렉토리에서 tags/weights/events 를 읽는다."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    return Config(
        tags=load_yaml(tags or base / "tags.yaml"),
        weights=load_yaml(weights or base / "weights.yaml"),
        events=load_yaml(events or base / "events.yaml"),
    )


def _sector_source_names(tag_cfg: dict[str, Any]) -> list[str]:
    """t_sector 정의에서 실제 섹터 태그명 목록을 만든다."""
    spec = tag_cfg.get("t_sector") or {}
    prefix = spec.get("source_prefix")
    if not prefix:
        return []
    count = int(spec.get("count", 12))
    return [f"{prefix}{i:02d}" for i in range(1, count + 1)]


def load_operating_data(
    source: str | Path | pd.DataFrame,
    config: Config,
    *,
    resample: str | None = None,
) -> Dataset:
    """CSV/Excel/DataFrame 을 읽어 표준 변수명 Dataset 으로 변환한다.

    Args:
        source: CSV·Excel 경로 또는 이미 읽어둔 DataFrame.
        config: load_config() 결과.
        resample: '10min' 등. 지정 시 시간 평균으로 리샘플링.
    """
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        path = Path(source)
        if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            raw = pd.read_excel(path)
        else:
            raw = pd.read_csv(path)

    tag_cfg = config.tags
    out = pd.DataFrame()
    available: set[str] = set()
    missing: list[str] = []

    for var in SCALAR_VARS:
        spec = tag_cfg.get(var) or {}
        src = spec.get("source")
        if src and src in raw.columns:
            out[var] = raw[src]
            available.add(var)
        elif spec.get("required"):
            raise ValueError(
                f"필수 태그 누락: 표준변수 '{var}' → 태그 '{src}' 를 데이터에서 찾을 수 없습니다."
            )
        else:
            missing.append(f"{var} ({src or '미지정'})")

    if "timestamp" not in out.columns:
        raise ValueError("timestamp 태그를 찾을 수 없습니다. config/tags.yaml 을 확인하세요.")

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    # --- 섹터별 온도 --------------------------------------------------------
    sector_cols: list[str] = []
    for i, src in enumerate(_sector_source_names(tag_cfg), start=1):
        if src in raw.columns:
            col = f"t_sector_{i:02d}"
            out[col] = raw[src].to_numpy()
            sector_cols.append(col)
    if sector_cols:
        available.add("t_sector")
    else:
        missing.append("t_sector (섹터별 출구온도)")

    if resample:
        out = out.set_index("timestamp").resample(resample).mean().dropna(how="all").reset_index()

    return Dataset(df=out, available=available, sector_cols=sector_cols, missing_tags=missing)
