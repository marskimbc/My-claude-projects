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
    """설정 파일을 묶어 들고 다니는 컨테이너."""

    tags: dict[str, Any]
    weights: dict[str, Any]
    events: dict[str, Any]
    fleet: dict[str, Any] = field(default_factory=dict)   # fleet.yaml (20대 레지스트리)

    @property
    def equipment(self) -> dict[str, Any]:
        return self.events.get("equipment", {})

    @property
    def baseline_spec(self) -> dict[str, Any]:
        """설비별 베이스라인 지정 (period / manual). fleet 경로에서만 채워진다."""
        return self.events.get("baseline") or {}

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
        if isinstance(rows, dict):   # fleet.yaml 은 설비별 dict — 단일 설비 경로에서는 비운다
            rows = []
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
    fleet: str | Path | None = None,
) -> Config:
    """config 디렉토리에서 tags/weights/events(/fleet) 를 읽는다.

    fleet.yaml 은 없어도 된다 — 없으면 단일 설비 모드로 동작한다.
    """
    base = Path(config_dir) if config_dir else CONFIG_DIR
    fleet_path = Path(fleet) if fleet else base / "fleet.yaml"
    return Config(
        tags=load_yaml(tags or base / "tags.yaml"),
        weights=load_yaml(weights or base / "weights.yaml"),
        events=load_yaml(events or base / "events.yaml"),
        fleet=load_yaml(fleet_path) if fleet_path.exists() else {},
    )


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def read_frames(
    source: str | Path | list[str | Path],
    *,
    timestamp_col: str = "TIMESTAMP",
) -> pd.DataFrame:
    """파일 하나 · 파일 목록 · 폴더를 받아 하나의 프레임으로 병합한다.

    1시간 단위로 데이터를 계속 누적하는 운영을 전제로 한다. 월별 export 를 폴더에
    쌓아 두면 그대로 읽히고, 구간이 겹치면 **나중 파일이 이깁니다**(재추출본 우선).
    """
    if isinstance(source, (list, tuple)):
        paths = [Path(p) for p in source]
    else:
        path = Path(source)
        if path.is_dir():
            paths = sorted(
                p for p in path.iterdir()
                if p.suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xls")
            )
            if not paths:
                raise ValueError(f"'{path}' 안에 읽을 수 있는 데이터 파일이 없습니다.")
        else:
            paths = [path]

    frames = [_read_one(p) for p in paths]
    if len(frames) == 1:
        return frames[0]

    merged = pd.concat(frames, ignore_index=True)
    if timestamp_col in merged.columns:
        merged[timestamp_col] = pd.to_datetime(merged[timestamp_col])
        merged = (
            merged.sort_values(timestamp_col)
            .drop_duplicates(subset=[timestamp_col], keep="last")
            .reset_index(drop=True)
        )
    return merged


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
