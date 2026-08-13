"""설비 레지스트리와 20대 통합 분석.

단일 설비 분석(`pipeline.analyze`)은 그대로 두고, 그 위에 fleet 계층을 얹는다.
설비별로는 기존 경로를 그대로 재사용하되 두 가지가 추가된다.

1. **설비 사양 상속** — overrides > series > types > defaults 순으로 해석해
   계열 하나만 적어도 호기 전체가 전개된다.
2. **동급기 상호비교** — 같은 계열의 정규화 차압 중앙값과 비교한다. 이 축이
   중요한 이유는 **베이스라인이 오염돼 있어도 유효**하기 때문이다. 자기 베이스라인
   대비 지표(A1)는 기준 자체가 틀리면 무력하지만, 옆 호기와의 비교는 영향받지 않는다.
   그래서 분석이 2패스로 나뉜다 — 전 설비를 일 집계까지 돌린 뒤에야 peer 를 구할 수 있다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_loader import CONFIG_DIR, SCALAR_VARS, Config, load_yaml

# 설비 ID 에서 계열과 호기를 떼어내는 규칙 (VRT-A31101A → VRT-A31101 + A)
EQUIPMENT_ID_RE = re.compile(r"^(?P<series>.+?)(?P<unit>[A-Z])$")

# 설비 사양으로 인정하는 키 (상속 대상)
SPEC_KEYS = (
    "type", "label", "media_type", "sector_count",
    "design_flow_cmm", "design_max_dp_mmh2o", "design_voc_ppm",
    "recommended_cleaning_interval_h", "dominant_fouling",
)


@dataclass
class EquipmentSpec:
    """상속이 모두 해석된 설비 한 대의 사양."""

    equipment_id: str          # VRT-A31101A
    series_id: str             # VRT-A31101
    unit: str                  # A
    type_key: str              # VRT | ORT
    spec: dict[str, Any] = field(default_factory=dict)
    maintenance_events: list[dict] = field(default_factory=list)
    baseline_spec: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return str(self.spec.get("label", self.type_key))

    @property
    def design_flow(self) -> float:
        return float(self.spec.get("design_flow_cmm", 1000.0))

    def to_events_block(self) -> dict[str, Any]:
        """기존 단일 설비 경로가 먹을 수 있는 events dict 로 변환한다."""
        equipment = dict(self.spec)
        equipment["name"] = self.equipment_id
        # 세정 주기당 설계 누적 VOC 부하 ≈ 설계농도 × 설계풍량 × 세정주기
        if "design_voc_load_ppm_cmm_h" not in equipment:
            voc = float(equipment.get("design_voc_ppm", 300.0))
            hours = float(equipment.get("recommended_cleaning_interval_h", 8760.0))
            equipment["design_voc_load_ppm_cmm_h"] = voc * self.design_flow * hours
        return {
            "equipment": equipment,
            "maintenance_events": list(self.maintenance_events),
            "baseline": dict(self.baseline_spec),
        }


def load_fleet(config_dir: str | Path | None = None, *, path: str | Path | None = None) -> list[EquipmentSpec]:
    """fleet.yaml 을 읽어 설비 목록으로 전개한다 (상속 해석 포함)."""
    target = Path(path) if path else Path(config_dir or CONFIG_DIR) / "fleet.yaml"
    if not target.exists():
        return []
    return parse_fleet(load_yaml(target))


def parse_fleet(raw: dict[str, Any]) -> list[EquipmentSpec]:
    """fleet.yaml 내용을 EquipmentSpec 목록으로 전개한다.

    상속 우선순위: overrides > series > types > defaults
    """
    defaults = raw.get("defaults") or {}
    types = raw.get("types") or {}
    overrides = raw.get("overrides") or {}
    baselines = raw.get("baselines") or {}
    events = raw.get("maintenance_events") or {}

    specs: list[EquipmentSpec] = []
    for entry in raw.get("series") or []:
        series_id = str(entry["id"])
        type_key = series_id.split("-", 1)[0]

        # 아래에서 위로 덮어쓴다
        merged: dict[str, Any] = {k: v for k, v in defaults.items() if k in SPEC_KEYS}
        merged.update({k: v for k, v in (types.get(type_key) or {}).items() if k in SPEC_KEYS})
        merged.update({k: v for k, v in entry.items() if k in SPEC_KEYS})

        for unit in entry.get("units") or []:
            equipment_id = f"{series_id}{unit}"
            unit_spec = dict(merged)
            unit_spec.update({
                k: v for k, v in (overrides.get(equipment_id) or {}).items() if k in SPEC_KEYS
            })
            specs.append(EquipmentSpec(
                equipment_id=equipment_id,
                series_id=series_id,
                unit=str(unit),
                type_key=type_key,
                spec=unit_spec,
                maintenance_events=list(events.get(equipment_id) or []),
                baseline_spec=dict(baselines.get(equipment_id) or {}),
            ))
    return specs


def unit_config(spec: EquipmentSpec, config: Config) -> Config:
    """설비 한 대짜리 Config 를 만든다 — 기존 단일 설비 경로를 그대로 재사용하기 위함."""
    return Config(
        tags=config.tags,
        weights=config.weights,
        events=spec.to_events_block(),
        fleet=config.fleet,
    )


# ---------------------------------------------------------------------------
# 태그 분해
# ---------------------------------------------------------------------------
def _fleet_tag_cfg(config: Config) -> dict[str, Any]:
    return (config.tags.get("fleet") or {})


def expected_columns(spec: EquipmentSpec, config: Config) -> dict[str, str]:
    """설비 하나가 쓰는 {실제 컬럼명 → 단일모드 source 명} 매핑을 만든다.

    단일모드 source 명(RTO_DP_BED 등)으로 되돌려 주면 기존 load_operating_data 가
    그대로 처리한다 — 필수 태그 검증·섹터 컬럼 수집까지 전부 재사용된다.
    """
    fleet_cfg = _fleet_tag_cfg(config)
    pattern = str(fleet_cfg.get("tag_pattern", "{equipment}_{suffix}"))
    suffixes = fleet_cfg.get("suffixes") or {}
    per_equipment = (fleet_cfg.get("equipment_overrides") or {}).get(spec.equipment_id, {})

    mapping: dict[str, str] = {}
    for var in SCALAR_VARS:
        if var == "timestamp":
            continue
        single_source = (config.tags.get(var) or {}).get("source")
        if not single_source:
            continue
        suffix = per_equipment.get(var) or suffixes.get(var)
        if not suffix:
            continue
        actual = pattern.format(equipment=spec.equipment_id, suffix=suffix)
        mapping[actual] = single_source

    # 섹터별 온도
    sector_cfg = config.tags.get("t_sector") or {}
    single_prefix = sector_cfg.get("source_prefix")
    fleet_prefix = fleet_cfg.get("sector_suffix_prefix")
    if single_prefix and fleet_prefix:
        for i in range(1, int(sector_cfg.get("count", 12)) + 1):
            actual = pattern.format(equipment=spec.equipment_id, suffix=f"{fleet_prefix}{i:02d}")
            mapping[actual] = f"{single_prefix}{i:02d}"

    return mapping


def split_by_equipment(
    df: pd.DataFrame, spec: EquipmentSpec, config: Config
) -> pd.DataFrame:
    """와이드 프레임에서 설비 한 대의 컬럼만 뽑아 단일모드 이름으로 되돌린다."""
    fleet_cfg = _fleet_tag_cfg(config)
    shared = fleet_cfg.get("shared") or {"timestamp": "TIMESTAMP"}

    out = pd.DataFrame(index=df.index)
    ts_col = shared.get("timestamp", "TIMESTAMP")
    if ts_col not in df.columns:
        raise ValueError(
            f"공용 시각 컬럼 '{ts_col}' 을 찾을 수 없습니다. config/tags.yaml 의 fleet.shared 를 확인하세요."
        )
    out[(config.tags.get("timestamp") or {}).get("source", "TIMESTAMP")] = df[ts_col]

    for actual, single_source in expected_columns(spec, config).items():
        if actual in df.columns:
            out[single_source] = df[actual].to_numpy()

    return out


def detect_equipment(df: pd.DataFrame, config: Config, specs: list[EquipmentSpec]) -> pd.DataFrame:
    """컬럼명을 스캔해 설비별 태그 확보 현황을 표로 만든다 (온보딩 진단용).

    매핑이 틀리면 조용히 지표가 빠져 점수만 이상해진다. 그 전에 잡기 위한 화면이다.
    """
    rows = []
    for spec in specs:
        expected = expected_columns(spec, config)
        found = [c for c in expected if c in df.columns]
        missing = [c for c in expected if c not in df.columns]
        rows.append({
            "설비": spec.equipment_id,
            "계열": spec.series_id,
            "종류": spec.label,
            "확보 태그": len(found),
            "전체 태그": len(expected),
            "상태": "✅ 정상" if not missing else ("⛔ 데이터 없음" if not found else "⚠️ 일부 누락"),
            "누락 태그": ", ".join(missing[:6]) + (" …" if len(missing) > 6 else ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fleet 분석
# ---------------------------------------------------------------------------
@dataclass
class FleetAnalysis:
    """20대 전체 분석 결과."""

    config: Config
    specs: dict[str, EquipmentSpec]
    units: dict[str, Any] = field(default_factory=dict)      # equipment_id → Analysis
    failed: dict[str, str] = field(default_factory=dict)     # equipment_id → 실패 사유
    peer_ratio: dict[str, pd.Series] = field(default_factory=dict)
    peer_count: dict[str, int] = field(default_factory=dict)
    detection: pd.DataFrame = field(default_factory=pd.DataFrame)   # 태그 감지 결과

    @property
    def series_ids(self) -> list[str]:
        seen: list[str] = []
        for spec in self.specs.values():
            if spec.series_id not in seen:
                seen.append(spec.series_id)
        return seen

    def series_units(self, series_id: str) -> list[str]:
        return [e for e, s in self.specs.items() if s.series_id == series_id]

    def ranking(self) -> pd.DataFrame:
        """위험도 순 정비 우선순위 표 — 20대에서 가장 먼저 필요한 정보."""
        rows = []
        for eq_id, analysis in self.units.items():
            spec = self.specs[eq_id]
            try:
                snap = analysis.snapshot()
            except ValueError:
                continue

            score_series = analysis.score
            prior = score_series.loc[score_series.index <= snap.date - pd.Timedelta(days=30)]
            delta = snap.score - float(prior.iloc[-1]) if not prior.empty else np.nan

            rul = analysis.rul.days_remaining
            guide = analysis.guidance(snap.date)
            rows.append({
                "설비": eq_id,
                "계열": spec.series_id,
                "종류": spec.label,
                "점수": round(snap.score, 1),
                "등급": snap.grade,
                "30일 변화": round(delta, 1) if np.isfinite(delta) else None,
                "잔여일": None if rul is None else round(rul),
                "판정": guide["headline"],
                "동급 대비": (
                    round(float(analysis.indicators.raw["A6"].iloc[-1]), 2)
                    if "A6" in analysis.indicators.raw.columns
                    and analysis.indicators.raw["A6"].notna().any()
                    else None
                ),
                "비교대수": self.peer_count.get(eq_id, 0),
                "베이스라인": analysis.baseline.source_label,
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("점수").reset_index(drop=True)

    def grade_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for analysis in self.units.values():
            try:
                grade = analysis.snapshot().grade
            except ValueError:
                continue
            counts[grade] = counts.get(grade, 0) + 1
        return counts


def analyze_fleet(
    source: str | Path | pd.DataFrame | list,
    config: Config | None = None,
    *,
    config_dir: str | Path | None = None,
    resample: str | None = None,
) -> FleetAnalysis:
    """20대 통합 데이터를 설비별로 분해해 전부 분석한다.

    동급기 비교 때문에 2패스로 돈다.
      패스 1: 설비별 전처리 → 일 집계
      패스 2: 계열별 peer 중앙값 산출
      패스 3: 설비별 지표(A6 포함) → 채점 → 추세 → 판정

    한 대가 실패해도 나머지는 그대로 나온다 (`failed` 에 사유 기록).
    """
    from . import pipeline
    from .io_loader import load_config, read_frames

    cfg = config or load_config(config_dir)
    specs = load_fleet(config_dir) if not cfg.fleet else parse_fleet(cfg.fleet)
    if not specs:
        raise ValueError("config/fleet.yaml 에 설비가 정의되어 있지 않습니다.")

    df = source if isinstance(source, pd.DataFrame) else read_frames(source)

    spec_map = {s.equipment_id: s for s in specs}
    result = FleetAnalysis(
        config=cfg, specs=spec_map, detection=detect_equipment(df, cfg, specs)
    )

    # --- 패스 1: 설비별 전처리 ---------------------------------------------
    prepared: dict[str, pipeline.Prepared] = {}
    for spec in specs:
        try:
            unit_df = split_by_equipment(df, spec, cfg)
            prepared[spec.equipment_id] = pipeline.prepare(
                unit_df, unit_config(spec, cfg), resample=resample
            )
        except Exception as exc:  # noqa: BLE001 — 한 대의 실패가 전체를 막으면 안 된다
            result.failed[spec.equipment_id] = f"{type(exc).__name__}: {exc}"

    # --- 패스 2: 계열별 동급기 중앙값 ---------------------------------------
    for series_id in {s.series_id for s in specs}:
        members = [e for e in prepared if spec_map[e].series_id == series_id]
        if len(members) < 2:
            for eq_id in members:
                result.peer_count[eq_id] = 0
            continue

        ratios = pd.DataFrame({eq_id: prepared[eq_id].dp_ratio for eq_id in members})
        for eq_id in members:
            others = ratios.drop(columns=[eq_id])
            peer_median = others.median(axis=1)
            # 최소 1대 이상의 유효 동급기가 있는 날만 비교한다
            enough = others.notna().sum(axis=1) >= 1
            result.peer_ratio[eq_id] = peer_median.where(enough)
            result.peer_count[eq_id] = len(members) - 1

    # --- 패스 3: 설비별 채점 ------------------------------------------------
    for eq_id, prep in prepared.items():
        try:
            result.units[eq_id] = pipeline.score_prepared(
                prep, peer_dp_ratio=result.peer_ratio.get(eq_id)
            )
        except Exception as exc:  # noqa: BLE001
            result.failed[eq_id] = f"{type(exc).__name__}: {exc}"

    return result
