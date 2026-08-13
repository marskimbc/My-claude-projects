"""테스트 공용 픽스처.

합성 데이터는 커밋하지 않으므로(.gitignore) 테스트가 직접 생성한다.
분석은 시나리오당 수 초가 걸리므로 세션 단위로 한 번만 수행한다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


#: fleet 테스트용 데이터 길이. 시나리오는 전체 기간에 정규화되어 있으므로
#: 짧게 잡아도 최종 열화 수준은 1년치와 같다.
FLEET_DAYS = 180


def _load_script(name: str):
    """data/sample 의 스크립트를 모듈로 불러온다 (패키지가 아니므로 직접 로드)."""
    path = PROJECT_ROOT / "data" / "sample" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_generator():
    return _load_script("generate_sample.py")


@pytest.fixture(scope="session")
def generator():
    return _load_generator()


@pytest.fixture(scope="session")
def config():
    from rto_health.io_loader import load_config

    return load_config(PROJECT_ROOT / "config")


@pytest.fixture(scope="session")
def scenarios(generator):
    """3개 시나리오의 원데이터."""
    return {name: generator.build_scenario(name, seed=42) for name in generator.SCENARIOS}


@pytest.fixture(scope="session")
def analyses(scenarios):
    """3개 시나리오의 분석 결과."""
    from rto_health.pipeline import analyze

    return {name: analyze(df, config_dir=PROJECT_ROOT / "config") for name, df in scenarios.items()}


@pytest.fixture(scope="session")
def normal(analyses):
    return analyses["normal"]


@pytest.fixture(scope="session")
def gradual(analyses):
    return analyses["gradual_fouling"]


@pytest.fixture(scope="session")
def rapid(analyses):
    return analyses["rapid_plugging"]


# --- 20대 fleet -------------------------------------------------------------

@pytest.fixture(scope="session")
def fleet_specs():
    from rto_health.fleet import load_fleet

    return load_fleet(PROJECT_ROOT / "config")


@pytest.fixture(scope="session")
def fleet_frame():
    """20대 통합 원데이터 (1시간 간격)."""
    gen = _load_script("generate_fleet_sample.py")
    return gen.build_fleet(days=FLEET_DAYS, freq_min=60, config_dir=PROJECT_ROOT / "config")


@pytest.fixture(scope="session")
def fleet(fleet_frame):
    """20대 전체 분석 결과."""
    from rto_health.fleet import analyze_fleet

    return analyze_fleet(fleet_frame, config_dir=PROJECT_ROOT / "config")
