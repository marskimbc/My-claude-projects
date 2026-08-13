"""Can Type RTO 축열재(세라믹) 막힘 사전예측 시스템.

운전 데이터에서 축열재 막힘 진행도를 정량화하여
  · 100점 만점 설비 건전도 점수 (가중 배점)
  · 변화추이 분석 (추세 기울기 · 변화점 · 잔여여유 예측)
  · 고장모드별 점검 가이드
를 산출한다.
"""

__version__ = "0.1.0"

__all__ = [
    "io_loader",
    "preprocess",
    "normalize",
    "baseline",
    "indicators",
    "scoring",
    "trend",
    "diagnose",
    "report",
    "pipeline",
]
