# 국내 주식 관리 · 지수 분석 · 매매 추천

코스피/코스닥 지수와 테마 ETF(양자·신재생·방산·조선·전력·인공지능·인프라),
금·은·구리·부동산 등 원자재/대체자산의 동향을 한 번에 훑고,
지수 분석 결과를 근거로 **국내 주식 종목 매매 후보와 진입·손절·목표가**까지 뽑아주는
파이썬 CLI 도구입니다.

> ⚠️ **투자 자문이 아닙니다.** 공개 시세를 규칙대로 계산한 참고 자료일 뿐이며,
> 최종 판단과 손익 책임은 사용자 본인에게 있습니다.

---

## 1. 무엇을 해주나

| 단계 | 내용 |
|---|---|
| ① 시장 국면 | 코스피·코스닥 추세 + 환율/금리/VIX/해외지수 → `위험선호 / 중립 / 위험회피` 판정과 **권장 주식 노출 비중** |
| ② 지수 동향 | 코스피, 코스피200, 코스닥, 코스닥150의 추세·수익률·RSI·52주 고점 대비 |
| ③ 매크로·원자재 | 금·은·구리·원유·리츠·원/달러·미10년물·VIX·S&P500·나스닥·SOX |
| ④ 테마 상대강도 | 테마별 ETF 바스켓의 코스피 대비 초과수익 → `주도 / 개선 / 둔화 / 소외` 로테이션 국면 |
| ⑤ 상관 분석 | 테마 ↔ 원자재/대체자산 상관계수 히트맵 |
| ⑥ 매매 추천 | 주도 테마의 ETF·개별종목 스코어링 → 액션(매수/관심/보유/축소/매도) + **진입가·손절·목표·수량·비중** |

부가 기능: 보유 종목 원장 관리와 평가손익, 점수 기반 리밸런싱 백테스트, HTML 대시보드.

---

## 2. 설치

```bash
git clone <repo> && cd my-claude-projects
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

파이썬 3.11 이상을 권장합니다.

---

## 3. 빠른 시작

```bash
# 시장 국면 + 지수 + 매크로 한눈에
python -m stockmgr market

# 테마 상대강도와 원자재 상관
python -m stockmgr themes --members

# 특정 테마의 상위 ETF/종목 스크리닝
python -m stockmgr screen --theme power --theme defense --top 5

# 종합 리포트 + 매매 추천 (자본 2천만원 기준) + HTML 저장
python -m stockmgr recommend --capital 20000000 --html reports/today.html

# 브라우저용 대시보드만 생성
python -m stockmgr report -o reports/today.html
```

### 데이터 소스가 막혀 있는 환경이라면

```bash
python -m stockmgr --mode offline recommend
```

`--mode offline` 은 **합성(가짜) 시세**로 전체 파이프라인을 돌립니다.
설치 확인, 화면 확인, 테스트용입니다. 리포트 상단에 경고가 붙고,
이 데이터로는 절대 매매 판단을 하면 안 됩니다.

기본값인 `--mode auto` 는 실시간 조회를 시도하고, 실패하면 합성 모드로 자동 강등하며
경고를 출력합니다. 실시간이 아니면 아예 실패하길 원하면 `--mode live` 를 쓰세요.

---

## 4. 명령어

| 명령 | 설명 | 주요 옵션 |
|---|---|---|
| `market` | 시장 국면·지수·매크로 | |
| `themes` | 테마 상대강도, 원자재 상관 | `--theme`, `--members` |
| `screen` | 테마별 상위 ETF/개별종목 | `--theme`, `--top` |
| `recommend` | 전체 리포트 + 매매 추천 | `--capital`, `--theme`, `--max`, `--html` |
| `report` | HTML 대시보드 생성 | `-o/--output`, `--capital` |
| `backtest` | 점수 기반 리밸런싱 백테스트 | `--theme`, `--top` |
| `portfolio` | 보유 종목 관리/평가 | `status`(기본) / `buy` / `sell` / `cash` |
| `cache` | 캐시 삭제 | `--namespace` |

공통 옵션: `--mode {auto,live,offline}`, `--date YYYY-MM-DD`, `--refresh`, `-v`.

### 보유 종목 관리 예시

```bash
python -m stockmgr portfolio cash --amount 20000000          # 현금 잔고 설정
python -m stockmgr portfolio buy  --ticker 069500 --shares 20 --price 34500 \
    --name "KODEX 200" --kind etf --stop 32800 --target 37500
python -m stockmgr portfolio sell --ticker 069500 --shares 10 --price 36000
python -m stockmgr portfolio                                  # 평가손익 + 점검 알림
```

원장은 `data/holdings.json` 에 저장됩니다(`--file` 로 변경 가능).
`portfolio` 는 손절/목표 도달 종목과 점수가 떨어진 종목을 따로 알려줍니다.

---

## 5. 데이터 출처

| 구분 | 출처 | 비고 |
|---|---|---|
| 코스피/코스닥 지수, ETF, 개별종목, 투자자별 수급, PER/PBR | KRX (`pykrx`) | 공식 공시 데이터 |
| 금·은·구리·원유·리츠·환율·미국금리·VIX·해외지수 | Yahoo Finance (`yfinance`) | 국내 미상장 자산 보완용 |

조회 결과는 `data/cache/` 에 CSV 로 캐시되며 기본 TTL 은 12시간입니다
(`settings.yaml`의 `data.cache_ttl_hours`). 새로 받고 싶으면 `--refresh`.

> **테마 ETF 는 종목코드를 하드코딩하지 않습니다.** 상장/폐지가 잦기 때문에,
> `config/themes.yaml` 에 이름 패턴만 정의해두고 실행 시점의 KRX ETF 전체 목록에서
> 매칭합니다. 새 테마를 추가하려면 키워드만 넣으면 됩니다.

---

## 6. 분석 로직

### 종합 점수 (0~100)

`config/settings.yaml` 의 `scoring.weights` 로 가중합합니다.

| 축 | 기본 가중 | 계산 근거 |
|---|---|---|
| 추세 `trend` | 35% | 20/60/120/200일선 정배열 정도, 60일선 기울기, ADX·DI 방향성 |
| 모멘텀 `momentum` | 25% | 1·3·6개월 수익률(기간 보정), RSI 과열/과매도 감점 |
| 상대강도 `relative_strength` | 20% | 코스피 대비 초과수익 |
| 수급 `flow` | 10% | 외국인+기관 20일 순매수의 자기 이력 대비 z-score |
| 리스크 `risk` | 10% | 60일 변동성, 120일 MDD, 52주 고점 대비 (낮을수록 고득점) |

각 축은 로지스틱 함수로 0~100 에 매핑되며, 0 편차가 50점입니다.

### 매매 신호

점수 임계값(`signals`)으로 `매수 / 관심 / 보유 / 비중축소 / 매도` 를 정한 뒤 보정합니다.

- RSI ≥ 75(과열) → 매수를 **관심**으로 강등
- 20일 평균 거래대금 미달 → 매수를 **관심**으로 강등
- 시장 국면이 `위험회피` → 신규 매수 보류, 축소 신호는 매도로 강화

### 포지션 사이징 (ATR 기반)

"얼마나 확신하는가"가 아니라 **"틀렸을 때 얼마를 잃는가"** 로 수량을 정합니다.

```
손절폭 = 2 × ATR(14)          목표가 = 진입 + 4 × ATR   (손익비 1:2)
수량   = (자본 × 1% × 국면노출 × 확신배수) ÷ 손절폭
```

여기에 상한 3중 장치가 걸립니다 — 종목당 15%, 테마당 30%, 추천 합계 80%.
국면이 나쁘면 노출 자체가 60%/30%로 줄어듭니다.

### 백테스트

리밸런싱 시점마다 **그 시점까지의 데이터만으로** 점수를 매겨 상위 N종목 동일가중 보유합니다.
왕복 수수료·세금(기본 15bp)만 반영하고 슬리피지·거래정지·상장폐지는 반영하지 않으므로,
결과는 낙관적으로 나옵니다. 로직 비교용으로만 쓰세요.

---

## 7. 설정

전부 YAML 이라 코드 수정 없이 조정할 수 있습니다.

- `config/settings.yaml` — 지표 기간, 점수 가중치, 신호 임계값, 리스크 한도, 백테스트
- `config/themes.yaml` — 벤치마크 지수 코드, 테마 키워드, 원자재, 매크로 심볼

테마 추가 예시:

```yaml
themes:
  bio:
    label: 바이오
    include: [바이오, 헬스케어, 제약]
    exclude: []
```

---

## 8. 프로젝트 구조

```
stockmgr/
├─ config.py              설정 로딩
├─ universe.py            테마 → 실제 ETF 해석 (키워드 매칭)
├─ cli.py                 명령줄 진입점
├─ data/
│  ├─ krx.py              pykrx 어댑터 (한글 컬럼 → 영문 표준)
│  ├─ macro.py            yfinance 어댑터
│  ├─ synthetic.py        오프라인 합성 시세 생성기
│  ├─ cache.py            CSV 디스크 캐시 (stale-if-error)
│  └─ loader.py           데이터 파사드 (live/offline/auto)
├─ analysis/
│  ├─ indicators.py       SMA·EMA·RSI·MACD·볼린저·ATR·ADX·OBV·스토캐스틱
│  ├─ score.py            5축 점수화
│  ├─ trend.py            추세 판정
│  ├─ regime.py           시장 국면
│  ├─ relative.py         테마 상대강도·로테이션
│  └─ correlation.py      테마 ↔ 원자재 상관
├─ strategy/
│  ├─ signals.py          매매 신호
│  ├─ screener.py         테마 → ETF → 구성종목 스크리닝
│  ├─ risk.py             손절/목표/포지션 사이징
│  └─ recommend.py        전체 파이프라인
├─ portfolio/             보유 원장, 평가손익
├─ backtest/engine.py     리밸런싱 백테스트
└─ report/                콘솔 / HTML 출력
```

---

## 9. 테스트

```bash
pip install pytest
python -m pytest tests -q
```

모든 테스트는 네트워크 없이 돌아갑니다. 실시간(pykrx) 경로는 한글 컬럼 응답을
흉내 낸 가짜 모듈로 변환 로직까지 검증합니다(`tests/test_krx_adapter.py`).

---

## 10. 알아둘 한계

- **지연 데이터입니다.** KRX 일봉 기준이며 실시간 호가·체결이 아닙니다. 장중 재계산은 부정확합니다.
- **점수는 예측이 아닙니다.** 과거 가격/수급의 요약일 뿐이며, 공시·실적·정책 같은 사건은 반영하지 않습니다.
- **테마 키워드 매칭은 완벽하지 않습니다.** ETF 이름이 실제 편입 종목을 항상 대변하진 않습니다. `themes --members` 로 실제 편입 목록을 확인하세요.
- 상관계수는 인과가 아닙니다.
- 사내망·프록시 환경에서는 `data.krx.co.kr` 과 Yahoo Finance 접근이 막혀 있을 수 있습니다. 이 경우 `--mode auto` 가 합성 모드로 강등하며, 그 리포트는 참고 자료가 아닙니다.
