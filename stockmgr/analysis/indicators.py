"""기술적 지표 (pandas 순수 구현, TA-Lib 불필요).

모든 함수는 입력과 같은 인덱스의 Series/DataFrame 을 돌려주고,
계산에 필요한 기간이 모자라면 NaN 을 남긴다(임의 값으로 채우지 않는다).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # 하락이 전혀 없던 구간은 RSI 100 으로 본다.
    return out.where(avg_loss.ne(0) | avg_gain.isna(), 100.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "hist": macd_line - signal_line,
    })


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower,
                         "width": width, "pct_b": pct_b})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder ADX/DI. 추세의 '세기'를 본다(방향은 DI 로)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    smoothing = dict(alpha=1 / period, adjust=False, min_periods=period)
    atr_ = true_range(df).ewm(**smoothing).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(**smoothing).mean() / atr_
    minus_di = 100 * minus_dm.ewm(**smoothing).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    # 완전한 추세 구간에서 부동소수 오차로 100 을 아주 살짝 넘길 수 있어 잘라낸다.
    return pd.DataFrame({
        "plus_di": plus_di.clip(0, 100),
        "minus_di": minus_di.clip(0, 100),
        "adx": dx.ewm(**smoothing).mean().clip(0, 100),
    })


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.DataFrame:
    low = df["low"].rolling(period, min_periods=period).min()
    high = df["high"].rolling(period, min_periods=period).max()
    k = 100 * (df["close"] - low) / (high - low).replace(0, np.nan)
    return pd.DataFrame({"k": k, "d": k.rolling(smooth, min_periods=smooth).mean()})


def returns(series: pd.Series, window: int) -> float:
    """window 영업일 수익률(%). 데이터가 모자라면 NaN."""
    clean = series.dropna()
    if len(clean) <= window:
        return float("nan")
    past, now = clean.iloc[-window - 1], clean.iloc[-1]
    if past == 0:
        return float("nan")
    return float((now / past - 1) * 100)


def annualized_vol(series: pd.Series, window: int = 60) -> float:
    daily = series.pct_change().dropna().tail(window)
    if len(daily) < 5:
        return float("nan")
    return float(daily.std(ddof=0) * np.sqrt(TRADING_DAYS) * 100)


def max_drawdown(series: pd.Series, window: int | None = None) -> float:
    """최대 낙폭(%, 음수). window 를 주면 최근 구간만."""
    clean = series.dropna()
    if window:
        clean = clean.tail(window)
    if clean.empty:
        return float("nan")
    peak = clean.cummax()
    return float(((clean / peak - 1) * 100).min())


def drawdown_from_high(series: pd.Series, window: int = 252) -> float:
    """52주 고점 대비 현재 위치(%, 음수)."""
    clean = series.dropna().tail(window)
    if clean.empty:
        return float("nan")
    return float((clean.iloc[-1] / clean.max() - 1) * 100)


def slope_pct(series: pd.Series, window: int = 20) -> float:
    """최근 window 구간 선형회귀 기울기를 평균값 대비 %/일 로 환산."""
    clean = series.dropna().tail(window)
    if len(clean) < max(3, window // 2):
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    slope = np.polyfit(x, clean.to_numpy(dtype=float), 1)[0]
    mean = clean.mean()
    if mean == 0:
        return float("nan")
    return float(slope / mean * 100)


def enrich(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """자주 쓰는 지표를 한 번에 붙인다."""
    p = params or {}
    out = df.copy()
    close = out["close"]
    out["ma_short"] = sma(close, p.get("ma_short", 20))
    out["ma_mid"] = sma(close, p.get("ma_mid", 60))
    out["ma_long"] = sma(close, p.get("ma_long", 120))
    out["ma_trend"] = sma(close, p.get("ma_trend", 200))
    out["rsi"] = rsi(close, p.get("rsi_period", 14))
    out = out.join(macd(close))
    out = out.join(bollinger(close, p.get("bb_period", 20), p.get("bb_std", 2.0))
                   [["pct_b", "width"]])
    out["atr"] = atr(out, p.get("atr_period", 14))
    out["atr_pct"] = out["atr"] / close * 100
    out = out.join(adx(out, p.get("adx_period", 14))[["adx", "plus_di", "minus_di"]])
    if "volume" in out.columns:
        out["vol_ma20"] = sma(out["volume"], 20)
    return out
