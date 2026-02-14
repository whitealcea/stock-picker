"""Factor calculation logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ATR_WINDOW,
    LOWVOL_WINDOW,
    MA_WINDOW,
    MOMENTUM_WINDOW,
    RSI_WINDOW,
    VOLUME_WINDOW,
)
from .data import MarketDataStore
from .utils import minmax_normalize

FACTOR_COLUMNS = ["momentum", "lowvol", "rsi2", "ma_break", "volume_ratio"]
NORM_COLUMNS = [f"{c}_n" for c in FACTOR_COLUMNS]


def calc_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Simple RSI implementation."""
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calc_atr(ohlcv: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """ATR based on high/low/close."""
    high = ohlcv["High"]
    low = ohlcv["Low"]
    close = ohlcv["Close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr


def compute_factor_snapshot(store: MarketDataStore, signal_date: pd.Timestamp) -> pd.DataFrame:
    """Compute cross-sectional factors at a specific signal date."""
    records: list[dict] = []

    for symbol, frame in store.symbol_frames.items():
        hist = frame.loc[:signal_date].copy()
        if len(hist) < max(MA_WINDOW, MOMENTUM_WINDOW + 1, VOLUME_WINDOW, ATR_WINDOW):
            continue
        hist = hist.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if len(hist) < max(MA_WINDOW, MOMENTUM_WINDOW + 1, VOLUME_WINDOW, ATR_WINDOW):
            continue

        close = hist["Close"]
        ret = close.pct_change()

        mom_base = close.iloc[-1 - MOMENTUM_WINDOW]
        momentum = close.iloc[-1] / mom_base - 1.0 if mom_base > 0 else np.nan

        vol20 = ret.iloc[-LOWVOL_WINDOW:].std(ddof=0)
        lowvol = 1.0 / vol20 if pd.notna(vol20) and vol20 > 0 else np.nan

        rsi2_raw = calc_rsi(close, window=RSI_WINDOW).iloc[-1]
        # 低RSI(売られ過ぎ)を高スコアにするため反転
        rsi2 = 100.0 - rsi2_raw if pd.notna(rsi2_raw) else np.nan

        ma200 = close.rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean().iloc[-1]
        ma_break = float(close.iloc[-1] > ma200) if pd.notna(ma200) else np.nan

        vol_mean = hist["Volume"].rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean().iloc[-1]
        volume_ratio = hist["Volume"].iloc[-1] / vol_mean if pd.notna(vol_mean) and vol_mean > 0 else np.nan

        atr20 = calc_atr(hist, window=ATR_WINDOW).iloc[-1]

        records.append(
            {
                "symbol": symbol,
                "momentum": momentum,
                "lowvol": lowvol,
                "rsi2": rsi2,
                "ma_break": ma_break,
                "volume_ratio": volume_ratio,
                "atr20": atr20,
                "close": close.iloc[-1],
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("symbol")
    df = df.replace([np.inf, -np.inf], np.nan)

    for factor in FACTOR_COLUMNS:
        df[f"{factor}_n"] = minmax_normalize(df[factor])

    df = df.dropna(subset=NORM_COLUMNS + ["atr20"])
    if df.empty:
        return df

    df["score"] = df[NORM_COLUMNS].mean(axis=1)
    return df.sort_values("score", ascending=False)
