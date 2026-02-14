"""Regime and portfolio construction logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MA_WINDOW


def is_risk_on(benchmark_frame: pd.DataFrame, signal_date: pd.Timestamp) -> bool:
    """Risk-on if benchmark close > 200-day moving average."""
    hist = benchmark_frame.loc[:signal_date]
    hist = hist.dropna(subset=["Close"])
    if len(hist) < MA_WINDOW:
        return False

    close = hist["Close"].iloc[-1]
    ma200 = hist["Close"].rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean().iloc[-1]
    if pd.isna(ma200):
        return False
    return bool(close > ma200)


def effective_k(top_k: int, risk_on: bool) -> int:
    """Apply regime-based position count adjustment."""
    if risk_on:
        return max(1, top_k)
    return max(1, top_k // 2)


def build_target_portfolio(
    factor_snapshot: pd.DataFrame,
    top_k: int,
    risk_on: bool,
) -> pd.DataFrame:
    """Select top-K symbols and allocate by inverse ATR."""
    if factor_snapshot.empty:
        return pd.DataFrame()

    k = effective_k(top_k, risk_on)
    selected = factor_snapshot.sort_values("score", ascending=False).head(k).copy()
    if selected.empty:
        return selected

    atr = selected["atr20"].replace([np.inf, -np.inf, 0.0], np.nan)
    inv_atr = 1.0 / atr
    inv_atr = inv_atr.replace([np.inf, -np.inf], np.nan).dropna()

    if inv_atr.empty or inv_atr.sum() <= 0:
        selected["weight"] = 1.0 / len(selected)
    else:
        weights = inv_atr / inv_atr.sum()
        selected["weight"] = 0.0
        selected.loc[weights.index, "weight"] = weights
        if selected["weight"].sum() <= 0:
            selected["weight"] = 1.0 / len(selected)
        else:
            selected["weight"] = selected["weight"] / selected["weight"].sum()

    selected["rank"] = np.arange(1, len(selected) + 1)
    return selected
