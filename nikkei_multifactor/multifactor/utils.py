"""Utility helpers for date handling, normalization, and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def parse_date(date_text: str) -> pd.Timestamp:
    """Parse yyyy-mm-dd text into normalized timestamp."""
    return pd.Timestamp(date_text).normalize()


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory if needed."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Scale series into [0, 1]. If constant, return 0.5 for valid rows."""
    series = series.astype(float)
    valid = series.replace([np.inf, -np.inf], np.nan)
    min_v = valid.min(skipna=True)
    max_v = valid.max(skipna=True)
    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(np.nan, index=series.index, dtype=float)
    if np.isclose(max_v, min_v):
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out[valid.notna()] = 0.5
        return out
    return (valid - min_v) / (max_v - min_v)


def align_to_previous_trading_day(
    trading_days: pd.DatetimeIndex, target: pd.Timestamp, field_name: str
) -> pd.Timestamp:
    """Round a target date down to the nearest available trading day."""
    if len(trading_days) == 0:
        raise ValueError("取引カレンダーが空です。データ取得に失敗した可能性があります。")
    target = target.normalize()
    eligible = trading_days[trading_days <= target]
    if len(eligible) == 0:
        raise ValueError(
            f"{field_name}={target.date()} より前の取引日が存在しません。"
        )
    return pd.Timestamp(eligible[-1]).normalize()


def previous_trading_day(
    trading_days: pd.DatetimeIndex, current: pd.Timestamp
) -> pd.Timestamp:
    """Return previous trading day for current day."""
    current = current.normalize()
    idx = trading_days.get_indexer([current])[0]
    if idx <= 0:
        raise ValueError(f"{current.date()} の前営業日が見つかりません。")
    return pd.Timestamp(trading_days[idx - 1]).normalize()


def next_trading_day(trading_days: pd.DatetimeIndex, current: pd.Timestamp) -> pd.Timestamp:
    """Return next trading day for current day."""
    current = current.normalize()
    idx = trading_days.get_indexer([current])[0]
    if idx == -1:
        raise ValueError(f"{current.date()} は取引カレンダーに含まれていません。")
    if idx + 1 >= len(trading_days):
        raise ValueError(f"{current.date()} の翌営業日が見つかりません。")
    return pd.Timestamp(trading_days[idx + 1]).normalize()


def first_trading_days_by_month(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return first trading day in each month within provided index."""
    if len(days) == 0:
        return []
    ser = pd.Series(days, index=days)
    grouped = ser.groupby(days.to_period("M")).first()
    return [pd.Timestamp(v).normalize() for v in grouped.tolist()]


def calc_max_drawdown(equity: pd.Series) -> float:
    """Compute max drawdown from equity curve."""
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calc_performance_metrics(
    daily_returns: pd.Series, equity: pd.Series, benchmark_returns: pd.Series | None = None
) -> dict[str, float]:
    """Compute core strategy metrics."""
    if daily_returns.empty:
        return {
            "total_return": float("nan"),
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "benchmark_total_return": float("nan"),
            "excess_return": float("nan"),
        }

    total_return = float(equity.iloc[-1] - 1.0)
    days = len(daily_returns)
    annual_return = float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1.0)
    annual_vol = float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else float("nan")
    max_dd = calc_max_drawdown(equity)

    benchmark_total = float("nan")
    excess_return = float("nan")
    if benchmark_returns is not None and len(benchmark_returns) == days:
        bench_equity = (1.0 + benchmark_returns).cumprod()
        benchmark_total = float(bench_equity.iloc[-1] - 1.0)
        excess_return = total_return - benchmark_total

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "benchmark_total_return": benchmark_total,
        "excess_return": excess_return,
    }


def dumps_pretty_json(data: dict) -> str:
    """JSON dump helper that handles numpy and timestamps."""

    def _default(obj):  # noqa: ANN001
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, pd.Timestamp):
            return obj.date().isoformat()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"JSON serialization is not supported for {type(obj)!r}")

    return json.dumps(data, ensure_ascii=False, indent=2, default=_default)


def ensure_unique_symbols(symbols: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for sym in symbols:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out
