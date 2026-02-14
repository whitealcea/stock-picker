"""Data loading layer based on yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from .config import MIN_REQUIRED_TRADING_DAYS
from .utils import ensure_unique_symbols

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class MarketDataStore:
    """Container for symbol OHLCV data and benchmark data."""

    symbol_frames: dict[str, pd.DataFrame]
    benchmark_frame: pd.DataFrame

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        """Trading day index derived from benchmark close series."""
        days = self.benchmark_frame.index[self.benchmark_frame["Close"].notna()]
        return pd.DatetimeIndex(days).sort_values()

    @property
    def symbols(self) -> list[str]:
        """List of symbols that passed data quality filtering."""
        return sorted(self.symbol_frames.keys())


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[REQUIRED_COLS]
    df = df.apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(how="all")
    return df


def _extract_symbol_frames(raw: pd.DataFrame, chunk: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))

        # Typical shape for group_by="ticker": columns = (ticker, field)
        if any(sym in level0 for sym in chunk):
            for sym in chunk:
                if sym in level0:
                    out[sym] = _clean_ohlcv(raw[sym])
        # Fallback shape: columns = (field, ticker)
        elif any(sym in level1 for sym in chunk):
            for sym in chunk:
                if sym in level1:
                    out[sym] = _clean_ohlcv(raw.xs(sym, axis=1, level=1))
        return out

    # Single symbol can come back as single-level columns.
    if len(chunk) == 1:
        out[chunk[0]] = _clean_ohlcv(raw)
    return out


def _download_chunk(chunk: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    start_s = start.date().isoformat()
    end_s = (end + pd.Timedelta(days=1)).date().isoformat()  # yfinance end is exclusive
    raw = yf.download(
        tickers=chunk,
        start=start_s,
        end=end_s,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
    )
    return _extract_symbol_frames(raw, chunk)


def download_market_data(
    symbols: list[str],
    benchmark: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int = 50,
) -> MarketDataStore:
    """Download OHLCV for symbols + benchmark and apply basic filtering."""
    universe = ensure_unique_symbols(symbols)
    symbol_frames: dict[str, pd.DataFrame] = {}

    for chunk in _chunked(universe, batch_size):
        frames = _download_chunk(chunk, start=start, end=end)
        for sym, df in frames.items():
            if df.empty:
                continue
            if df["Close"].notna().sum() < MIN_REQUIRED_TRADING_DAYS:
                continue
            symbol_frames[sym] = df

    benchmark_map = _download_chunk([benchmark], start=start, end=end)
    benchmark_frame = benchmark_map.get(benchmark)
    if benchmark_frame is None or benchmark_frame.empty:
        raise RuntimeError(f"ベンチマーク {benchmark} の取得に失敗しました。")

    if not symbol_frames:
        raise RuntimeError(
            "有効な銘柄データが取得できませんでした。銘柄リストまたは日付範囲を見直してください。"
        )

    return MarketDataStore(symbol_frames=symbol_frames, benchmark_frame=benchmark_frame)
