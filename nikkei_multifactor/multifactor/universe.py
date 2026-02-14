"""Universe construction utilities for Nikkei225 symbols."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .utils import ensure_unique_symbols

NIKKEI_COMPONENT_URLS = [
    "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225",
    "https://en.wikipedia.org/wiki/Nikkei_225",
]

_CODE_COL_PATTERN = re.compile(r"(code|銘柄コード|証券コード|ticker)", flags=re.IGNORECASE)
_FOUR_DIGIT_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to yfinance format for Japan equities."""
    s = str(symbol).strip().upper()
    s = s.replace(" ", "")
    if not s:
        return s
    if s.endswith(".T"):
        return s
    if s.isdigit() and len(s) == 4:
        return f"{s}.T"
    return s


def _is_jp_ticker(symbol: str) -> bool:
    s = normalize_symbol(symbol)
    return bool(re.fullmatch(r"\d{4}\.T", s))


def _flatten_columns(df: pd.DataFrame) -> list[str]:
    flat: list[str] = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat.append(" ".join([str(x) for x in col if str(x) != ""]).strip())
        else:
            flat.append(str(col))
    return flat


def _extract_codes_from_table(df: pd.DataFrame) -> list[str]:
    table = df.copy()
    table.columns = _flatten_columns(table)

    candidate_cols = [c for c in table.columns if _CODE_COL_PATTERN.search(c)]
    if not candidate_cols:
        return []

    codes: list[str] = []
    for col in candidate_cols:
        text_values = table[col].astype(str)
        for text in text_values:
            matches = _FOUR_DIGIT_PATTERN.findall(text)
            for m in matches:
                codes.append(m)
    return codes


def fetch_nikkei225_symbols(min_symbols: int = 200) -> list[str]:
    """Fetch Nikkei225 component symbols from public pages."""
    raw_codes: list[str] = []
    for url in NIKKEI_COMPONENT_URLS:
        try:
            tables = pd.read_html(url)
        except Exception:
            continue
        for table in tables:
            raw_codes.extend(_extract_codes_from_table(table))

    symbols = [normalize_symbol(code) for code in raw_codes if code.isdigit() and len(code) == 4]
    symbols = ensure_unique_symbols(symbols)

    if len(symbols) < min_symbols:
        raise RuntimeError(
            "日経225構成銘柄の自動取得に失敗しました。"
            " --symbols_csv で銘柄一覧CSV（symbol列 or 1列目に 7203.T 等）を指定してください。"
        )
    return symbols


def load_symbols_from_csv(path: str | Path) -> list[str]:
    """Load symbols from a CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"銘柄CSVが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"銘柄CSVが空です: {csv_path}")

    lower_map = {str(c).lower(): c for c in df.columns}
    if "symbol" in lower_map:
        raw = df[lower_map["symbol"]]
    else:
        # symbol列が無い場合は、最も「4桁コードらしい」列を採用
        best_col = None
        best_score = -1
        for col in df.columns:
            values = df[col].dropna().astype(str)
            score = int(sum(_is_jp_ticker(v) for v in values))
            if score > best_score:
                best_score = score
                best_col = col
        raw = df[best_col] if best_col is not None else df.iloc[:, 0]

    symbols = [normalize_symbol(v) for v in raw.dropna().astype(str).tolist()]
    symbols = [s for s in symbols if s]
    symbols = ensure_unique_symbols(symbols)
    if not symbols:
        raise ValueError(f"銘柄CSVからシンボルを抽出できませんでした: {csv_path}")
    return symbols


def get_universe(symbols_csv: str | None = None) -> list[str]:
    """Build final tradable universe."""
    if symbols_csv:
        return load_symbols_from_csv(symbols_csv)
    return fetch_nikkei225_symbols()
