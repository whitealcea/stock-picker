"""Configuration values for the Nikkei225 multi-factor project."""

from __future__ import annotations

# Universe / benchmark
DEFAULT_BENCHMARK = "^N225"

# Factor windows (trading days)
MOMENTUM_WINDOW = 126
LOWVOL_WINDOW = 20
RSI_WINDOW = 2
MA_WINDOW = 200
VOLUME_WINDOW = 20
ATR_WINDOW = 20

# Data requirements
MIN_HISTORY_BUFFER_DAYS = 650  # calendar days; enough for rolling indicators
MIN_REQUIRED_TRADING_DAYS = 220

# Portfolio defaults
DEFAULT_TOP_K = 20
TRADING_DAYS_PER_YEAR = 252
