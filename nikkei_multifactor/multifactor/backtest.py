"""Signal generation and backtest simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import MarketDataStore
from .factors import compute_factor_snapshot
from .strategy import build_target_portfolio, effective_k, is_risk_on
from .utils import (
    align_to_previous_trading_day,
    calc_performance_metrics,
    first_trading_days_by_month,
    next_trading_day,
    previous_trading_day,
)


@dataclass
class PickResult:
    asof_aligned: pd.Timestamp
    risk_on: bool
    effective_k: int
    picks: pd.DataFrame


@dataclass
class BacktestResult:
    asof_aligned: pd.Timestamp
    to_date_aligned: pd.Timestamp
    entry_date: pd.Timestamp
    equity_curve: pd.DataFrame
    summary: dict[str, Any]
    rebalances: list[dict[str, Any]]


def generate_picks(store: MarketDataStore, asof: pd.Timestamp, top_k: int) -> PickResult:
    """Generate picks at asof close (or previous trading day if non-trading)."""
    calendar = store.trading_days
    asof_aligned = align_to_previous_trading_day(calendar, asof, "asof")

    snapshot = compute_factor_snapshot(store, signal_date=asof_aligned)
    risk_on = is_risk_on(store.benchmark_frame, signal_date=asof_aligned)
    selected = build_target_portfolio(snapshot, top_k=top_k, risk_on=risk_on)
    eff_k = effective_k(top_k, risk_on)

    return PickResult(
        asof_aligned=asof_aligned,
        risk_on=risk_on,
        effective_k=eff_k,
        picks=selected,
    )


@dataclass
class _RebalanceDecision:
    exec_date: pd.Timestamp
    signal_date: pd.Timestamp
    risk_on: bool
    target: pd.DataFrame


def _symbol_intra_return(frame: pd.DataFrame, day: pd.Timestamp) -> float:
    if day not in frame.index:
        return 0.0
    row = frame.loc[day]
    op, cl = row["Open"], row["Close"]
    if pd.isna(op) or pd.isna(cl) or op <= 0:
        return 0.0
    return float(cl / op - 1.0)


def _symbol_overnight_return(frame: pd.DataFrame, day: pd.Timestamp, prev_day: pd.Timestamp) -> float:
    if day not in frame.index or prev_day not in frame.index:
        return 0.0
    op = frame.loc[day, "Open"]
    prev_close = frame.loc[prev_day, "Close"]
    if pd.isna(op) or pd.isna(prev_close) or prev_close <= 0:
        return 0.0
    return float(op / prev_close - 1.0)


def _symbol_close_to_close_return(frame: pd.DataFrame, day: pd.Timestamp, prev_day: pd.Timestamp) -> float:
    if day not in frame.index or prev_day not in frame.index:
        return 0.0
    cl = frame.loc[day, "Close"]
    prev_close = frame.loc[prev_day, "Close"]
    if pd.isna(cl) or pd.isna(prev_close) or prev_close <= 0:
        return 0.0
    return float(cl / prev_close - 1.0)


def _weighted_return(
    weights: dict[str, float],
    store: MarketDataStore,
    return_func,
    day: pd.Timestamp,
    prev_day: pd.Timestamp | None = None,
) -> float:
    ret = 0.0
    for symbol, w in weights.items():
        frame = store.symbol_frames.get(symbol)
        if frame is None:
            continue
        if prev_day is None:
            sym_ret = return_func(frame, day)
        else:
            sym_ret = return_func(frame, day, prev_day)
        ret += w * sym_ret
    return float(ret)


def _calc_turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    symbols = set(old_weights) | set(new_weights)
    return float(sum(abs(new_weights.get(s, 0.0) - old_weights.get(s, 0.0)) for s in symbols))


def _build_rebalance_plan(
    store: MarketDataStore,
    asof_aligned: pd.Timestamp,
    to_date_aligned: pd.Timestamp,
    top_k: int,
) -> tuple[list[_RebalanceDecision], pd.Timestamp, pd.DatetimeIndex]:
    calendar = store.trading_days
    entry_date = next_trading_day(calendar, asof_aligned)

    bt_days = calendar[(calendar >= entry_date) & (calendar <= to_date_aligned)]
    if len(bt_days) == 0:
        raise ValueError("バックテスト期間が空です。to_date を見直してください。")

    month_first_days = first_trading_days_by_month(bt_days)
    rebalance_exec_dates = sorted(set([entry_date] + month_first_days))

    plan: list[_RebalanceDecision] = []
    for exec_date in rebalance_exec_dates:
        signal_date = asof_aligned if exec_date == entry_date else previous_trading_day(calendar, exec_date)
        snapshot = compute_factor_snapshot(store, signal_date=signal_date)
        risk_on = is_risk_on(store.benchmark_frame, signal_date=signal_date)
        target = build_target_portfolio(snapshot, top_k=top_k, risk_on=risk_on)
        plan.append(
            _RebalanceDecision(
                exec_date=exec_date,
                signal_date=signal_date,
                risk_on=risk_on,
                target=target,
            )
        )
    return plan, entry_date, bt_days


def run_backtest(
    store: MarketDataStore,
    asof: pd.Timestamp,
    to_date: pd.Timestamp,
    top_k: int,
) -> BacktestResult:
    """Run backtest from next business day open after asof until to_date."""
    calendar = store.trading_days
    asof_aligned = align_to_previous_trading_day(calendar, asof, "asof")
    to_date_aligned = align_to_previous_trading_day(calendar, to_date, "to_date")
    if to_date_aligned <= asof_aligned:
        raise ValueError("to_date は asof より後の日付を指定してください。")

    plan, entry_date, bt_days = _build_rebalance_plan(
        store=store,
        asof_aligned=asof_aligned,
        to_date_aligned=to_date_aligned,
        top_k=top_k,
    )
    plan_map = {p.exec_date: p for p in plan}

    current_weights: dict[str, float] = {}
    prev_day: pd.Timestamp | None = None
    equity = 1.0
    benchmark_equity = 1.0
    rows: list[dict[str, Any]] = []
    rebalance_log: list[dict[str, Any]] = []

    benchmark_frame = store.benchmark_frame

    for day in bt_days:
        day = pd.Timestamp(day).normalize()
        is_rebalance_day = day in plan_map
        turnover = 0.0

        if is_rebalance_day:
            decision = plan_map[day]
            new_weights = (
                decision.target["weight"].to_dict()
                if not decision.target.empty and "weight" in decision.target.columns
                else {}
            )
            turnover = _calc_turnover(current_weights, new_weights)

            if prev_day is None:
                # 初回は寄りで新規建てするため当日の日中リターンのみ
                daily_ret = _weighted_return(
                    weights=new_weights,
                    store=store,
                    return_func=_symbol_intra_return,
                    day=day,
                    prev_day=None,
                )
            else:
                # リバランス日は、前日引け->当日寄り(旧ウェイト) + 当日日中(新ウェイト)
                overnight = _weighted_return(
                    weights=current_weights,
                    store=store,
                    return_func=_symbol_overnight_return,
                    day=day,
                    prev_day=prev_day,
                )
                intraday = _weighted_return(
                    weights=new_weights,
                    store=store,
                    return_func=_symbol_intra_return,
                    day=day,
                    prev_day=None,
                )
                daily_ret = (1.0 + overnight) * (1.0 + intraday) - 1.0

            current_weights = new_weights

            rebalance_log.append(
                {
                    "exec_date": day.date().isoformat(),
                    "signal_date": decision.signal_date.date().isoformat(),
                    "risk_on": decision.risk_on,
                    "effective_k": len(current_weights),
                    "symbols": sorted(current_weights.keys()),
                }
            )
        else:
            if prev_day is None:
                daily_ret = 0.0
            else:
                daily_ret = _weighted_return(
                    weights=current_weights,
                    store=store,
                    return_func=_symbol_close_to_close_return,
                    day=day,
                    prev_day=prev_day,
                )

        # Benchmark: 初日は寄り->引け、2日目以降は引け->引けで評価
        if prev_day is None:
            benchmark_ret = _symbol_intra_return(benchmark_frame, day)
        else:
            benchmark_ret = _symbol_close_to_close_return(benchmark_frame, day, prev_day)

        equity *= 1.0 + daily_ret
        benchmark_equity *= 1.0 + benchmark_ret

        rows.append(
            {
                "date": day,
                "daily_return": daily_ret,
                "equity": equity,
                "benchmark_daily_return": benchmark_ret,
                "benchmark_equity": benchmark_equity,
                "holdings": len(current_weights),
                "turnover": turnover,
            }
        )
        prev_day = day

    equity_curve = pd.DataFrame(rows).set_index("date")
    metrics = calc_performance_metrics(
        daily_returns=equity_curve["daily_return"],
        equity=equity_curve["equity"],
        benchmark_returns=equity_curve["benchmark_daily_return"],
    )

    summary: dict[str, Any] = {
        "asof_input": asof.date().isoformat(),
        "asof_aligned": asof_aligned.date().isoformat(),
        "entry_date": entry_date.date().isoformat(),
        "to_date_input": to_date.date().isoformat(),
        "to_date_aligned": to_date_aligned.date().isoformat(),
        "top_k_requested": int(top_k),
        "num_rebalances": len(rebalance_log),
        "final_equity": float(equity_curve["equity"].iloc[-1]),
        "final_benchmark_equity": float(equity_curve["benchmark_equity"].iloc[-1]),
        **metrics,
    }

    return BacktestResult(
        asof_aligned=asof_aligned,
        to_date_aligned=to_date_aligned,
        entry_date=entry_date,
        equity_curve=equity_curve,
        summary=summary,
        rebalances=rebalance_log,
    )
