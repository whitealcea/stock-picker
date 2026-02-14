"""CLI entrypoint for Nikkei225 multi-factor picks/backtest."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from multifactor.backtest import generate_picks, run_backtest
from multifactor.config import DEFAULT_BENCHMARK, DEFAULT_TOP_K, MIN_HISTORY_BUFFER_DAYS
from multifactor.data import download_market_data
from multifactor.universe import get_universe
from multifactor.utils import dumps_pretty_json, ensure_output_dir, parse_date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nikkei225 multi-factor stock picker and backtester"
    )
    parser.add_argument("--mode", required=True, choices=["picks", "backtest"])
    parser.add_argument("--asof", required=True, help="シグナル基準日 (YYYY-MM-DD)")
    parser.add_argument("--to_date", help="バックテスト終了日 (YYYY-MM-DD)")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="選択銘柄数 K")
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="ベンチマークシンボル (default: ^N225)",
    )
    parser.add_argument(
        "--symbols_csv",
        help="銘柄一覧CSV（symbol列 or 1列目）。未指定時は日経225構成銘柄を自動取得。",
    )
    parser.add_argument("--output_dir", default="outputs", help="出力先ディレクトリ")
    return parser


def _prepare_data(
    asof: pd.Timestamp,
    end_date: pd.Timestamp,
    benchmark: str,
    symbols_csv: str | None,
):
    symbols = get_universe(symbols_csv=symbols_csv)
    start_date = min(asof, end_date) - pd.Timedelta(days=MIN_HISTORY_BUFFER_DAYS)
    return download_market_data(
        symbols=symbols,
        benchmark=benchmark,
        start=start_date,
        end=end_date,
    )


def run_cli(args: argparse.Namespace) -> int:
    asof = parse_date(args.asof)
    output_dir = ensure_output_dir(args.output_dir)

    if args.top_k <= 0:
        raise ValueError("--top_k は1以上を指定してください。")

    if args.mode == "picks":
        store = _prepare_data(
            asof=asof,
            end_date=asof,
            benchmark=args.benchmark,
            symbols_csv=args.symbols_csv,
        )
        picks_result = generate_picks(store, asof=asof, top_k=args.top_k)
        if picks_result.picks.empty:
            raise RuntimeError("買い候補が0件です。asofや銘柄ユニバースを見直してください。")

        output = picks_result.picks.reset_index().rename(columns={"index": "symbol"})
        output.insert(1, "asof_aligned", picks_result.asof_aligned.date().isoformat())
        output.insert(2, "risk_on", picks_result.risk_on)
        output.insert(3, "effective_k", picks_result.effective_k)

        picks_path = output_dir / f"picks_{picks_result.asof_aligned.date().isoformat()}.csv"
        output.to_csv(picks_path, index=False)
        print(f"[picks] rows={len(output)}")
        print(f"[picks] asof_aligned={picks_result.asof_aligned.date().isoformat()}")
        print(f"[picks] saved={picks_path}")
        return 0

    if args.mode == "backtest":
        if not args.to_date:
            raise ValueError("--mode backtest では --to_date が必須です。")
        to_date = parse_date(args.to_date)

        store = _prepare_data(
            asof=asof,
            end_date=to_date,
            benchmark=args.benchmark,
            symbols_csv=args.symbols_csv,
        )
        bt_result = run_backtest(store, asof=asof, to_date=to_date, top_k=args.top_k)

        equity = bt_result.equity_curve.reset_index()
        equity["date"] = pd.to_datetime(equity["date"]).dt.date.astype(str)
        equity_path = output_dir / "equity_curve.csv"
        equity.to_csv(equity_path, index=False)

        summary_data = dict(bt_result.summary)
        summary_data["rebalances"] = bt_result.rebalances
        summary_path = output_dir / "summary.json"
        summary_path.write_text(dumps_pretty_json(summary_data), encoding="utf-8")

        print(f"[backtest] asof_aligned={bt_result.asof_aligned.date().isoformat()}")
        print(f"[backtest] entry_date={bt_result.entry_date.date().isoformat()}")
        print(f"[backtest] to_date_aligned={bt_result.to_date_aligned.date().isoformat()}")
        print(f"[backtest] final_equity={bt_result.summary['final_equity']:.6f}")
        print(f"[backtest] saved={summary_path} , {equity_path}")
        return 0

    raise ValueError(f"未対応モードです: {args.mode}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_cli(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
