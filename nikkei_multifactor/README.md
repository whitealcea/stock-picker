# Nikkei225 Multi-Factor Backtest (Standalone)

日経225向けのマルチファクター銘柄選定とバックテストを行う、独立実行可能なPythonプロジェクトです。

## 実装済み要件

- `yfinance` で日本株 (`.T`) と `^N225` を日足OHLCV取得
- ファクター
  - モメンタム（126日リターン）
  - 低ボラ（20日標準偏差の逆数）
  - RSI(2)（低RSIを高スコア化）
  - 200日移動平均ブレイク
  - 出来高比率（当日出来高 / 20日平均）
- 各ファクターを0-1正規化し、等重み合成スコアを算出
- レジーム判定（`^N225` が 200MA より上ならリスクオン、下なら保有銘柄数を半減）
- 上位K銘柄をATR(20)逆数で等リスク配分
- 月次リバランス
- 先見バイアス回避
  - シグナルは当日引けデータまで
  - 約定は翌営業日寄り
  - 月次リバランスも前営業日シグナル→当日寄り約定
- `asof` が非営業日の場合は直近営業日に丸め

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行例

### 銘柄抽出

```bash
python main.py --mode picks --asof 2026-01-14
```

出力: `outputs/picks_2026-01-14.csv`（例）

### バックテスト

```bash
python main.py --mode backtest --asof 2026-01-14 --to_date 2026-02-14
```

出力:

- `outputs/summary.json`
- `outputs/equity_curve.csv`

## オプション

- `--top_k`: 上位選択銘柄数（デフォルト20）
- `--symbols_csv`: 銘柄CSV指定（`symbol`列 or 1列目）
  - 未指定時は日経225構成銘柄の自動取得を試行
- `--benchmark`: ベンチマーク（デフォルト `^N225`）
- `--output_dir`: 出力先（デフォルト `outputs`）
