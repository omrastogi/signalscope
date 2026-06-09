"""
Rebuild rl_ready_dataset.csv from raw sources for the full 2020-2025 date range.

Reads:
  data/raw/market/{TICKER}_prices.csv    — OHLCV prices (2020-2025)
  data/raw/sec_filings/{TICKER}_filings_ai.csv — SEC AI keyword density

Outputs:
  data/processed/rl_ready_dataset.csv   — env-ready feature table

Columns produced:
  date, ticker, close, volume, daily_return,
  volume_zscore, rolling_volatility_20d, sec_ai_density

ai_trend_score is intentionally omitted here — environment._build_dataset()
merges it from google_trends_features.csv via merge_asof at load time.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW      = Path(__file__).parent / "raw"
PROCESSED = Path(__file__).parent / "processed"

# Tickers to include — GOOGL raw file is renamed to GOOG for env consistency
TICKER_MAP = {
    "AAPL":  "AAPL",
    "AMZN":  "AMZN",
    "GOOGL": "GOOG",
    "META":  "META",
    "MSFT":  "MSFT",
    "NVDA":  "NVDA",
}

WINDOW = 20   # days for rolling volatility and volume z-score


def load_prices(raw_ticker: str, env_ticker: str) -> pd.DataFrame:
    path = RAW / "market" / f"{raw_ticker}_prices.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    price_col = "adj_close" if "adj_close" in df.columns else "close"
    df["close"]        = df[price_col].astype(float)
    df["volume"]       = df["volume"].astype(float)
    df["daily_return"] = df["close"].pct_change()

    # 20-day rolling volatility (annualised std of daily returns)
    df["rolling_volatility_20d"] = (
        df["daily_return"].rolling(WINDOW, min_periods=5).std() * np.sqrt(252)
    )

    # Volume z-score vs 20-day rolling mean/std
    vol_mean = df["volume"].rolling(WINDOW, min_periods=5).mean()
    vol_std  = df["volume"].rolling(WINDOW, min_periods=5).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    df["ticker"] = env_ticker
    return df[["date", "ticker", "close", "volume", "daily_return",
               "volume_zscore", "rolling_volatility_20d"]]


def load_sec_density(raw_ticker: str, env_ticker: str, dates: pd.Series) -> pd.Series:
    path = RAW / "sec_filings" / f"{raw_ticker}_filings_ai.csv"
    if not path.exists():
        return pd.Series(0.0, index=dates.index, name="sec_ai_density")

    sec = pd.read_csv(path, parse_dates=["filing_date"])
    sec = sec.sort_values("filing_date")

    # Daily density: mean intensity on the filing date, then forward-fill
    daily = (
        sec.groupby("filing_date")["ai_intensity_per_1000_words"]
        .mean()
        .reindex(dates)
        .ffill()
        .fillna(0.0)
    )
    daily.index = dates.index
    daily.name  = "sec_ai_density"
    return daily


def main():
    frames = []
    for raw_ticker, env_ticker in TICKER_MAP.items():
        print(f"Processing {raw_ticker} -> {env_ticker}")
        df = load_prices(raw_ticker, env_ticker)
        sec = load_sec_density(raw_ticker, env_ticker, df["date"])
        df["sec_ai_density"] = sec.values
        frames.append(df)

    out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    out_path = PROCESSED / "rl_ready_dataset.csv"
    out.to_csv(out_path, index=False)

    print(f"\nWrote {len(out):,} rows to {out_path}")
    print(f"Date range : {out['date'].min().date()} -> {out['date'].max().date()}")
    print(f"Tickers    : {sorted(out['ticker'].unique())}")
    print(f"Rows/ticker: {len(out) // out['ticker'].nunique()}")


if __name__ == "__main__":
    main()
