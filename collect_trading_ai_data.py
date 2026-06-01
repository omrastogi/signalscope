#!/usr/bin/env python3
"""
Collect data for: Stock Trading with Tabular Q-Learning.

Data sources:
1. Market prices and volume via yfinance.
2. AI search interest via pytrends / Google Trends. 
3. AI-related keyword intensity from SEC EDGAR filings.
* For non-daily frequency data, we use forward filling

Output:
- raw/market/{TICKER}_prices.csv
- raw/google_trends/{TICKER}_trends.csv
- raw/sec_filings/{TICKER}_filings_ai.csv
- processed/{TICKER}_daily_features.csv
- processed/all_daily_features.csv
- metadata.json

Example:
python collect_trading_ai_data.py \
  --start 2020-01-01 \
  --end 2025-12-31 \
  --tickers NVDA MSFT GOOGL META AMZN AAPL AMD \
  --sec-user-agent "name email" \
  --out data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE: Dict[str, Dict[str, str]] = {
    "NVDA": {"company": "NVIDIA", "cik": "0001045810", "trend_keyword": "Nvidia AI"},
    "MSFT": {"company": "Microsoft", "cik": "0000789019", "trend_keyword": "Microsoft AI"},
    "GOOGL": {"company": "Alphabet", "cik": "0001652044", "trend_keyword": "Google AI"},
    "META": {"company": "Meta Platforms", "cik": "0001326801", "trend_keyword": "Meta AI"},
    "AMZN": {"company": "Amazon", "cik": "0001018724", "trend_keyword": "Amazon AI"},
    "AAPL": {"company": "Apple", "cik": "0000320193", "trend_keyword": "Apple AI"},
    "AMD": {"company": "Advanced Micro Devices", "cik": "0000002488", "trend_keyword": "AMD AI"},
    "ORCL": {"company": "Oracle", "cik": "0001341439", "trend_keyword": "Oracle AI"},
    "TSLA": {"company": "Tesla", "cik": "0001318605", "trend_keyword": "Tesla AI"},
    "CRM": {"company": "Salesforce", "cik": "0001108524", "trend_keyword": "Salesforce AI"},
    "ADBE": {"company": "Adobe", "cik": "0000796343", "trend_keyword": "Adobe AI"},
}

AI_TERM_PATTERNS: List[str] = [
    r"\bartificial intelligence\b",
    r"\bgenerative\s+ai\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bneural networks?\b",
    r"\blarge language models?\b",
    r"\bLLMs?\b",
    r"\bfoundation models?\b",
    r"\bAI\b",
]


@dataclass
class CollectorConfig:
    start: str
    end: str
    tickers: List[str]
    out_dir: Path
    sec_user_agent: str
    sec_forms: List[str]
    geo: str
    trend_threshold: float
    volume_threshold: float
    train_end: str
    trend_shift_days: int
    filing_shift_days: int
    skip_trends: bool
    skip_sec: bool
    sec_delay_seconds: float
    max_sec_filings_per_ticker: Optional[int]


def parse_args() -> CollectorConfig:
    parser = argparse.ArgumentParser(
        description="Collect market, Google Trends, and SEC filing AI-signal data for the TradingAI Q-learning project."
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date, inclusive, YYYY-MM-DD.")
    parser.add_argument("--end", default="2025-12-31", help="End date, inclusive, YYYY-MM-DD.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["NVDA", "MSFT", "GOOGL", "META", "AMZN", "AAPL", "AMD"],
        help="Ticker symbols to collect.",
    )
    parser.add_argument("--out", default="data", help="Output directory.")
    parser.add_argument(
        "--sec-user-agent",
        default=os.environ.get("SEC_USER_AGENT", ""),
        help='Required by SEC. Example: "Your Name your_email@example.com". Can also set SEC_USER_AGENT env var.',
    )
    parser.add_argument(
        "--sec-forms",
        nargs="+",
        default=["10-K", "10-Q", "8-K"],
        help="SEC filing forms to include.",
    )
    parser.add_argument("--geo", default="US", help="Google Trends geography; use US for this project.")
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.02,
        help="5-trading-day price trend threshold. Default 0.02 means +/-2%%.",
    )
    parser.add_argument(
        "--volume-threshold",
        type=float,
        default=1.5,
        help="Abnormal-volume threshold as volume / 20-day average volume.",
    )
    parser.add_argument(
        "--train-end",
        default="2023-12-31",
        help="End date for fitting low/medium/high bucket thresholds without test leakage.",
    )
    parser.add_argument(
        "--trend-shift-days",
        type=int,
        default=5,
        help="Business-day lag applied to Google Trends signal to reduce look-ahead risk.",
    )
    parser.add_argument(
        "--filing-shift-days",
        type=int,
        default=1,
        help="Business-day lag applied to SEC filing signal to reduce same-day look-ahead risk.",
    )
    parser.add_argument("--skip-trends", action="store_true", help="Skip Google Trends collection and fill trend signal with NaN.")
    parser.add_argument("--skip-sec", action="store_true", help="Skip SEC filing collection and fill filing signal with 0.")
    parser.add_argument(
        "--sec-delay-seconds",
        type=float,
        default=0.15,
        help="Delay between SEC HTTP requests. Keep this polite; SEC rate guidance is at most 10 requests/second.",
    )
    parser.add_argument(
        "--max-sec-filings-per-ticker",
        type=int,
        default=None,
        help="Optional cap for SEC filings per ticker after date/form filtering. Useful for a quick dry run.",
    )
    args = parser.parse_args()

    tickers = [t.upper().strip() for t in args.tickers]
    unknown = [t for t in tickers if t not in DEFAULT_UNIVERSE]
    if unknown:
        raise SystemExit(
            f"Unknown ticker(s) without built-in CIK mapping: {unknown}. "
            "Add them to DEFAULT_UNIVERSE in the script before running."
        )

    if not args.skip_sec and not args.sec_user_agent.strip():
        raise SystemExit(
            "SEC requires a descriptive User-Agent. Re-run with, for example:\n"
            "  --sec-user-agent \"Your Name your_email@example.com\"\n"
            "or set environment variable SEC_USER_AGENT."
        )

    return CollectorConfig(
        start=args.start,
        end=args.end,
        tickers=tickers,
        out_dir=Path(args.out),
        sec_user_agent=args.sec_user_agent.strip(),
        sec_forms=[f.upper() for f in args.sec_forms],
        geo=args.geo,
        trend_threshold=args.trend_threshold,
        volume_threshold=args.volume_threshold,
        train_end=args.train_end,
        trend_shift_days=args.trend_shift_days,
        filing_shift_days=args.filing_shift_days,
        skip_trends=args.skip_trends,
        skip_sec=args.skip_sec,
        sec_delay_seconds=args.sec_delay_seconds,
        max_sec_filings_per_ticker=args.max_sec_filings_per_ticker,
    )


def make_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "market": out_dir / "raw" / "market",
        "trends": out_dir / "raw" / "google_trends",
        "sec": out_dir / "raw" / "sec_filings",
        "sec_text": out_dir / "raw" / "sec_filings_text",
        "processed": out_dir / "processed",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def end_exclusive(end: str) -> str:
    return (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Missing dependency yfinance. Install with: pip install -r requirements.txt") from exc

    print(f"[market] Downloading {ticker} prices...")
    df = yf.download(
        ticker,
        start=start,
        end=end_exclusive(end),
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No market data returned for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance versions differ in MultiIndex layout. Drop a ticker level if present.
        for level in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(level):
                df = df.xs(ticker, level=level, axis=1)
                break

    df = df.reset_index()
    rename = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["ticker"] = ticker
    return df.sort_values("date")


def fetch_google_trends(keyword: str, start: str, end: str, geo: str) -> pd.DataFrame:
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise RuntimeError("Missing dependency pytrends. Install with: pip install -r requirements.txt") from exc

    print(f"[trends] Downloading Google Trends for {keyword!r}...")
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    timeframe = f"{start} {end}"

    # Google Trends can rate-limit. Use a few retries with increasing waits.
    last_exc: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop="")
            df = pytrends.interest_over_time()
            if df.empty:
                return pd.DataFrame(columns=["date", "google_trends_ai_interest", "trend_keyword"])
            df = df.reset_index().rename(columns={"date": "date", keyword: "google_trends_ai_interest"})
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df["trend_keyword"] = keyword
            return df[["date", "google_trends_ai_interest", "trend_keyword"]].sort_values("date")
        except Exception as exc:  # pytrends raises several non-specific request exceptions
            last_exc = exc
            wait = 15 * attempt
            print(f"[trends] Attempt {attempt} failed for {keyword!r}: {exc}. Waiting {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Google Trends failed for {keyword!r}: {last_exc}")


def sec_session(user_agent: str):
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Missing dependency requests. Install with: pip install -r requirements.txt") from exc
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def sec_get(session, url: str, delay_seconds: float, as_json: bool = False):
    time.sleep(delay_seconds)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    if as_json:
        return response.json()
    return response.content


def cik10(cik: str) -> str:
    return str(int(cik)).zfill(10)


def cik_no_leading(cik: str) -> str:
    return str(int(cik))


def recent_filings_to_df(recent: Dict[str, List[Any]]) -> pd.DataFrame:
    if not recent:
        return pd.DataFrame()
    keys = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"]
    available = {k: recent.get(k, []) for k in keys if k in recent}
    if not available:
        return pd.DataFrame()
    return pd.DataFrame(available)


def fetch_sec_filing_index(session, cik: str, delay_seconds: float) -> pd.DataFrame:
    cik_padded = cik10(cik)
    base = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    root = sec_get(session, base, delay_seconds=delay_seconds, as_json=True)

    frames = []
    recent_df = recent_filings_to_df(root.get("filings", {}).get("recent", {}))
    if not recent_df.empty:
        frames.append(recent_df)

    for file_info in root.get("filings", {}).get("files", []):
        name = file_info.get("name")
        if not name:
            continue
        url = f"https://data.sec.gov/submissions/{name}"
        try:
            data = sec_get(session, url, delay_seconds=delay_seconds, as_json=True)
            df = recent_filings_to_df(data)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"[sec] Warning: failed to read submissions file {name}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["accessionNumber"])
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce").dt.normalize()
    return df.dropna(subset=["filingDate", "accessionNumber", "form"]).sort_values("filingDate")


def sec_archive_url(cik: str, accession_number: str, primary_document: str) -> str:
    accession_clean = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading(cik)}/{accession_clean}/{primary_document}"


def html_bytes_to_text(content: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Missing dependency beautifulsoup4. Install with: pip install -r requirements.txt") from exc

    # Most SEC primary documents are HTML or inline XBRL. Fall back to replacement decoding.
    raw = content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "table"]):
        # Tables often include duplicated XBRL values. Remove them for a cleaner narrative-text signal.
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_ai_terms(text: str) -> Tuple[int, int, float]:
    if not text:
        return 0, 0, 0.0
    word_count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    hit_count = 0
    for pattern in AI_TERM_PATTERNS:
        hit_count += len(re.findall(pattern, text, flags=re.IGNORECASE))
    intensity = hit_count / max(word_count / 1000.0, 1.0)
    return hit_count, word_count, intensity


def fetch_sec_ai_signal(
    ticker: str,
    cik: str,
    start: str,
    end: str,
    forms: List[str],
    user_agent: str,
    delay_seconds: float,
    text_dir: Path,
    max_filings: Optional[int],
) -> pd.DataFrame:
    print(f"[sec] Downloading SEC filing index for {ticker}...")
    session = sec_session(user_agent)
    idx = fetch_sec_filing_index(session, cik, delay_seconds=delay_seconds)
    if idx.empty:
        return pd.DataFrame()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    idx = idx[
        (idx["filingDate"] >= start_ts)
        & (idx["filingDate"] <= end_ts)
        & (idx["form"].str.upper().isin([f.upper() for f in forms]))
    ].copy()
    idx = idx.sort_values("filingDate")
    if max_filings is not None and len(idx) > max_filings:
        print(f"[sec] Limiting {ticker} to first {max_filings} filings after filtering.")
        idx = idx.head(max_filings)

    ticker_text_dir = text_dir / ticker
    ticker_text_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for _, row in idx.iterrows():
        accession = row["accessionNumber"]
        primary_doc = row.get("primaryDocument", "")
        form = row.get("form", "")
        filing_date = row["filingDate"]
        if not primary_doc or not isinstance(primary_doc, str):
            continue

        text_path = ticker_text_dir / f"{accession}_{form.replace('/', '-')}.txt"
        url = sec_archive_url(cik, accession, primary_doc)

        try:
            if text_path.exists() and text_path.stat().st_size > 0:
                text = text_path.read_text(encoding="utf-8", errors="replace")
            else:
                print(f"[sec] {ticker} {filing_date.date()} {form} {accession}")
                content = sec_get(session, url, delay_seconds=delay_seconds, as_json=False)
                text = html_bytes_to_text(content)
                text_path.write_text(text, encoding="utf-8")

            hits, words, intensity = count_ai_terms(text)
            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik10(cik),
                    "filing_date": filing_date,
                    "form": form,
                    "accession_number": accession,
                    "primary_document": primary_doc,
                    "filing_url": url,
                    "ai_keyword_hits": hits,
                    "word_count": words,
                    "ai_intensity_per_1000_words": intensity,
                }
            )
        except Exception as exc:
            print(f"[sec] Warning: failed {ticker} {accession}: {exc}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("filing_date")


def bucket_low_medium_high(series: pd.Series, train_mask: pd.Series) -> Tuple[pd.Series, Dict[str, Optional[float]]]:
    train = series[train_mask].dropna()
    if train.empty:
        return pd.Series(["low"] * len(series), index=series.index), {"q33": None, "q66": None}

    q33 = float(train.quantile(0.33))
    q66 = float(train.quantile(0.66))
    if math.isclose(q33, q66):
        unique_vals = sorted(train.unique())
        if len(unique_vals) >= 3:
            q33 = float(unique_vals[len(unique_vals) // 3])
            q66 = float(unique_vals[(2 * len(unique_vals)) // 3])
        elif len(unique_vals) == 2:
            q33 = float(unique_vals[0])
            q66 = float(unique_vals[0])
        else:
            q33 = q66 = float(unique_vals[0])

    labels = pd.Series(index=series.index, dtype="object")
    labels[series <= q33] = "low"
    labels[(series > q33) & (series <= q66)] = "medium"
    labels[series > q66] = "high"
    labels = labels.fillna("low")
    return labels, {"q33": q33, "q66": q66}


def bucket_filing_intensity(series: pd.Series, train_mask: pd.Series) -> Tuple[pd.Series, Dict[str, Optional[float]]]:
    # Many days have no filing signal. Treat zero as low, and split positive values into medium/high.
    train_pos = series[train_mask & (series > 0)].dropna()
    if train_pos.empty:
        labels = pd.Series(np.where(series > 0, "high", "low"), index=series.index)
        return labels, {"positive_q66": None}
    q66 = float(train_pos.quantile(0.66))
    labels = pd.Series("low", index=series.index, dtype="object")
    labels[(series > 0) & (series <= q66)] = "medium"
    labels[series > q66] = "high"
    return labels, {"positive_q66": q66}


def align_and_make_features(
    ticker: str,
    price_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame],
    filings_df: Optional[pd.DataFrame],
    config: CollectorConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    px = price_df.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.sort_values("date").set_index("date")

    features = px[["open", "high", "low", "close", "adj_close", "volume"]].copy()
    features["ticker"] = ticker
    features["return_1d"] = features["adj_close"].pct_change()
    features["trend_5d_return"] = features["adj_close"].pct_change(5)
    features["volume_ma_20"] = features["volume"].rolling(20, min_periods=5).mean()
    features["volume_ratio_20"] = features["volume"] / features["volume_ma_20"]

    if trends_df is not None and not trends_df.empty:
        gt = trends_df.copy()
        gt["date"] = pd.to_datetime(gt["date"]).dt.normalize()
        gt = gt.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        g_series = gt["google_trends_ai_interest"].reindex(features.index, method="ffill")
        features["google_trends_ai_interest_raw"] = g_series.shift(config.trend_shift_days)
    else:
        features["google_trends_ai_interest_raw"] = np.nan

    if filings_df is not None and not filings_df.empty:
        fd = filings_df.copy()
        fd["filing_date"] = pd.to_datetime(fd["filing_date"]).dt.normalize()
        daily_f = (
            fd.groupby("filing_date")["ai_intensity_per_1000_words"]
            .mean()
            .sort_index()
            .rename("sec_ai_intensity_raw")
        )
        f_series = daily_f.reindex(features.index, method="ffill").fillna(0.0)
        features["sec_ai_intensity_raw"] = f_series.shift(config.filing_shift_days).fillna(0.0)
    else:
        features["sec_ai_intensity_raw"] = 0.0

    train_mask = features.index <= pd.Timestamp(config.train_end)
    g_bucket, g_thresholds = bucket_low_medium_high(features["google_trends_ai_interest_raw"].fillna(0), train_mask)
    f_bucket, f_thresholds = bucket_filing_intensity(features["sec_ai_intensity_raw"].fillna(0), train_mask)

    features["G_bucket"] = g_bucket
    features["F_bucket"] = f_bucket
    features["T_bucket"] = np.select(
        [
            features["trend_5d_return"] < -config.trend_threshold,
            features["trend_5d_return"] > config.trend_threshold,
        ],
        ["down", "up"],
        default="neutral",
    )
    features["V_bucket"] = np.where(features["volume_ratio_20"] >= config.volume_threshold, "high", "normal")

    # P_t is portfolio state and should be created inside the trading environment, not in the static dataset.
    features["state_without_position"] = (
        features["G_bucket"].astype(str)
        + "|"
        + features["F_bucket"].astype(str)
        + "|"
        + features["T_bucket"].astype(str)
        + "|"
        + features["V_bucket"].astype(str)
    )

    features = features.reset_index()
    metadata = {
        "ticker": ticker,
        "G_bucket_thresholds": g_thresholds,
        "F_bucket_thresholds": f_thresholds,
        "trend_threshold": config.trend_threshold,
        "volume_threshold": config.volume_threshold,
        "trend_shift_days": config.trend_shift_days,
        "filing_shift_days": config.filing_shift_days,
    }
    return features, metadata


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] {path}")


def main() -> None:
    config = parse_args()
    dirs = make_dirs(config.out_dir)

    all_features: List[pd.DataFrame] = []
    all_metadata: Dict[str, Any] = {
        "config": {
            "start": config.start,
            "end": config.end,
            "tickers": config.tickers,
            "sec_forms": config.sec_forms,
            "geo": config.geo,
            "train_end": config.train_end,
            "trend_threshold": config.trend_threshold,
            "volume_threshold": config.volume_threshold,
            "trend_shift_days": config.trend_shift_days,
            "filing_shift_days": config.filing_shift_days,
        },
        "universe": {t: DEFAULT_UNIVERSE[t] for t in config.tickers},
        "tickers": {},
        "notes": [
            "P_bucket is not included because portfolio position is generated by the trading environment.",
            "Google Trends and SEC signals are shifted to reduce look-ahead bias.",
            "G/F bucket thresholds are fit on dates <= train_end unless fallback thresholds are needed.",
        ],
    }

    for ticker in config.tickers:
        meta = DEFAULT_UNIVERSE[ticker]
        print("=" * 80)
        print(f"Collecting {ticker}: {meta['company']}")

        prices = fetch_prices(ticker, config.start, config.end)
        save_csv(prices, dirs["market"] / f"{ticker}_prices.csv")

        trends = pd.DataFrame()
        if not config.skip_trends:
            try:
                trends = fetch_google_trends(meta["trend_keyword"], config.start, config.end, config.geo)
            except Exception as exc:
                print(f"[trends] Warning: using NaN trend data for {ticker}: {exc}")
        save_csv(trends, dirs["trends"] / f"{ticker}_trends.csv")

        filings = pd.DataFrame()
        if not config.skip_sec:
            try:
                filings = fetch_sec_ai_signal(
                    ticker=ticker,
                    cik=meta["cik"],
                    start=config.start,
                    end=config.end,
                    forms=config.sec_forms,
                    user_agent=config.sec_user_agent,
                    delay_seconds=config.sec_delay_seconds,
                    text_dir=dirs["sec_text"],
                    max_filings=config.max_sec_filings_per_ticker,
                )
            except Exception as exc:
                print(f"[sec] Warning: using zero SEC filing signal for {ticker}: {exc}")
        save_csv(filings, dirs["sec"] / f"{ticker}_filings_ai.csv")

        features, ticker_metadata = align_and_make_features(ticker, prices, trends, filings, config)
        save_csv(features, dirs["processed"] / f"{ticker}_daily_features.csv")
        all_features.append(features)
        all_metadata["tickers"][ticker] = ticker_metadata

    if all_features:
        combined = pd.concat(all_features, ignore_index=True).sort_values(["ticker", "date"])
        save_csv(combined, dirs["processed"] / "all_daily_features.csv")

    metadata_path = config.out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(all_metadata, indent=2, default=str), encoding="utf-8")
    print(f"[save] {metadata_path}")
    print("Done.")


if __name__ == "__main__":
    main()
