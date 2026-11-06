"""Loading price data.

The repo ships a snapshot of the original project's data so everything runs offline and
results are reproducible. Prices in the snapshot are rebased to 100 on the first day, which
is all that returns-based analysis needs. Use `download_prices` to refresh from Yahoo Finance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import BENCHMARK, TICKERS

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "dow30_prices_snapshot.csv"


def load_snapshot(path: Path | str = SNAPSHOT) -> tuple[pd.DataFrame, pd.Series]:
    """Return (stock prices, DJIA prices) from a CSV with a Date index and a DJIA column."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    bench = df[BENCHMARK]
    stocks = df.drop(columns=BENCHMARK)
    return stocks, bench


def download_prices(
    start: str = "2020-07-01",
    end: str = "2024-12-31",
    cache: Path | str | None = None,
    rebase: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download Dow 30 and DJIA closes from Yahoo Finance (end date is exclusive).

    Defaults match the original project window. With `cache` set, the result is written to
    CSV in the same layout as the snapshot so `load_snapshot` can read it back.
    """
    import yfinance as yf  # imported lazily so the rest of the package works offline

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    dji = yf.download("^DJI", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(dji, pd.DataFrame):
        dji = dji.iloc[:, 0]
    df = raw[TICKERS].copy()
    df[BENCHMARK] = dji
    df = df.sort_index().ffill().dropna()
    if rebase:
        df = df / df.iloc[0] * 100
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, float_format="%.10f")
    return df[TICKERS], df[BENCHMARK]
