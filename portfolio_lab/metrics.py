"""Performance metrics computed from daily simple returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def annualized_return(r: pd.Series) -> float:
    """Compound annual growth rate implied by a series of daily returns."""
    n = len(r)
    if n == 0:
        return float("nan")
    return float((1 + r).prod() ** (TRADING_DAYS / n) - 1)


def annualized_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def summarize(r: pd.Series, rf: float = 0.0, start_value: float = 10_000) -> dict:
    """Headline numbers for one return series."""
    r = r.dropna()
    ann_ret = annualized_return(r)
    ann_vol = annualized_vol(r)
    return {
        "Total return": float((1 + r).prod() - 1),
        "Annual return": ann_ret,
        "Annual volatility": ann_vol,
        "Sharpe ratio": (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan"),
        "Worst drop (max drawdown)": float(drawdown_series(r).min()),
        "Growth of $10,000": float(start_value * (1 + r).prod()),
    }


def summary_table(returns: dict[str, pd.Series], rf: float = 0.0) -> pd.DataFrame:
    """One row per strategy, one column per metric."""
    return pd.DataFrame({name: summarize(r, rf) for name, r in returns.items()}).T
