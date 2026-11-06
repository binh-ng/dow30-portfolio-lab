"""Honest testing: optimize on the past, judge on data the optimizer never saw.

Two approaches:
  * train_test: fit once on an early window, then hold those weights on the later window
  * walk_forward: refit on a rolling window every few weeks, using only data available then
Portfolios are held at constant weights (rebalanced daily) between refits, matching the
original project's convention. Transaction costs are charged on each refit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimize import PortfolioResult, StrategyConfig


def portfolio_returns(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily returns of a constant-weight mix."""
    rets = prices.pct_change().dropna()
    return rets.reindex(columns=weights.index).fillna(0.0).dot(weights)


@dataclass
class TrainTestResult:
    result: PortfolioResult
    train_returns: pd.Series
    test_returns: pd.Series
    train_end: pd.Timestamp


def train_test(prices: pd.DataFrame, cfg: StrategyConfig, train_end: str | pd.Timestamp) -> TrainTestResult:
    """Fit on data up to `train_end`, then hold the weights on everything after it."""
    train_end = pd.Timestamp(train_end)
    train_prices = prices.loc[:train_end]
    if len(train_prices) < 60:
        raise ValueError("Need at least 60 trading days of training data.")
    last_train_day = train_prices.index[-1]
    result = cfg.build(train_prices)
    train_ret = portfolio_returns(train_prices, result.weights)
    test_prices = prices.loc[last_train_day:]
    test_ret = portfolio_returns(test_prices, result.weights)
    return TrainTestResult(result, train_ret, test_ret, last_train_day)


@dataclass
class WalkForwardResult:
    returns: pd.Series
    weights: pd.DataFrame  # one row per refit date
    turnover: pd.Series


def walk_forward(
    prices: pd.DataFrame,
    cfg: StrategyConfig,
    lookback: int = 252,
    step: int = 63,
    cost_bps: float = 5.0,
) -> WalkForwardResult:
    """Refit every `step` trading days on the trailing `lookback` days.

    Weights chosen at the close of day t only use prices up to day t and are applied to
    returns from day t+1 onward, so there is no peeking.
    """
    rets = prices.pct_change()
    n = len(prices)
    refits = list(range(lookback, n - 1, step))
    if not refits:
        raise ValueError("Not enough data for the chosen lookback.")

    pieces, weight_rows, turnovers = [], {}, {}
    prev = pd.Series(0.0, index=prices.columns)
    for j, t in enumerate(refits):
        window = prices.iloc[t - lookback : t + 1]
        w = cfg.build(window).weights.reindex(prices.columns).fillna(0.0)
        end = refits[j + 1] if j + 1 < len(refits) else n - 1
        seg = rets.iloc[t + 1 : end + 1].dot(w)
        turnover = float((w - prev).abs().sum())
        seg.iloc[0] -= turnover * cost_bps / 1e4
        pieces.append(seg)
        weight_rows[prices.index[t]] = w
        turnovers[prices.index[t]] = turnover
        prev = w
    out = pd.concat(pieces)
    return WalkForwardResult(out, pd.DataFrame(weight_rows).T, pd.Series(turnovers))


def benchmark_returns(series: pd.Series, start: pd.Timestamp | None = None) -> pd.Series:
    """Daily returns for a price series (e.g. the DJIA), optionally from a start date."""
    s = series if start is None else series.loc[start:]
    return s.pct_change().dropna()


def equal_weight_returns(prices: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.Series:
    p = prices if start is None else prices.loc[start:]
    return p.pct_change().dropna().mean(axis=1)


def align(*series: pd.Series) -> list[pd.Series]:
    """Trim every series to the dates they all share."""
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    return [s.loc[idx] for s in series]
