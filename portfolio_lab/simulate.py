"""Monte Carlo simulation of portfolio value.

Everything runs on log returns, so compounding is handled the same way in every output
(the original project mixed two conventions). Two ways to generate scenarios:
  * "bootstrap": resample blocks of real historical days (keeps fat tails and short streaks)
  * "fitted": draw from the best-fitting distribution among a short, sensible candidate list
A drift option lets you ask "what if returns look less rosy than 2020 to 2024?"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import TRADING_DAYS

CANDIDATES = {
    "normal": stats.norm,
    "student-t": stats.t,
    "skew-normal": stats.skewnorm,
    "johnson-su": stats.johnsonsu,
}

DRIFT_CHOICES = {
    "As in history": None,
    "Half of history": "half",
    "Long-run stock average (7% a year)": 0.07,
    "Flat (0% a year)": 0.0,
}


@dataclass
class FitResult:
    name: str
    params: tuple
    aic: float
    table: pd.DataFrame


def fit_distribution(returns: pd.Series) -> FitResult:
    """Fit each candidate to daily returns and keep the lowest AIC."""
    x = returns.dropna().values
    rows = []
    for name, dist in CANDIDATES.items():
        try:
            params = dist.fit(x)
            ll = np.sum(dist.logpdf(x, *params))
            if not np.isfinite(ll):
                continue
            k = len(params)
            aic = 2 * k - 2 * ll
            ks = stats.kstest(x, dist.name, args=params)
            rows.append({"distribution": name, "AIC": aic, "KS statistic": ks.statistic, "params": params})
        except Exception:
            continue
    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    best = table.iloc[0]
    return FitResult(best["distribution"], best["params"], float(best["AIC"]), table.drop(columns="params"))


def _target_daily_log_drift(hist_daily_log_mean: float, drift) -> float:
    if drift is None:
        return hist_daily_log_mean
    if drift == "half":
        return hist_daily_log_mean / 2
    return float(np.log1p(drift) / TRADING_DAYS)


def _block_bootstrap(x: np.ndarray, horizon: int, n_sims: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(x) - block + 1, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :horizon]
    return x[idx].T  # shape (horizon, n_sims)


@dataclass
class SimulationResult:
    paths: np.ndarray  # shape (horizon + 1, n_sims), portfolio value over time
    method: str
    detail: str
    start_value: float
    horizon: int

    @property
    def final(self) -> np.ndarray:
        return self.paths[-1]

    def bands(self, quantiles=(5, 25, 50, 75, 95)) -> pd.DataFrame:
        return pd.DataFrame(
            {f"p{q}": np.percentile(self.paths, q, axis=1) for q in quantiles}
        )

    def summary(self) -> dict:
        f = self.final
        v0 = self.start_value
        q5, q50, q95 = np.percentile(f, [5, 50, 95])
        worst_tail = f[f <= q5]
        return {
            "Median ending value": float(q50),
            "Likely range low (5th percentile)": float(q5),
            "Likely range high (95th percentile)": float(q95),
            "Chance of ending below start": float((f < v0).mean()),
            "Chance of losing 10% or more": float((f <= 0.9 * v0).mean()),
            "Average of the worst 5% of outcomes": float(worst_tail.mean()),
        }


def simulate(
    daily_returns: pd.Series,
    horizon: int = 21,
    n_sims: int = 5000,
    start_value: float = 10_000,
    method: str = "bootstrap",
    drift="As in history",
    block: int = 5,
    seed: int = 24,
) -> SimulationResult:
    """Simulate `n_sims` paths of `horizon` trading days."""
    x = daily_returns.dropna().values
    logs = np.log1p(x)
    hist_mean = float(logs.mean())
    drift_value = DRIFT_CHOICES.get(drift, drift) if isinstance(drift, str) else drift
    target = _target_daily_log_drift(hist_mean, drift_value)
    rng = np.random.default_rng(seed)

    if method == "bootstrap":
        sims = _block_bootstrap(logs, horizon, n_sims, block, rng)
        detail = f"block bootstrap (blocks of {block} days)"
    elif method == "fitted":
        fit = fit_distribution(pd.Series(x))
        dist = CANDIDATES[fit.name]
        simple = dist.rvs(*fit.params, size=(horizon, n_sims), random_state=rng)
        simple = np.clip(simple, -0.95, None)  # a daily loss beyond 95% is not physically sensible
        sims = np.log1p(simple)
        detail = f"fitted {fit.name} distribution"
    else:
        raise ValueError("method must be 'bootstrap' or 'fitted'")

    # Shift so the average daily log return equals the chosen drift.
    sims = sims - sims.mean() + target
    cum = np.vstack([np.zeros((1, n_sims)), np.cumsum(sims, axis=0)])
    return SimulationResult(start_value * np.exp(cum), method, detail, start_value, horizon)
