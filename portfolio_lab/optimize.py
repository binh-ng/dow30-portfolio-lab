"""Mean-variance optimization built on PyPortfolioOpt.

Two ways to estimate inputs are supported:
  * "raw": plain historical mean and sample covariance (what the original project used)
  * "stabilized": Ledoit-Wolf shrunk covariance and expected returns pulled toward the
    cross-sectional average, which cuts the optimizer's habit of chasing past winners.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, objective_functions, risk_models

from .config import SECTORS


@dataclass
class PortfolioResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    objective: str
    risk_free: float
    note: str = ""

    @property
    def holdings(self) -> pd.Series:
        """Non-zero weights, largest first."""
        w = self.weights[self.weights > 1e-6]
        return w.sort_values(ascending=False)


@dataclass
class Inputs:
    mu: pd.Series
    cov: pd.DataFrame
    stabilized: bool = False


def estimate_inputs(prices: pd.DataFrame, stabilized: bool = True, mu_shrink: float = 0.5) -> Inputs:
    """Expected annual returns and covariance from a window of prices."""
    mu = expected_returns.mean_historical_return(prices)
    if stabilized:
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        mu = (1 - mu_shrink) * mu + mu_shrink * mu.mean()
    else:
        cov = risk_models.sample_cov(prices)
    return Inputs(mu=mu, cov=cov, stabilized=stabilized)


def _effective_cap(n_assets: int, max_weight: float) -> float:
    """A cap below 1/n cannot sum to 100%, so nudge it up to the smallest feasible value."""
    return float(max(max_weight, 1.0 / n_assets + 1e-6))


def _build_ef(
    inputs: Inputs,
    max_weight: float,
    sector_caps: float | None,
    l2_gamma: float,
    min_weights: dict | None = None,
) -> EfficientFrontier:
    cap = _effective_cap(len(inputs.mu), max_weight)
    bounds = (0, cap)
    if min_weights:
        lows = [min(float(min_weights.get(t, 0.0)), cap) for t in inputs.mu.index]
        if sum(lows) > 1 + 1e-9:
            raise ValueError("The minimum amounts for your must-have stocks add up to more than 100%.")
        bounds = [(lo, cap) for lo in lows]
    ef = EfficientFrontier(inputs.mu, inputs.cov, weight_bounds=bounds)
    if sector_caps is not None and sector_caps < 1.0:
        mapper = {t: SECTORS.get(t, "Other") for t in inputs.mu.index}
        sectors = sorted(set(mapper.values()))
        # A sector cap is only feasible when enough sectors exist to hold 100%.
        upper = {s: max(sector_caps, 1.0 / len(sectors) + 1e-6) for s in sectors}
        lower = {s: 0.0 for s in sectors}
        ef.add_sector_constraints(mapper, lower, upper)
    if l2_gamma > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
    return ef


def _finish(ef: EfficientFrontier, inputs: Inputs, objective: str, rf: float, note: str = "") -> PortfolioResult:
    w = pd.Series(ef.clean_weights(cutoff=1e-4, rounding=6)).reindex(inputs.mu.index).fillna(0.0)
    w = w / w.sum()
    ret = float(w @ inputs.mu)
    vol = float(np.sqrt(w.values @ inputs.cov.values @ w.values))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return PortfolioResult(w, ret, vol, sharpe, objective, rf, note)


def optimize_portfolio(
    inputs: Inputs,
    objective: str = "max_sharpe",
    max_weight: float = 0.25,
    rf: float = 0.0,
    sector_cap: float | None = None,
    l2_gamma: float = 0.0,
    target_return: float | None = None,
    min_weights: dict | None = None,
) -> PortfolioResult:
    """Solve for weights.

    objective: "max_sharpe", "min_vol", "target_return" or "equal".
    """
    n = len(inputs.mu)
    if objective == "equal":
        w = pd.Series(1.0 / n, index=inputs.mu.index)
        vol = float(np.sqrt(w.values @ inputs.cov.values @ w.values))
        ret = float(w @ inputs.mu)
        return PortfolioResult(w, ret, vol, (ret - rf) / vol, "equal", rf)

    ef = _build_ef(inputs, max_weight, sector_cap, l2_gamma, min_weights)
    if objective == "max_sharpe":
        if (inputs.mu <= rf).all():
            ef = _build_ef(inputs, max_weight, sector_cap, l2_gamma, min_weights)
            ef.min_volatility()
            return _finish(ef, inputs, "min_vol", rf, "No stock beat the risk-free rate, so this is the lowest-risk mix instead.")
        ef.max_sharpe(risk_free_rate=rf)
    elif objective == "min_vol":
        ef.min_volatility()
    elif objective == "target_return":
        ef.efficient_return(target_return)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return _finish(ef, inputs, objective, rf)


def max_feasible_return(mu: pd.Series, max_weight: float) -> float:
    """Highest return reachable when no stock may exceed `max_weight` (ignores sector caps)."""
    cap = _effective_cap(len(mu), max_weight)
    remaining, total = 1.0, 0.0
    for m in mu.sort_values(ascending=False):
        take = min(cap, remaining)
        total += take * m
        remaining -= take
        if remaining <= 1e-12:
            break
    return float(total)


def efficient_frontier(
    inputs: Inputs,
    max_weight: float = 0.25,
    rf: float = 0.0,
    sector_cap: float | None = None,
    l2_gamma: float = 0.0,
    n_points: int = 30,
) -> pd.DataFrame:
    """Frontier points between the minimum-volatility and the highest feasible return."""
    low = optimize_portfolio(inputs, "min_vol", max_weight, rf, sector_cap, l2_gamma)
    high = max_feasible_return(inputs.mu, max_weight) * 0.995
    rows = []
    for target in np.linspace(low.expected_return, max(high, low.expected_return), n_points):
        try:
            res = optimize_portfolio(inputs, "target_return", max_weight, rf, sector_cap, l2_gamma, target_return=target)
        except Exception:
            continue
        rows.append({"return": res.expected_return, "volatility": res.volatility, "sharpe": res.sharpe})
    return pd.DataFrame(rows)


def portfolio_for_risk_level(
    inputs: Inputs,
    level: float,
    max_weight: float = 0.25,
    rf: float = 0.0,
    sector_cap: float | None = None,
    l2_gamma: float = 0.0,
    min_weights: dict | None = None,
) -> PortfolioResult:
    """Pick a point on the frontier. level 0 is the lowest-risk mix, level 1 is the most aggressive."""
    low = optimize_portfolio(inputs, "min_vol", max_weight, rf, sector_cap, l2_gamma, min_weights=min_weights)
    high = max_feasible_return(inputs.mu, max_weight) * 0.995
    level = float(np.clip(level, 0.0, 1.0))
    target = low.expected_return + level * (max(high, low.expected_return) - low.expected_return)
    for shrink in (1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0):
        t = low.expected_return + shrink * (target - low.expected_return)
        try:
            return optimize_portfolio(inputs, "target_return", max_weight, rf, sector_cap, l2_gamma,
                                      target_return=t, min_weights=min_weights)
        except Exception:
            continue
    return low


def random_portfolios(inputs: Inputs, n: int = 3000, rf: float = 0.0, seed: int = 7) -> pd.DataFrame:
    """Random long-only mixes, used as the cloud behind the efficient frontier chart."""
    rng = np.random.default_rng(seed)
    k = len(inputs.mu)
    alphas = rng.choice([0.15, 0.4, 1.0], size=n)
    weights = np.vstack([rng.dirichlet(np.full(k, a)) for a in alphas])
    rets = weights @ inputs.mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, inputs.cov.values, weights))
    return pd.DataFrame({"return": rets, "volatility": vols, "sharpe": (rets - rf) / vols})


def stock_risk_return(inputs: Inputs) -> pd.DataFrame:
    """Each stock on its own: annual return vs annual volatility."""
    return pd.DataFrame({
        "return": inputs.mu,
        "volatility": np.sqrt(np.diag(inputs.cov.values)),
    }, index=inputs.mu.index)


@dataclass
class StrategyConfig:
    """A named, repeatable recipe for building a portfolio."""
    name: str
    objective: str = "max_sharpe"
    max_weight: float = 0.25
    stabilized: bool = True
    mu_shrink: float = 0.5
    l2_gamma: float = 0.0
    sector_cap: float | None = None
    rf: float = 0.0
    risk_level: float = 0.5  # only used when objective == "risk_level" (0 cautious, 1 aggressive)
    min_weights: dict | None = None  # must-have stocks: {ticker: minimum share}
    max_holdings: int | None = None  # hold at most this many stocks
    extra: dict = field(default_factory=dict)

    def _solve_core(self, inputs: Inputs) -> PortfolioResult:
        mw = {t: v for t, v in (self.min_weights or {}).items() if t in inputs.mu.index} or None
        if self.objective == "risk_level":
            return portfolio_for_risk_level(
                inputs, self.risk_level, self.max_weight, self.rf, self.sector_cap, self.l2_gamma, mw
            )
        return optimize_portfolio(
            inputs, self.objective, self.max_weight, self.rf, self.sector_cap, self.l2_gamma, min_weights=mw
        )

    def solve(self, inputs: Inputs) -> PortfolioResult:
        first = self._solve_core(inputs)
        k = self.max_holdings
        if not k or int((first.weights > 1e-6).sum()) <= k:
            return first
        # Too many names: keep the must-haves plus the largest positions, then re-solve on just those.
        must = [t for t in (self.min_weights or {}) if t in inputs.mu.index]
        ranked = [t for t in first.weights.sort_values(ascending=False).index if t not in must]
        keep = (must + ranked)[: max(k, len(must))]
        sub = Inputs(inputs.mu[keep], inputs.cov.loc[keep, keep], inputs.stabilized)
        res = self._solve_core(sub)
        weights = res.weights.reindex(inputs.mu.index).fillna(0.0)
        return PortfolioResult(weights, res.expected_return, res.volatility, res.sharpe, res.objective,
                               res.risk_free, f"Limited to the {len(keep)} stocks that matter most.")

    def build(self, prices: pd.DataFrame) -> PortfolioResult:
        return self.solve(estimate_inputs(prices, self.stabilized, self.mu_shrink))
