"""Core checks: replication of the original results, constraints, no look-ahead, simulation sanity."""
import numpy as np
import pandas as pd
import pytest

from portfolio_lab.backtest import train_test, walk_forward
from portfolio_lab.config import SECTORS
from portfolio_lab.data import load_snapshot
from portfolio_lab.metrics import summarize
from portfolio_lab.optimize import StrategyConfig, estimate_inputs, optimize_portfolio, portfolio_for_risk_level
from portfolio_lab.simulate import simulate


@pytest.fixture(scope="module")
def data():
    return load_snapshot()


def test_snapshot_is_clean(data):
    px, dj = data
    assert px.shape == (1132, 30)
    assert not px.isna().any().any() and not dj.isna().any()
    assert px.index.is_monotonic_increasing and px.index.is_unique


def test_replicates_original_course_results(data):
    px, _ = data
    raw = estimate_inputs(px, stabilized=False)
    r = optimize_portfolio(raw, "max_sharpe", max_weight=1.0, rf=0.0)
    assert r.weights["WMT"] == pytest.approx(0.33151, abs=2e-3)
    assert r.weights["IBM"] == pytest.approx(0.13082, abs=2e-3)
    assert (r.expected_return, r.volatility, r.sharpe) == pytest.approx((0.236, 0.145, 1.62), abs=0.01)
    mv = optimize_portfolio(raw, "min_vol", max_weight=1.0, rf=0.0)
    assert (mv.expected_return, mv.volatility) == pytest.approx((0.108, 0.113), abs=0.002)


@pytest.mark.parametrize("objective", ["max_sharpe", "min_vol"])
@pytest.mark.parametrize("stabilized", [True, False])
def test_weights_obey_budget_and_caps(data, objective, stabilized):
    px, _ = data
    r = optimize_portfolio(estimate_inputs(px, stabilized), objective, max_weight=0.15, rf=0.03)
    assert r.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (r.weights >= -1e-9).all() and (r.weights <= 0.15 + 1e-6).all()


def test_sector_cap_is_respected(data):
    px, _ = data
    r = optimize_portfolio(estimate_inputs(px, True), "max_sharpe", 0.25, 0.03, sector_cap=0.30)
    by_sector = r.weights.groupby(lambda t: SECTORS[t]).sum()
    assert by_sector.max() <= 0.30 + 1e-4


def test_infeasible_cap_is_raised_to_feasible(data):
    px, _ = data
    r = optimize_portfolio(estimate_inputs(px.iloc[:, :10], True), "min_vol", max_weight=0.01)
    assert r.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_risk_dial_moves_along_the_frontier(data):
    px, _ = data
    inp = estimate_inputs(px, True)
    calm = portfolio_for_risk_level(inp, 0.0, 0.25, 0.03)
    bold = portfolio_for_risk_level(inp, 1.0, 0.25, 0.03)
    assert bold.expected_return > calm.expected_return and bold.volatility > calm.volatility


def test_train_test_uses_only_past_data(data):
    px, _ = data
    cfg = StrategyConfig("t", "max_sharpe", 0.25, stabilized=True, rf=0.03)
    base = train_test(px, cfg, "2022-12-31")
    altered = px.copy()
    altered.loc["2023-01-01":] *= np.random.default_rng(0).uniform(0.5, 1.5, size=altered.loc["2023-01-01":].shape)
    again = train_test(altered, cfg, "2022-12-31")
    pd.testing.assert_series_equal(base.result.weights, again.result.weights)
    assert base.test_returns.index[0] > base.train_end


def test_walk_forward_has_no_lookahead(data):
    px, _ = data
    cfg = StrategyConfig("t", "max_sharpe", 0.25, stabilized=True, rf=0.03)
    base = walk_forward(px, cfg)
    cut = base.weights.index[5]
    altered = px.copy()
    after = altered.index > cut
    altered.loc[after] *= np.random.default_rng(1).uniform(0.7, 1.3, size=(after.sum(), px.shape[1]))
    again = walk_forward(altered, cfg)
    pd.testing.assert_frame_equal(base.weights.loc[:cut], again.weights.loc[:cut])


def test_simulation_is_reproducible_and_sane(data):
    px, _ = data
    r = px.pct_change().dropna().mean(axis=1)
    a = simulate(r, 21, 2000, drift="Flat (0% a year)", seed=3)
    b = simulate(r, 21, 2000, drift="Flat (0% a year)", seed=3)
    np.testing.assert_allclose(a.paths, b.paths)
    assert a.paths.shape == (22, 2000) and (a.paths[0] == a.start_value).all()
    s = a.summary()
    assert 0.35 < s["Chance of ending below start"] < 0.65
    assert s["Likely range low (5th percentile)"] < s["Median ending value"] < s["Likely range high (95th percentile)"]


def test_fitted_method_runs(data):
    px, _ = data
    r = px.pct_change().dropna().mean(axis=1)
    assert simulate(r, 10, 500, method="fitted").final.shape == (500,)


def test_metrics_match_hand_calculation():
    r = pd.Series([0.01] * 252)
    m = summarize(r, rf=0.0)
    assert m["Annual return"] == pytest.approx(1.01 ** 252 - 1)
    assert m["Worst drop (max drawdown)"] == 0


def test_download_path_layout_and_cache(monkeypatch, tmp_path):
    """Yahoo is not called for real: a stand-in returns the same shape yfinance does."""
    import yfinance as yf
    from portfolio_lab.config import TICKERS
    from portfolio_lab.data import download_prices, load_snapshot as load

    idx = pd.bdate_range("2024-01-01", periods=30)

    def fake(tickers, **kwargs):
        cols = list(tickers) if isinstance(tickers, (list, tuple)) else [tickers]
        frame = pd.DataFrame(np.random.default_rng(0).uniform(50, 150, (30, len(cols))), index=idx, columns=cols)
        return pd.concat({"Close": frame}, axis=1)

    monkeypatch.setattr(yf, "download", fake)
    stocks, bench = download_prices("2024-01-01", "2024-03-01", cache=tmp_path / "prices.csv")
    assert list(stocks.columns) == TICKERS and len(bench) == 30
    assert np.allclose(stocks.iloc[0], 100) and bench.iloc[0] == pytest.approx(100)
    s2, b2 = load(tmp_path / "prices.csv")
    assert s2.shape == stocks.shape and b2.name == "DJIA"


def test_must_have_stocks_get_their_minimum(data):
    px, _ = data
    inp = estimate_inputs(px, True)
    r = optimize_portfolio(inp, "min_vol", 0.25, 0.03, min_weights={"BA": 0.05, "DIS": 0.05})
    assert r.weights["BA"] >= 0.05 - 1e-6 and r.weights["DIS"] >= 0.05 - 1e-6
    assert r.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_must_have_minimums_over_100_percent_raise(data):
    px, _ = data
    inp = estimate_inputs(px, True)
    with pytest.raises(ValueError):
        optimize_portfolio(inp, "min_vol", 1.0, 0.03, min_weights={t: 0.2 for t in px.columns[:8]})


@pytest.mark.parametrize("k", [4, 6, 10])
@pytest.mark.parametrize("level", [0.0, 0.5, 1.0])
def test_max_holdings_is_respected(data, k, level):
    px, _ = data
    cfg = StrategyConfig("t", "risk_level", 0.4, True, rf=0.03, risk_level=level, max_holdings=k,
                         min_weights={"IBM": 0.05})
    r = cfg.build(px)
    assert (r.weights > 1e-6).sum() <= k
    assert r.weights["IBM"] >= 0.05 - 1e-6
    assert r.weights.sum() == pytest.approx(1.0, abs=1e-6) and r.weights.max() <= 0.4 + 1e-6


def test_risk_tolerance_orders_portfolios_with_few_stocks(data):
    px, _ = data
    vols = []
    for level in (0.0, 0.5, 1.0):
        cfg = StrategyConfig("t", "risk_level", 0.4, True, rf=0.03, risk_level=level, max_holdings=8)
        vols.append(cfg.build(px).volatility)
    assert vols[0] <= vols[1] + 1e-9 <= vols[2] + 2e-9
