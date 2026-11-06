"""Dow 30 Portfolio Lab: an interactive app for people who do not code.

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from portfolio_lab import plots
from portfolio_lab.backtest import (align, benchmark_returns, equal_weight_returns, portfolio_returns, walk_forward)
from portfolio_lab.config import NAMES, SECTORS, TICKERS
from portfolio_lab.data import load_snapshot
from portfolio_lab.metrics import summarize, summary_table
from portfolio_lab.optimize import (StrategyConfig, efficient_frontier, estimate_inputs, random_portfolios,
                                    stock_risk_return)
from portfolio_lab.simulate import DRIFT_CHOICES, simulate

st.set_page_config(page_title="Dow 30 Portfolio Lab", page_icon="📈", layout="wide")

HORIZONS = {"1 week": 5, "1 month": 21, "3 months": 63, "1 year": 252}


@st.cache_data(show_spinner=False)
def get_data():
    return load_snapshot()


@st.cache_data(show_spinner=False)
def get_inputs(prices: pd.DataFrame, stabilized: bool):
    return estimate_inputs(prices, stabilized)


@st.cache_data(show_spinner="Mapping the risk and reward landscape...")
def get_frontier(prices: pd.DataFrame, stabilized: bool, max_weight: float, rf: float, sector_cap: float | None):
    inputs = estimate_inputs(prices, stabilized)
    frontier = efficient_frontier(inputs, max_weight, rf, sector_cap, n_points=22)
    return frontier, random_portfolios(inputs, 2000, rf), stock_risk_return(inputs)


@st.cache_data(show_spinner="Re-testing every quarter...")
def get_walk_forward(prices: pd.DataFrame, cfg: StrategyConfig):
    return walk_forward(prices, cfg)


@st.cache_data(show_spinner=False)
def get_simulation(returns: pd.Series, horizon: int, n_sims: int, start: float, drift: str, method: str):
    return simulate(returns, horizon, n_sims, start, method=method, drift=drift)


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


prices_all, djia = get_data()

# ---------------------------------------------------------------- sidebar
RISK_STEPS = ["Very cautious", "Cautious", "Balanced", "Growth", "Aggressive"]
RISK_TEXT = {
    "Very cautious": "The smoothest ride. Smaller gains, smaller drops.",
    "Cautious": "Leans toward steadier stocks. Modest swings.",
    "Balanced": "A middle path between steady and ambitious.",
    "Growth": "Accepts bigger swings for more growth.",
    "Aggressive": "Chases the highest past returns. Big swings and fewer stocks doing the heavy lifting.",
}
ALL_SECTORS = sorted(set(SECTORS.values()))


def label(ticker: str) -> str:
    return f"{ticker}  ·  {NAMES.get(ticker, ticker)}"


with st.sidebar:
    st.header("Build your portfolio")

    st.subheader("1. How much are you investing?")
    start_value = st.number_input("Amount ($)", 1_000, 10_000_000, 10_000, step=1_000, key="amount")

    st.subheader("2. How much risk is OK?")
    risk_label = st.select_slider("Risk tolerance", RISK_STEPS, value="Balanced", key="risk", label_visibility="collapsed")
    st.caption(RISK_TEXT[risk_label])
    risk_level = RISK_STEPS.index(risk_label) / (len(RISK_STEPS) - 1)

    st.subheader("3. Which stocks?")
    sectors_sel = st.multiselect("Sectors to include", ALL_SECTORS, default=ALL_SECTORS, key="sectors")
    pool = [t for t in TICKERS if SECTORS[t] in sectors_sel]
    leave_out = st.multiselect("Leave out specific stocks", pool, format_func=label, key="leave_out")
    chosen = [t for t in pool if t not in leave_out]
    must = st.multiselect("Stocks I definitely want", chosen, format_func=label, key="must",
                          help="These always get at least the minimum share you set below.")
    must_min = st.slider("Minimum in each of those", 1, 15, 3, format="%d%%", key="must_min") / 100 if must else 0.0

    st.subheader("4. How many stocks?")
    lo, hi = max(3, len(must)), len(chosen)
    if hi > lo:
        max_holdings = st.slider("Hold at most this many", lo, hi, min(10, hi), key="max_holdings",
                                 help="Fewer stocks is simpler to manage but gives up some diversification.")
    else:
        max_holdings = hi

    st.divider()
    st.subheader("Test it, or use it")
    mode = st.radio("Mode", ["Test it on the past", "Build today's mix from all data"], label_visibility="collapsed",
                    help="Test mode hides the later years from the optimizer, then shows what happened.", key="mode")
    first, last = prices_all.index[0].date(), prices_all.index[-1].date()
    train_end = pd.Timestamp(last)
    if mode.startswith("Test"):
        lo_d = (prices_all.index[0] + pd.Timedelta(days=365)).date()
        hi_d = (prices_all.index[-1] - pd.Timedelta(days=90)).date()
        train_end = pd.Timestamp(st.slider("Learn from data up to", lo_d, hi_d, pd.Timestamp("2022-12-31").date(),
                                           format="MMM YYYY", key="train_end"))

    st.divider()
    st.subheader("What could happen next")
    horizon_label = st.select_slider("How far ahead", list(HORIZONS), value="1 month", key="horizon")
    drift_label = st.selectbox("Return assumption", list(DRIFT_CHOICES), index=2, key="drift",
                               help="2020 to 2024 was a strong stretch. The more cautious options guard against assuming it repeats.")

    with st.expander("Advanced settings"):
        max_weight = st.slider("Most in any one stock", 5, 100, 25, step=5, format="%d%%", key="max_weight",
                               help="A lower cap forces the money to spread out.") / 100
        sector_pct = st.slider("Most in any one sector", 20, 100, 100, step=5, format="%d%%", key="sector_cap",
                               help="100% means no sector limit.")
        sector_cap = None if sector_pct >= 100 else sector_pct / 100
        stabilized = st.checkbox("Use steadier estimates (recommended)", value=True, key="stabilized",
                                 help="Pulls extreme past returns toward the average so the app does not bet everything on last period's winners.")
        rf = st.slider("Return from a safe savings account", 0.0, 6.0, 3.0, step=0.25, format="%.2f%%", key="rf",
                       help="Used to judge whether extra risk was worth it.") / 100
        best_sharpe = st.checkbox("Ignore my risk setting and find the best return per unit of risk", value=False, key="best_sharpe")
        method = st.radio("Scenario engine", ["bootstrap", "fitted"], key="engine", format_func=lambda m: {
            "bootstrap": "Replay real days in short blocks", "fitted": "Draw from a fitted distribution"}[m])
        n_sims = st.select_slider("Number of simulated futures", [1000, 2000, 5000, 10000], value=5000, key="n_sims")

objective = "max_sharpe" if best_sharpe else "risk_level"

# ---------------------------------------------------------------- compute
st.title("Dow 30 Portfolio Lab")
st.caption("Explore how a mix of Dow Jones stocks could be built, how it would have held up, and what might happen next. Educational project, not financial advice.")

if len(chosen) < 5:
    st.warning("Pick at least 5 stocks in the sidebar so there is something to mix.")
    st.stop()

prices = prices_all[chosen]
train_prices = prices.loc[:train_end] if mode.startswith("Test") else prices
cfg = StrategyConfig("You", objective, max_weight, stabilized, rf=rf, sector_cap=sector_cap, risk_level=risk_level,
                     min_weights={t: must_min for t in must} or None, max_holdings=max_holdings)
try:
    inputs = get_inputs(train_prices, stabilized)
    result = cfg.solve(inputs)
except Exception as exc:  # infeasible limits, solver trouble, etc.
    st.error("These settings leave no valid mix. Try allowing more stocks, a higher per-stock limit, or smaller minimums.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

if max_weight < 1 / min(len(chosen), max_holdings):
    st.info(f"With {min(len(chosen), max_holdings)} stocks, each must hold at least {1 / min(len(chosen), max_holdings):.1%}. The per-stock limit was nudged up.")
if result.note and not result.note.startswith("Limited"):
    st.info(result.note)

holdings = result.holdings
train_ret = portfolio_returns(train_prices, result.weights)
test_mode = mode.startswith("Test")
last_train_day = train_prices.index[-1]

tab_mix, tab_past, tab_future, tab_market, tab_about = st.tabs(
    ["Your portfolio", "How it would have done", "What could happen next", "Explore the market", "About and limits"])

# ---------------------------------------------------------------- tab 1
with tab_mix:
    top = ", ".join(f"{NAMES.get(t, t)} ({w:.0%})" for t, w in holdings.head(3).items())
    lead = (f"**{money(start_value)} spread across {len(holdings)} stocks at a '{risk_label.lower()}' risk setting.**"
            if objective == "risk_level" else
            f"**{money(start_value)} spread across {len(holdings)} stocks, chosen for the best return per unit of risk.**")
    st.markdown(
        f"{lead} Biggest positions: {top}. On the data it learned from ({train_prices.index[0]:%b %Y} to {last_train_day:%b %Y}) "
        f"this mix would have earned about **{pct(result.expected_return)} a year**, with ups and downs of roughly "
        f"**{pct(result.volatility)}** a year. That is a description of the past, not a forecast."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Typical yearly return (in the past)", pct(result.expected_return))
    c2.metric("Yearly ups and downs", pct(result.volatility), help="Annual volatility: how widely returns swing.")
    c3.metric("Return per unit of risk", f"{result.sharpe:.2f}", help="Sharpe ratio. Higher is better, but only if it holds up on new data.")

    st.subheader("Your shopping list")
    shop = pd.DataFrame({
        "Company": [NAMES.get(t, t) for t in holdings.index],
        "Sector": [SECTORS.get(t, "") for t in holdings.index],
        "Share (%)": (holdings.values * 100).round(2),
        "Amount ($)": (holdings.values * start_value).round(0),
    }, index=holdings.index)
    st.dataframe(
        shop, width="stretch",
        column_config={
            "Share (%)": st.column_config.ProgressColumn("Share of portfolio", min_value=0, max_value=100, format="%.1f%%"),
            "Amount ($)": st.column_config.NumberColumn("Amount", format="dollar"),
        },
    )
    st.download_button("Download my portfolio (CSV)", shop.to_csv().encode(), "my_portfolio.csv", "text/csv")

    left, right = st.columns([3, 2])
    left.plotly_chart(plots.weights_chart(result.weights), width="stretch", config={"displayModeBar": False})
    right.plotly_chart(plots.sector_chart(result.weights), width="stretch", config={"displayModeBar": False})

    frontier, cloud, stock_rr = get_frontier(train_prices, stabilized, max_weight, rf, sector_cap)
    eq = estimate_inputs(train_prices, stabilized)
    eq_vol = float((eq.cov.values.sum() / len(eq.mu) ** 2) ** 0.5)
    pts = {"Your mix": (result.volatility, result.expected_return), "Equal weight": (eq_vol, float(eq.mu.mean()))}
    st.plotly_chart(plots.frontier_chart(cloud, frontier, stock_rr, pts), width="stretch", config={"displayModeBar": False})
    st.caption("Each faint dot is a random mix. The black line is the best return available at each level of risk. Hover over any dot to see what it is. "
               "Holding fewer stocks or adding must-haves can place your mix a little off the line, which is the price of simplicity.")

# ---------------------------------------------------------------- tab 2
with tab_past:
    comparators = {"Equal weight (your stocks)": equal_weight_returns(prices), "DJIA": benchmark_returns(djia)}
    if test_mode:
        test_ret = portfolio_returns(prices.loc[last_train_day:], result.weights)
        full = pd.concat([train_ret, test_ret])
        series = align(full, *comparators.values())
        series = dict(zip(["Your mix", *comparators], series))
        promised = summarize(train_ret, rf)["Sharpe ratio"]
        delivered = summarize(test_ret, rf)["Sharpe ratio"]
        a, b, c = st.columns(3)
        a.metric("Return per unit of risk on data it learned from", f"{promised:.2f}")
        b.metric("Return per unit of risk on data it never saw", f"{delivered:.2f}", delta=f"{delivered - promised:+.2f}", delta_color="off")
        test_series = dict(zip(series, align(*[series[k].loc[last_train_day:] for k in series])))
        mine, dj_test = summarize(test_series["Your mix"], rf), summarize(test_series["DJIA"], rf)
        c.metric(f"After {last_train_day:%b %Y}: you vs DJIA", pct(mine["Total return"]), delta=f"{mine['Total return'] - dj_test['Total return']:+.1%} vs DJIA")
        verdict = "ahead of" if mine["Total return"] > dj_test["Total return"] else "behind"
        st.markdown(
            f"On the years it studied, your mix looked like **{promised:.2f}**. On the years it never saw, it scored **{delivered:.2f}**, "
            f"and finished **{verdict}** the DJIA. A big gap between those two numbers is a warning that the optimizer was fitting noise."
        )
        st.plotly_chart(plots.growth_chart(series, start_value, split_date=last_train_day, title=f"Growth of {money(start_value)}"),
                        width="stretch", config={"displayModeBar": False})
        st.subheader(f"Scorecard after {last_train_day:%b %Y}")
        shown = summary_table(test_series, rf).copy()
        compare_returns = test_series
    else:
        series = dict(zip(["Your mix", *comparators], align(train_ret, *comparators.values())))
        st.info("You are in 'build today's mix' mode, so every year of data was used to build it. Looking back would flatter it. "
                "Switch to **Test it on the past** in the sidebar for an honest check.")
        st.plotly_chart(plots.growth_chart(series, start_value, title=f"Growth of {money(start_value)} (the optimizer saw all of this)"),
                        width="stretch", config={"displayModeBar": False})
        shown = summary_table(series, rf).copy()
        compare_returns = series

    for col in ("Total return", "Annual return", "Annual volatility", "Worst drop (max drawdown)"):
        shown[col] = shown[col].map(pct)
    shown["Sharpe ratio"] = shown["Sharpe ratio"].map(lambda v: f"{v:.2f}")
    shown["Growth of $10,000"] = shown["Growth of $10,000"].map(money)
    st.dataframe(shown, width="stretch")
    st.plotly_chart(plots.drawdown_chart(compare_returns), width="stretch", config={"displayModeBar": False})

    st.subheader("A tougher test: re-optimize every quarter")
    st.caption("Each quarter the app re-learns from the last 12 months only, pays a small trading cost, and holds the new mix. No peeking at the future.")
    try:
        wf = get_walk_forward(prices, cfg)
        start_wf = wf.returns.index[0] - pd.Timedelta(days=1)
        wf_series = {"Your strategy (refit quarterly)": wf.returns,
                     "Equal weight (your stocks)": equal_weight_returns(prices, start_wf),
                     "DJIA": benchmark_returns(djia, start_wf)}
        wf_series = dict(zip(wf_series, align(*wf_series.values())))
        st.plotly_chart(plots.growth_chart(wf_series, start_value, title=f"Growth of {money(start_value)}, refit every quarter"),
                        width="stretch", config={"displayModeBar": False})
        w = summary_table(wf_series, rf).copy()
        for col in ("Total return", "Annual return", "Annual volatility", "Worst drop (max drawdown)"):
            w[col] = w[col].map(pct)
        w["Sharpe ratio"] = w["Sharpe ratio"].map(lambda v: f"{v:.2f}")
        w["Growth of $10,000"] = w["Growth of $10,000"].map(money)
        st.dataframe(w, width="stretch")
        st.caption(f"Average yearly turnover is high ({wf.turnover.mean():.0%} of the portfolio changes per refit), which is one reason real-world results can disappoint.")
    except Exception as exc:
        st.warning("The quarterly test could not run with these settings.")
        st.caption(f"Technical detail: {exc}")

# ---------------------------------------------------------------- tab 3
with tab_future:
    horizon = HORIZONS[horizon_label]
    sim = get_simulation(train_ret, horizon, n_sims, float(start_value), drift_label, method)
    s = sim.summary()
    st.markdown(
        f"Imagine investing **{money(start_value)}** in this mix for **{horizon_label}**. The app replays {n_sims:,} possible futures "
        f"using {sim.detail}. Return assumption: **{drift_label.lower()}**."
    )
    a, b, c, d = st.columns(4)
    a.metric("Typical ending value", money(s["Median ending value"]))
    b.metric("Bad case (1 in 20)", money(s["Likely range low (5th percentile)"]), help="Only 5% of simulated futures ended lower than this.")
    c.metric("Good case (1 in 20)", money(s["Likely range high (95th percentile)"]), help="Only 5% of simulated futures ended higher than this.")
    d.metric("Chance of ending below start", pct(s["Chance of ending below start"], 0))
    st.plotly_chart(plots.fan_chart(sim.bands(), float(start_value), f"Where {money(start_value)} could be over {horizon_label}"),
                    width="stretch", config={"displayModeBar": False})
    st.plotly_chart(plots.outcomes_histogram(sim.final, float(start_value)), width="stretch", config={"displayModeBar": False})
    st.markdown(f"If things go badly (the worst 1 in 20 futures), the average ending value is about **{money(s['Average of the worst 5% of outcomes'])}**.")
    st.caption("Try switching the return assumption in the sidebar. The past few years were unusually good for stocks, so the more cautious settings are worth a look.")

# ---------------------------------------------------------------- tab 4
with tab_market:
    st.plotly_chart(plots.stock_scatter(stock_risk_return(inputs), result.weights), width="stretch", config={"displayModeBar": False})
    st.caption("Blue dots are stocks your mix holds. Stocks high and to the left offered more return for less risk in the data used.")
    picks = st.multiselect("Compare a few stocks against the DJIA", chosen, default=chosen[:3])
    if picks:
        rets = {t: prices[t].pct_change().dropna() for t in picks}
        rets["DJIA"] = benchmark_returns(djia)
        st.plotly_chart(plots.growth_chart(dict(zip(rets, align(*rets.values()))), start_value, title=f"Growth of {money(start_value)}"),
                        width="stretch", config={"displayModeBar": False})
    st.plotly_chart(plots.correlation_heatmap(prices.pct_change().dropna()), width="stretch", config={"displayModeBar": False})
    st.caption("Darker squares mean two stocks tend to rise and fall together. Mixing stocks that do not move together is what makes diversification work.")

# ---------------------------------------------------------------- tab 5
with tab_about:
    st.markdown(f"""
**What this is.** A tool built on mean-variance optimization (Harry Markowitz, 1952). It looks at how 30 Dow stocks moved in the past
and searches for the mix with the best return for its risk. The data covers {first:%b %Y} to {last:%b %Y}.

**Why the test mode matters.** A mix tuned on past data nearly always looks great on that same data. The honest question is whether it
keeps working afterwards, so the app lets you hide the later years from the optimizer and then reveals what happened.

**Steadier estimates.** Plain historical averages are noisy, and optimizers love noise. The steadier option shrinks the risk estimates
(Ledoit-Wolf) and pulls expected returns halfway toward the average of all 30 stocks.

**Limits to keep in mind**
- About {(last - first).days / 365.25:.1f} years of data is a short sample. Past returns do not predict future returns.
- The stock list is the Dow lineup for most of 2020 to 2024. Amazon, Nvidia and Sherwin-Williams later replaced Walgreens, Intel and Dow Inc.
- The DJIA is a price index and leaves out dividends, so the equal-weight line is the fairer comparison for stock mixes.
- Simulations assume future days look like past days. Real markets can break patterns, especially in a crisis.
- Trading costs are a flat 0.05% per trade in the quarterly test. Taxes are ignored.

**This is an educational project, not financial advice.** Talk to a licensed advisor before investing.
""")
