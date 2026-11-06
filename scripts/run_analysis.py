"""Run the full analysis and write tables to results/ and an interactive report to docs/index.html.

    python scripts/run_analysis.py                     # uses the bundled data snapshot
    python scripts/run_analysis.py --refresh           # re-download prices from Yahoo Finance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.io as pio

from portfolio_lab import plots
from portfolio_lab.backtest import (align, benchmark_returns, equal_weight_returns, train_test, walk_forward)
from portfolio_lab.data import SNAPSHOT, download_prices, load_snapshot
from portfolio_lab.metrics import summarize, summary_table
from portfolio_lab.optimize import (StrategyConfig, efficient_frontier, estimate_inputs, optimize_portfolio,
                                    random_portfolios, stock_risk_return)
from portfolio_lab.simulate import simulate


def pct(x):
    return f"{x:.1%}"


def fmt_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in ("Total return", "Annual return", "Annual volatility", "Worst drop (max drawdown)"):
        d[c] = d[c].map(pct)
    d["Sharpe ratio"] = d["Sharpe ratio"].map(lambda v: f"{v:.2f}")
    d["Growth of $10,000"] = d["Growth of $10,000"].map(lambda v: f"${v:,.0f}")
    return d.to_html(classes="tbl", border=0, escape=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download fresh prices from Yahoo Finance")
    ap.add_argument("--rf", type=float, default=0.03, help="annual risk-free rate (default 3%%)")
    ap.add_argument("--train-end", default="2022-12-31")
    args = ap.parse_args()
    rf, train_end = args.rf, args.train_end

    if args.refresh:
        px, dj = download_prices(cache=SNAPSHOT)
    else:
        px, dj = load_snapshot()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)

    cfgs = {
        "Max Sharpe (original method)": StrategyConfig("orig", "max_sharpe", 0.25, stabilized=False, rf=rf),
        "Max Sharpe (stabilized)": StrategyConfig("stab", "max_sharpe", 0.25, stabilized=True, rf=rf),
        "Min volatility": StrategyConfig("minvol", "min_vol", 0.25, stabilized=True, rf=rf),
    }

    # 1. Replication of the original course results (0% risk-free, no cap for the first row)
    raw = estimate_inputs(px, stabilized=False)
    rep_rows = {}
    for label, obj, cap in (("Max Sharpe, no cap", "max_sharpe", 1.0), ("Max Sharpe, 25% cap", "max_sharpe", 0.25),
                            ("Min volatility", "min_vol", 1.0)):
        r = optimize_portfolio(raw, obj, cap, rf=0.0)
        rep_rows[label] = {"Expected return": r.expected_return, "Volatility": r.volatility, "Sharpe": r.sharpe,
                           "Top holding": f"{r.holdings.index[0]} {r.holdings.iloc[0]:.1%}"}
    replication = pd.DataFrame(rep_rows).T
    replication.to_csv(ROOT / "results" / "replication_of_original.csv")

    # 2. Train on early data, test on later data
    tt = {k: train_test(px, c, train_end) for k, c in cfgs.items()}
    split = next(iter(tt.values())).train_end
    test_series = {k: v.test_returns for k, v in tt.items()}
    test_series["Equal weight (all 30)"] = equal_weight_returns(px, split)
    test_series["DJIA"] = benchmark_returns(dj, split)
    names, vals = list(test_series), align(*test_series.values())
    test_series = dict(zip(names, vals))
    train_test_table = summary_table(test_series, rf)
    train_test_table.to_csv(ROOT / "results" / "train_test_metrics.csv")

    promise = pd.DataFrame({
        "Promised (data it was fitted on)": {k: summarize(v.train_returns, rf)["Sharpe ratio"] for k, v in tt.items()},
        "Delivered (unseen data)": {k: summarize(v.test_returns, rf)["Sharpe ratio"] for k, v in tt.items()},
    })
    promise.to_csv(ROOT / "results" / "promised_vs_delivered_sharpe.csv")

    # 3. Walk-forward: refit every quarter on the trailing year
    wf = {k: walk_forward(px, c) for k, c in cfgs.items()}
    wf_series = {k: v.returns for k, v in wf.items()}
    first = next(iter(wf_series.values())).index[0]
    wf_series["Equal weight (all 30)"] = equal_weight_returns(px, first - pd.Timedelta(days=1))
    wf_series["DJIA"] = benchmark_returns(dj, first - pd.Timedelta(days=1))
    names, vals = list(wf_series), align(*wf_series.values())
    wf_series = dict(zip(names, vals))
    wf_table = summary_table(wf_series, rf)
    wf_table.to_csv(ROOT / "results" / "walk_forward_metrics.csv")

    # 4. Portfolios built from all data (what the tool would suggest today)
    full_inputs = estimate_inputs(px, stabilized=True)
    today = {k: c.build(px) for k, c in cfgs.items()}
    pd.DataFrame({k: v.weights for k, v in today.items()}).round(4).to_csv(ROOT / "results" / "weights_all_data.csv")

    # 5. Simulation for the stabilized max Sharpe mix, one trading month ahead
    mix_returns = (px.pct_change().dropna() * today["Max Sharpe (stabilized)"].weights).sum(axis=1)
    sims = {d: simulate(mix_returns, 21, 5000, drift=d) for d in ("As in history", "Long-run stock average (7% a year)")}
    sim_rows = {d: s.summary() for d, s in sims.items()}
    pd.DataFrame(sim_rows).T.to_csv(ROOT / "results" / "monte_carlo_21_days.csv")

    # 6. Figures
    cloud = random_portfolios(raw, 3000, rf=rf)
    frontier = efficient_frontier(raw, max_weight=1.0, rf=rf, n_points=35)
    stocks = stock_risk_return(raw)
    pts = {}
    for name, c in (("Original max Sharpe", optimize_portfolio(raw, "max_sharpe", 0.25, rf)),
                    ("Min volatility", optimize_portfolio(raw, "min_vol", 0.25, rf)),
                    ("Equal weight", optimize_portfolio(raw, "equal", 0.25, rf))):
        pts[name] = (c.volatility, c.expected_return)
    figs = {
        "promise": plots.promise_vs_reality_chart(promise),
        "growth_test": plots.growth_chart(test_series, title=f"After {split:%b %Y}: growth of $10,000 on data the optimizer never saw"),
        "growth_wf": plots.growth_chart(wf_series, title="Refit every quarter, trade only on what was known then"),
        "dd": plots.drawdown_chart(wf_series),
        "frontier": plots.frontier_chart(cloud, frontier, stocks, pts),
        "weights": plots.weights_chart(today["Max Sharpe (stabilized)"].weights, "Today's stabilized max Sharpe mix"),
        "sector": plots.sector_chart(today["Max Sharpe (stabilized)"].weights),
        "fan": plots.fan_chart(sims["As in history"].bands(), 10_000, "One trading month ahead, if the past repeats"),
        "fan_cautious": plots.fan_chart(sims["Long-run stock average (7% a year)"].bands(), 10_000, "Same mix, with a more cautious return assumption"),
        "hist": plots.outcomes_histogram(sims["Long-run stock average (7% a year)"].final, 10_000),
        "corr": plots.correlation_heatmap(px.pct_change().dropna()),
    }
    for k, f in figs.items():
        f.update_layout(height=min(f.layout.height or 460, 560))

    def div(key, first=False):
        return pio.to_html(figs[key], full_html=False, include_plotlyjs=True if first else False,
                           config={"displayModeBar": False, "responsive": True})

    best_in = promise.loc["Max Sharpe (original method)", "Promised (data it was fitted on)"]
    best_out = promise.loc["Max Sharpe (original method)", "Delivered (unseen data)"]
    djia_tt = train_test_table.loc["DJIA"]
    orig_tt = train_test_table.loc["Max Sharpe (original method)"]
    djia_wf = wf_table.loc["DJIA"]
    stab_wf = wf_table.loc["Max Sharpe (stabilized)"]
    sim_c = sim_rows["Long-run stock average (7% a year)"]
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dow 30 Portfolio Lab: results</title>
<style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--bg:#fcfcfb;--blue:#2a78d6}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Inter,-apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 20px 64px}} h1{{font-size:30px;margin:0 0 6px}} h2{{font-size:21px;margin:44px 0 6px}}
p.lead{{color:var(--ink2);margin:0 0 20px;max-width:720px}} p{{max-width:760px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:20px 0}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px}} .card b{{display:block;font-size:26px;line-height:1.2}} .card span{{color:var(--ink2);font-size:13px}}
.tbl{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}} .tbl th,.tbl td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}} .tbl th:first-child,.tbl td:first-child{{text-align:left}}
.tbl thead th{{color:var(--ink2);font-weight:600}} .note{{color:var(--ink2);font-size:13px;max-width:760px}} .wrap{{overflow-x:auto}}
.callout{{border-left:4px solid var(--blue);background:#eef4fc;padding:10px 14px;border-radius:0 8px 8px 0;max-width:760px}}
</style></head><body><main>
<h1>Dow 30 Portfolio Lab</h1>
<p class="lead">Can math pick a better mix of Dow stocks? This report tests the idea honestly: build a portfolio from the past, then see what it did on data it never saw.</p>
<div class="cards">
<div class="card"><b>{best_in:.2f} to {best_out:.2f}</b><span>Sharpe ratio of the original max-Sharpe mix: on the data it was fitted on vs afterwards</span></div>
<div class="card"><b>{pct(orig_tt['Total return'])} vs {pct(djia_tt['Total return'])}</b><span>Original mix vs the DJIA after {split:%b %Y}</span></div>
<div class="card"><b>{pct(stab_wf['Annual return'])} vs {pct(djia_wf['Annual return'])}</b><span>Yearly return, refit every quarter, vs the DJIA</span></div>
<div class="card"><b>{sim_c['Chance of ending below start']:.0%}</b><span>Chance of losing money in one month (cautious return assumption)</span></div>
</div>
<h2>1. Promised vs delivered</h2>
<p>The optimizer looks great on the years it studied. The real test is what happens next. Here the portfolios were built using data through {split:%b %Y} only.</p>
{div('promise', first=True)}
<div class="callout">The gap between the two bars is the cost of "learning" from noise. Picking last period's winners did not predict the next period's winners.</div>
<h2>2. How they did on unseen data</h2>
{div('growth_test')}
<div class="wrap">{fmt_table(train_test_table)}</div>
<p class="note">Sharpe ratio uses a {rf:.0%} risk-free rate. The DJIA is a price index and leaves out dividends, so equal weight is the fairer yardstick for stock mixes.</p>
<h2>3. A fairer test: refit every quarter</h2>
<p>Each quarter the strategy re-learns from the trailing 12 months and trades with a 0.05% cost. This is closer to how a real investor would use it.</p>
<p>Results here are mixed and the gaps are small. Over about three and a half years, a difference of a point or two a year can easily be luck, so read this as "no reliable edge", not "a winner".</p>
{div('growth_wf')}
<div class="wrap">{fmt_table(wf_table)}</div>
{div('dd')}
<h2>4. The risk and reward map</h2>
<p>Every dot is a possible mix. The black line is the best return available for each level of risk, based on 2020 to 2024 history.</p>
{div('frontier')}
<h2>5. What it would hold today, and what could happen next</h2>
{div('weights')}{div('sector')}
<p>Simulated one trading month ahead. History from 2020 to 2024 was unusually strong for stocks, so the second view assumes a long-run average return instead.</p>
{div('fan')}{div('fan_cautious')}{div('hist')}
<h2>6. Which stocks move together</h2>
{div('corr')}
<h2>Limits to keep in mind</h2>
<p class="note">About four and a half years of data is a short sample. Past returns do not predict future returns. The stock list is the Dow lineup for most of 2020 to 2024, not today's lineup. Educational project, not financial advice.</p>
</main></body></html>"""
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\nREPLICATION\n", replication.round(4))
    print("\nPROMISED vs DELIVERED (Sharpe)\n", promise.round(2))
    print(f"\nTRAIN/TEST after {split.date()}\n", train_test_table.round(3))
    print(f"\nWALK-FORWARD from {first.date()}\n", wf_table.round(3))
    print("\nTURNOVER per refit (avg)", {k: round(v.turnover.mean(), 2) for k, v in wf.items()})
    print("\nTODAY max sharpe stabilized\n", today["Max Sharpe (stabilized)"].holdings.round(3).to_dict())
    print("\nMC 21d\n", pd.DataFrame(sim_rows).T.round(3))
    print("\nwrote docs/index.html", round((ROOT / "docs" / "index.html").stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
