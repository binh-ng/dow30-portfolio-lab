# Changelog

v1 is the DATA 612 course submission (notebook, report, slides). v2 keeps the same core method and adds honest testing, safer claims, and an app for non-technical users.

## Replication check

Before changing anything, the v2 code was run on the v1 data and reproduced the v1 numbers: WMT 33.15% and IBM 13.08% in the unconstrained max-Sharpe mix, return 23.6%, volatility 14.5%, Sharpe 1.62, and the min-volatility mix at 10.8% return and 11.3% volatility. One thing this exposed: v1 implicitly used a 0% risk-free rate. v2 makes it a visible setting (default 3%).

## The big change: out-of-sample testing

| | v1 | v2 |
|---|---|---|
| How performance was judged | Weights fitted on 2020 to 2024, then compared with the DJIA over the same 2020 to 2024 | Fit on early data, judge on later data. Plus a quarterly walk-forward with trading costs |
| Result | "Consistently outperformed the DJIA" | The max-Sharpe mix's Sharpe ratio fell from 1.65 to -0.03 on unseen data. The original method (25% cap) returned 5.2% vs DJIA 28.4% after Dec 2022. Walk-forward results were close to the benchmark |
| Conclusion | Optimized portfolios beat the market | No reliable edge shown. The v1 outperformance came from fitting to the same data it was tested on |

## Claims corrected

| v1 statement | Issue | v2 |
|---|---|---|
| "Consistently outperformed the DJIA" | In-sample result | Replaced with out-of-sample evidence |
| "Monte Carlo validated robustness" | A model fitted to the same data cannot validate itself | Described as "a range of possible outcomes" |
| "Tight confidence intervals" | The 20-day range was about -6% to +10% | Shown as a fan chart with plain wording |
| "95% CI" | It is a percentile range of simulated outcomes, not a confidence interval | Labeled "bad case / good case (1 in 20)" |
| "Histograms support normality" | Daily moves reached -26% and +26%, and v1's own best-fit result was a heavy-tailed `genhyperbolic` | Candidates include heavy-tailed fits, with the choice reported. Bootstrap needs no distribution |

## Consistency fixes

- Data window: the report said Jan 1, 2020 and the code started 2020-07-01. v2 documents the real window.
- Prices: the report said "unadjusted", but the code output is labeled adjusted. v2 documents this.
- Min-volatility "constrained" and "unconstrained" were identical (the cap never bound), and the Monte Carlo printout labeled the constrained one "Unconstrained". v2 reports one min-volatility result.
- Monte Carlo used two compounding conventions, `exp(cumsum(r))` for value and `cumprod(1+r)` for returns, so the two summaries disagreed (2.03% vs 1.95% mean). v2 uses log returns everywhere.
- The report listed sector constraints, which were never implemented. v2 implements them (sector cap slider).
- The report credited Seaborn and Gurobi, which the notebook does not use. Typos fixed: "Sharpio", "PORFOLIO".

## Modeling additions

- Steadier estimates: Ledoit-Wolf covariance plus expected returns shrunk halfway toward the cross-sectional mean. This is standard practice for reducing reliance on noisy past winners (v1 held 11 of 30 stocks, with 33% in one name). It helped only slightly on unseen data (Sharpe 0.03 vs -0.03), so it is a sensible default, not a fix.
- Risk-free rate is an explicit setting.
- Cautious-to-aggressive dial (a point on the efficient frontier).
- Return-assumption options for simulation (history, half of history, 7% a year, flat), since 2020 to 2024 was unusually strong.
- Equal-weight benchmark alongside the DJIA (which is price-only).
- Metrics: annual return, volatility, Sharpe, max drawdown, growth of $10,000.

## New visuals

Promised vs delivered bars, growth of $10,000 with the train/test split marked, drawdowns, efficient frontier with random mixes and single stocks, weights and sector exposure, simulation fan chart and outcome histogram, clustered correlation heatmap, stock risk vs reward scatter. Every chart has hover details. Colors follow a fixed palette, with the DJIA always gray.

## New for non-technical users

- Streamlit app: goal picker, sliders, plain-English summaries, a "test it on the past" mode, and a download button for weights.
- Static interactive report in `docs/index.html` (works on GitHub Pages with no install).

## Engineering

- Code moved from one notebook into a small package with 15 passing tests (replication, constraints, sector cap, no look-ahead in train/test and walk-forward, simulation sanity, download layout).
- Bundled data snapshot so results reproduce offline. `requirements.txt`, MIT license, `.gitignore`.
- Original notebook kept in `notebooks/` with outputs cleared (the original was 10.6 MB and would not render well on GitHub).

## Known limitations

- Prices come from a bundled snapshot (Jul 2020 to Dec 2024). `python scripts/run_analysis.py --refresh` pulls fresh data from Yahoo Finance.
- Sector labels and the Dow lineup note are from general knowledge and may differ slightly from index provider data.
- About 4.5 years of data is a short sample, so conclusions about any strategy are tentative.

## Customization (v2.1)

Everything a non-technical user would want to change is in the sidebar:

- **Amount:** any amount from $1,000. The app turns weights into dollars per stock.
- **Risk tolerance:** five steps from Very cautious to Aggressive. Each step moves the mix along the efficient frontier.
- **Which stocks:** pick sectors, leave out specific stocks, and mark must-haves with a minimum share.
- **Portfolio size:** hold at most N stocks. The optimizer keeps the must-haves plus the strongest positions, then re-solves.
- **Plain-English errors:** impossible requests (for example, minimums that add up to more than 100%) get a clear message.
- 27 tests cover the limits above, and a scenario sweep of the app ran across every control.
