"""Plotly figures shared by the Streamlit app and the static report.

Colors follow a fixed-order categorical palette (blue, orange, aqua, yellow, magenta, violet).
A strategy keeps the same color on every chart. The DJIA is always neutral gray.
Sequential magnitude (frontier cloud, correlation) uses a single blue ramp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage

from .config import SECTORS
from .metrics import drawdown_series

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def color_for(name: str, i: int = 0) -> str:
    n = name.lower()
    if "djia" in n or "dow jones" in n:
        return INK2
    if "equal" in n:
        return ORANGE
    if "min" in n or "lowest" in n:
        return AQUA
    if "original" in n or "raw" in n:
        return YELLOW
    if "custom" in n or "your" in n:
        return VIOLET
    if "sharpe" in n or "best" in n:
        return BLUE
    return CATEGORICAL[i % len(CATEGORICAL)]


def _base(fig: go.Figure, title: str, height: int = 460, xtitle: str | None = None, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=17, color=INK)),
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=13, color=INK2),
        margin=dict(l=60, r=24, t=70, b=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=INK2)),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def growth_chart(
    returns: dict[str, pd.Series],
    start_value: float = 10_000,
    split_date: pd.Timestamp | None = None,
    title: str = "Growth of $10,000",
) -> go.Figure:
    fig = go.Figure()
    for i, (name, r) in enumerate(returns.items()):
        wealth = start_value * (1 + r).cumprod()
        is_bench = "djia" in name.lower()
        fig.add_trace(go.Scatter(
            x=wealth.index, y=wealth.values, name=name, mode="lines",
            line=dict(color=color_for(name, i), width=2, dash="dash" if is_bench else "solid"),
            hovertemplate="%{y:$,.0f}",
        ))
    if split_date is not None:
        fig.add_vline(x=split_date, line=dict(color=INK2, width=1, dash="dot"))
        fig.add_annotation(
            x=split_date, y=1, yref="paper", yanchor="bottom", xanchor="left",
            text="  Optimizer only saw data to the left", showarrow=False, font=dict(color=INK2, size=12),
        )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return _base(fig, title, ytitle="Portfolio value")


def drawdown_chart(returns: dict[str, pd.Series], title: str = "How far each portfolio fell from its peak") -> go.Figure:
    fig = go.Figure()
    for i, (name, r) in enumerate(returns.items()):
        dd = drawdown_series(r)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name, mode="lines",
            line=dict(color=color_for(name, i), width=2, dash="dash" if "djia" in name.lower() else "solid"),
            hovertemplate="%{y:.1%}",
        ))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%")
    return _base(fig, title, ytitle="Below previous high")


def frontier_chart(
    cloud: pd.DataFrame,
    frontier: pd.DataFrame,
    stocks: pd.DataFrame,
    portfolios: dict[str, tuple[float, float]],
    title: str = "Risk and reward: every mix of these stocks",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cloud["volatility"], y=cloud["return"], mode="markers", name="Random mixes",
        marker=dict(size=5, color=cloud["sharpe"], colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)],
                    opacity=0.55, colorbar=dict(title="Return per<br>unit of risk", thickness=12, len=0.6)),
        hovertemplate="Risk %{x:.1%}<br>Return %{y:.1%}<extra>Random mix</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=stocks["volatility"], y=stocks["return"], mode="markers", name="Single stocks",
        text=stocks.index, marker=dict(size=8, color="white", line=dict(color=INK2, width=1.5)),
        hovertemplate="%{text}<br>Risk %{x:.1%}<br>Return %{y:.1%}<extra>Single stock</extra>",
    ))
    if len(frontier):
        fig.add_trace(go.Scatter(
            x=frontier["volatility"], y=frontier["return"], mode="lines", name="Best possible mixes (efficient frontier)",
            line=dict(color=INK, width=2), hovertemplate="Risk %{x:.1%}<br>Return %{y:.1%}<extra>Frontier</extra>",
        ))
    for i, (name, (vol, ret)) in enumerate(portfolios.items()):
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text", name=name, text=[name], textposition="top center",
            textfont=dict(color=INK, size=12),
            marker=dict(size=14, color=color_for(name, i), line=dict(color="white", width=2)),
            hovertemplate=f"{name}<br>Risk %{{x:.1%}}<br>Return %{{y:.1%}}<extra></extra>",
        ))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return _base(fig, title, height=520, xtitle="Risk (how bumpy the ride is, per year)", ytitle="Expected return per year")


def weights_chart(weights: pd.Series, title: str = "Where the money goes", color: str = BLUE, top: int = 15) -> go.Figure:
    w = weights[weights > 1e-4].sort_values(ascending=True).tail(top)
    fig = go.Figure(go.Bar(
        x=w.values, y=w.index, orientation="h", marker=dict(color=color),
        text=[f"{v:.1%}" for v in w.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%", range=[0, max(w.max() * 1.2, 0.05)])
    return _base(fig, title, height=max(260, 28 * len(w) + 110), xtitle="Share of portfolio")


def sector_chart(weights: pd.Series, title: str = "Exposure by sector", color: str = BLUE) -> go.Figure:
    sec = weights.groupby(lambda t: SECTORS.get(t, "Other")).sum()
    sec = sec[sec > 1e-4].sort_values()
    fig = go.Figure(go.Bar(
        x=sec.values, y=sec.index, orientation="h", marker=dict(color=color),
        text=[f"{v:.0%}" for v in sec.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%", range=[0, max(sec.max() * 1.2, 0.05)])
    return _base(fig, title, height=max(240, 34 * len(sec) + 110), xtitle="Share of portfolio")


def promise_vs_reality_chart(table: pd.DataFrame, title: str = "What the optimizer promised vs what happened next") -> go.Figure:
    """table: index = strategy name, columns 'Promised (data it was fitted on)' and 'Delivered (unseen data)'."""
    fig = go.Figure()
    for col, color in zip(table.columns, (BLUE, ORANGE)):
        fig.add_trace(go.Bar(
            name=col, x=table.index, y=table[col], marker=dict(color=color),
            text=[f"{v:.2f}" for v in table[col]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>" + col + ": %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    ymax = float(table.values.max())
    ymin = float(min(table.values.min(), 0))
    # Leave room under zero so labels on tiny negative bars do not collide with the category names.
    lower = min(ymin * 1.4, -0.14 * ymax) if ymin < 0 else 0
    fig.update_yaxes(range=[lower, ymax * 1.2])
    return _base(fig, title, ytitle="Sharpe ratio (return per unit of risk)")


def fan_chart(bands: pd.DataFrame, start_value: float, title: str = "Where the portfolio could be") -> go.Figure:
    x = bands.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bands["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p5"], fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                             line=dict(width=0), name="90% of outcomes land here", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p25"], fill="tonexty", fillcolor="rgba(42,120,214,0.30)",
                             line=dict(width=0), name="Middle half of outcomes", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p50"], line=dict(color=BLUE, width=2.5), name="Typical (median) outcome",
                             hovertemplate="%{y:$,.0f}"))
    for q, label in (("p95", "Good case (95th percentile)"), ("p5", "Bad case (5th percentile)")):
        fig.add_trace(go.Scatter(x=x, y=bands[q], line=dict(color=BLUE, width=1, dash="dot"), name=label,
                                 showlegend=False, hovertemplate=label + ": %{y:$,.0f}"))
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[start_value, start_value], mode="lines", name="Starting amount",
                             line=dict(color=INK2, width=1, dash="dash"), hoverinfo="skip"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return _base(fig, title, xtitle="Trading days from today", ytitle="Portfolio value")


def outcomes_histogram(final: np.ndarray, start_value: float, title: str = "How the possible endings spread out") -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=final, nbinsx=60, marker=dict(color=BLUE, line=dict(color="white", width=1)),
        hovertemplate="Ending value %{x:$,.0f}<br>%{y} of simulated outcomes<extra></extra>",
    ))
    p_loss = float((final < start_value).mean())
    fig.add_vline(x=start_value, line=dict(color=INK, width=1.5, dash="dash"))
    fig.add_annotation(x=start_value, y=1, yref="paper", yanchor="bottom", xanchor="right",
                       text=f"Start. {p_loss:.0%} of outcomes end below this  ", showarrow=False, font=dict(color=INK, size=12))
    fig.update_xaxes(tickprefix="$", tickformat=",")
    return _base(fig, title, xtitle="Portfolio value at the end", ytitle="Number of simulated outcomes")


def correlation_heatmap(returns: pd.DataFrame, title: str = "Which stocks move together") -> go.Figure:
    corr = returns.corr()
    order = leaves_list(linkage(1 - corr.values[np.triu_indices(len(corr), 1)], method="average"))
    corr = corr.iloc[order, order]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=0, zmax=1,
        colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)],
        colorbar=dict(title="Move<br>together", thickness=12),
        hovertemplate="%{y} and %{x}<br>Correlation %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return _base(fig, title, height=620)


def stock_scatter(stocks: pd.DataFrame, highlight: pd.Series | None = None, title: str = "Every stock: reward vs risk") -> go.Figure:
    held = highlight[highlight > 1e-4].index if highlight is not None else []
    fig = go.Figure()
    for label, mask, fill in (
        ("In your portfolio", stocks.index.isin(held), BLUE),
        ("Not held", ~stocks.index.isin(held), "white"),
    ):
        s = stocks[mask]
        if not len(s):
            continue
        fig.add_trace(go.Scatter(
            x=s["volatility"], y=s["return"], mode="markers+text", text=s.index, textposition="top center",
            textfont=dict(size=11, color=INK2), name=label,
            marker=dict(size=11, color=fill, line=dict(color=BLUE if fill == "white" else "white", width=1.5)),
            hovertemplate="%{text}<br>Risk %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
        ))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return _base(fig, title, height=520, xtitle="Risk per year", ytitle="Past return per year")
