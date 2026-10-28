"""Plotly figures. Puts are rose, calls are teal, 'today' is ink."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .surface import (SurfaceParams, atm_vol, call_delta_of_strike, forward, iv, metrics,
                      strike_from_x)

INK = "#24324F"
PUT = "#C2417A"
CALL = "#1F8A8A"
MUTED = "#98A2B3"
GOLD = "#D99A00"
GAIN = "#2E8B57"
LOSS = "#C0392B"
FONT = dict(family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif", size=13, color="#344054")
GRID = "#EEF1F4"


def _layout(fig, title=None, height=360, **kw):
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15, color="#1F2937")) if title else None,
                      height=height, margin=dict(l=10, r=10, t=44 if title else 12, b=10),
                      font=FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=12)),
                      hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color="#1F2937")),
                      **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False, title_font=dict(size=12, color="#667085"))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title_font=dict(size=12, color="#667085"))
    return fig


def _pct(v):
    return np.asarray(v) * 100


def smile_points(p: SurfaceParams, days: int, n: int = 121, width: float = 2.6):
    T = days / 365
    x = np.linspace(-width, width, n)
    K = strike_from_x(p, x, T)
    return K, iv(p, K, T), call_delta_of_strike(p, K, T)


def fig_smile(p: SurfaceParams, days: int, axis: str = "pct", overlays=(),
              show_move: bool = True, title=None, height=380):
    """Implied vol by strike for one expiry. overlays: [(label, params, dash)]."""
    T = days / 365
    fig = go.Figure()

    def xs(K, D, q):
        if axis == "strike":
            return K
        if axis == "delta":
            return D
        return K / q.spot * 100

    for ov in overlays:
        label, q, dash = ov[:3]
        K, v, D = smile_points(q, ov[3] if len(ov) > 3 else days)
        fig.add_scatter(x=xs(K, D, q), y=_pct(v), name=label, mode="lines",
                        line=dict(color=MUTED, dash=dash, width=2))

    K, v, D = smile_points(p, days)
    X = xs(K, D, p)
    atm_i = np.argmin(np.abs(K - forward(p, T)))
    fig.add_scatter(x=X[: atm_i + 1], y=_pct(v[: atm_i + 1]), name="Downside strikes (puts)",
                    mode="lines", line=dict(color=PUT, width=3))
    fig.add_scatter(x=X[atm_i:], y=_pct(v[atm_i:]), name="Upside strikes (calls)",
                    mode="lines", line=dict(color=CALL, width=3))

    if show_move:
        a = float(atm_vol(p, T))
        lo, hi = p.spot * np.exp(-a * np.sqrt(T)), p.spot * np.exp(a * np.sqrt(T))
        if axis == "delta":
            lo_x, hi_x = (float(call_delta_of_strike(p, lo, T)), float(call_delta_of_strike(p, hi, T)))
        elif axis == "strike":
            lo_x, hi_x = lo, hi
        else:
            lo_x, hi_x = lo / p.spot * 100, hi / p.spot * 100
        fig.add_vrect(x0=min(lo_x, hi_x), x1=max(lo_x, hi_x), fillcolor=GOLD, opacity=0.08,
                      line_width=0, annotation_text="likely range",
                      annotation_position="top left", annotation_font_size=11)

    xt = {"pct": "Strike as % of today's price", "strike": "Strike price", "delta": "Delta"}[axis]
    fig.update_xaxes(title=xt)
    if axis == "delta":
        fig.update_xaxes(autorange="reversed", tickvals=[0.9, 0.75, 0.5, 0.25, 0.1],
                         ticktext=["10Δ put", "25Δ put", "ATM", "25Δ call", "10Δ call"],
                         range=[0.97, 0.03])
    fig.update_yaxes(title="Implied volatility", ticksuffix="%")
    return _layout(fig, title, height)


def fig_term(p: SurfaceParams, expiries, event_label=None, overlays=(), height=340, title=None):
    d = np.array(expiries, float)
    m = [metrics(p, x) for x in d]
    fig = go.Figure()
    for label, q, dash in overlays:
        fig.add_scatter(x=d, y=_pct(atm_vol(q, d / 365)), name=label, mode="lines",
                        line=dict(color=MUTED, dash=dash))
    fig.add_scatter(x=d, y=[_pct(x["put25"]) for x in m], name="Downside put (25Δ)", mode="lines+markers",
                    line=dict(color=PUT), marker=dict(size=6))
    fig.add_scatter(x=d, y=[_pct(x["atm"]) for x in m], name="At the money", mode="lines+markers",
                    line=dict(color=INK, width=3), marker=dict(size=7))
    fig.add_scatter(x=d, y=[_pct(x["call25"]) for x in m], name="Upside call (25Δ)", mode="lines+markers",
                    line=dict(color=CALL), marker=dict(size=6))
    if p.event_days is not None and event_label:
        fig.add_vline(x=p.event_days, line=dict(color=GOLD, dash="dot", width=2),
                      annotation_text=event_label, annotation_position="top right")
    fig.update_xaxes(title="Days until expiry")
    fig.update_yaxes(title="Implied volatility", ticksuffix="%")
    return _layout(fig, title, height)


def fig_surface3d(p: SurfaceParams, expiries, height=520):
    d = np.array([x for x in expiries if x >= 7], float)
    sd = float(atm_vol(p, 0.25)) * 0.5
    m = np.linspace(np.exp(-2.2 * sd), np.exp(2.2 * sd), 41)
    Z = np.array([_pct(iv(p, p.spot * m, x / 365)) for x in d])
    fig = go.Figure(go.Surface(x=m * 100, y=d, z=Z, colorscale=[[0, CALL], [0.5, "#E9EDF3"], [1, PUT]],
                               colorbar=dict(title="IV %"), contours_z=dict(show=False)))
    fig.update_layout(scene=dict(xaxis_title="Strike % of price", yaxis_title="Days",
                                 zaxis_title="IV %", camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))),
                      height=height, margin=dict(l=0, r=0, t=10, b=0), font=FONT)
    return fig


def fig_gauge(pct: float, height=150):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct, number=dict(suffix="th pct", valueformat=".0f", font=dict(size=22)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], tickfont=dict(size=10)),
                   bar=dict(color=INK, thickness=0.28),
                   steps=[dict(range=[0, 10], color="#F3D3DF"), dict(range=[10, 25], color="#F8E7EE"),
                          dict(range=[25, 75], color="#EEF1F5"), dict(range=[75, 90], color="#F8E7EE"),
                          dict(range=[90, 100], color="#F3D3DF")])))
    fig.update_layout(height=height, margin=dict(l=30, r=30, t=14, b=0), font=FONT)
    return fig


def fig_history(series, label, signed=False, height=340, extra=None):
    s = series.dropna() * 100
    fig = go.Figure()
    p10, p50, p90 = s.quantile(0.1), s.median(), s.quantile(0.9)
    fig.add_hrect(y0=p10, y1=p90, fillcolor=MUTED, opacity=0.12, line_width=0,
                  annotation_text="usual range", annotation_position="top left",
                  annotation_font_size=11)
    fig.add_hline(y=p50, line=dict(color=MUTED, dash="dot"), annotation_text="median",
                  annotation_position="bottom right", annotation_font_size=11)
    if extra is not None:
        fig.add_scatter(x=extra[1].index, y=extra[1] * 100, name=extra[0], mode="lines",
                        line=dict(color=MUTED, width=1.5))
    fig.add_scatter(x=s.index, y=s, name=label, mode="lines", line=dict(color=INK, width=2.2))
    fig.add_scatter(x=[s.index[-1]], y=[s.iloc[-1]], name="Today", mode="markers",
                    marker=dict(color=GOLD, size=11, line=dict(color=INK, width=1)))
    fig.update_yaxes(title="vol points" if signed else None, ticksuffix="" if signed else "%")
    return _layout(fig, None, height)


def fig_waterfall(attr: dict, height=320):
    keys = ["Time passing", "Price move", "Volatility", "Protection cost"]
    fig = go.Figure(go.Waterfall(
        x=keys + ["Overall"], y=[attr[k] for k in keys] + [attr["Total"]],
        measure=["relative"] * 4 + ["total"],
        increasing=dict(marker=dict(color=GAIN)), decreasing=dict(marker=dict(color=LOSS)),
        totals=dict(marker=dict(color=INK)),
        connector=dict(line=dict(color=GRID)),
        text=[(f"+${attr[k]:,.0f}" if attr[k] >= 0 else f"−${-attr[k]:,.0f}") for k in keys + ["Total"]],
        textposition="outside"))
    fig.update_yaxes(title="Gain / loss", tickformat="$,.0f")
    return _layout(fig, None, height, showlegend=False)


def fig_heatmap(grid, spot_moves, vol_moves, height=380):
    lim = float(np.nanmax(np.abs(grid))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=grid, x=[f"{m:+.0%}" for m in spot_moves], y=[f"{v:+.0f}" for v in vol_moves],
        colorscale=[[0, LOSS], [0.5, "#F5F6F8"], [1, GAIN]], zmin=-lim, zmax=lim,
        text=[[f"{v / 1000:,.1f}k" if abs(v) >= 1000 else f"{v:,.0f}" for v in row] for row in grid],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="P&L $"), hovertemplate="Price %{x}, vol %{y} pts: $%{z:,.0f}<extra></extra>"))
    fig.update_xaxes(title="Price move")
    fig.update_yaxes(title="Vol shift (pts)")
    return _layout(fig, None, height, hovermode="closest")


def fig_payoff(spots, at_h, today, spot, horizon, height=300):
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_vline(x=spot, line=dict(color=MUTED, dash="dot"), annotation_text="today",
                  annotation_position="top", annotation_font_size=11)
    fig.add_scatter(x=spots, y=today, name="If the price moved today", mode="lines",
                    line=dict(color=MUTED, dash="dash"))
    fig.add_scatter(x=spots, y=at_h, name=f"At expiry ({horizon} days)", mode="lines",
                    line=dict(color=INK, width=3), fill="tozeroy", fillcolor="rgba(36,50,79,0.06)")
    fig.update_xaxes(title="Underlying price")
    fig.update_yaxes(title="Gain / loss", tickformat="$,.0f")
    return _layout(fig, None, height)


def fig_bars(labels, values, title=None, height=280, xtitle=None, ytitle=None):
    colors = [GAIN if v >= 0 else LOSS for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[(f"+${v:,.0f}" if v >= 0 else f"−${-v:,.0f}") for v in values],
                    textposition="outside", width=0.5))
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle, tickformat="$,.0f")
    return _layout(fig, title, height, showlegend=False)


def fig_compare_smiles(items, days=30, normalize=True, height=380):
    """items: [(ticker, params)] on a common delta axis."""
    palette = [INK, PUT, CALL, GOLD, "#7B61A8", "#5E7D4A"]
    fig = go.Figure()
    for i, (tk, p) in enumerate(items):
        K, v, D = smile_points(p, days, width=3.0, n=181)
        keep = (D > 0.05) & (D < 0.95)
        K, v, D = K[keep], v[keep], D[keep]
        a = float(atm_vol(p, days / 365))
        y = v / a if normalize else _pct(v)
        fig.add_scatter(x=D, y=y, name=tk, mode="lines", line=dict(color=palette[i % 6], width=2.5))
    fig.update_xaxes(autorange="reversed", tickvals=[0.9, 0.75, 0.5, 0.25, 0.1],
                     ticktext=["10Δ put", "25Δ put", "ATM", "25Δ call", "10Δ call"],
                     range=[0.95, 0.05], title="Delta")
    fig.update_yaxes(title="IV ÷ ATM IV" if normalize else "Implied vol (%)")
    return _layout(fig, None, height)


def fig_quadrant(rows, height=420):
    """rows: [(ticker, iv_pct, skew_pct)]"""
    fig = go.Figure()
    fig.add_hline(y=50, line=dict(color="#D0D5DD", dash="dot", width=1))
    fig.add_vline(x=50, line=dict(color="#D0D5DD", dash="dot", width=1))
    for x, y, txt in [(75, 93, "Options pricey · protection pricey"),
                      (25, 93, "Options cheap · protection pricey"),
                      (75, 43, "Options pricey · protection cheap"),
                      (25, 43, "Options cheap · protection cheap")]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(size=11, color=MUTED))
    fig.add_scatter(x=[r[1] for r in rows], y=[r[2] for r in rows], text=[r[0] for r in rows],
                    mode="markers+text", textposition="top center",
                    marker=dict(size=16, color=INK, line=dict(color="white", width=2)),
                    hovertemplate="%{text}: IV %{x:.0f}th pct, skew %{y:.0f}th pct<extra></extra>")
    fig.update_xaxes(range=[-6, 106], title="Option prices vs. own past year (percentile)", showgrid=False)
    fig.update_yaxes(range=[-8, 108], title="Protection cost vs. own past year (percentile)", showgrid=False)
    return _layout(fig, None, height, showlegend=False, hovermode="closest")


def fig_distribution(p: SurfaceParams, days: int, height=300):
    """Lognormal price distribution implied by ATM vol, with ±1σ shaded."""
    T = days / 365
    a = float(atm_vol(p, T))
    sd = a * np.sqrt(T)
    z = np.linspace(-3.2, 3.2, 300)
    S = p.spot * np.exp(z * sd)
    dens = np.exp(-0.5 * z**2) / (S * sd * np.sqrt(2 * np.pi))
    fig = go.Figure()
    inside = np.abs(z) <= 1
    fig.add_scatter(x=S, y=dens, mode="lines", line=dict(color=INK, width=2), name="Range of possible prices",
                    hoverinfo="skip")
    fig.add_scatter(x=S[inside], y=dens[inside], fill="tozeroy", mode="none",
                    fillcolor="rgba(217,154,0,0.25)", name="About 2 in 3 outcomes", hoverinfo="skip")
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title=f"Price in {days} days")
    return _layout(fig, None, height)
