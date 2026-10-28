"""Valuation, greeks, scenario shocks and P&L attribution for a set of legs.

A leg is a dict: {"type": "call"|"put"|"stock", "qty": int, "strike": float, "dte": int}.
Options are in contracts of 100 shares.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pricing import bs_greeks, bs_price
from .surface import (SurfaceParams, atm_vol, forward, iv, metrics, roll,
                      slope_for_skew_change, smile_raw)

MULT = 100


@dataclass(frozen=True)
class Shock:
    spot_pct: float = 0.0      # e.g. -0.10
    vol_pts: float = 0.0       # parallel shift, vol points
    front_pts: float = 0.0     # extra shift that fades with expiry
    skew_pts: float = 0.0      # change in 30d 25Δ put-minus-call skew
    days: int = 0              # calendar days forward
    sticky: str = "strike"     # "strike" or "delta"


def apply_shock(p: SurfaceParams, s: Shock) -> SurfaceParams:
    q = roll(p, s.days)
    q = q.replace(spot=p.spot * (1 + s.spot_pct),
                  atm_short=max(q.atm_short + (s.vol_pts + s.front_pts) / 100, 0.02),
                  atm_long=max(q.atm_long + s.vol_pts / 100, 0.02))
    if s.skew_pts:
        q = q.replace(slope=slope_for_skew_change(q, s.skew_pts))
    return q


def _cp(leg):
    return 1.0 if leg["type"] == "call" else -1.0


def leg_value(leg, p_new: SurfaceParams, days: float, spots, S0: float, sticky="delta"):
    """Dollar value of a leg for an array of spot prices, `days` from now, on surface p_new.

    sticky='strike': each strike keeps the vol it had (surface is pinned to strikes).
    sticky='delta':  the smile moves with the stock (vol is read at the new moneyness).
    """
    spots = np.asarray(spots, float)
    if leg["type"] == "stock":
        return leg["qty"] * spots
    T = (leg["dte"] - days) / 365.0
    K = leg["strike"]
    if T <= 0:
        return leg["qty"] * MULT * np.maximum(_cp(leg) * (spots - K), 0.0)
    a = atm_vol(p_new, T)
    F = spots * np.exp((p_new.r - p_new.q) * T)
    k_eff = K * spots / S0 if sticky == "strike" else K
    x = np.log(k_eff / F) / (a * np.sqrt(T))
    vol = smile_raw(a, p_new.slope, p_new.curv, x)
    return leg["qty"] * MULT * bs_price(spots, K, T, p_new.r, p_new.q, vol, _cp(leg))


def book_value(legs, p_new, days, spots, S0, sticky="delta"):
    spots = np.asarray(spots, float)
    total = np.zeros_like(spots)
    for leg in legs:
        total = total + leg_value(leg, p_new, days, spots, S0, sticky)
    return total


def value_now(legs, p: SurfaceParams) -> float:
    return float(book_value(legs, p, 0, [p.spot], p.spot)[0])


def leg_table(legs, p: SurfaceParams) -> pd.DataFrame:
    rows = []
    for leg in legs:
        if leg["type"] == "stock":
            rows.append({"leg": leg, "iv": np.nan, "price": p.spot,
                         "value": leg["qty"] * p.spot, "delta": float(leg["qty"]),
                         "gamma": 0.0, "vega": 0.0, "theta": 0.0})
            continue
        T = leg["dte"] / 365.0
        vol = float(iv(p, leg["strike"], T))
        px = float(bs_price(p.spot, leg["strike"], T, p.r, p.q, vol, _cp(leg)))
        g = bs_greeks(p.spot, leg["strike"], T, p.r, p.q, vol, _cp(leg))
        n = leg["qty"] * MULT
        rows.append({"leg": leg, "iv": vol, "price": px, "value": n * px,
                     "delta": n * float(g["delta"]), "gamma": n * float(g["gamma"]),
                     "vega": n * float(g["vega"]), "theta": n * float(g["theta"])})
    return pd.DataFrame(rows)


def skew_exposure(legs, p: SurfaceParams, pts: float = 1.0) -> float:
    """$ P&L if 30d 25Δ put-minus-call skew rises by `pts` vol points (spot and ATM unchanged)."""
    q = p.replace(slope=slope_for_skew_change(p, pts))
    return float(book_value(legs, q, 0, [p.spot], p.spot)[0] - value_now(legs, p))


def totals(legs, p: SurfaceParams) -> dict:
    t = leg_table(legs, p)
    if t.empty:
        return {k: 0.0 for k in ["value", "delta", "dollar_delta_1pct", "gamma_1pct",
                                 "vega", "theta", "skew"]}
    return {
        "value": float(t["value"].sum()),
        "delta": float(t["delta"].sum()),
        "dollar_delta_1pct": float(t["delta"].sum() * p.spot * 0.01),
        "gamma_1pct": float(t["gamma"].sum() * p.spot * 0.01),   # change in share-delta per 1% move
        "vega": float(t["vega"].sum()),
        "theta": float(t["theta"].sum()),
        "skew": skew_exposure(legs, p),
    }


def attribution(legs, p: SurfaceParams, s: Shock) -> dict:
    """Step-by-step P&L: time passing -> stock move -> vol level -> skew."""
    S0 = p.spot
    S1 = S0 * (1 + s.spot_pct)
    v0 = value_now(legs, p)
    pt = roll(p, s.days)
    v1 = float(book_value(legs, pt, s.days, [S0], S0, s.sticky)[0])
    v2 = float(book_value(legs, pt, s.days, [S1], S0, s.sticky)[0])
    pv = pt.replace(atm_short=max(pt.atm_short + (s.vol_pts + s.front_pts) / 100, 0.02),
                    atm_long=max(pt.atm_long + s.vol_pts / 100, 0.02))
    v3 = float(book_value(legs, pv, s.days, [S1], S0, s.sticky)[0])
    pk = pv.replace(slope=slope_for_skew_change(pv, s.skew_pts)) if s.skew_pts else pv
    v4 = float(book_value(legs, pk, s.days, [S1], S0, s.sticky)[0])
    return {"Time passing": v1 - v0, "Price move": v2 - v1, "Volatility": v3 - v2,
            "Protection cost": v4 - v3, "Total": v4 - v0, "shocked": pk}


def per_leg_pnl(legs, p: SurfaceParams, s: Shock) -> list[float]:
    q = apply_shock(p, s)
    S1 = p.spot * (1 + s.spot_pct)
    out = []
    for leg in legs:
        now = float(leg_value(leg, p, 0, [p.spot], p.spot)[0])
        new = float(leg_value(leg, q, s.days, [S1], p.spot, s.sticky)[0])
        out.append(new - now)
    return out


def pnl_grid(legs, p: SurfaceParams, s: Shock, spot_moves, vol_moves) -> np.ndarray:
    """P&L for a grid of spot moves (columns) x vol shifts (rows); other shock terms held."""
    base = value_now(legs, p)
    spots = p.spot * (1 + np.asarray(spot_moves))
    out = np.empty((len(vol_moves), len(spots)))
    for i, dv in enumerate(vol_moves):
        q = apply_shock(p, Shock(0, s.vol_pts + dv, s.front_pts, s.skew_pts, s.days, s.sticky))
        out[i] = book_value(legs, q, s.days, spots, p.spot, s.sticky) - base
    return out


def vega_by_expiry(legs, p: SurfaceParams) -> pd.DataFrame:
    t = leg_table(legs, p)
    if t.empty:
        return pd.DataFrame(columns=["dte", "vega"])
    t["dte"] = [l.get("dte", np.nan) for l in t["leg"]]
    t = t.dropna(subset=["dte"])
    return t.groupby("dte", as_index=False)["vega"].sum()


def payoff_curves(legs, p: SurfaceParams, width_sd: float = 3.5, n: int = 241):
    """P&L at the first expiry and P&L today, across a range of stock prices."""
    opt = [l for l in legs if l["type"] != "stock"]
    horizon = min(l["dte"] for l in opt) if opt else 30
    sd = float(atm_vol(p, horizon / 365)) * np.sqrt(horizon / 365)
    spots = p.spot * np.exp(np.linspace(-width_sd, width_sd, n) * sd)
    cost = value_now(legs, p)
    ph = roll(p, horizon)
    at_h = book_value(legs, ph, horizon, spots, p.spot, "delta") - cost
    today = book_value(legs, p, 0, spots, p.spot, "delta") - cost
    return spots, at_h, today, horizon


def structure_stats(legs, p: SurfaceParams) -> dict:
    """Premium, approximate max gain/loss and breakevens at the first expiry."""
    opt = [l for l in legs if l["type"] != "stock"]
    horizon = min(l["dte"] for l in opt) if opt else 30
    cost = value_now(legs, p)
    ph = roll(p, horizon)
    sd = float(atm_vol(p, horizon / 365)) * np.sqrt(horizon / 365)
    # Realistic range (±5 standard deviations) plus a near-zero price for short puts
    wide = np.concatenate([[p.spot * 0.001], p.spot * np.exp(np.linspace(-5, 5, 2001) * sd)])
    pnl = book_value(legs, ph, horizon, wide, p.spot, "delta") - cost
    # Far above the strikes every call behaves like stock, so the net count decides
    upside_units = sum(l["qty"] * (MULT if l["type"] == "call" else 1)
                       for l in legs if l["type"] in ("call", "stock"))
    unlimited_up = upside_units > 0
    unlimited_loss = upside_units < 0
    spots, at_h, _, _ = payoff_curves(legs, p)
    sign = np.sign(at_h)
    idx = np.where(np.diff(sign) != 0)[0]
    bes = [float(spots[i] - at_h[i] * (spots[i + 1] - spots[i]) / (at_h[i + 1] - at_h[i]))
           for i in idx]
    stock_cost = sum(l["qty"] * p.spot for l in legs if l["type"] == "stock")
    return {
        "premium": cost - stock_cost,   # >0 you pay, <0 you collect
        "max_gain": None if unlimited_up else float(pnl.max()),
        "max_loss": None if unlimited_loss else float(pnl.min()),
        "breakevens": bes,
        "horizon": horizon,
    }
