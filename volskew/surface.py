"""A small parametric volatility surface, plus the skew metrics built from it.

The surface is described by a handful of intuitive numbers (see SurfaceParams):

* ATM term structure: a short-end level that decays toward a long-run level,
  plus an optional one-off event (earnings, FDA decision) that adds a lump of
  variance to every expiry after the event date.
* Smile: for each expiry, implied vol is the ATM vol scaled by a quadratic in
  standardised moneyness x = ln(K/F) / (ATM vol * sqrt(T)), i.e. "how many
  standard deviations away from the forward is this strike?"
      iv = atm * (1 + slope * x + curv * x^2)
  slope < 0 is classic equity put skew, slope > 0 is call skew, curv lifts
  both wings.

Real market data will later be fitted to the same parameters (or replaced by
an SVI fit) without changing any of the screens.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

X_CLIP = 3.0
_XGRID = np.linspace(-4.0, 4.0, 1601)
MIN_T = 1.0 / 365.0


@dataclass(frozen=True)
class SurfaceParams:
    spot: float
    r: float
    q: float
    atm_short: float          # ATM vol at the very front, ignoring any event
    atm_long: float           # long-run ATM vol
    tau: float                # speed the front decays toward the long end (years)
    slope: float              # skew; negative = puts richer, positive = calls richer
    curv: float               # smile curvature; higher = richer wings
    event_days: float | None = None   # calendar days to the next event (None = none)
    event_move: float = 0.0   # market's expected one-day event move (decimal)

    def replace(self, **kw) -> "SurfaceParams":
        return dataclasses.replace(self, **kw)


# ----------------------------------------------------------------------------
# Raw (array-friendly) building blocks
# ----------------------------------------------------------------------------
def atm_raw(atm_short, atm_long, tau, ev_days, ev_move, T):
    """ATM vol for arrays of parameters. ev_days may be NaN for 'no event'."""
    T = np.maximum(np.asarray(T, float), MIN_T)
    base = atm_long + (atm_short - atm_long) * np.exp(-T / tau)
    base = np.maximum(base, 0.01)
    ev_days = np.asarray(ev_days, float)
    has_ev = np.isfinite(ev_days) & (ev_days >= 0) & (T * 365.0 >= np.nan_to_num(ev_days, nan=1e9))
    var = base**2 * T + np.where(has_ev, np.asarray(ev_move, float) ** 2, 0.0)
    return np.sqrt(var / T)


def smile_raw(atm, slope, curv, x):
    xc = np.clip(x, -X_CLIP, X_CLIP)
    return np.maximum(atm * (1.0 + slope * xc + curv * xc**2), 0.25 * atm)


def delta_point_raw(atm, slope, curv, T, delta, kind):
    """IV and standardised moneyness at a given forward delta.

    kind='put' with delta=0.25 means the 25-delta put; 'call' likewise.
    Works on arrays of parameters (one row per date).
    """
    atm = np.asarray(atm, float)[..., None]
    slope = np.asarray(slope, float)[..., None]
    curv = np.asarray(curv, float)[..., None]
    T = np.maximum(np.asarray(T, float), MIN_T)
    T = np.broadcast_to(T, atm.shape[:-1])[..., None]
    sig = smile_raw(atm, slope, curv, _XGRID)
    sq = np.sqrt(T)
    d1 = (-_XGRID * atm * sq + 0.5 * sig**2 * T) / (sig * sq)
    call_delta = ndtr(d1)
    target = delta if kind == "call" else 1.0 - delta
    idx = np.abs(call_delta - target).argmin(axis=-1)[..., None]
    iv = np.take_along_axis(sig, idx, axis=-1)[..., 0]
    x = _XGRID[idx[..., 0]]
    return iv, x


# ----------------------------------------------------------------------------
# Convenience wrappers on SurfaceParams
# ----------------------------------------------------------------------------
def _ev(p: SurfaceParams):
    return np.nan if p.event_days is None else p.event_days


def forward(p: SurfaceParams, T):
    return p.spot * np.exp((p.r - p.q) * np.asarray(T, float))


def atm_vol(p: SurfaceParams, T):
    return atm_raw(p.atm_short, p.atm_long, p.tau, _ev(p), p.event_move, T)


def std_moneyness(p: SurfaceParams, K, T):
    T = np.maximum(np.asarray(T, float), MIN_T)
    return np.log(np.asarray(K, float) / forward(p, T)) / (atm_vol(p, T) * np.sqrt(T))


def iv(p: SurfaceParams, K, T):
    """Implied vol for strike(s) K and expiry(ies) T in years."""
    T = np.maximum(np.asarray(T, float), MIN_T)
    a = atm_vol(p, T)
    x = np.log(np.asarray(K, float) / forward(p, T)) / (a * np.sqrt(T))
    return smile_raw(a, p.slope, p.curv, x)


def strike_from_x(p: SurfaceParams, x, T):
    T = np.maximum(np.asarray(T, float), MIN_T)
    return forward(p, T) * np.exp(np.asarray(x) * atm_vol(p, T) * np.sqrt(T))


def call_delta_of_strike(p: SurfaceParams, K, T):
    T = np.maximum(np.asarray(T, float), MIN_T)
    sig = iv(p, K, T)
    d1 = (np.log(forward(p, T) / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    return ndtr(d1)


def strike_for_delta(p: SurfaceParams, T, delta, kind):
    a = atm_vol(p, T)
    _, x = delta_point_raw(a, p.slope, p.curv, T, delta, kind)
    return float(strike_from_x(p, x, T))


def round_strike(K, spot):
    step = strike_step(spot)
    return float(np.round(K / step) * step)


def strike_step(spot):
    if spot < 30:
        return 0.5
    if spot < 100:
        return 1.0
    if spot < 250:
        return 2.5
    return 5.0


def metrics(p: SurfaceParams, days: float = 30) -> dict:
    """Headline skew numbers for one expiry (vols in decimals)."""
    T = days / 365.0
    a = float(atm_vol(p, T))
    out = {"atm": a}
    for d, tag in [(0.25, "25"), (0.10, "10")]:
        out[f"put{tag}"] = float(delta_point_raw(a, p.slope, p.curv, T, d, "put")[0])
        out[f"call{tag}"] = float(delta_point_raw(a, p.slope, p.curv, T, d, "call")[0])
    out["rr25"] = out["put25"] - out["call25"]
    out["rr10"] = out["put10"] - out["call10"]
    out["fly25"] = 0.5 * (out["put25"] + out["call25"]) - a
    out["term"] = float(atm_vol(p, 90 / 365.0) - atm_vol(p, 30 / 365.0))
    return out


def slope_for_skew_change(p: SurfaceParams, pts: float, days: float = 30) -> float:
    """New slope that moves the 25-delta put-minus-call skew by `pts` vol points."""
    if pts == 0:
        return p.slope
    h = 0.01
    base = metrics(p, days)["rr25"]
    bumped = metrics(p.replace(slope=p.slope - h), days)["rr25"]
    g = (bumped - base) / h
    if abs(g) < 1e-9:
        return p.slope
    return p.slope - (pts / 100.0) / g


def roll(p: SurfaceParams, days: float) -> SurfaceParams:
    """Move the calendar forward: the event clock ticks down, and drops off once passed."""
    if days <= 0 or p.event_days is None:
        return p
    left = p.event_days - days
    return p.replace(event_days=left if left >= 0 else None)
