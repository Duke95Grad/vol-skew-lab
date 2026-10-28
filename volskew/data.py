"""Data layer.

Every screen talks to a DataProvider, never to the simulator directly, so a
future release can drop in a real-data provider (option chains -> fitted
surface parameters, closing prices -> realised vol, a broker file -> positions)
without touching the UI.

The DemoDataProvider builds one year of simulated history for five real
securities. Spot levels approximate late-October 2026; vols, skew, history,
positions and event dates are simulated, each tuned so "today" tells a
different story.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from .surface import SurfaceParams, atm_raw, delta_point_raw, round_strike, strike_step

AS_OF = pd.Timestamp("2026-10-27")
N_DAYS = 252
RATE = 0.04
EXPIRIES = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
TRADING_TO_CAL = 365.0 / 252.0

HISTORY_COLUMNS = [
    "spot", "atm_short", "atm_long", "tau", "slope", "curv", "event_days", "event_move",
    "iv30", "rr25", "fly25", "term", "rv20", "vrp",
]


@dataclass
class Archetype:
    ticker: str
    name: str
    blurb: str
    story: str
    spot: float
    q: float
    atm_short: float
    atm_long: float
    tau: float
    slope: float
    curv: float
    vol_of_vol: float
    rho: float                      # correlation of spot returns with vol shocks
    event_move: float = 0.0
    event_every: int | None = None  # trading days between recurring events
    next_event_td: int | None = None
    event_label: str = "Earnings report"
    targets: dict = field(default_factory=dict)   # where "today" lands
    year_return: float = 0.0        # approximate trailing 1-year price change
    positions: list = field(default_factory=list)
    seed: int = 0


ARCHETYPES: dict[str, Archetype] = {a.ticker: a for a in [
    # Spot levels approximate late-October 2026 closes. Everything else (vol
    # levels, history, skew, event timing) is simulated for illustration.
    Archetype(
        ticker="SPX", year_return=0.13, name="S&P 500 Index", seed=34,
        blurb="Cash-settled index options. Typically the steepest put skew in the market.",
        story="Downside protection is unusually expensive right now.",
        spot=7800.0, q=0.012, atm_short=0.145, atm_long=0.175, tau=0.35,
        slope=-0.20, curv=0.035, vol_of_vol=1.0, rho=-0.8,
        targets={"atm_short": 0.172, "slope": -0.315, "curv": 0.042},
        positions=[
            {"type": "put", "qty": 2, "m": 0.97, "dte": 45},
            {"type": "put", "qty": -2, "m": 0.90, "dte": 45},
            {"type": "call", "qty": -1, "m": 1.05, "dte": 30},
        ],
    ),
    Archetype(
        ticker="NVDA", year_return=0.12, name="NVIDIA Corp.", seed=11,
        blurb="Mega-cap semiconductor leader. Moderate put skew; large earnings moves.",
        story="Options are quiet overall, but earnings in about three weeks are lifting the front month.",
        spot=215.0, q=0.0002, atm_short=0.40, atm_long=0.42, tau=0.4,
        slope=-0.08, curv=0.03, vol_of_vol=0.7, rho=-0.6,
        event_move=0.07, event_every=63, next_event_td=16, event_label="Earnings report",
        targets={"atm_short": 0.31, "atm_long": 0.37},
        positions=[
            {"type": "stock", "qty": 300},
            {"type": "call", "qty": -3, "m": 1.08, "dte": 30},
            {"type": "put", "qty": 3, "m": 0.90, "dte": 30},
        ],
    ),
    Archetype(
        ticker="PG", year_return=0.056, name="Procter & Gamble Co.", seed=54,
        blurb="Household-products bellwether. Low volatility, steady dividend.",
        story="Options are cheap and downside protection is unusually affordable.",
        spot=160.0, q=0.027, atm_short=0.17, atm_long=0.18, tau=0.4,
        slope=-0.11, curv=0.022, vol_of_vol=0.6, rho=-0.5,
        event_move=0.03, event_every=63, next_event_td=60, event_label="Earnings report",
        targets={"atm_short": 0.123, "atm_long": 0.15, "slope": -0.05},
        positions=[{"type": "stock", "qty": 1000}],
    ),
    Archetype(
        ticker="GME", year_return=0.12, name="GameStop Corp.", seed=23,
        blurb="Retail-investor favorite. Calls are often priced above puts.",
        story="Implied vol is near the top of its range and upside calls are in heavy demand.",
        spot=26.5, q=0.0, atm_short=0.55, atm_long=0.55, tau=0.3,
        slope=0.04, curv=0.035, vol_of_vol=1.1, rho=0.3,
        event_move=0.10, event_every=63, next_event_td=29, event_label="Earnings report",
        targets={"atm_short": 0.88, "atm_long": 0.68, "slope": 0.10},
        positions=[
            {"type": "call", "qty": -10, "m": 1.25, "dte": 30},
            {"type": "call", "qty": 10, "m": 1.50, "dte": 30},
            {"type": "put", "qty": -5, "m": 0.85, "dte": 21},
        ],
    ),
    Archetype(
        ticker="TSLA", year_return=-0.17, name="Tesla, Inc.", seed=34,
        blurb="High-volatility mega-cap. Skew is often flat, and wings tend to be rich on both sides.",
        story="Earnings just passed, yet far out-of-the-money options on both sides are unusually rich.",
        spot=375.0, q=0.0, atm_short=0.50, atm_long=0.52, tau=0.5,
        slope=-0.02, curv=0.06, vol_of_vol=0.8, rho=-0.3,
        event_move=0.08, event_every=63, next_event_td=62, event_label="Earnings report",
        targets={"curv": 0.105, "atm_short": 0.60},
        positions=[
            {"type": "call", "qty": 5, "m": 1.00, "dte": 30},
            {"type": "put", "qty": 5, "m": 1.00, "dte": 30},
        ],
    ),
]}


def _ou(rng, n, x0, mean, kappa, sig, shocks=None):
    dt = 1 / 252
    x = np.empty(n)
    x[0] = x0
    z = rng.standard_normal(n) if shocks is None else shocks
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (mean - x[t - 1]) * dt + sig * np.sqrt(dt) * z[t]
    return x


def _blend_tail(path, target, length=20):
    w = np.linspace(0, 1, length) ** 1.5
    out = path.copy()
    out[-length:] = path[-length:] * (1 - w) + target * w
    return out


@lru_cache(maxsize=None)
def _simulate(ticker: str) -> pd.DataFrame:
    a = ARCHETYPES[ticker]
    rng = np.random.default_rng(a.seed)
    warm = 25
    n = N_DAYS + warm
    dt = 1 / 252

    zv = rng.standard_normal(n)
    zs = a.rho * zv + np.sqrt(1 - a.rho**2) * rng.standard_normal(n)

    ls = _ou(rng, n, np.log(a.atm_short), np.log(a.atm_short), 6.0, a.vol_of_vol, zv)
    ll = _ou(rng, n, np.log(a.atm_long), np.log(a.atm_long), 2.0, 0.35 * a.vol_of_vol,
             0.8 * zv + 0.6 * rng.standard_normal(n))
    slope = _ou(rng, n, a.slope, a.slope, 4.0, 0.7 * abs(a.slope) + 0.05)
    curv = _ou(rng, n, a.curv, a.curv, 4.0, 0.7 * a.curv + 0.02)

    # Event calendar (index positions; the last row is "today")
    events = []
    if a.next_event_td is not None:
        future = (n - 1) + a.next_event_td
        events.append(future)
        if a.event_every:
            k = future - a.event_every
            while k >= 0:
                events.append(k)
                k -= a.event_every
    events = np.array(sorted(events)) if events else np.array([], dtype=int)
    idx = np.arange(n)
    if len(events):
        nxt = np.array([events[events >= t][0] if (events >= t).any() else -1 for t in idx])
        ev_days = np.where(nxt >= 0, (nxt - idx) * TRADING_TO_CAL, np.nan)
    else:
        ev_days = np.full(n, np.nan)

    # Spot path: realised vol runs a bit under implied (the usual risk premium)
    rv_true = np.exp(ls) * 0.88
    rets = (RATE - a.q - 0.5 * rv_true**2) * dt + rv_true * np.sqrt(dt) * zs
    past_events = set(int(e) for e in events if e < n - 1)
    for e in past_events:
        rets[e + 1] += rng.normal() * a.event_move * 0.9
    path = np.cumsum(rets) - np.sum(rets)          # log price relative to today (ends at 0)
    # Tilt the path so the trailing year shows roughly the real-world move;
    # a straight-line tilt leaves daily volatility untouched.
    first = n - N_DAYS                             # first row that gets displayed
    w = np.clip((np.arange(n) - (n - 1)) / (n - 1 - first), -1, 0)   # -1 at first row -> 0 today
    drift_now = path[first]
    path = path + w * drift_now + w * np.log1p(a.year_return)
    spot = a.spot * np.exp(path)

    atm_s, atm_l = np.exp(ls), np.exp(ll)
    t = a.targets
    if "atm_short" in t:
        atm_s = _blend_tail(atm_s, t["atm_short"])
    if "atm_long" in t:
        atm_l = _blend_tail(atm_l, t["atm_long"])
    if "slope" in t:
        slope = _blend_tail(slope, t["slope"])
    if "curv" in t:
        curv = _blend_tail(curv, t["curv"])

    T30, T90 = 30 / 365, 90 / 365
    ev_move = np.full(n, a.event_move)
    tau = np.full(n, a.tau)
    iv30 = atm_raw(atm_s, atm_l, tau, ev_days, ev_move, T30)
    iv90 = atm_raw(atm_s, atm_l, tau, ev_days, ev_move, T90)
    p25, _ = delta_point_raw(iv30, slope, curv, T30, 0.25, "put")
    c25, _ = delta_point_raw(iv30, slope, curv, T30, 0.25, "call")

    lr = np.diff(np.log(spot), prepend=np.log(spot[0]))
    rv20 = pd.Series(lr).rolling(20).std().to_numpy() * np.sqrt(252)

    df = pd.DataFrame({
        "spot": spot, "atm_short": atm_s, "atm_long": atm_l, "tau": tau,
        "slope": slope, "curv": curv, "event_days": ev_days, "event_move": ev_move,
        "iv30": iv30, "rr25": p25 - c25, "fly25": 0.5 * (p25 + c25) - iv30,
        "term": iv90 - iv30, "rv20": rv20,
    })
    df["vrp"] = df["iv30"] - df["rv20"]
    df = df.iloc[warm:].reset_index(drop=True)
    df.index = pd.bdate_range(end=AS_OF, periods=N_DAYS)
    df.index.name = "date"
    return df


class DemoDataProvider:
    """Simulated data. Swap for a real provider with the same methods later."""

    label = "Simulated market data"
    as_of = AS_OF
    rate = RATE
    expiries = EXPIRIES

    def tickers(self) -> list[str]:
        return list(ARCHETYPES)

    def info(self, ticker: str) -> dict:
        a = ARCHETYPES[ticker]
        return {"ticker": a.ticker, "name": a.name, "blurb": a.blurb, "story": a.story,
                "event_label": a.event_label, "q": a.q}

    def history(self, ticker: str) -> pd.DataFrame:
        """Daily history with the columns in HISTORY_COLUMNS, indexed by date."""
        return _simulate(ticker)

    def params_on(self, ticker: str, row: pd.Series) -> SurfaceParams:
        a = ARCHETYPES[ticker]
        ev = row["event_days"]
        return SurfaceParams(
            spot=float(row["spot"]), r=RATE, q=a.q,
            atm_short=float(row["atm_short"]), atm_long=float(row["atm_long"]),
            tau=float(row["tau"]), slope=float(row["slope"]), curv=float(row["curv"]),
            event_days=None if not np.isfinite(ev) else float(ev),
            event_move=float(row["event_move"]),
        )

    def current_params(self, ticker: str) -> SurfaceParams:
        h = self.history(ticker)
        return self.params_on(ticker, h.iloc[-1])

    def typical_params(self, ticker: str) -> SurfaceParams:
        """A 'normal day' surface: one-year medians of the shape, no event."""
        h = self.history(ticker)
        p = self.current_params(ticker)
        return p.replace(atm_short=float(h["atm_short"].median()),
                         atm_long=float(h["atm_long"].median()),
                         slope=float(h["slope"].median()), curv=float(h["curv"].median()),
                         event_days=None)

    def positions(self, ticker: str) -> list[dict]:
        """Demo positions with real strikes. Options: qty in contracts (x100)."""
        a = ARCHETYPES[ticker]
        S = self.current_params(ticker).spot
        out = []
        for pos in a.positions:
            leg = {"type": pos["type"], "qty": pos["qty"]}
            if pos["type"] != "stock":
                leg["strike"] = round_strike(S * pos["m"], S)
                leg["dte"] = pos["dte"]
            out.append(leg)
        return out


def leg_label(leg: dict) -> str:
    """Plain label, e.g. 'Buy 2 · 7,565 put · 45 days' or 'Own 300 shares'."""
    if leg["type"] == "stock":
        return f"{'Own' if leg['qty'] > 0 else 'Short'} {abs(leg['qty']):,} shares"
    side = "Buy" if leg["qty"] > 0 else "Sell"
    k = leg["strike"]
    ks = f"{k:,.0f}" if k >= 100 or float(k).is_integer() else f"{k:,.2f}".rstrip("0").rstrip(".")
    return f"{side} {abs(leg['qty'])} · {ks} {leg['type']} · {leg['dte']} days"


__all__ = ["DemoDataProvider", "ARCHETYPES", "EXPIRIES", "leg_label", "strike_step"]
