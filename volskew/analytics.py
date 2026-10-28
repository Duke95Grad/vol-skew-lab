"""Historical context: percentiles, ranks and plain-English readings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metric:
    key: str
    novice: str
    pro: str
    help: str
    high: str
    low: str


METRICS: dict[str, Metric] = {m.key: m for m in [
    Metric("iv30", "Option prices", "ATM IV (30d)",
           "How much movement options are pricing over the next month, as a yearly percentage "
           "(30-day at-the-money implied volatility).",
           "Options cost more than usual. The market expects bigger moves.",
           "Options cost less than usual. The market expects a calm stretch."),
    Metric("rr25", "Cost of downside protection", "25Δ risk reversal (put − call)",
           "How much more puts cost than calls, in volatility points (30-day 25-delta put "
           "minus 25-delta call). Positive means puts are pricier.",
           "Puts cost more than usual compared with calls. Investors are paying up for protection.",
           "Puts cost less than usual compared with calls. Protection is relatively cheap."),
    Metric("fly25", "Cost of far-away strikes", "25Δ butterfly",
           "How much extra options far from today's price cost, on both sides, compared with "
           "at-the-money options (average of 25-delta put and call vol, minus ATM vol).",
           "Options far from today's price cost more than usual. The market is pricing a big surprise.",
           "Options far from today's price are relatively cheap."),
    Metric("term", "Near-term vs. longer-term", "Term structure (3M − 1M ATM)",
           "3-month implied volatility minus 1-month. Usually positive. Negative means "
           "near-term options cost more, often because of an event or current stress.",
           "Longer-dated options cost more than usual relative to near-term ones.",
           "Near-term options cost more than usual relative to longer-dated ones."),
    Metric("rv20", "Recent actual movement", "Realized vol (20d)",
           "How much the price has actually moved over the last 20 trading days, as a yearly "
           "percentage. Compare it with what options are pricing.",
           "The price has been moving more than usual lately.",
           "The price has been unusually quiet lately."),
    Metric("vrp", "Priced vs. actual movement", "IV30 − RV20",
           "Implied minus realized volatility. Usually positive, because option sellers expect "
           "to be paid for taking risk.",
           "Options are pricing much more movement than the price has actually shown.",
           "The price has been moving about as much as options expect, or more."),
]}

HEADLINE_KEYS = ["iv30", "rr25", "fly25", "term", "rv20", "vrp"]


def percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float((s <= value).mean() * 100)


def iv_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    lo, hi = s.min(), s.max()
    return float(100 * (value - lo) / (hi - lo)) if hi > lo else 50.0


def pct_phrase(pct: float, lower_is: str = "lower") -> str:
    """'higher than 82% of the past year', with sensible wording at the extremes."""
    if pct >= 99.5:
        return "the highest of the past year"
    if pct <= 0.5:
        return "the lowest of the past year"
    return f"higher than on {pct:.0f}% of days in the past year"


def band_word(pct: float) -> str:
    if pct >= 90:
        return "near the top of its 1-year range"
    if pct >= 75:
        return "higher than usual"
    if pct > 25:
        return "in its normal range"
    if pct > 10:
        return "lower than usual"
    return "near the bottom of its 1-year range"


def band_color(pct: float) -> str:
    if pct >= 90 or pct <= 10:
        return "red"
    if pct >= 75 or pct <= 25:
        return "orange"
    return "green"


def fmt_pts(v: float, signed: bool = False) -> str:
    """Vol as percentage points, e.g. 0.184 -> '18.4%'; signed for spreads."""
    return f"{v * 100:+.1f} pts" if signed else f"{v * 100:.1f}%"


def fmt_metric(key: str, v: float) -> str:
    return fmt_pts(v, signed=key in ("rr25", "fly25", "term", "vrp"))


def snapshot(hist: pd.DataFrame, lookback: int = 252) -> dict:
    """Current value, percentile and stats for every headline metric."""
    h = hist.iloc[-lookback:]
    out = {}
    for k in HEADLINE_KEYS:
        s = h[k].dropna()
        v = float(s.iloc[-1])
        out[k] = {
            "value": v,
            "pct": percentile(s, v),
            "rank": iv_rank(s, v),
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p90": float(s.quantile(0.90)),
            "min": float(s.min()),
            "max": float(s.max()),
            "z": float((v - s.mean()) / s.std()) if s.std() > 0 else 0.0,
            "month_ago": float(s.iloc[-22]) if len(s) > 22 else float("nan"),
        }
    return out


def reading(key: str, snap: dict) -> str:
    m = METRICS[key]
    pct = snap[key]["pct"]
    v = snap[key]["value"]
    if pct >= 75:
        text = m.high
    elif pct <= 25:
        text = m.low
    else:
        text = "In its usual range."
    if key == "rr25" and v < 0:
        text += " Right now calls cost more than puts (call skew)."
    if key == "term" and v < -0.005 and pct > 25:
        text += " Near-term options currently cost more than longer-dated ones."
    return text


def unusual(snap: dict, hi: float = 85, lo: float = 15) -> list[str]:
    keys = [k for k in HEADLINE_KEYS if snap[k]["pct"] >= hi or snap[k]["pct"] <= lo]
    return sorted(keys, key=lambda k: -abs(snap[k]["pct"] - 50))


def summary_sentences(snap: dict, event: tuple[str, float] | None) -> list[str]:
    """Two or three plain-English sentences describing today's surface."""
    iv = snap["iv30"]
    rr = snap["rr25"]
    s = []
    s.append(f"Options are pricing about {iv['value'] * 100:.0f}% yearly movement, "
             f"which is {band_word(iv['pct'])} ({pct_phrase(iv['pct'])}).")
    if rr["value"] >= 0:
        s.append(f"Downside puts trade {rr['value'] * 100:.1f} vol points above upside calls; "
                 f"that skew is {band_word(rr['pct'])}.")
    else:
        s.append(f"Upside calls trade {-rr['value'] * 100:.1f} vol points above downside puts "
                 f"(call skew); puts-vs-calls is {band_word(rr['pct'])}.")
    if event:
        label, days = event
        s.append(f"The next {label.lower()} is expected in about {days:.0f} days, so expiries "
                 f"after that date carry extra vol for the expected jump.")
    elif snap["term"]["value"] < -0.01:
        s.append("Near-term options are pricier than longer-dated ones (an inverted curve).")
    return s
