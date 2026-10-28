"""Turn what the surface is saying into example trade structures.

Every card is an educational example tied to a specific reading of the
surface, with the reasoning, what needs to happen, and what can go wrong.
Nothing here is a recommendation.
"""
from __future__ import annotations

from .analytics import pct_phrase
from .surface import SurfaceParams, metrics, round_strike, strike_for_delta


def _opt(p, kind, qty, delta, dte):
    K = strike_for_delta(p, dte / 365, delta, kind)
    return {"type": kind, "qty": qty, "strike": round_strike(K, p.spot), "dte": dte}


def _atm(p, kind, qty, dte):
    return {"type": kind, "qty": qty, "strike": round_strike(p.spot, p.spot), "dte": dte}


def _fix_dup(legs):
    """If rounding made two strikes of a spread identical, nudge the further one out."""
    from .surface import strike_step
    seen = {}
    for leg in legs:
        if leg["type"] == "stock":
            continue
        key = (leg["type"], leg["strike"], leg["dte"])
        if key in seen:
            step = strike_step(leg["strike"])
            leg["strike"] += step if leg["type"] == "call" else -step
        seen[(leg["type"], leg["strike"], leg["dte"])] = True
    return legs


def build_cards(p: SurfaceParams, snap: dict, holdings: list[dict], event: tuple | None) -> list[dict]:
    m30 = metrics(p, 30)
    rr, iv, fly, term, vrp = (snap[k] for k in ["rr25", "iv30", "fly25", "term", "vrp"])
    long_shares = sum(l["qty"] for l in holdings if l["type"] == "stock" and l["qty"] > 0)
    lots = max(1, int(long_shares // 100)) if long_shares else 1
    cards = []

    # 1. Pricey protection ------------------------------------------------------
    if rr["value"] > 0 and rr["pct"] >= 75:
        why = (f"A typical downside put is priced at {m30['put25'] * 100:.1f}% implied volatility, "
               f"versus {m30['call25'] * 100:.1f}% for the matching upside call. That "
               f"{rr['value'] * 100:.1f}-point gap is {pct_phrase(rr['pct'])}.")
        cards.append({
            "key": "put_spread", "level": "Beginner-friendly", "risk": "Defined risk",
            "condition": "Downside protection is expensive",
            "title": "Put spread: buy protection, sell the priciest part",
            "why": why,
            "what": "Buy a put a little below today's price and sell a put further below. The "
                    "further put is where prices are most inflated, so selling it pays for much "
                    "of the protection.",
            "needs": "A moderate drop toward the lower strike.",
            "risks": "Protection stops at the lower strike, so a crash beyond it isn't covered. "
                     "If the price rises, you lose what you paid.",
            "legs": _fix_dup([_opt(p, "put", 1, 0.40, 45), _opt(p, "put", -1, 0.20, 45)]),
        })
        cards.append({
            "key": "risk_reversal", "level": "Advanced", "risk": "Large loss possible",
            "condition": "Downside protection is expensive",
            "title": "Risk reversal: sell the pricey put, buy the cheaper call",
            "why": why + " Selling the expensive side to pay for the cheaper side puts that gap to work.",
            "what": "Sell a downside put and buy an upside call, often for close to zero cost. "
                    "It behaves like owning the underlying between the two strikes.",
            "needs": "The price to hold steady or rise, and protection to get cheaper.",
            "risks": "The sold put loses dollar for dollar below its strike, so a sharp drop can "
                     "cause large losses. That drop is exactly what the market is charging extra for.",
            "legs": _fix_dup([_opt(p, "put", -1, 0.25, 45), _opt(p, "call", 1, 0.25, 45)]),
        })

    # 2. Cheap protection ---------------------------------------------------------
    if rr["value"] >= 0 and rr["pct"] <= 25:
        cheap = ("the lowest of the past year" if rr["pct"] <= 0.5 else
                 f"lower than on {100 - rr['pct']:.0f}% of days in the past year")
        why = (f"Puts cost only {rr['value'] * 100:.1f} volatility points more than calls, "
               f"{cheap}. Protection is on sale.")
        if long_shares:
            legs = [{"type": "stock", "qty": lots * 100},
                    _opt(p, "put", lots, 0.25, 60), _opt(p, "call", -lots, 0.25, 60)]
            title = "Collar your shares while protection is cheap"
            what = (f"You own {long_shares:,.0f} shares. Buy a put below today's price and pay "
                    "for it by selling a call above. Right now the put is cheap compared with the call.")
            risks = "Your gains stop at the call strike until the options expire."
        else:
            legs = [_opt(p, "put", 1, 0.25, 60)]
            title = "Buy a put while protection is cheap"
            what = "A put below today's price costs less than usual right now for the same protection."
            risks = "If nothing happens, the put loses value over time and you lose what you paid."
        cards.append({
            "key": "cheap_protection", "level": "Beginner-friendly", "risk": "Defined risk",
            "condition": "Downside protection is cheap", "title": title, "why": why,
            "what": what,
            "needs": "Any meaningful drop, or protection simply returning to its usual price.",
            "risks": risks, "legs": _fix_dup(legs),
        })

    # 3. Calls cost more than puts --------------------------------------------------
    if rr["value"] < 0:
        why = (f"Upside calls are priced {-rr['value'] * 100:.1f} volatility points above downside "
               "puts. Traders are paying up for upside, which is unusual for most stocks.")
        if long_shares:
            legs = [{"type": "stock", "qty": lots * 100}, _opt(p, "call", -lots, 0.25, 30)]
            title, level = "Covered call: sell the pricey upside", "Beginner-friendly"
            what = f"Sell a call above today's price against your {long_shares:,.0f} shares and collect the inflated premium."
            risks = "If the price rallies past the strike, your gains stop there."
        else:
            legs = _fix_dup([_opt(p, "call", -1, 0.25, 30), _opt(p, "call", 1, 0.10, 30)])
            title, level = "Call spread: sell the pricey upside, capped", "Intermediate"
            what = ("Sell a call above today's price and buy one further up to cap the risk. "
                    "You collect premium where prices are most inflated.")
            risks = ("Pricey calls often come with a squeeze. A sharp rally through both strikes "
                     "gives the maximum loss.")
        cards.append({
            "key": "call_skew", "level": level, "risk": "Defined risk",
            "condition": "Calls cost more than puts", "title": title, "why": why, "what": what,
            "needs": "The price to stay below the sold strike while call demand cools.",
            "risks": risks, "legs": legs,
        })

    # 4. Options price more movement than is happening -----------------------------
    if iv["pct"] >= 70 and vrp["value"] > 0:
        risk = ("A large move through either sold strike. High prices are often high for a reason"
                + (f": the next {event[0].lower()} is about {event[1]:.0f} days away, inside this "
                   "expiry." if event and event[1] < 30 else ", so check for upcoming events first."))
        cards.append({
            "key": "condor", "level": "Intermediate", "risk": "Defined risk",
            "condition": "Options price more movement than is happening",
            "title": "Iron condor: get paid while the price stays in a range",
            "why": (f"Options are pricing {iv['value'] * 100:.0f}% yearly movement "
                    f"({pct_phrase(iv['pct'])}), but the price has actually been moving at "
                    f"{snap['rv20']['value'] * 100:.0f}%: a {vrp['value'] * 100:.0f}-point cushion."),
            "what": "Sell a put below and a call above today's price, and buy options further out "
                    "on each side to cap the loss. You keep the premium if the price stays between "
                    "the sold strikes.",
            "needs": "The price to stay in range and option prices to ease.",
            "risks": risk,
            "legs": _fix_dup([_opt(p, "put", 1, 0.07, 30), _opt(p, "put", -1, 0.16, 30),
                              _opt(p, "call", -1, 0.16, 30), _opt(p, "call", 1, 0.07, 30)]),
        })

    # 5. Cheap options --------------------------------------------------------------
    if iv["pct"] <= 25:
        low = ("the lowest of the past year" if iv["pct"] <= 0.5 else
               f"lower than on {100 - iv['pct']:.0f}% of days in the past year")
        cards.append({
            "key": "straddle", "level": "Intermediate", "risk": "Defined risk",
            "condition": "Options are cheap",
            "title": "Straddle: own movement while it's on sale",
            "why": f"Options are pricing {iv['value'] * 100:.0f}% yearly movement, {low}.",
            "what": "Buy a call and a put at today's price. You profit from a big move in either "
                    "direction, or from option prices rising back to normal.",
            "needs": "A move bigger than the breakevens, or option prices rising.",
            "risks": "The position loses a little value every day. Cheap options can stay cheap.",
            "legs": [_atm(p, "call", 1, 60), _atm(p, "put", 1, 60)],
        })

    # 6. Event lifting near-term options --------------------------------------------
    if event and event[1] <= 45:
        label, days = event
        front = next(d for d in [14, 21, 30, 45, 60] if d >= days + 2)
        back = front + 30
        cards.append({
            "key": "calendar", "level": "Intermediate", "risk": "Defined risk",
            "condition": f"An upcoming {label.lower()} is lifting near-term prices",
            "title": "Calendar spread: sell the event month, own a later month",
            "why": (f"The next {label.lower()} is about {days:.0f} days away. Options expiring in "
                    f"{front} days are priced at {float(metrics(p, front)['atm']) * 100:.0f}% "
                    f"implied volatility, versus {float(metrics(p, back)['atm']) * 100:.0f}% for "
                    f"{back} days. The nearer month carries the event premium."),
            "what": "Sell a call at today's price in the month that includes the event, and buy "
                    "the same strike one month later. After the event, near-term prices usually "
                    "drop sharply (the vol crush).",
            "needs": "A smaller move on the event than the market expects.",
            "risks": "A big jump away from the strike hurts both options, and the later month can "
                     "lose value too.",
            "legs": [_atm(p, "call", -1, front), _atm(p, "call", 1, back)],
        })
    elif term["value"] < -0.01 and term["pct"] <= 20:
        cards.append({
            "key": "calendar_inv", "level": "Intermediate", "risk": "Defined risk",
            "condition": "Near-term options are unusually pricey",
            "title": "Calendar spread: sell the pricey near month",
            "why": (f"Options expiring in 3 months are priced {-term['value'] * 100:.1f} points "
                    "below those expiring in 1 month. Normally it's the other way round, so "
                    "near-term options carry extra stress premium."),
            "what": "Sell a 30-day call at today's price and buy a 60-day call at the same strike.",
            "needs": "Near-term worry to fade while the price stays near the strike.",
            "risks": "If stress grows, near-term prices can climb further and the price can jump.",
            "legs": [_atm(p, "call", -1, 30), _atm(p, "call", 1, 60)],
        })

    # 7. Pricey far-away strikes ------------------------------------------------------
    if fly["pct"] >= 85 and not any(c["key"] == "condor" for c in cards):
        cards.append({
            "key": "wings", "level": "Advanced", "risk": "Defined risk",
            "condition": "Far-away strikes are pricey",
            "title": "Wide iron condor: sell the pricey far-away options",
            "why": (f"Options far from today's price cost {fly['value'] * 100:.1f} points extra, "
                    f"{pct_phrase(fly['pct'])}. The market is paying up for a big surprise either way."),
            "what": "Sell puts and calls well away from today's price, and buy protection even "
                    "further out.",
            "needs": "No extreme move in either direction.",
            "risks": "Pricey far-away options often signal a possible big jump. A large move "
                     "gives the maximum loss.",
            "legs": _fix_dup([_opt(p, "put", 1, 0.03, 30), _opt(p, "put", -1, 0.10, 30),
                              _opt(p, "call", -1, 0.10, 30), _opt(p, "call", 1, 0.03, 30)]),
        })

    if not cards:
        cards.append({
            "key": "none", "level": "", "risk": "", "condition": "Nothing stands out",
            "title": "No clear signal from option prices today",
            "why": "Option prices, protection cost and term structure are all in their usual ranges.",
            "what": "That's useful to know: options are fairly priced against their own history, "
                    "so any trade should come from your view on the price itself.",
            "needs": "", "risks": "", "legs": [],
        })
    return cards
