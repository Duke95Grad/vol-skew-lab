"""Vol & Skew Lab: a friendly volatility and skew analysis app.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import pathlib
import sys

# Lets the app find its volskew folder even when the repo keeps it in a subfolder
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from volskew import charts, learn, ui
from volskew.analytics import (HEADLINE_KEYS, METRICS, fmt_metric, reading, snapshot, unusual)
from volskew.data import DemoDataProvider, leg_label
from volskew.ideas import build_cards
from volskew.scenario import (Shock, attribution, leg_table, payoff_curves, per_leg_pnl,
                              pnl_grid, structure_stats, totals, vega_by_expiry)
from volskew.surface import atm_vol, metrics

st.set_page_config(page_title="Vol & Skew Lab", page_icon="📐", layout="wide")
ui.inject_css()

CHART_CONFIG = {"displayModeBar": False, "displaylogo": False}


@st.cache_resource
def get_provider():
    return DemoDataProvider()


provider = get_provider()
TICKERS = provider.tickers()


def show_chart(fig, key):
    st.plotly_chart(fig, key=key, config=CHART_CONFIG)


def money(v: float, signed: bool = False) -> str:
    if signed:
        return f"+${v:,.0f}" if v >= 0 else f"−${-v:,.0f}"
    return f"${v:,.0f}" if v >= 0 else f"−${-v:,.0f}"


def esc(text: str) -> str:
    """Escape dollar signs so Streamlit markdown doesn't read them as math."""
    return text.replace("$", "\\$")


def gain_word(v: float) -> str:
    return "gain" if v >= 0 else "lose"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("Vol & Skew Lab")
    st.caption("See what option prices are saying, in plain English.")
    ticker = st.selectbox("Security", TICKERS, key="ticker",
                          format_func=lambda t: f"{t} · {provider.info(t)['name']}")
    info = provider.info(ticker)
    st.caption(info["blurb"])
    st.divider()
    pro = st.toggle("Show advanced details", key="pro",
                    help="Adds trader terminology, extra charts, grids and model numbers.")
    st.divider()
    st.badge("Simulated market data", color="orange", icon=":material/science:")
    st.caption(f"As of {provider.as_of:%B %d, %Y}. Prices are close to recent levels. "
               "Volatilities, history, event dates and positions are simulated for demonstration.")
    with st.expander("Glossary", icon=":material/menu_book:"):
        st.markdown(
            "**Implied volatility (IV):** how much movement option prices imply, as a yearly %.\n\n"
            "**Realized volatility:** how much the price actually moved.\n\n"
            "**Strike:** the price at which an option lets you buy (call) or sell (put).\n\n"
            "**Delta:** roughly the chance an option finishes in the money. A 25-delta put is "
            "a typical out-of-the-money put.\n\n"
            "**Skew:** how much more puts cost than calls, in volatility points.\n\n"
            "**Wings:** options far from today's price.\n\n"
            "**Term structure:** implied volatility by expiry date.\n\n"
            "**Percentile:** where today ranks against the past year. 90th means higher than "
            "on 90% of days.\n\n"
            "**Vega:** profit or loss if implied volatility rises 1 point.\n\n"
            "**Theta:** profit or loss from one day passing.")


def tip(text: str, icon: str = "💡"):
    """Guidance shown only in the simple view."""
    if not pro:
        st.info(text, icon=icon)


def label(key: str) -> str:
    return METRICS[key].pro if pro else METRICS[key].novice


# ----------------------------------------------------------------------------
# Shared state for the selected security
# ----------------------------------------------------------------------------
hist = provider.history(ticker)
p = provider.current_params(ticker)
typical = provider.typical_params(ticker)
month_ago = provider.params_on(ticker, hist.iloc[-22])
snap = snapshot(hist)
event = (info["event_label"], p.event_days) if p.event_days is not None else None
positions = provider.positions(ticker)
m30 = metrics(p, 30)

# ----------------------------------------------------------------------------
# Header: today at a glance
# ----------------------------------------------------------------------------
st.markdown(f"## {ticker} <span style='color:#667085;font-weight:500'>· {info['name']} · "
            f"{p.spot:,.2f}</span>", unsafe_allow_html=True)


def glance_cards():
    iv, rr = snap["iv30"], snap["rr25"]
    move = m30["atm"] * np.sqrt(30 / 365) * 100
    t_iv = ui.tone_for(iv["pct"])
    iv_chip = {"high": "Higher than usual", "low": "Lower than usual", "normal": "Typical"}[t_iv]
    c1 = ui.card("Option prices", ui.chip(iv_chip, t_iv), f"±{move:.1f}% next month",
                 f"That's {iv['value'] * 100:.0f}% implied volatility. {ui.pct_sentence(iv['pct'])}")

    if rr["value"] < 0:
        rr_chip, t_rr = "Calls cost more", "high"
        rr_val = f"Calls +{-rr['value'] * 100:.1f} pts"
        rr_s = "Upside calls cost more than downside puts, which is unusual. Traders are chasing upside."
    else:
        t_rr = ui.tone_for(rr["pct"])
        rr_chip = {"high": "Expensive", "low": "Cheap", "normal": "Typical"}[t_rr]
        rr_val = f"Puts +{rr['value'] * 100:.1f} pts"
        rr_s = f"Puts cost more than calls by this much. {ui.pct_sentence(rr['pct'])}"
    c2 = ui.card("Downside protection", ui.chip(rr_chip, t_rr), rr_val, rr_s)

    if event and event[1] <= 60:
        c3 = ui.card("Coming up", ui.chip(event[0], "event"), f"In about {event[1]:.0f} days",
                     "Expiries after this date include extra volatility for the expected move. "
                     "Date is an estimate.")
    else:
        odd = [k for k in unusual(snap) if k not in ("iv30", "rr25")]
        if odd:
            k = odd[0]
            c3 = ui.card("Also worth noting", ui.chip(METRICS[k].novice, ui.tone_for(snap[k]["pct"])),
                         fmt_metric(k, snap[k]["value"]), reading(k, snap))
        else:
            c3 = ui.card("Everything else", ui.chip("Typical", "normal"), "Nothing unusual",
                         "Other measures are in their usual ranges, and no event is close.")
    cols = st.columns(3)
    for col, html_card in zip(cols, (c1, c2, c3)):
        with col:
            ui.html_block(html_card)


def pro_strip():
    c = st.columns(5)
    c[0].metric("ATM IV 30d", fmt_metric("iv30", snap["iv30"]["value"]),
                f"{(snap['iv30']['value'] - snap['iv30']['month_ago']) * 100:+.1f} vs 1M",
                delta_color="off", border=True,
                help="30-day at-the-money implied volatility, with the change over the past month.")
    c[1].metric("IV pct (1y)", f"{snap['iv30']['pct']:.0f}", delta_color="off", border=True,
                help="Share of the past year when 30-day IV was below today's level.")
    c[2].metric("25Δ RR", fmt_metric("rr25", snap["rr25"]["value"]),
                f"{snap['rr25']['pct']:.0f} pct", delta_color="off", delta_arrow="off", border=True,
                help="30-day 25-delta put IV minus 25-delta call IV, with its 1-year percentile. "
                     "Positive = puts richer.")
    c[3].metric("IV − RV", fmt_metric("vrp", snap["vrp"]["value"]),
                f"{snap['vrp']['pct']:.0f} pct", delta_color="off", delta_arrow="off", border=True,
                help="30-day implied minus 20-day realized volatility, with its 1-year percentile.")
    c[4].metric("Next event", f"{event[1]:.0f}d" if event else "None", event[0] if event else None,
                delta_color="off", delta_arrow="off", border=True,
                help="Calendar days to the next scheduled event. Dates are estimates.")


if pro:
    pro_strip()
else:
    glance_cards()
st.write("")

tabs = st.tabs([":material/school: Start here", ":material/ssid_chart: Surface",
                ":material/history: vs. past year", ":material/science: What if?",
                ":material/lightbulb: Trade ideas", ":material/work: My positions",
                ":material/compare_arrows: Compare"])

# ============================================================================
# 1. Start here
# ============================================================================
with tabs[0]:
    learn.render(ticker, info, p, typical, hist, snap, provider, show_chart, pro)

# ============================================================================
# 2. Surface
# ============================================================================
with tabs[1]:
    ui.intro("How much options cost at each strike and expiry. Higher on the chart = pricier.")
    days = st.select_slider("Expiry", provider.expiries, value=30, key="surf_days",
                            format_func=lambda d: f"{d} days")
    axis_labels = {"pct": "% of today's price", "delta": "Delta", "strike": "Strike price"}
    with st.expander("Chart options", icon=":material/tune:", expanded=pro):
        o1, o2 = st.columns(2)
        axis = o1.radio("Horizontal axis", list(axis_labels), format_func=axis_labels.get,
                        index=1 if pro else 0, key="surf_axis", horizontal=True)
        ov_typ = o2.checkbox("Show a typical day", value=True, key="surf_ov_t",
                             help="The usual shape and level over the past year, with no event.")
        ov_month = o2.checkbox("Show 1 month ago", key="surf_ov_m")

    chart_col, read_col = st.columns([3, 1], gap="large")
    with chart_col:
        overlays = []
        if ov_month:
            overlays.append(("1 month ago", month_ago, "dash"))
        if ov_typ:
            overlays.append(("Typical day", typical, "dot"))
        show_chart(charts.fig_smile(p, days, axis, overlays,
                                    title=f"The smile: implied volatility by strike, {days}-day expiry"),
                   "surf_smile")
    with read_col:
        mt = metrics(p, days)
        st.markdown("**At this expiry**")
        st.markdown(f"Downside put: **{mt['put25'] * 100:.1f}%**  \n"
                    f"At the money: **{mt['atm'] * 100:.1f}%**  \n"
                    f"Upside call: **{mt['call25'] * 100:.1f}%**")
        gap = mt["rr25"] * 100
        st.caption((f"Puts cost {gap:.1f} points more than calls." if gap >= 0 else
                    f"Calls cost {-gap:.1f} points more than puts.") +
                   " Puts and calls shown are 25-delta.")
    if not pro:
        with st.expander("How to read this chart", icon=":material/help:"):
            st.markdown(
                "- Each point is one strike. The height is its implied volatility: higher means "
                "that option is pricier for its distance from today's price.\n"
                "- The **rose** half covers strikes below today's price, where people buy puts "
                "for protection. The **teal** half covers strikes above it.\n"
                "- The shaded band is the **likely range**: where the price ends up about two "
                "times in three, based on what options are pricing.\n"
                "- The dotted line is a **typical day**. If today's line sits higher on the "
                "left, protection costs more than usual.")

    st.write("")
    term_col, term_read = st.columns([3, 1], gap="large")
    with term_col:
        show_chart(charts.fig_term(p, provider.expiries, info["event_label"],
                                   overlays=[("At the money, 1 month ago", month_ago, "dash")] if ov_month else (),
                                   title="Term structure: implied volatility by expiry"), "surf_term")
    with term_read:
        st.markdown("**What this shows**")
        st.markdown(reading("term", snap))
        if event and event[1] < 120:
            before = [d for d in provider.expiries if d < event[1]]
            after = [d for d in provider.expiries if d >= event[1]]
            if before and after:
                st.markdown(f"Next {event[0].lower()} in about {event[1]:.0f} days. The last "
                            f"expiry before it: **{metrics(p, before[-1])['atm'] * 100:.0f}%**. "
                            f"The first after it: **{metrics(p, after[0])['atm'] * 100:.0f}%**.")
            st.caption(f"Expected move on the day: about ±{p.event_move * 100:.1f}%. "
                       "Event dates are estimates.")

    if pro:
        st.write("")
        show_chart(charts.fig_surface3d(p, provider.expiries), "surf_3d")
        grid = pd.DataFrame({d: [metrics(p, d)[k] * 100 for k in ["put10", "put25", "atm", "call25", "call10"]]
                             for d in provider.expiries},
                            index=["10Δ put", "25Δ put", "ATM", "25Δ call", "10Δ call"])
        grid.columns = [f"{d}d" for d in grid.columns]
        st.markdown("**IV grid (%)**")
        st.dataframe(grid.style.format("{:.1f}"))
        with st.expander("Model parameters"):
            st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.__dict__.items()})

# ============================================================================
# 3. vs. past year
# ============================================================================
with tabs[2]:
    ui.intro("Is today normal? Each measure is ranked against this security's own past year.")
    lb = 252
    if pro:
        lb_map = {"3 months": 63, "6 months": 126, "1 year": 252}
        lb = lb_map[st.segmented_control("Compare against", list(lb_map), default="1 year",
                                         key="hist_lb", required=True)]
    hsnap = snapshot(hist, lb)

    odd = unusual(hsnap)
    if odd:
        st.markdown("**What stands out today**")
        for k in odd[:3]:
            st.info(f"**{label(k)}**: {ui.pct_sentence(hsnap[k]['pct'])} {reading(k, hsnap)}",
                    icon=":material/flag:")
    else:
        st.success("Everything is in its usual range today.", icon=":material/check_circle:")

    def measure_row(k):
        s = hsnap[k]
        with st.container(border=True):
            a, b = st.columns([2, 3], gap="large", vertical_alignment="center")
            with a:
                st.markdown(f"**{label(k)}**", help=METRICS[k].help)
                st.markdown(f"Today **{fmt_metric(k, s['value'])}** · usually {fmt_metric(k, s['median'])}")
            with b:
                ui.html_block(ui.pbar(s["pct"]))
            line = reading(k, hsnap) if not pro else (
                f"{s['pct']:.0f}th percentile · rank {s['rank']:.0f} · z {s['z']:+.1f} · "
                f"range {fmt_metric(k, s['min'])} to {fmt_metric(k, s['max'])}")
            st.caption(line)

    main_keys = ["iv30", "rr25", "vrp"]
    more_keys = ["fly25", "term", "rv20"]
    for k in main_keys:
        measure_row(k)
    if pro:
        for k in more_keys:
            measure_row(k)
    else:
        with st.expander("Three more measures", icon=":material/add:"):
            for k in more_keys:
                measure_row(k)

    st.write("")
    st.markdown("**Look at the history**")
    k_sel = st.selectbox("Measure", HEADLINE_KEYS, format_func=label, index=1, key="hist_metric",
                         label_visibility="collapsed")
    extra = ("Actual movement (20-day)", hist["rv20"].iloc[-lb:]) if k_sel == "iv30" else None
    show_chart(charts.fig_history(hist[k_sel].iloc[-lb:], label(k_sel),
                                  signed=k_sel in ("rr25", "fly25", "term", "vrp"), extra=extra),
               "hist_chart")
    if k_sel == "iv30":
        tip("The grey line is how much the price actually moved. When the dark line (what "
            "options price) sits well above it, option buyers are paying for movement that "
            "hasn't shown up.")

# ============================================================================
# 4. What if?
# ============================================================================
SC_DEFAULTS = {"sc_spot": 0, "sc_vol": 0, "sc_front": 0, "sc_skew": 0.0, "sc_days": 0,
               "sc_sticky": "Sticky strike"}
for k, v in SC_DEFAULTS.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "Sharp selloff": ("The price drops 10% over two days. Volatility jumps, most of all in "
                      "near-term options, and protection gets pricier.",
                      dict(sc_spot=-10, sc_vol=6, sc_front=4, sc_skew=3.0, sc_days=2)),
    "Slow grind higher": ("The price rises 5% over a month as worry fades.",
                          dict(sc_spot=5, sc_vol=-2, sc_front=-1, sc_skew=-1.0, sc_days=30)),
    "Event passes": ("It's the day after the next scheduled event. The price is unchanged, "
                     "but the extra volatility priced for the event drains out (the vol crush).", None),
    "Melt-up": ("The price jumps 12% in a week on heavy call buying. Volatility rises and "
                "protection gets cheaper relative to calls.",
                dict(sc_spot=12, sc_vol=4, sc_front=2, sc_skew=-3.0, sc_days=5)),
    "Quiet two weeks": ("Two weeks pass with no real move, and volatility eases.",
                        dict(sc_spot=0, sc_vol=-3, sc_front=-2, sc_skew=-0.5, sc_days=14)),
}


def on_story():
    name = st.session_state.get("sc_story")
    if not name:
        st.session_state.update({k: v for k, v in SC_DEFAULTS.items() if k != "sc_sticky"})
        return
    vals = PRESETS[name][1]
    if vals is None:
        d = int(np.ceil(p.event_days)) + 1 if p.event_days is not None else 1
        vals = dict(sc_spot=0, sc_vol=0, sc_front=0, sc_skew=0.0, sc_days=d)
    st.session_state.update(vals)


def on_slider():
    st.session_state["sc_story"] = None


with tabs[3]:
    ui.intro("Pick a story to see how positions would react, and why.")
    staged = st.session_state.get("staged")
    sources = ["My positions"]
    if staged and staged["ticker"] == ticker:
        sources.append(f"Idea: {staged['title']}")
    if st.session_state.get("sc_src") not in sources:
        st.session_state["sc_src"] = sources[0]
    if len(sources) > 1:
        st.radio("Test", sources, horizontal=True, key="sc_src")
    src = st.session_state["sc_src"]
    legs = positions if src == "My positions" else staged["legs"]

    st.segmented_control("Story", list(PRESETS), key="sc_story", on_change=on_story,
                         label_visibility="collapsed")
    story = st.session_state.get("sc_story")
    if story:
        st.caption(PRESETS[story][0] + (" (No event is scheduled for this security.)"
                                        if story == "Event passes" and not event else ""))

    with st.expander("Adjust the story yourself", icon=":material/tune:", expanded=pro):
        s1, s2 = st.columns(2, gap="large")
        with s1:
            st.slider("Price move", -30, 30, step=1, key="sc_spot", format="%+d%%", on_change=on_slider)
            st.slider("Days that pass", 0, 60, step=1, key="sc_days", on_change=on_slider,
                      help="Options that expire in the meantime settle at their intrinsic value.")
            st.radio("When the price moves, each strike…", ["Sticky strike", "Sticky delta"],
                     key="sc_sticky", horizontal=True,
                     format_func=lambda x: {"Sticky strike": "keeps its own vol",
                                            "Sticky delta": "slides with the price"}[x],
                     help="Keeps its own vol (sticky strike): a falling price moves you onto "
                          "pricier put strikes. Slides with the price (sticky delta): the whole "
                          "smile moves along with the price.")
        with s2:
            st.slider("Volatility change (points)", -15, 25, step=1, key="sc_vol", format="%+d",
                      on_change=on_slider, help="Every expiry's implied volatility moves by this much.")
            st.slider("Change in protection cost (points)", -8.0, 10.0, step=0.5, key="sc_skew",
                      format="%+.1f", on_change=on_slider,
                      help="Positive: puts get pricier relative to calls, as in most selloffs.")
            if pro:
                st.slider("Extra near-term vol change (points)", -10, 15, step=1, key="sc_front",
                          on_change=on_slider, help="Added on top, fading out for longer expiries.")

    shock = Shock(st.session_state.sc_spot / 100, st.session_state.sc_vol, st.session_state.sc_front,
                  st.session_state.sc_skew, st.session_state.sc_days,
                  "strike" if st.session_state.sc_sticky == "Sticky strike" else "delta")
    attr = attribution(legs, p, shock)
    shocked = attr["shocked"]
    total = attr["Total"]
    drivers = {k: attr[k] for k in ["Time passing", "Price move", "Volatility", "Protection cost"]}
    top = max(drivers, key=lambda k: abs(drivers[k]))
    is_blank = shock == Shock(sticky=shock.sticky)

    if is_blank:
        st.info("Choose a story above to see what would happen to these positions.",
                icon=":material/touch_app:")
    else:
        r1, r2 = st.columns([1, 2], gap="large", vertical_alignment="center")
        with r1:
            tone = "gain" if total >= 0 else "loss"
            names = {"Time passing": "time passing", "Price move": "the price move",
                     "Volatility": "the change in volatility", "Protection cost": "the change in protection cost"}
            sentence = f"The biggest piece comes from {names[top]} ({money(drivers[top], True)})."
            if event and shock.days > event[1]:
                sentence += (f" Time passing includes the {event[0].lower()}, so the extra "
                             "event volatility drains out.")
            ui.html_block(ui.card("Result", ui.chip("Gain" if total >= 0 else "Loss", tone),
                                  money(total, True), sentence))
        with r2:
            show_chart(charts.fig_waterfall(attr, height=300), "sc_wf")

        pl = per_leg_pnl(legs, p, shock)
        show_chart(charts.fig_bars([leg_label(l) for l in legs], pl, title="Result by position",
                                   height=260), "sc_legs")

        opt_dtes = [l["dte"] for l in legs if l["type"] != "stock"]
        view_days = min(opt_dtes) if opt_dtes else 30
        smile_fig = charts.fig_smile(shocked, max(view_days - shock.days, 1), "pct",
                                     overlays=[("Before", p, "dash", view_days)], show_move=False,
                                     title=f"Smile before and after ({view_days}-day expiry)")
        if pro:
            show_chart(smile_fig, "sc_smile")
            st.markdown("**Result grid: price move × volatility change** (other settings held)")
            spot_moves = np.array([-0.2, -0.15, -0.1, -0.07, -0.05, -0.03, 0, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2])
            vol_moves = np.array([-10, -6, -3, 0, 3, 6, 10, 15])
            show_chart(charts.fig_heatmap(pnl_grid(legs, p, shock, spot_moves, vol_moves),
                                          spot_moves, vol_moves), "sc_heat")
        else:
            with st.expander("See how the smile changes", icon=":material/ssid_chart:"):
                show_chart(smile_fig, "sc_smile")

# ============================================================================
# 5. Trade ideas
# ============================================================================


def stage(card):
    st.session_state["staged"] = {"ticker": ticker, "title": card["title"], "legs": card["legs"]}
    st.session_state["sc_src"] = f"Idea: {card['title']}"
    st.toast(f"Ready on the What if? tab: {card['title']}", icon="🧪")


with tabs[4]:
    ui.intro("Example option setups that match what's unusual today. For learning, not recommendations.")
    cards = build_cards(p, snap, positions, event)
    show_adv = pro or st.toggle("Include advanced ideas", key="ideas_adv",
                                help="Advanced ideas can involve large or open-ended losses.")
    shown = [c for c in cards if show_adv or c["level"] != "Advanced"]
    hidden = len(cards) - len(shown)
    for i, card in enumerate(shown):
        with st.container(border=True):
            tone = {"Beginner-friendly": "normal", "Intermediate": "low", "Advanced": "high"}.get(card["level"], "neutral")
            chips = ui.chip(card["condition"], "neutral")
            if card["level"]:
                chips += " " + ui.chip(card["level"], tone) + " " + ui.chip(
                    card["risk"], "gain" if card["risk"] == "Defined risk" else "loss")
            ui.html_block(chips)
            st.markdown(f"### {card['title']}")
            st.markdown(f"**Why now:** {card['why']}")
            st.markdown(f"**The setup:** {card['what']}")
            if not card["legs"]:
                continue
            left, right = st.columns([3, 2], gap="large")
            spots, at_h, today_c, hzn = payoff_curves(card["legs"], p)
            with left:
                show_chart(charts.fig_payoff(spots, at_h, today_c, p.spot, hzn, height=280),
                           f"idea_pay_{i}")
            with right:
                stt = structure_stats(card["legs"], p)
                st.markdown("**Positions**  \n" + "  \n".join(leg_label(l) for l in card["legs"]))
                prem = stt["premium"]
                x1, x2 = st.columns(2)
                x1.metric("You pay" if prem >= 0 else "You collect", money(abs(prem)))
                x2.metric("Most you can lose", "Unlimited" if stt["max_loss"] is None
                          else money(abs(stt["max_loss"])))
                x1.metric("Most you can make", "Unlimited" if stt["max_gain"] is None
                          else money(stt["max_gain"]))
                bes = stt["breakevens"]
                bfmt = "{:,.0f}" if p.spot >= 1000 else "{:,.2f}"
                x2.markdown("**Breakeven" + ("s" if len(bes) > 1 else "") + "**  \n" +
                            ("  \n".join(bfmt.format(b) for b in bes) or "n/a"))
                if pro:
                    tt = totals(card["legs"], p)
                    st.caption(f"Δ {tt['delta']:+,.0f} sh · vega {money(tt['vega'], True)}/pt · "
                               f"θ {money(tt['theta'], True)}/day · skew {money(tt['skew'], True)}/pt")
                st.caption(f"Gain and loss figures are at the first expiry ({stt['horizon']} days).")
            n1, n2 = st.columns(2, gap="large")
            n1.success(f"**What needs to happen:** {card['needs']}", icon=":material/check:")
            n2.warning(f"**What can go wrong:** {card['risks']}", icon=":material/warning:")
            st.button("Test it on the What if? tab", key=f"stage_{i}", on_click=stage, args=(card,),
                      icon=":material/science:")
    if hidden:
        plural = hidden > 1
        st.caption(f"{hidden} advanced idea{'s' if plural else ''} hidden. Switch on "
                   f"“Include advanced ideas” to see {'them' if plural else 'it'}.")

# ============================================================================
# 6. My positions
# ============================================================================
with tabs[5]:
    ui.intro(f"What your {ticker} positions are exposed to, in dollars.")
    tt = totals(positions, p)
    has_opts = any(l["type"] != "stock" for l in positions)

    def exp_card(title, v, sentence):
        return ui.card(title, ui.chip("Gain" if v >= 0 else "Loss", "gain" if v >= 0 else "loss"),
                       money(v, True), sentence)

    cards_html = [
        exp_card(f"If {ticker} rises 1%", tt["dollar_delta_1pct"],
                 "From the price move alone. A 1% drop does roughly the opposite."),
        exp_card("If volatility rises 1 point", tt["vega"],
                 ("You benefit when options get pricier (long volatility)." if tt["vega"] > 1 else
                  "You lose when options get pricier (short volatility)." if tt["vega"] < -1 else
                  "Little exposure to the overall level of volatility.")),
        exp_card("Each day that passes", tt["theta"],
                 "With nothing else changing. Option sellers usually earn this; buyers pay it."),
        exp_card("If protection gets 1 point pricier", tt["skew"],
                 ("You're exposed to downside options getting pricier, which tends to happen in "
                  "selloffs (short the downside wing)." if tt["skew"] < -1 else
                  "You benefit when downside protection gets pricier." if tt["skew"] > 1 else
                  "Little exposure to the price of protection.")),
    ]
    cols = st.columns(4)
    for col, h in zip(cols, cards_html):
        with col:
            ui.html_block(h)
    if not has_opts:
        st.write("")
        st.info("These positions have no options, so only the price move matters. The Trade "
                "ideas tab shows ways to add protection.", icon=":material/info:")

    st.write("")
    lt = leg_table(positions, p)
    show = pd.DataFrame({"Position": [leg_label(l) for l in lt["leg"]], "Value": lt["value"]})
    fmt = {"Value": money}
    if pro:
        show["IV"] = lt["iv"] * 100
        show["Price"] = lt["price"]
        show["Delta (sh)"] = lt["delta"]
        show["Vega ($/pt)"] = lt["vega"]
        show["Theta ($/day)"] = lt["theta"]
        fmt.update({"IV": "{:.1f}%", "Price": "{:,.2f}", "Delta (sh)": "{:+,.0f}",
                    "Vega ($/pt)": lambda v: money(v, True), "Theta ($/day)": lambda v: money(v, True)})
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Positions**")
        st.dataframe(show.style.format(fmt, na_rep="—"), hide_index=True)
        st.caption("Demo positions. Loading your own is planned for a future release.")
    with right:
        vb = vega_by_expiry(positions, p)
        if not vb.empty:
            show_chart(charts.fig_bars([f"{int(d)} days" for d in vb["dte"]], list(vb["vega"]),
                                       title="Volatility exposure by expiry ($ per point)", height=260),
                       "pos_vega")

    with st.expander("All securities", icon=":material/table_view:", expanded=pro):
        rows = []
        for t in TICKERS:
            pt = provider.current_params(t)
            lg = provider.positions(t)
            x = totals(lg, pt)
            rows.append({"Security": t, "Value": x["value"], "If +1% move": x["dollar_delta_1pct"],
                         "If vol +1 pt": x["vega"], "Per day": x["theta"],
                         "If protection +1 pt": x["skew"]})
        port = pd.DataFrame(rows).set_index("Security")
        port.loc["Total"] = port.sum()
        f = {c: (lambda v: money(v, True)) for c in port.columns if c != "Value"}
        f["Value"] = money
        st.dataframe(port.style.format(f))

# ============================================================================
# 7. Compare
# ============================================================================
with tabs[6]:
    ui.intro("How each security compares with its own past year, side by side.")
    sel = st.multiselect("Securities", TICKERS, default=TICKERS, key="cmp_sel")
    if not sel:
        st.info("Pick at least one security.", icon=":material/info:")
    else:
        rows, quad = [], []
        for t in sel:
            sn = snapshot(provider.history(t))
            quad.append((t, sn["iv30"]["pct"], sn["rr25"]["pct"]))
            rows.append({"Security": t,
                         "Option prices": ui.pct_sentence(sn["iv30"]["pct"]) if not pro else sn["iv30"]["pct"],
                         "Protection cost": ui.pct_sentence(sn["rr25"]["pct"]) if not pro else sn["rr25"]["pct"],
                         "IV": sn["iv30"]["value"] * 100, "Skew (pts)": sn["rr25"]["value"] * 100,
                         "Wings (pts)": sn["fly25"]["value"] * 100, "IV − RV (pts)": sn["vrp"]["value"] * 100})
        q1, q2 = st.columns([3, 2], gap="large")
        with q1:
            show_chart(charts.fig_quadrant(quad), "cmp_quad")
        with q2:
            st.markdown("**How to read this map**")
            st.markdown("Each dot compares a security with **its own** past year, not with the "
                        "others.\n\n- **Right:** options cost more than usual.\n- **Top:** "
                        "downside protection costs more than usual.\n- **Corners** are where "
                        "the most unusual pricing is.")
        df = pd.DataFrame(rows).set_index("Security")
        if not pro:
            df = df[["Option prices", "Protection cost", "IV"]]
            st.dataframe(df.style.format({"IV": "{:.0f}%"}), column_config={
                "IV": st.column_config.Column("Implied vol")})
        else:
            df.columns = ["IV pct", "Skew pct", "IV", "Skew (pts)", "Wings (pts)", "IV − RV (pts)"]
            st.dataframe(df.style.format("{:.1f}").format("{:.0f}", subset=["IV pct", "Skew pct"]))

        with st.expander("Compare smile shapes", icon=":material/ssid_chart:", expanded=pro):
            cdays = st.select_slider("Expiry", provider.expiries, value=30, key="cmp_days",
                                     format_func=lambda d: f"{d} days")
            items = [(t, provider.current_params(t)) for t in sel]
            show_chart(charts.fig_compare_smiles(items, cdays, True), "cmp_smiles")
            st.caption("Each line is divided by its own at-the-money vol, so a 16%-vol index and an "
                       "80%-vol stock share one scale. Rising to the left = puts pricier; rising "
                       "to the right = calls pricier; a deep U = rich wings.")

st.divider()
st.caption("Vol & Skew Lab is an educational prototype. The securities are real, but the market "
           "data shown (implied volatilities, history, event dates, positions) is simulated and "
           "does not reflect actual quotes. Nothing here is investment advice or a recommendation "
           "to buy or sell any security or strategy.")
