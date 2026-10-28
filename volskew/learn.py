"""Guided lesson and quiz. Every step uses the currently selected security."""
from __future__ import annotations

import numpy as np
import streamlit as st

from . import charts, ui
from .surface import metrics

STEPS = [
    "What implied volatility means",
    "Each strike has its own price: the smile",
    "Why protection usually costs more: skew",
    "Each expiry has its own price: term structure",
    "Is today normal? Comparing with the past year",
    "What could change: scenarios",
]

QUIZ = [
    ("A stock is at $100 and its 30-day implied volatility is 32%. About how big a move "
     "are options pricing for the next month?",
     ["About $3", "About $9", "About $32", "About $16"], 1,
     "32% ÷ 3.5 ≈ 9%, so about ±$9. Dividing yearly vol by about 3.5 gives a one-month move."),
    ("The 25-delta put has an implied vol of 28% and the 25-delta call 22%. What does that tell you?",
     ["Calls cost more than puts", "The market charges extra for downside protection",
      "The stock is about to fall", "Options are cheap"], 1,
     "When puts carry a higher vol than calls, protection costs extra. It's a price, not a forecast."),
    ("A company reports earnings in 5 days. Which expiry usually has the highest implied vol?",
     ["The 3-day expiry", "The 7-day expiry", "The 6-month expiry", "They're all the same"], 1,
     "The 7-day expiry is the shortest one that includes the report, so the expected jump is "
     "packed into the fewest days."),
    ("Skew is at the 95th percentile of its past year. What's the most accurate reading?",
     ["A crash is coming", "Protection is unusually expensive compared with this stock's own past",
      "You should buy puts", "Implied vol is high"], 1,
     "Percentiles compare today with the stock's own history. Pricey protection doesn't mean a "
     "drop is coming, and the overall vol level is a separate measure."),
    ("Implied vol is 40%, but the stock has only been moving at 22%. What does that gap mean?",
     ["Options are pricing much more movement than has been happening",
      "The stock will definitely get more volatile", "Options are underpriced",
      "Nothing; the two aren't related"], 0,
     "That gap is what option sellers get paid for taking risk. It can disappear fast if the "
     "stock starts moving."),
]


def _nav():
    i = st.session_state.get("learn_step", 0)
    c1, c2, c3 = st.columns([1, 4, 1], vertical_alignment="center")
    with c1:
        if st.button("Back", disabled=i == 0, key="learn_back", width="stretch",
                     icon=":material/arrow_back:"):
            st.session_state.learn_step = i - 1
            st.rerun()
    with c2:
        st.progress((i + 1) / len(STEPS), text=f"Step {i + 1} of {len(STEPS)} · {STEPS[i]}")
    with c3:
        if st.button("Next", disabled=i == len(STEPS) - 1, key="learn_next", type="primary",
                     width="stretch", icon=":material/arrow_forward:", icon_position="right"):
            st.session_state.learn_step = i + 1
            st.rerun()
    return i


def _two(chart_fn, text: str, tip: str | None = None, tip_icon: str = "💡"):
    left, right = st.columns([3, 2], gap="large")
    with left:
        chart_fn()
    with right:
        st.markdown(text)
        if tip:
            st.info(tip, icon=tip_icon)


def render(ticker, info, p, typical, hist, snap, provider, show_chart):
    with st.container(border=True):
        st.markdown("**How to use this app**")
        ui.html_block(ui.steps([
            ("Pick a security in the sidebar.",
             "Every screen updates for it, including this lesson."),
            ("Read the three cards at the top.",
             "They sum up what options are saying today, in plain English."),
            ("Explore the tabs from left to right.",
             "Surface → compare with the past year → test scenarios → see ideas."),
        ]))
    st.subheader("Learn the basics in six short steps")
    i = _nav()
    m30 = metrics(p, 30)
    a30 = m30["atm"]

    if i == 0:
        move = p.spot * a30 * np.sqrt(30 / 365)
        _two(lambda: show_chart(charts.fig_distribution(p, 30), "learn_dist"),
             f"An option's price depends on how much the underlying might move before the "
             f"option expires. **Implied volatility** (IV) is the size of move that option "
             f"prices imply, stated as a yearly percentage.\n\n"
             f"{ticker}'s 30-day IV is **{a30 * 100:.0f}%**. Over one month, that works out to "
             f"a typical move of about **±{move:,.2f}** (±{move / p.spot * 100:.1f}%).\n\n"
             f"Put simply: options are priced as if there's roughly a two-in-three chance "
             f"{ticker} ends the month between **{p.spot - move:,.2f}** and "
             f"**{p.spot + move:,.2f}**.",
             "Quick math: divide yearly IV by about 3.5 for a one-month move, or by 16 for a one-day move.")

    elif i == 1:
        _two(lambda: show_chart(charts.fig_smile(p, 30, "pct"), "learn_smile"),
             "If every strike had the same implied volatility, this line would be flat. It "
             "isn't. Each strike has its own IV, and the curve connecting them is called the "
             "**smile**.\n\n"
             f"For {ticker}, 30 days out: a downside put trades at **{m30['put25'] * 100:.1f}%**, "
             f"the at-the-money option at **{a30 * 100:.1f}%**, and an upside call at "
             f"**{m30['call25'] * 100:.1f}%**.",
             "We label strikes by **delta**. A 25-delta put has roughly a 1-in-4 chance of "
             "finishing in the money. Using delta lets you compare a $25 stock with a "
             "7,800-point index on the same scale.")

    elif i == 2:
        others = [(t, provider.current_params(t)) for t in ["SPX", "GME"]]
        if ticker not in ("SPX", "GME"):
            others.append((ticker, p))
        _two(lambda: show_chart(charts.fig_compare_smiles(others, 30), "learn_skew"),
             "**Skew** is the tilt of the smile. For most stocks and indexes the downside "
             "(puts) costs more, for three reasons:\n\n"
             "- Investors who own stock buy puts for protection, so demand is steady.\n"
             "- Markets tend to fall faster than they rise, and volatility jumps when they fall.\n"
             "- Since the 1987 crash, option sellers have charged extra for crash insurance.\n\n"
             "Some names tilt the other way. When traders chase upside, calls can cost more "
             "than puts (**call skew**), as GME often shows.",
             f"Skew here = the 25-delta put's IV minus the 25-delta call's IV. "
             f"{ticker} is at {m30['rr25'] * 100:+.1f} points today.", "📏")

    elif i == 3:
        txt = ("Different expiries carry different implied volatilities too. Usually "
               "longer-dated options carry a bit more, because more can happen over more time.\n\n")
        if p.event_days is not None and p.event_days < 120:
            txt += (f"{ticker}'s next {info['event_label'].lower()} is expected in about "
                    f"{p.event_days:.0f} days (dotted line). The first expiry after it carries "
                    "extra volatility, because the whole expected jump is packed into a short "
                    "window. Once the event passes, that extra drops out overnight. Traders call "
                    "this the **vol crush**.")
        else:
            txt += ("When near-term options cost more than longer-dated ones, the market is "
                    "either stressed right now or expecting a near-term event.")
        _two(lambda: show_chart(charts.fig_term(p, provider.expiries, info["event_label"]),
                                "learn_term"), txt)

    elif i == 4:
        rr = snap["rr25"]
        _two(lambda: show_chart(charts.fig_history(hist["rr25"], "Cost of downside protection", True),
                                "learn_hist"),
             "A number on its own says little. Is 20% implied volatility high? For a utility, "
             "yes. For a biotech, it's quiet. So the app compares each security with **its "
             "own past year**.\n\n"
             f"{ticker}'s cost of protection today: {ui.pct_sentence(rr['pct'])}\n\n"
             "An unusual reading doesn't predict direction. It tells you something is priced "
             "differently than normal, which is where the ideas on the **Trade ideas** tab come from.")

    elif i == 5:
        from .scenario import Shock, apply_shock
        shocked = apply_shock(p, Shock(-0.10, 6, 4, 3, 2))
        _two(lambda: show_chart(charts.fig_smile(shocked, 30, "pct",
                                                 overlays=[("Before", p, "dash")], show_move=False),
                                "learn_scen"),
             "The whole surface moves over time. In a sharp selloff, implied volatility usually "
             "rises, near-term options rise most, and protection gets pricier. The chart shows a "
             "10% drop with those changes (dashed line = today).\n\n"
             "The **What if?** tab lets you apply stories like this to positions and see how "
             "much of the result came from the price move, from volatility, from skew and from "
             "time passing.",
             "You've finished the lesson. Try the quiz below, then explore the other tabs.", "✅")

    st.write("")
    with st.expander("Check what you learned: 5-question quiz", icon=":material/quiz:",
                     expanded=(i == len(STEPS) - 1)):
        score = answered = 0
        for qi, (q, opts, ans, why) in enumerate(QUIZ):
            choice = st.radio(f"**{qi + 1}.** {q}", opts, index=None, key=f"quiz_{qi}")
            if choice is not None:
                answered += 1
                if opts.index(choice) == ans:
                    score += 1
                    st.success(f"Correct. {why}")
                else:
                    st.error(f"Not quite. {why}")
        if answered == len(QUIZ):
            st.markdown(f"**Score: {score} of {len(QUIZ)}**")
