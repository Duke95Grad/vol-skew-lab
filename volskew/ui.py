"""Small presentation helpers: colors, status chips, percentile bars, cards.

Color meaning is consistent across the whole app:
  * Readings vs. history use blue (lower than usual), green (typical) and
    amber (higher than usual). High vol isn't "bad", so no red here.
  * Red and green are reserved for profit and loss.
  * Rose = puts / downside, teal = calls / upside, navy = today.
"""
from __future__ import annotations

import html

import streamlit as st

TONES = {
    "low": ("#EAF2FB", "#1D4E89"),
    "normal": ("#EAF6EF", "#22603D"),
    "high": ("#FDF3E3", "#8A5300"),
    "neutral": ("#F2F4F7", "#475467"),
    "event": ("#F1ECFB", "#5B3E9E"),
    "gain": ("#EAF6EF", "#22603D"),
    "loss": ("#FCECEA", "#8E2A20"),
}

_CSS = """
<style>
.vs-card {border:1px solid #E4E7EC; border-radius:12px; padding:16px 18px; background:#FFFFFF;
          min-height:150px; box-sizing:border-box;}
.vs-card .vs-k {font-size:0.78rem; letter-spacing:.04em; text-transform:uppercase; color:#667085;
                font-weight:600; margin-bottom:8px;}
.vs-card .vs-v {font-size:1.55rem; font-weight:650; color:#1F2937; margin:6px 0 4px 0; line-height:1.2;}
.vs-card .vs-s {font-size:0.92rem; color:#475467; line-height:1.45;}
.vs-chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
          font-weight:600; white-space:nowrap;}
.vs-pbar {margin:6px 0 2px 0;}
.vs-track {position:relative; height:10px; border-radius:6px; display:flex; overflow:visible;}
.vs-z {height:10px;}
.vs-z.l {background:#DCE8F6; width:25%; border-radius:6px 0 0 6px;}
.vs-z.m {background:#EDF0F3; width:50%;}
.vs-z.h {background:#F8E6C9; width:25%; border-radius:0 6px 6px 0;}
.vs-dot {position:absolute; top:50%; width:18px; height:18px; border-radius:50%; background:#24324F;
         border:3px solid #FFFFFF; box-shadow:0 0 0 1px #24324F33; transform:translate(-50%,-50%);}
.vs-ticks {display:flex; justify-content:space-between; font-size:0.75rem; color:#98A2B3; margin-top:6px;}
.vs-intro {font-size:1rem; color:#475467; margin:0.2rem 0 0.8rem 0;}
.vs-step {display:flex; gap:12px; align-items:flex-start; margin-bottom:10px;}
.vs-num {flex:0 0 28px; height:28px; border-radius:50%; background:#2F4B7C; color:#fff;
         font-weight:700; display:flex; align-items:center; justify-content:center; font-size:0.9rem;}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def chip(text: str, tone: str = "neutral") -> str:
    bg, fg = TONES.get(tone, TONES["neutral"])
    return f'<span class="vs-chip" style="background:{bg};color:{fg}">{html.escape(text)}</span>'


def tone_for(pct: float) -> str:
    if pct >= 75:
        return "high"
    if pct <= 25:
        return "low"
    return "normal"


def pct_sentence(pct: float) -> str:
    """'Higher than on 85% of days in the past year.' with clean extremes."""
    if pct >= 99.5:
        return "The highest level of the past year."
    if pct <= 0.5:
        return "The lowest level of the past year."
    if pct >= 50:
        return f"Higher than on {pct:.0f}% of days in the past year."
    return f"Lower than on {100 - pct:.0f}% of days in the past year."


def pbar(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="vs-pbar"><div class="vs-track"><span class="vs-z l"></span>'
            f'<span class="vs-z m"></span><span class="vs-z h"></span>'
            f'<span class="vs-dot" style="left:{pct:.1f}%"></span></div>'
            f'<div class="vs-ticks"><span>Lower than usual</span><span>Typical</span>'
            f'<span>Higher than usual</span></div></div>')


def card(key: str, chip_html: str, value: str, sentence: str, extra: str = "") -> str:
    return (f'<div class="vs-card"><div class="vs-k">{html.escape(key)}</div>{chip_html}'
            f'<div class="vs-v">{html.escape(value)}</div>'
            f'<div class="vs-s">{html.escape(sentence)}</div>{extra}</div>')


def html_block(s: str):
    st.markdown(s, unsafe_allow_html=True)


def intro(text: str):
    """One-line answer to 'what does this tab tell me?'."""
    st.markdown(f'<div class="vs-intro">{text}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> str:
    out = []
    for i, (title, body) in enumerate(items, 1):
        out.append(f'<div class="vs-step"><div class="vs-num">{i}</div><div><b>{html.escape(title)}</b>'
                   f'<br><span style="color:#475467">{html.escape(body)}</span></div></div>')
    return "".join(out)
