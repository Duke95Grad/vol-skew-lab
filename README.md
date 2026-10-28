# Vol & Skew Lab

A friendly volatility and volatility-skew analysis app built with Streamlit. It explains
what an options surface is saying, compares today with the past year, stress-tests
positions, and connects unusual readings to example option structures.

**Market data is simulated in this release.** The five securities are real; spot prices
approximate late-October 2026 levels and the one-year price paths roughly match the
actual trailing-year move. Implied vols, skew, history, event dates and positions are
simulated so each one tells a different story:

| Ticker | What it is | Today's story |
|---|---|---|
| SPX | S&P 500 Index | Put skew at a 1-year high: protection is expensive |
| NVDA | NVIDIA | Earnings in ~3 weeks lifting the front month |
| PG | Procter & Gamble | Vol and skew near 1-year lows: protection is cheap |
| GME | GameStop | High vol and call skew (calls priced over puts) |
| TSLA | Tesla | Elevated vol with very rich wings on both sides |

## What's inside

The top of every screen shows three plain-English cards: option prices, the cost of downside
protection, and what's coming up. In the advanced view they're replaced by five numbers
(ATM IV, IV percentile, 25Δ skew, IV − RV, next event). Below them are seven tabs:

- **Start here**: how to use the app, a six-step lesson that uses the selected security, and a quiz.
- **Surface**: the smile (implied vol by strike) and term structure (by expiry), with a
  typical-day overlay and a "How to read this chart" guide.
- **vs. past year**: where each measure ranks against its own past year, on a simple
  lower / typical / higher bar, plus what stands out today.
- **What if?**: pick a story (selloff, event passes, melt-up…) and see the result split into
  time, price, volatility and protection cost. Fine-tune it with sliders.
- **Trade ideas**: example setups triggered by unusual readings, with what you pay, the most
  you can make or lose, and what needs to happen. Advanced ideas are hidden by default.
- **My positions**: exposure in dollars ("if the price rises 1%…", "each day that passes…").
- **Compare**: all securities on one map of option prices vs. protection cost.

The **Show advanced details** switch in the sidebar adds trader terminology, the 3D
surface, an IV grid, P&L heat maps, greeks and percentile statistics.

Colors are used consistently: blue / green / amber mean lower than usual / typical / higher
than usual; red and green are only used for losses and gains; rose is the downside (puts),
teal the upside (calls).

## Run it on your computer

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Your browser opens at http://localhost:8501.

## Put it on Streamlit Community Cloud

Same process as the OCC PM app and the Options Risk Dashboard.

1. **Create the repository.** On github.com click **New**, name it (for example
   `vol-skew-lab`), choose Public or Private, and click **Create repository**.
2. **Upload the files.** On the new repo's page click **uploading an existing file**.
   Drag in everything from this folder: `app.py`, `requirements.txt`, `README.md`,
   `.gitignore`, the whole `volskew` folder and the `.streamlit` folder. Click **Commit
   changes**.
   - Hidden folders: if your file browser hides `.streamlit`, that's fine to skip; the
     app runs without it (you just lose the colour theme). On a Mac press
     Cmd+Shift+. in Finder to show hidden files.
   - Check that the repo shows a `volskew` folder containing ten `.py` files.
3. **Deploy.** Go to share.streamlit.io, click **Create app**, then **Deploy a public
   app from GitHub**. Pick the repository, branch `main`, and main file `app.py`.
   Click **Deploy**. The first build takes two or three minutes.
4. **Updating later.** Edit or re-upload a file on GitHub and commit; the app picks up
   the change automatically within a minute or so.

## Code layout

```
app.py                 Streamlit screens (all seven tabs)
volskew/data.py        DataProvider: demo securities, history, positions
volskew/surface.py     Parametric vol surface, delta strikes, skew metrics
volskew/pricing.py     Black-Scholes prices and greeks (continuous dividend yield)
volskew/scenario.py    Shocks, P&L attribution, payoffs, position greeks
volskew/analytics.py   Percentiles, plain-English readings
volskew/ideas.py       Rules that turn surface readings into idea cards
volskew/charts.py      Plotly figures
volskew/learn.py       Guided lesson and quiz
volskew/ui.py          Colors, status chips, percentile bars, cards
```

## Plugging in real data later

Every screen reads through the `DataProvider` methods in `volskew/data.py`, so a real
provider only has to return the same shapes:

- `history(ticker)`: a daily DataFrame indexed by date with the columns in
  `HISTORY_COLUMNS` (spot, surface parameters, and the derived `iv30`, `rr25`, `fly25`,
  `term`, `rv20`, `vrp`).
- `current_params(ticker)`: a `SurfaceParams` for today.
- `positions(ticker)`: a list of legs, e.g.
  `{"type": "put", "qty": -2, "strike": 5225, "dte": 45}` or
  `{"type": "stock", "qty": 300}`.

The natural path is: pull end-of-day option chains, fit the surface parameters for each
day (the model is ATM term structure + slope + curvature in standard-deviation
moneyness; an SVI fit can replace it), and compute the derived columns with the same
functions the demo uses. Rates are a flat 4% for now; a per-expiry yield curve would slot
into `SurfaceParams`.

## Model notes

- Implied vol for strike K and expiry T: `atm(T) × (1 + slope·x + curv·x²)`, where
  `x = ln(K/F) / (atm(T)·√T)`, clipped at ±3.
- ATM term structure decays from a short-end level toward a long-run level; a scheduled
  event adds its expected one-day move as a lump of variance to expiries after it.
- Skew = 25Δ put IV − 25Δ call IV (forward deltas, 30 days). Butterfly = average of the
  two minus ATM. Term = 90-day ATM − 30-day ATM.
- Scenario skew shocks are expressed in vol points of 30-day 25Δ skew and mapped to the
  slope parameter numerically.

*Educational tool. All data is simulated. Nothing here is investment advice.*
