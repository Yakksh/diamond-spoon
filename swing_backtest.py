#!/usr/bin/env python3
"""
swing_backtest.py - can a rules-based system "catch the peaks" of a stock?

It runs seven things on the same price history and compares them honestly:

  ORACLE (hindsight)  Buys every major trough and sells every major peak, found
                      with scipy.signal.find_peaks on the WHOLE series. It uses
                      future data, so nobody can trade it. It is the ceiling,
                      shown only so you can measure how much a real system loses.
  ZIGZAG swing        The real-time version of the oracle. A trough only counts
                      once price has rallied X% off the lowest close (then buy);
                      a peak only counts once price has fallen X% off the highest
                      close since buying (then sell). You always buy a bit above
                      the bottom and sell a bit below the top - that gap is the
                      price of not knowing the future.
  TREND + ATR stop    Buys strength inside an uptrend (20-EMA above 50-EMA, price
                      above the 200-day average, close at a 20-day high). Sells
                      when the close drops k x ATR below the highest close since
                      entry (a "Chandelier exit"), or the EMAs cross back down.
  MEAN REVERSION      Buys panic: close below the lower Bollinger Band (20, 2)
                      with RSI(14) under 30. Sells at the middle band or RSI > 70,
                      with a -10% stop and a 40-day time stop.
  200-DAY TREND       Holds while the close is above its 200-day average. A band
                      (default 2%) on either side cuts the in-out whipsaw.
  UPTREND DIP         Buys a sharp 2-day dip (RSI(2) under 10) while the close is
                      above the 200-day average. Sells when the close gets back
                      above the 5-day average, or after 10 days.
  BUY & HOLD          The benchmark every other line has to beat.

Realism built in
  * Signals use only data up to day t; orders fill at day t+1's open (close if
    the data has no Open column). The oracle alone fills at the exact close.
  * Transaction cost per side (default 0.15%: STT, stamp duty, exchange fees,
    DP charge and a little slippage, approximately - check your broker).
  * Indian capital-gains tax on listed equity, netted per financial year:
    STCG 20% if held <= 12 months, LTCG 12.5% above Rs 1.25 lakh/FY, + 4% cess.
    Losses carry forward (short-term losses offset any gain, long-term losses
    only long-term gains). Today's rates are applied to the whole history.
    Rates as of FY 2026-27; update the constants if a Budget changes them.
  * --cash-yield lets money earn interest while a strategy is out of the stock
    (e.g. 0.06 for a liquid fund). Tax on that interest is not modelled.
  * Yahoo serves today's bar while NSE is still trading; it is dropped until
    16:00 IST, so a live price is never mistaken for a close.
  * --walkforward tunes each strategy's main knob on the first 60% of the data
    and reports it on the unseen last 40%. If a strategy only wins in-sample,
    it was fitted to noise.

Usage
  pip install yfinance pandas numpy scipy matplotlib
  python swing_backtest.py RELIANCE.NS
  python swing_backtest.py RELIANCE.NS TCS.NS HDFCBANK.NS --years 10 --walkforward
  python swing_backtest.py RELIANCE.NS --plot reliance.png
  python swing_backtest.py RELIANCE.NS --cash-yield 0.06
  python swing_backtest.py --csv prices.csv          # Date + Close (Open/High/Low optional)

  For many stocks at once see evaluate.py; for rotating between stocks see rotation.py.

NSE tickers end in .NS on Yahoo Finance (BSE: .BO). Prices are split/bonus
adjusted (auto_adjust=True); if you use your own CSV, make sure it is adjusted
too, or a bonus issue will look like a crash.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Tax and cost constants (India, listed equity with STT paid) ---------------
STCG_RATE = 0.20          # held 12 months or less
LTCG_RATE = 0.125         # held more than 12 months
LTCG_EXEMPT = 125_000     # per financial year, per person
CESS = 0.04               # Health & Education Cess on the tax
DEFAULT_COST = 0.0015     # per side, approximate


# ==============================================================================
# Data
# ==============================================================================
def load_yahoo(ticker: str, years: int) -> pd.DataFrame:
    import yfinance as yf

    # One extra year so the 200-day average is warmed up before trading starts.
    df = yf.download(ticker, period=f"{years + 1}y", auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No data for {ticker}. Check the symbol (NSE needs .NS).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = _clean(df)
    if out.attrs.get("bad_ticks"):
        print(f"{ticker}: dropped bad bars from Yahoo on {out.attrs['bad_ticks']}",
              file=sys.stderr)
    return drop_live_bar(out)


def drop_live_bar(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop today's bar until NSE has closed (15:30 IST, plus a margin): before
    then Yahoo's 'Close' is just the latest traded price."""
    now = now if now is not None else pd.Timestamp.now(tz="Asia/Kolkata")
    if len(df) and df.index[-1].date() == now.date() and now.hour < 16:
        return df.iloc[:-1]
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(".", "") for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c)
    parsed = pd.to_datetime(df[date_col], format="ISO8601", errors="coerce")
    if parsed.isna().any():   # e.g. NSE-style 28-Oct-2026 or 28/10/2026
        parsed = pd.to_datetime(df[date_col], format="mixed", dayfirst=True)
    df[date_col] = parsed
    df = df.set_index(date_col).sort_index()
    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    df = df.rename(columns=rename)
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if "Close" not in keep:
        raise SystemExit("Data needs at least a Close column.")
    out = df[keep].apply(pd.to_numeric, errors="coerce").dropna()
    out = out[(out > 0).all(axis=1)]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return drop_bad_ticks(out)


def drop_bad_ticks(df: pd.DataFrame, factor: float = 1.8, max_len: int = 3) -> pd.DataFrame:
    """Drop runs of up to `max_len` bars that sit `factor`x or more away from the
    bars on both sides, e.g. Yahoo's NIFTYBEES on 19-20 Dec 2019 at a tenth of
    the real price. Real crashes don't snap back to within 5% in days, so they
    stay. Dropped dates are listed in df.attrs["bad_ticks"]."""
    lc = np.log(df["Close"].to_numpy())
    far, back = np.log(factor), 0.05
    bad: list[int] = []
    for i in np.flatnonzero(np.abs(np.diff(lc)) > far):   # bar i -> i+1 is a big jump
        if bad and i <= bad[-1]:
            continue
        for n in range(1, max_len + 1):                   # bars i+1..i+n off, i+n+1 back
            j = i + n + 1
            if j < len(lc) and abs(lc[j] - lc[i]) < back and \
                    all(abs(lc[k] - lc[i]) > far for k in range(i + 1, j)):
                bad.extend(range(i + 1, j))
                break
    out = df.drop(df.index[bad])
    out.attrs["bad_ticks"] = [d.date() for d in df.index[bad]]
    return out


# ==============================================================================
# Indicators (all causal: value on day t uses data up to day t only)
# ==============================================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range. Falls back to close-to-close moves if no High/Low."""
    if {"High", "Low"} <= set(df.columns):
        prev = df["Close"].shift()
        tr = pd.concat(
            [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()],
            axis=1,
        ).max(axis=1)
    else:
        tr = df["Close"].diff().abs()
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ==============================================================================
# Strategies -> target position series (1 = want to hold, 0 = want to be flat)
# ==============================================================================
def strat_zigzag(df: pd.DataFrame, pct: float = 10.0) -> pd.Series:
    c = df["Close"].to_numpy()
    tgt = np.zeros(len(c))
    pos, lo, hi = 0, c[0], c[0]
    for i, px in enumerate(c):
        if pos == 0:
            lo = min(lo, px)
            if px >= lo * (1 + pct / 100):      # trough confirmed
                pos, hi = 1, px
        else:
            hi = max(hi, px)
            if px <= hi * (1 - pct / 100):      # peak confirmed
                pos, lo = 0, px
        tgt[i] = pos
    return pd.Series(tgt, df.index)


def strat_trend(df: pd.DataFrame, k: float = 3.0, fast: int = 20, slow: int = 50,
                filt: int = 200) -> pd.Series:
    c = df["Close"]
    ef, es = ema(c, fast).to_numpy(), ema(c, slow).to_numpy()
    sma = c.rolling(filt).mean().to_numpy()
    hi20 = c.rolling(20).max().to_numpy()
    a = atr(df).to_numpy()
    cv = c.to_numpy()
    tgt = np.zeros(len(cv))
    pos, peak = 0, 0.0
    for i, px in enumerate(cv):
        if pos == 0:
            if ef[i] > es[i] and px > sma[i] and px >= hi20[i]:   # NaN compares False
                pos, peak = 1, px
        else:
            peak = max(peak, px)
            if px < peak - k * a[i] or ef[i] < es[i]:
                pos = 0
        tgt[i] = pos
    return pd.Series(tgt, df.index)


def strat_meanrev(df: pd.DataFrame, rsi_buy: float = 30, n: int = 20, nstd: float = 2.0,
                  stop: float = 0.10, max_days: int = 40) -> pd.Series:
    c = df["Close"]
    mid = c.rolling(n).mean()
    lower = (mid - nstd * c.rolling(n).std()).to_numpy()
    mid, r, cv = mid.to_numpy(), rsi(c).to_numpy(), c.to_numpy()
    tgt = np.zeros(len(cv))
    pos, entry, days = 0, 0.0, 0
    for i, px in enumerate(cv):
        if pos == 0:
            if px < lower[i] and r[i] < rsi_buy:
                pos, entry, days = 1, px, 0
        else:
            days += 1
            if px >= mid[i] or r[i] > 70 or px <= entry * (1 - stop) or days >= max_days:
                pos = 0
        tgt[i] = pos
    return pd.Series(tgt, df.index)


def strat_sma200(df: pd.DataFrame, band: float = 0.02, n: int = 200) -> pd.Series:
    c = df["Close"].to_numpy()
    sma = df["Close"].rolling(n).mean().to_numpy()
    tgt = np.zeros(len(c))
    pos = 0
    for i, px in enumerate(c):
        if pos == 0 and px > sma[i] * (1 + band):      # NaN compares False
            pos = 1
        elif pos == 1 and px < sma[i] * (1 - band):
            pos = 0
        tgt[i] = pos
    return pd.Series(tgt, df.index)


def strat_dip(df: pd.DataFrame, rsi2: float = 10, max_days: int = 10) -> pd.Series:
    c = df["Close"]
    sma200, sma5 = c.rolling(200).mean().to_numpy(), c.rolling(5).mean().to_numpy()
    r, cv = rsi(c, 2).to_numpy(), c.to_numpy()
    tgt = np.zeros(len(cv))
    pos, days = 0, 0
    for i, px in enumerate(cv):
        if pos == 0:
            if px > sma200[i] and r[i] < rsi2:
                pos, days = 1, 0
        else:
            days += 1
            if px > sma5[i] or days >= max_days:
                pos = 0
        tgt[i] = pos
    return pd.Series(tgt, df.index)


def strat_buyhold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, df.index)


def oracle_swings(close: pd.Series, prom_pct: float = 12.0) -> list[tuple[int, str]]:
    """Major troughs ('T') and peaks ('P') in hindsight, alternating, starting with T."""
    from scipy.signal import find_peaks

    v = close.to_numpy()
    prom = prom_pct / 100 * np.median(v)
    peaks, _ = find_peaks(v, prominence=prom)
    troughs, _ = find_peaks(-v, prominence=prom)
    events = sorted([(i, "T") for i in troughs] + [(i, "P") for i in peaks])
    clean: list[tuple[int, str]] = []
    for i, kind in events:
        if clean and clean[-1][1] == kind:          # two troughs in a row: keep the lower
            j = clean[-1][0]
            if (kind == "T" and v[i] < v[j]) or (kind == "P" and v[i] > v[j]):
                clean[-1] = (i, kind)
        else:
            clean.append((i, kind))
    if clean and clean[0][1] == "P":
        clean = clean[1:]
    return clean


def strat_oracle(df: pd.DataFrame, prom_pct: float = 12.0) -> pd.Series:
    tgt = np.zeros(len(df))
    sw = oracle_swings(df["Close"], prom_pct)
    for (t_i, _), nxt in zip(sw[::2], sw[1::2] + [None] * (len(sw) % 2)):
        end = nxt[0] if nxt else len(df)
        tgt[t_i:end] = 1
    return pd.Series(tgt, df.index)


# ==============================================================================
# Backtest engine
# ==============================================================================
@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_px: float
    exit_px: float
    pnl: float           # rupees, after costs
    still_open: bool = False


@dataclass
class Result:
    name: str
    equity: pd.Series
    trades: list[Trade]
    fills: list[tuple[pd.Timestamp, float, str]]
    tax: float
    capital: float

    @property
    def years(self) -> float:
        return max((self.equity.index[-1] - self.equity.index[0]).days / 365.25, 1e-9)

    def summary(self) -> dict:
        final = self.equity.iloc[-1]
        post = final - self.tax
        dd = (self.equity / self.equity.cummax() - 1).min()
        closed = [t for t in self.trades if not t.still_open]
        wins = [t for t in closed if t.pnl > 0]
        return {
            "strategy": self.name,
            "return %": 100 * (final / self.capital - 1),
            "after tax %": 100 * (post / self.capital - 1),
            "CAGR % (post-tax)": 100 * ((max(post, 1) / self.capital) ** (1 / self.years) - 1),
            "max drawdown %": 100 * dd,
            "trades": len(self.trades),
            "win rate %": 100 * len(wins) / len(closed) if closed else float("nan"),
            "time in market %": 100 * self._exposure,
        }

    _exposure: float = 0.0


def fy_of(d: pd.Timestamp) -> int:
    """Indian financial year (April-March), labelled by its starting year."""
    return d.year if d.month >= 4 else d.year - 1


def tax_by_fy(realized: list[tuple[pd.Timestamp, float, bool]]) -> dict[int, float]:
    """Capital-gains tax owed for each financial year.

    `realized` holds (sale date, gain in rupees, held > 12 months). A short-term
    loss offsets short- or long-term gains; a long-term loss only long-term gains.
    Whatever is left unused carries forward to later years (the 8-year limit on
    carry-forward is ignored)."""
    by_fy: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])   # [short, long]
    for d, gain, long_term in realized:
        by_fy[fy_of(d)][1 if long_term else 0] += gain
    out: dict[int, float] = {}
    st_cf = lt_cf = 0.0                  # losses carried forward, as positive numbers
    for fy in sorted(by_fy):
        st, lt = by_fy[fy]
        if st < 0:                       # this year's short-term loss hits long-term gain...
            lt, st = lt + st, 0.0
        use = min(st_cf, st)             # ...and so do short-term losses carried in
        st, st_cf = st - use, st_cf - use
        use = min(st_cf, max(lt, 0))
        lt, st_cf = lt - use, st_cf - use
        use = min(lt_cf, max(lt, 0))     # long-term losses only ever touch long-term gains
        lt, lt_cf = lt - use, lt_cf - use
        if lt < 0:
            lt_cf, lt = lt_cf - lt, 0.0
        out[fy] = (STCG_RATE * st + LTCG_RATE * max(lt - LTCG_EXEMPT, 0)) * (1 + CESS)
    return out


def capital_gains_tax(trades: list[Trade]) -> float:
    return sum(tax_by_fy([(t.exit_date, t.pnl, (t.exit_date - t.entry_date).days > 365)
                          for t in trades]).values())


def backtest(df: pd.DataFrame, target: pd.Series, name: str, capital: float = 100_000,
             cost: float = DEFAULT_COST, same_close: bool = False,
             cash_yield: float = 0.0) -> Result:
    close = df["Close"].to_numpy()
    fill_px = close if same_close or "Open" not in df else df["Open"].to_numpy()
    tgt = target.to_numpy()
    dates = df.index
    daily = (1 + cash_yield) ** (1 / 252)

    cash, shares = capital, 0.0
    entry_date, entry_cost, entry_px = None, 0.0, 0.0
    trades: list[Trade] = []
    fills: list[tuple[pd.Timestamp, float, str]] = []
    equity = np.empty(len(df))
    in_mkt = 0

    for i in range(len(df)):
        # Decision made at the close of day i-1 is filled today (oracle: same day).
        want = tgt[i] if same_close else (tgt[i - 1] if i > 0 else 0)
        px = fill_px[i]
        if i > 0 and shares == 0:      # idle cash earned a day's interest overnight
            cash *= daily
        if want == 1 and shares == 0:
            shares = cash / (px * (1 + cost))          # fractional shares for simplicity
            entry_cost, entry_px, entry_date = cash, px, dates[i]
            cash = 0.0
            fills.append((dates[i], px, "buy"))
        elif want == 0 and shares > 0:
            proceeds = shares * px * (1 - cost)
            trades.append(Trade(entry_date, dates[i], entry_px, px, proceeds - entry_cost))
            cash, shares = proceeds, 0.0
            fills.append((dates[i], px, "sell"))
        in_mkt += shares > 0
        equity[i] = cash + shares * close[i]

    if shares > 0:   # value the open position as if sold at the last close (for tax)
        proceeds = shares * close[-1] * (1 - cost)
        trades.append(Trade(entry_date, dates[-1], entry_px, close[-1],
                            proceeds - entry_cost, still_open=True))
        equity[-1] = proceeds

    res = Result(name, pd.Series(equity, dates), trades, fills,
                 capital_gains_tax(trades), capital)
    res._exposure = in_mkt / len(df)
    return res


# ==============================================================================
# Running it
# ==============================================================================
STRATS = {
    "Zigzag swing": (strat_zigzag, "pct", [5, 7.5, 10, 12.5, 15, 20], 10.0),
    "Trend + ATR stop": (strat_trend, "k", [2, 3, 4, 5], 3.0),
    "Mean reversion": (strat_meanrev, "rsi_buy", [25, 30, 35], 30),
    "200-day trend": (strat_sma200, "band", [0, 0.01, 0.02, 0.05], 0.02),
    "Uptrend dip": (strat_dip, "rsi2", [5, 10, 15], 10),
}


def run_all(df: pd.DataFrame, start: pd.Timestamp, capital: float, cost: float,
            prom_pct: float, params: dict | None = None,
            cash_yield: float = 0.0) -> list[Result]:
    """Compute signals on the full history (warm-up included), trade from `start`."""
    params = params or {}
    window = df.index >= start
    ev = df[window]
    results = [backtest(ev, strat_oracle(ev, prom_pct), "ORACLE (hindsight, untradeable)",
                        capital, cost, same_close=True, cash_yield=cash_yield)]
    for name, (fn, knob, _grid, default) in STRATS.items():
        val = params.get(name, default)
        sig = fn(df, **{knob: val})[window]
        results.append(backtest(ev, sig, f"{name} ({knob}={val:g})", capital, cost,
                                cash_yield=cash_yield))
    results.append(backtest(ev, strat_buyhold(ev), "Buy & hold", capital, cost))
    return results


def walk_forward(df: pd.DataFrame, start: pd.Timestamp, capital: float, cost: float,
                 train_frac: float = 0.6, cash_yield: float = 0.0) -> pd.DataFrame:
    ev_idx = df.index[df.index >= start]
    split = ev_idx[int(len(ev_idx) * train_frac)]
    train = (df.index >= start) & (df.index < split)
    test = df.index >= split
    rows = []
    for name, (fn, knob, grid, _default) in STRATS.items():
        scored = []
        for val in grid:
            sig = fn(df, **{knob: val})
            r = backtest(df[train], sig[train], name, capital, cost, cash_yield=cash_yield)
            scored.append((r.equity.iloc[-1] - r.tax, val))
        best = max(scored)[1]
        sig = fn(df, **{knob: best})
        r = backtest(df[test], sig[test], name, capital, cost, cash_yield=cash_yield)
        s = r.summary()
        rows.append({"strategy": f"{name} ({knob}={best:g}, picked in-sample)",
                     "out-of-sample after tax %": s["after tax %"],
                     "max drawdown %": s["max drawdown %"], "trades": s["trades"]})
    bh = backtest(df[test], strat_buyhold(df[test]), "Buy & hold", capital, cost).summary()
    rows.append({"strategy": "Buy & hold", "out-of-sample after tax %": bh["after tax %"],
                 "max drawdown %": bh["max drawdown %"], "trades": bh["trades"]})
    print(f"\nWalk-forward: tuned {start.date()} -> {split.date()}, "
          f"tested {split.date()} -> {df.index[-1].date()} (unseen)")
    return pd.DataFrame(rows).set_index("strategy")


def plot(df: pd.DataFrame, start: pd.Timestamp, results: list[Result], prom_pct: float,
         path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    buy_c, sell_c = "#2a78d6", "#eb6834"       # categorical slots 1 and 2
    ev = df[df.index >= start]
    zz = next(r for r in results if r.name.startswith("Zigzag"))
    sw = oracle_swings(ev["Close"], prom_pct)

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot(ev.index, ev["Close"], color=muted, lw=1.4, label="Close (adjusted)")
    for kind, col, lab in (("T", buy_c, "Hindsight trough"), ("P", sell_c, "Hindsight peak")):
        pts = [i for i, k in sw if k == kind]
        ax.scatter(ev.index[pts], ev["Close"].iloc[pts], s=110, facecolors="none",
                   edgecolors=col, linewidths=2, label=lab, zorder=3)
    for side, col, mk, lab in (("buy", buy_c, "^", "Zigzag buy (real time)"),
                               ("sell", sell_c, "v", "Zigzag sell (real time)")):
        pts = [(d, p) for d, p, s in zz.fills if s == side]
        if pts:
            d, p = zip(*pts)
            ax.scatter(d, p, s=70, marker=mk, color=col, edgecolors=surface,
                       linewidths=1.5, label=lab, zorder=4)
    ax.set_title(title, loc="left", color=ink, fontsize=13)
    ax.tick_params(colors=muted)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d4d3cf")
    ax.grid(axis="y", color="#ebeae6", lw=0.8)
    ax.set_ylabel("Price (Rs)", color=muted)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    print(f"Chart saved to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("tickers", nargs="*", help="e.g. RELIANCE.NS TCS.NS")
    ap.add_argument("--csv", help="use a local CSV instead of Yahoo Finance")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST, help="per side, 0.0015 = 0.15%%")
    ap.add_argument("--prominence", type=float, default=12.0,
                    help="oracle: min swing size, %% of median price")
    ap.add_argument("--cash-yield", type=float, default=0.0,
                    help="yearly interest on idle cash, 0.06 = 6%%")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--plot", help="save a chart (single ticker) to this PNG path")
    args = ap.parse_args()

    from pathlib import Path
    sources = [(Path(args.csv).stem, None)] if args.csv else [(t, t) for t in args.tickers]
    if not sources:
        ap.error("give at least one ticker or --csv")

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    for label, ticker in sources:
        df = load_csv(args.csv) if args.csv else load_yahoo(ticker, args.years)
        start = df.index[-1] - pd.DateOffset(years=args.years)
        if start <= df.index[0]:
            start = df.index[0]
            print("Note: no warm-up history before the window, so the trend strategy "
                  "(needs ~200 days) sits out the start.", file=sys.stderr)
        start = df.index[df.index >= start][0]
        results = run_all(df, start, args.capital, args.cost, args.prominence,
                          cash_yield=args.cash_yield)
        table = pd.DataFrame([r.summary() for r in results]).set_index("strategy")
        print(f"\n=== {label}: {start.date()} -> {df.index[-1].date()} | capital "
              f"Rs {args.capital:,.0f} | cost {args.cost:.2%}/side | idle cash "
              f"{args.cash_yield:.1%}/yr ===")
        print(table.to_string())
        if args.walkforward:
            print(walk_forward(df, start, args.capital, args.cost,
                               cash_yield=args.cash_yield).to_string())
        if args.plot:
            plot(df, start, results, args.prominence, args.plot,
                 f"{label}: hindsight swings vs real-time zigzag signals")


if __name__ == "__main__":
    main()
