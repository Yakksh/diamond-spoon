#!/usr/bin/env python3
"""
odds.py - if I sell now, what are the odds I can buy back cheaper?

The plan it tests, for each buy-back level (say 10% below today's close):

  1. sell at the next open;
  2. leave a limit order to buy back at that level for the next N trading days
     (it fills the first day the low reaches it, at the open if it gaps below);
  3. if it never fills, give up and buy back at the close on day N.

The score is how many shares you end with per share you sold, after costs and
(optionally) tax: above 1 means the round trip left you with more shares than
just holding would have.

It answers that four ways, side by side:

  SIMULATED, past drift   10,000 paths built from the stock's own past days,
                          drawn in blocks (about 10 days on average) so calm and
                          stormy spells stay together. Includes the stock's
                          average past rise - often the shakiest assumption.
  SIMULATED, no drift     The same, with the average daily move removed.
  HISTORY, every day      The plan run from every past day, on what really happened.
  HISTORY, days like now  Only past days on the same side of the 200-day average
                          with a fall from the 1-year high within 7.5 points of
                          today's. Overlapping windows, so the separate episodes
                          are what count, and they are few.

These are odds from the past, not a forecast: the future can do what the last
ten years never did. The tool also leaves out that buying back restarts the
12-month clock for long-term capital-gains tax.

Usage
  python odds.py RELIANCE
  python odds.py RELIANCE --days 126 --levels 10 20
  python odds.py RELIANCE --cost-basis 900 --bought 2021-03-15 --qty 100
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import swing_backtest as sb
import universe


# ==============================================================================
# Paths: simulated or real, all as (rows, days) arrays of open, low, close
# ==============================================================================
def day_ratios(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each day's open, low and close as a multiple of the previous close."""
    prev = df["Close"].shift()
    low = df["Low"] if "Low" in df else df[["Open", "Close"]].min(axis=1)
    o, l, c = (df["Open"] / prev), (low / prev), (df["Close"] / prev)
    keep = prev.notna()
    return o[keep].to_numpy(), l[keep].to_numpy(), c[keep].to_numpy()


def simulate(ratios, start: float, days: int, replays: int, block: float,
             rng: np.random.Generator, drift: bool = True):
    """Stationary block bootstrap: each step continues the current run of past
    days, or with probability 1/block jumps to a random past day."""
    o, l, c = ratios
    if not drift:                                   # remove the average daily move
        k = np.exp(-np.log(c).mean())
        o, l, c = o * k, l * k, c * k
    n = len(c)
    idx = np.empty((replays, days), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, replays)
    for d in range(1, days):
        jump = rng.random(replays) < 1 / block
        idx[:, d] = np.where(jump, rng.integers(0, n, replays), (idx[:, d - 1] + 1) % n)
    closes = start * np.cumprod(c[idx], axis=1)
    prev = np.hstack([np.full((replays, 1), start), closes[:, :-1]])
    return prev * o[idx], prev * l[idx], closes


def history_windows(df: pd.DataFrame, days: int, since: pd.Timestamp):
    """For every day t from `since` with `days` more days after it: the start
    close and the next `days` opens, lows and closes."""
    from numpy.lib.stride_tricks import sliding_window_view as win

    low = df["Low"] if "Low" in df else df[["Open", "Close"]].min(axis=1)
    o, l, c = (s.to_numpy() for s in (df["Open"], low, df["Close"]))
    t = np.flatnonzero((df.index >= since)[: len(df) - days])
    return (c[t], win(o[1:], days)[t], win(l[1:], days)[t], win(c[1:], days)[t], t)


# ==============================================================================
# The plan
# ==============================================================================
def round_trip(start, opens, lows, closes, level: float, cost: float, tax_frac: float):
    """Shares after selling at the first open and buying back, per share sold.

    start: close the levels are measured from, one per row (or a scalar).
    Returns (filled, shares_ratio), one value per row."""
    start = np.broadcast_to(np.asarray(start, dtype=float), (opens.shape[0],))
    limit = start * (1 - level)
    hit = lows <= limit[:, None]
    filled = hit.any(axis=1)
    first = hit.argmax(axis=1)
    rows = np.arange(len(start))
    fill_px = np.where(filled, np.minimum(opens[rows, first], limit), closes[:, -1])
    sale = opens[:, 0] * (1 - cost) * (1 - tax_frac)
    return filled, sale / (fill_px * (1 + cost))


def tax_fraction(price: float, cost_basis: float | None, bought: pd.Timestamp | None,
                 qty: float | None, exempt_left: float, cost: float,
                 today: pd.Timestamp) -> tuple[float, str]:
    """Tax on selling today as a share of the sale proceeds, and a one-line note."""
    if cost_basis is None:
        return 0.0, "Tax not included (pass --cost-basis and --bought to include it)."
    long_term = bought is not None and (today - bought).days > 365
    gain = price * (1 - cost) - cost_basis
    if gain <= 0:
        return 0.0, f"No tax: selling at about Rs {price:,.0f} is a loss on your cost."
    if long_term:
        taxable = max(gain * qty - exempt_left, 0) / qty if qty else gain
        per_share = sb.LTCG_RATE * taxable * (1 + sb.CESS)
        kind = (f"long-term, {sb.LTCG_RATE:.1%} + cess"
                + (f", after Rs {exempt_left:,.0f} of exemption" if qty else
                   ", no exemption assumed (pass --qty)"))
    else:
        per_share = sb.STCG_RATE * gain * (1 + sb.CESS)
        kind = f"short-term, {sb.STCG_RATE:.0%} + cess"
    frac = per_share / (price * (1 - cost))
    return frac, (f"Tax on selling: about Rs {per_share:,.1f} a share ({kind}), "
                  f"{frac:.1%} of the sale.")


def summarise(filled: np.ndarray, ratio: np.ndarray) -> dict:
    change = 100 * (ratio - 1)
    return {"limit fills %": 100 * filled.mean(),
            "more shares %": 100 * (ratio > 1).mean(),
            "median shares %": np.median(change),
            "average shares %": change.mean(),
            "bad case (10th pct) %": np.percentile(change, 10)}


def episodes(t: np.ndarray) -> int:
    """Separate runs of consecutive days."""
    return int(len(t) and 1 + (np.diff(t) > 1).sum())


# ==============================================================================
# Running it
# ==============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("name", help="NSE symbol, e.g. RELIANCE")
    ap.add_argument("--days", type=int, default=63, help="trading days to wait (63 = ~3 months)")
    ap.add_argument("--levels", type=float, nargs="+", default=[5, 10, 15, 20],
                    help="buy-back levels, %% below today's close")
    ap.add_argument("--years", type=int, default=10, help="history to learn from")
    ap.add_argument("--replays", type=int, default=10_000)
    ap.add_argument("--block", type=float, default=10, help="average block length, days")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost", type=float, default=sb.DEFAULT_COST, help="per side, 0.0015 = 0.15%%")
    ap.add_argument("--cost-basis", type=float, help="your average buy price per share")
    ap.add_argument("--bought", help="when you bought (YYYY-MM-DD), for short/long-term tax")
    ap.add_argument("--qty", type=float, help="shares you would sell (for the LTCG exemption)")
    ap.add_argument("--exempt-left", type=float, default=sb.LTCG_EXEMPT,
                    help="LTCG exemption still unused this financial year")
    args = ap.parse_args()

    df = universe.load(args.name.upper())
    end = df.index[-1]
    since = end - pd.DateOffset(years=args.years)
    last = float(df["Close"].iloc[-1])
    levels = [x / 100 for x in args.levels]
    frac, tax_note = tax_fraction(last, args.cost_basis,
                                  pd.Timestamp(args.bought) if args.bought else None,
                                  args.qty, args.exempt_left, args.cost, pd.Timestamp.now())
    breakeven = last * (1 - args.cost) * (1 - frac) / (1 + args.cost)

    hist = df[df.index >= since - pd.DateOffset(days=1)]
    ratios = day_ratios(hist)
    rng = np.random.default_rng(args.seed)
    sims = {"SIMULATED, past drift": simulate(ratios, last, args.days, args.replays,
                                               args.block, rng, drift=True),
            "SIMULATED, no drift": simulate(ratios, last, args.days, args.replays,
                                             args.block, rng, drift=False)}

    # Today's state, and the same state on every past day.
    c = df["Close"]
    above = (c > c.rolling(200).mean()).to_numpy()
    fall = (c / c.rolling(252).max() - 1).to_numpy()
    start, o, l, cl, t = history_windows(df, args.days, since)
    like = (above[t] == above[-1]) & (np.abs(fall[t] - fall[-1]) <= 0.075)

    print(f"\n{args.name.upper()}: last close Rs {last:,.2f} on {end.date()} | "
          f"{'above' if above[-1] else 'below'} its 200-day average | "
          f"{100 * fall[-1]:+.1f}% from its 1-year high")
    print(f"Plan: sell at the next open, then try to buy back within {args.days} trading days; "
          f"if the limit never fills, buy back at the close on day {args.days}.")
    print(tax_note)
    print(f"Break-even: to end with the same number of shares you must buy back at or below "
          f"about Rs {breakeven:,.1f} ({100 * (breakeven / last - 1):+.1f}%).")
    print(f"Learning from {hist.index[0].date()} -> {end.date()}; cost {args.cost:.2%} a side.")

    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    blocks = [(name, (last, *paths)) for name, paths in sims.items()]
    blocks.append((f"HISTORY, every day ({len(t)} start days)",
                   (start, o, l, cl)))
    n_like = int(like.sum())
    blocks.append((f"HISTORY, days like now ({n_like} start days in "
                   f"{episodes(t[like])} separate episodes)",
                   (start[like], o[like], l[like], cl[like])))
    for name, (s0, opens, lows, closes) in blocks:
        print(f"\n=== {name} ===")
        if len(opens) < 20:
            print("Too few past days like this to say anything.")
            continue
        rows = []
        for lv in levels:
            filled, ratio = round_trip(s0, opens, lows, closes, lv, args.cost, frac)
            rows.append({"buy back at": f"Rs {last * (1 - lv):,.0f} (-{100 * lv:g}%)",
                         **summarise(filled, ratio)})
        print(pd.DataFrame(rows).set_index("buy back at").to_string())
        end_move = 100 * (closes[:, -1] / np.broadcast_to(s0, (len(closes),)) - 1)
        p10, p50, p90 = np.percentile(end_move, [10, 50, 90])
        print(f"Price after {args.days} days vs start: 10th pct {p10:+.1f}%, median "
              f"{p50:+.1f}%, 90th pct {p90:+.1f}%; lower than start "
              f"{100 * (end_move < 0).mean():.0f}% of the time.")
    print("\nmore shares % = how often the round trip beat simply holding. median / average "
          "shares % = typical and average gain or loss in share count; when the average is "
          "below zero, the plan loses shares on balance even if it usually wins. Buying "
          "back restarts the 12-month clock for long-term tax.")


if __name__ == "__main__":
    main()
