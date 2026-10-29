#!/usr/bin/env python3
"""
evaluate.py - does a rule work on stocks in general, or only on the one you tried?

Runs every rule in swing_backtest.STRATS, at its default setting (no tuning),
across a list of stocks and answers three questions:

  1. Beat buy & hold?  Post-tax CAGR of the rule minus buy & hold's, stock by
     stock, over the whole window and over each half on its own. A rule that
     only wins in one half was lucky in that half.
  2. Does the signal know anything?  Where the price was 1, 3 and 6 months
     after each buy and sell signal, next to the same numbers for every day.
     If prices fall after a sell signal no more often than after a random day,
     the rule cannot tell you when to get out, whatever its backtest says.
  3. Sell, then buy back cheaper?  Every time a rule sold a stock and later
     bought it back: how often was the buyback price lower than the sale?

Usage
  python evaluate.py                         # Nifty 50 (10y+ history) + fallen large caps
  python evaluate.py --universe nifty --cash-yield 0.06
  python evaluate.py RELIANCE TCS INFY       # your own list of NSE symbols
  python evaluate.py --out results.csv       # per-stock numbers for a spreadsheet
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import swing_backtest as sb
import universe

HORIZONS = {"1m": 21, "3m": 63, "6m": 126}     # trading days
WARMUP = 260                                   # bars needed before the window starts


def windows(end: pd.Timestamp, years: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    start = end - pd.DateOffset(years=years)
    mid = start + (end - start) / 2
    return {"whole": (start, end), "1st half": (start, mid), "2nd half": (mid, end)}


def signal_changes(target: np.ndarray) -> list[tuple[int, str]]:
    """(fill bar, side) for each change in the target: decided at a close, filled
    at the next bar's open."""
    return [(i + 1, "buy" if target[i] == 1 else "sell")
            for i in range(1, len(target) - 1) if target[i] != target[i - 1]]


def study(data: dict[str, pd.DataFrame], years: int, cost: float, cash_yield: float):
    end = max(df.index[-1] for df in data.values())
    wins = windows(end, years)
    perf, events, base, cycles = [], [], [], []
    for name, df in data.items():
        if (df.index < wins["whole"][0]).sum() < WARMUP:
            print(f"  skipped {name}: needs {years} years plus ~1 year of warm-up history",
                  file=sys.stderr)
            continue
        sigs = {rule: fn(df, **{knob: default})
                for rule, (fn, knob, _grid, default) in sb.STRATS.items()}

        for period, (a, b) in wins.items():
            m = (df.index >= a) & (df.index <= b)
            w = df[m]
            bh = sb.backtest(w, sb.strat_buyhold(w), "Buy & hold", cost=cost).summary()
            for rule, sig in sigs.items():
                s = sb.backtest(w, sig[m], rule, cost=cost, cash_yield=cash_yield).summary()
                perf.append({"stock": name, "period": period, "rule": rule,
                             "cagr": s["CAGR % (post-tax)"], "max_dd": s["max drawdown %"],
                             "trades": s["trades"], "bh_cagr": bh["CAGR % (post-tax)"],
                             "bh_max_dd": bh["max drawdown %"]})

        # Signal and buy-back tests use the whole window only.
        a, b = wins["whole"]
        m = (df.index >= a) & (df.index <= b)
        opn = df["Open"].to_numpy() if "Open" in df else df["Close"].to_numpy()
        days = np.flatnonzero(m)
        for h_name, h in HORIZONS.items():
            ok = days[days + h < len(df)]
            base.append(pd.DataFrame({"h": h_name, "fwd": opn[ok + h] / opn[ok] - 1}))
        for rule, sig in sigs.items():
            changes = [(i, side) for i, side in signal_changes(sig.to_numpy()) if m[i]]
            for i, side in changes:
                for h_name, h in HORIZONS.items():
                    if i + h < len(df):
                        events.append({"rule": rule, "side": side, "h": h_name,
                                       "fwd": opn[i + h] / opn[i] - 1})
            for (i, s1), (j, s2) in zip(changes, changes[1:]):
                if s1 == "sell" and s2 == "buy":
                    cycles.append({"stock": name, "rule": rule,
                                   "premium": opn[j] / opn[i] - 1, "days_out": j - i})
    return (pd.DataFrame(perf), pd.DataFrame(events),
            pd.concat(base, ignore_index=True), pd.DataFrame(cycles), wins)


def beat_table(perf: pd.DataFrame) -> pd.DataFrame:
    p = perf.assign(edge=perf.cagr - perf.bh_cagr, dd_saved=perf.max_dd - perf.bh_max_dd)
    return p.groupby("rule", sort=False).agg(
        beat_bh=("edge", lambda e: f"{(e > 0).sum()}/{len(e)}"),
        median_edge=("edge", "median"),
        median_cagr=("cagr", "median"),
        bh_median_cagr=("bh_cagr", "median"),
        median_dd_saved=("dd_saved", "median"),
        median_trades=("trades", "median"),
    )


def signal_table(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        b = base[base.h == h].fwd
        rows.append({"signal": "any day (baseline)", "h": h, "% lower": 100 * (b < 0).mean(),
                     "median move %": 100 * b.median(), "% down 20+": 100 * (b < -0.2).mean()})
        for (rule, side), g in events[events.h == h].groupby(["rule", "side"], sort=False):
            rows.append({"signal": f"{rule}: {side}", "h": h,
                         "% lower": 100 * (g.fwd < 0).mean(),
                         "median move %": 100 * g.fwd.median(),
                         "% down 20+": 100 * (g.fwd < -0.2).mean()})
    t = pd.DataFrame(rows).pivot(index="signal", columns="h")
    t = t[[(c, h) for c in ("% lower", "median move %") for h in HORIZONS]
          + [("% down 20+", "6m")]]
    order = ["any day (baseline)"] + sorted(i for i in t.index if i != "any day (baseline)")
    return t.loc[order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("names", nargs="*", help="NSE symbols, e.g. RELIANCE TCS (default: --universe)")
    ap.add_argument("--universe", choices=universe.UNIVERSES, default="all")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--cost", type=float, default=sb.DEFAULT_COST, help="per side, 0.0015 = 0.15%%")
    ap.add_argument("--cash-yield", type=float, default=0.0,
                    help="yearly interest on idle cash, 0.06 = 6%%")
    ap.add_argument("--out", help="save per-stock results to this CSV")
    args = ap.parse_args()

    names = universe.resolve(args.names, args.universe)
    print(f"Loading {len(names)} stocks...", file=sys.stderr)
    data = universe.load_many(names)
    perf, events, base, cycles, wins = study(data, args.years, args.cost, args.cash_yield)
    if perf.empty:
        sys.exit("No stock had enough history.")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    n = perf.stock.nunique()
    print(f"\n{n} stocks | rules at default settings | cost {args.cost:.2%}/side | "
          f"idle cash {args.cash_yield:.1%}/yr | CAGR after tax, edge in percentage points")
    for period, (a, b) in wins.items():
        print(f"\n=== 1. Beat buy & hold? {period}: {a.date()} -> {b.date()} ===")
        print(beat_table(perf[perf.period == period]).to_string())

    a, b = wins["whole"]
    print(f"\n=== 2. Does the signal know anything? ({a.date()} -> {b.date()}, "
          f"filled at the next open) ===")
    print("% lower = price lower that long after the signal; compare each row "
          "with the baseline.")
    print(signal_table(events, base).to_string())

    print("\n=== 3. Sell, then buy back cheaper? (each sell followed by the "
          "same rule's next buy) ===")
    if cycles.empty:
        print("No sell-then-buy cycles.")
    else:
        print(cycles.groupby("rule", sort=False).agg(
            cycles=("premium", "size"),
            cheaper_pct=("premium", lambda p: 100 * (p < 0).mean()),
            median_buyback_vs_sale_pct=("premium", lambda p: 100 * p.median()),
            median_days_out=("days_out", "median"),
        ).to_string())

    if args.out:
        perf.to_csv(args.out, index=False)
        print(f"\nPer-stock results saved to {args.out}")


if __name__ == "__main__":
    main()
