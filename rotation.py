#!/usr/bin/env python3
"""
rotation.py - instead of timing one stock, hold whichever stocks are strongest.

At the start of each month it ranks every stock in the list by its return over
the past 12 months, leaving out the most recent month (the standard "12-1
momentum": last month's winners tend to give some of it back). It holds the
top N in equal amounts and sells a stock once it ranks below `--exit-rank`.
A stock it keeps is not topped up or trimmed, which keeps trading - and
short-term tax - down.

Compared on the same rupees, costs and tax with:
  * equal-weight buy & hold of the same list (bought once, never touched)
  * NIFTYBEES (Nifty 50 ETF, dividends included). This is the fair benchmark:
    the index swapped its members as it went, so it carries no survivorship bias.

Realism
  * Ranks use closes up to the last trading day of the month; trades fill at
    the next day's open, at --cost per side.
  * Tax: shares are sold oldest lot first, gains are netted per financial year
    with loss carry-forward (swing_backtest.tax_by_fy), and each year's tax is
    paid at the first rebalance after 31 March, so it stops compounding. At
    the end everything is sold and the rest of the tax paid, for every line.
  * --market-filter moves to cash (earning --cash-yield) while NIFTYBEES closes
    below its 200-day average, and buys back in when it is above.

Survivorship bias: the stock list is today's names plus a hand-picked set of
fallen ones (universe.py), not the index as it stood on each date. Momentum
is flattered by a list of known survivors, so trust the NIFTYBEES comparison
more than the equal-weight one.

Usage
  python rotation.py
  python rotation.py --top 10 --exit-rank 20 --market-filter
  python rotation.py --universe nifty --years 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import swing_backtest as sb
import universe
from evaluate import windows

MONTH = 21     # trading days
YEAR = 252


@dataclass
class Lot:
    shares: float
    cost_per_share: float    # rupees paid per share, costs included
    date: pd.Timestamp


class Book:
    """Cash, tax lots and realised gains for a portfolio of stocks."""

    def __init__(self, capital: float, cost: float):
        self.cash = capital
        self.cost = cost
        self.lots: dict[str, list[Lot]] = {}
        self.realized: list[tuple[pd.Timestamp, float, bool]] = []
        self.traded = 0.0
        self.tax_paid = 0.0
        self.paid_fys: set[int] = set()

    def shares(self, name: str) -> float:
        return sum(lot.shares for lot in self.lots.get(name, []))

    def value(self, prices: pd.Series) -> float:
        return self.cash + sum(self.shares(n) * prices[n] for n in self.lots)

    def buy(self, name: str, rupees: float, px: float, date: pd.Timestamp) -> None:
        if rupees <= 0:
            return
        shares = rupees / (px * (1 + self.cost))
        self.lots.setdefault(name, []).append(Lot(shares, rupees / shares, date))
        self.cash -= rupees
        self.traded += rupees

    def sell(self, name: str, px: float, date: pd.Timestamp,
             shares: float | None = None) -> None:
        """Sell `shares` (default: all), oldest lot first."""
        lots = self.lots[name]
        left = self.shares(name) if shares is None else shares
        while left > 1e-12 and lots:
            lot = lots[0]
            take = min(lot.shares, left)
            proceeds = take * px * (1 - self.cost)
            self.realized.append((date, proceeds - take * lot.cost_per_share,
                                  (date - lot.date).days > 365))
            self.cash += proceeds
            self.traded += take * px
            lot.shares -= take
            left -= take
            if lot.shares <= 1e-12:
                lots.pop(0)
        if not lots:
            del self.lots[name]

    def pay_tax(self, before_fy: int, date: pd.Timestamp, prices: pd.Series) -> None:
        """Pay the tax for every financial year before `before_fy` not yet paid,
        selling a slice of every holding if cash runs short."""
        owed = sb.tax_by_fy(self.realized)
        due = sum(t for fy, t in owed.items() if fy < before_fy and fy not in self.paid_fys)
        self.paid_fys |= {fy for fy in owed if fy < before_fy}
        if due > self.cash and self.lots:
            held = sum(self.shares(n) * prices[n] for n in self.lots)
            frac = min((due - self.cash) / (held * (1 - self.cost)), 1.0)
            for name in list(self.lots):
                self.sell(name, prices[name], date, self.shares(name) * frac)
        self.cash -= due
        self.tax_paid += due


Decider = Callable[[int, list[str], set[str]], list[str]]


def run(opens: pd.DataFrame, closes: pd.DataFrame, start: int, end: int,
        rebalance_rows: set[int], decide: Decider, capital: float, cost: float,
        cash_yield: float = 0.0) -> dict:
    """Trade rows start..end inclusive. On a rebalance row r, decide(r - 1, held,
    tradeable) names the stocks to hold, using data up to row r - 1 only."""
    book = Book(capital, cost)
    marks = closes.ffill()
    daily = (1 + cash_yield) ** (1 / YEAR)
    dates = closes.index
    equity = np.empty(end - start + 1)
    in_cash = 0
    for k, r in enumerate(range(start, end + 1)):
        if k:
            book.cash *= daily
        d = dates[r]
        if r in rebalance_rows:
            px = opens.iloc[r]
            tradeable = set(px.dropna().index)
            target = decide(r - 1, sorted(book.lots), tradeable)
            for name in [n for n in book.lots if n not in target and n in tradeable]:
                book.sell(name, px[name], d)
            book.pay_tax(sb.fy_of(d), d, px.fillna(marks.iloc[r - 1]))
            new = [n for n in target if n not in book.lots and n in tradeable]
            budget = book.cash / len(new) if new else 0.0
            for name in new:
                book.buy(name, budget, px[name], d)
        equity[k] = book.value(marks.iloc[r])
        in_cash += book.cash > 0.5 * equity[k]

    traded = book.traded                         # turnover excludes the final sale
    last = marks.iloc[end]
    for name in list(book.lots):                 # sell everything at the last close
        book.sell(name, last[name], dates[end])
    book.pay_tax(sb.fy_of(dates[end]) + 1, dates[end], last)

    years = (dates[end] - dates[start]).days / 365.25
    eq = pd.Series(equity, dates[start:end + 1])
    final = book.cash
    return {
        "CAGR % (post-tax)": 100 * ((max(final, 1) / capital) ** (1 / years) - 1),
        "max drawdown %": 100 * (eq / eq.cummax() - 1).min(),
        "turnover %/yr": 100 * (traded / 2) / eq.mean() / years,
        "tax paid % of capital": 100 * book.tax_paid / capital,
        "time in cash %": 100 * in_cash / len(eq),
    }


def momentum_decider(closes: pd.DataFrame, top: int, exit_rank: int, lookback: int,
                     skip: int, risk_on: pd.Series | None = None) -> tuple[Decider, pd.DataFrame]:
    marks = closes.ffill(limit=5)
    score = marks.shift(skip) / marks.shift(lookback) - 1
    score[closes.isna()] = np.nan                 # no rank for a stock that is not trading

    def decide(row: int, held: list[str], tradeable: set[str]) -> list[str]:
        if risk_on is not None and not risk_on.iloc[row]:
            return []
        s = score.iloc[row].dropna().sort_values(ascending=False)
        rank = {n: i + 1 for i, n in enumerate(s.index)}
        keep = [n for n in held if rank.get(n, len(rank) + 1) <= exit_rank]
        new = [n for n in s.index if n not in keep and n in tradeable]
        return keep + new[:max(top - len(keep), 0)]

    return decide, score


def buy_once(names: list[str] | None = None) -> Decider:
    """Equal-weight buy & hold: buy on the first rebalance, never trade again."""
    def decide(row: int, held: list[str], tradeable: set[str]) -> list[str]:
        if held:
            return held
        return sorted(tradeable if names is None else set(names) & tradeable)
    return decide


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("names", nargs="*", help="NSE symbols (default: --universe)")
    ap.add_argument("--universe", choices=universe.UNIVERSES, default="all")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--top", type=int, default=10, help="how many stocks to hold")
    ap.add_argument("--exit-rank", type=int, help="sell once rank is worse than this "
                    "(default: same as --top, i.e. no buffer)")
    ap.add_argument("--lookback", type=int, default=12, help="months of past return to rank on")
    ap.add_argument("--skip", type=int, default=1, help="most recent months left out")
    ap.add_argument("--market-filter", action="store_true",
                    help="hold cash while NIFTYBEES is below its 200-day average")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--cost", type=float, default=sb.DEFAULT_COST, help="per side, 0.0015 = 0.15%%")
    ap.add_argument("--cash-yield", type=float, default=0.0,
                    help="yearly interest on idle cash, 0.06 = 6%%")
    args = ap.parse_args()
    exit_rank = args.exit_rank or args.top

    names = universe.resolve(args.names, args.universe)
    print(f"Loading {len(names)} stocks + {universe.BENCHMARK}...", file=sys.stderr)
    data = universe.load_many(names + [universe.BENCHMARK])
    bench = data.pop(universe.BENCHMARK)
    closes = pd.DataFrame({n: df["Close"] for n, df in data.items()}).sort_index()
    opens = pd.DataFrame({n: df["Open"] for n, df in data.items()}).reindex(closes.index)
    bclose = bench["Close"].reindex(closes.index).ffill()
    bopen = bench["Open"].reindex(closes.index)
    risk_on = (bclose > bclose.rolling(200).mean()) if args.market_filter else None
    decide, score = momentum_decider(closes, args.top, exit_rank, args.lookback * MONTH,
                                     args.skip * MONTH, risk_on)

    label = (f"Momentum top {args.top}, exit rank {exit_rank}, {args.lookback}-{args.skip}"
             + (", market filter" if args.market_filter else ""))
    month_start = np.flatnonzero(closes.index.month != np.roll(closes.index.month, 1))
    rows = []
    for period, (a, b) in windows(closes.index[-1], args.years).items():
        start = int(np.searchsorted(closes.index, a))
        end = int(np.searchsorted(closes.index, b, side="right")) - 1
        rebal = {r for r in month_start if start < r <= end} | {start + 1}
        lines = {
            label: run(opens, closes, start, end, rebal, decide, args.capital, args.cost,
                       args.cash_yield),
            f"Equal-weight buy & hold ({len(data)} stocks)":
                run(opens, closes, start, end, {start + 1}, buy_once(), args.capital, args.cost),
            f"{universe.BENCHMARK} buy & hold":
                run(bopen.to_frame("B"), bclose.to_frame("B"), start, end, {start + 1},
                    buy_once(), args.capital, args.cost),
        }
        for line, res in lines.items():
            rows.append({"period": f"{period}: {closes.index[start].date()} -> "
                                   f"{closes.index[end].date()}", "strategy": line, **res})

    pd.set_option("display.width", 170)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    print(f"\n{len(data)} stocks | capital Rs {args.capital:,.0f} | cost {args.cost:.2%}/side"
          f" | idle cash {args.cash_yield:.1%}/yr | tax paid every April, rest at the end")
    print(pd.DataFrame(rows).set_index(["period", "strategy"]).to_string())

    held = decide(len(closes) - 1, [], set(closes.columns[closes.iloc[-1].notna()]))
    s = score.iloc[-1]
    print(f"\nRule's picks on {closes.index[-1].date()} (if starting fresh today; "
          f"12-1 month return):")
    print("  " + ", ".join(f"{n} {100 * s[n]:+.0f}%" for n in held) if held
          else "  none (market filter says cash)")
    print("Survivorship bias: the list is today's names plus hand-picked fallen ones; "
          "see universe.py.")


if __name__ == "__main__":
    main()
