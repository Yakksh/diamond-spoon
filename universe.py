"""
universe.py - stock lists, and a disk cache for Yahoo prices.

The lists are hand-made, from memory - check them against NSE before trusting
any result that depends on exactly which stocks are in them.

  NIFTY   Today's Nifty 50 members with 10+ years of NSE history.
  FALLEN  Large caps that were in the index around 2016-17 and later dropped
          out or collapsed.

Both lists are biased, in opposite directions. Testing only today's members
flatters buy & hold, because the losers have already been removed. Adding
FALLEN back pulls the other way, because those names were picked knowing they
fell. Read every result with both biases in mind.

Prices are cached as data/<SYMBOL>.csv (full history, split/bonus/dividend
adjusted) and re-downloaded once they are more than `max_age_hours` old.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from swing_backtest import _clean, drop_live_bar, load_csv

NIFTY = """
RELIANCE HDFCBANK ICICIBANK INFY TCS ITC LT SBIN BHARTIARTL KOTAKBANK AXISBANK
HINDUNILVR BAJFINANCE BAJAJFINSV MARUTI M&M SUNPHARMA TITAN ULTRACEMCO ASIANPAINT
NESTLEIND HCLTECH WIPRO TECHM NTPC POWERGRID ONGC COALINDIA TATASTEEL JSWSTEEL
HINDALCO GRASIM ADANIPORTS ADANIENT BAJAJ-AUTO HEROMOTOCO EICHERMOT CIPLA DRREDDY
APOLLOHOSP INDUSINDBK TATACONSUM TRENT BEL SHRIRAMFIN BPCL
""".split()

FALLEN = """
YESBANK ZEEL VEDL IDEA IOC GAIL LUPIN AUROPHARMA BOSCHLTD ACC AMBUJACEM BHEL UPL PNB
""".split()

UNIVERSES = {"nifty": NIFTY, "fallen": FALLEN, "all": NIFTY + FALLEN}

BENCHMARK = "NIFTYBEES"   # Nifty 50 ETF; adjusted closes include its dividends

DATA_DIR = Path(__file__).resolve().parent / "data"


def yahoo_symbol(name: str) -> str:
    """RELIANCE -> RELIANCE.NS; anything already carrying a suffix is left alone."""
    return name if "." in name or name.startswith("^") else f"{name}.NS"


def load(name: str, max_age_hours: float = 12, retries: int = 2) -> pd.DataFrame:
    path, gone = DATA_DIR / f"{name}.csv", DATA_DIR / f"{name}.missing"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return load_csv(str(path))
    if gone.exists() and time.time() - gone.stat().st_mtime < 24 * 3600:
        raise ValueError(f"no Yahoo data for {name} (checked in the last day)")

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # its own error spam
    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(yahoo_symbol(name), period="max", auto_adjust=True,
                             progress=False)
        except Exception as e:                  # rate limits surface as exceptions
            print(f"  {name}: {e}", file=sys.stderr)
        if not df.empty:
            break
        time.sleep(2 ** (attempt + 1))
    if df.empty:
        if path.exists():
            print(f"  {name}: download failed, using the cached copy", file=sys.stderr)
            return load_csv(str(path))
        DATA_DIR.mkdir(exist_ok=True)
        gone.touch()
        raise ValueError(f"no Yahoo data for {name} ({yahoo_symbol(name)})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = _clean(df)
    if df.attrs.get("bad_ticks"):
        print(f"  {name}: dropped bad bars from Yahoo on {df.attrs['bad_ticks']}",
              file=sys.stderr)
    df = drop_live_bar(df)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index_label="Date")
    return df


def load_many(names: list[str], max_age_hours: float = 12,
              nse_fallback: bool = False) -> dict[str, pd.DataFrame]:
    """Yahoo prices for each name. With nse_fallback, a name Yahoo lacks (delisted,
    merged) is rebuilt from NSE's own files instead (nse.price_series)."""
    out = {}
    for name in names:
        try:
            out[name] = load(name, max_age_hours)
        except ValueError as e:
            if not nse_fallback:
                print(f"  skipped: {e}", file=sys.stderr)
                continue
            import nse
            try:
                out[name], notes = nse.price_series(name)
                print(f"  {name}: not on Yahoo, prices rebuilt from NSE's files", file=sys.stderr)
                for note in notes:
                    print(f"    {note}", file=sys.stderr)
            except ValueError as e2:
                print(f"  skipped: {e}; {e2}", file=sys.stderr)
    return out


def resolve(args_names: list[str], universe: str) -> list[str]:
    """Explicit symbols on the command line win over a named universe."""
    return [n.upper() for n in args_names] if args_names else UNIVERSES[universe]
