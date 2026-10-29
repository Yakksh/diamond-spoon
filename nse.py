#!/usr/bin/env python3
"""
nse.py - a point-in-time "Nifty-like" stock list from NSE's own daily files.

NSE publishes every trading day's prices for every listed stock (the
"bhavcopy"), including stocks that were later delisted, merged or renamed, so
a list built from it on each past date has no survivorship bias. NSE does not
publish past Nifty 50 memberships in a form this can read, so the list is a
proxy:

  at each month end, among stocks with stock futures that day (Nifty 50
  members must be in the F&O segment), the N with the highest median daily
  traded value over the last 6 months, traded on 90%+ of those days.

Traded value is not free-float market cap, so the proxy differs from the real
index; `python nse.py check` reports how much it overlaps today's Nifty 50.

Renames and splits: a rename keeps the ISIN, a split keeps the symbol, so a
stock is tracked as whatever is linked to it by either. It is reported under
its latest symbol.

NSE's previous-close column is NOT adjusted for bonuses and splits (RELIANCE
on 28 Oct 2024 still shows the pre-bonus close), so the prices here are used
for ranking only. Returns come from Yahoo's adjusted series (universe.py).

Files are cached under data/nse/: one parquet per month of daily rows, and the
list of stocks with futures on each month's last trading day.

Usage
  python nse.py download                        # ~2,700 small files since 2016; resumable
  python nse.py build --top 50                  # -> data/nse/universe.csv
  python nse.py check                           # overlap with today's Nifty 50
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd

ARCHIVE = "https://archives.nseindia.com/content"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
UDIFF_FROM = pd.Timestamp("2024-07-08")    # NSE's new file format starts around here
SERIES = ("EQ", "BE", "BZ")                 # normal, trade-for-trade, and suspended-ish
NSE_DIR = Path(__file__).resolve().parent / "data" / "nse"
UNIVERSE_CSV = NSE_DIR / "universe.csv"


# ==============================================================================
# Download
# ==============================================================================
def _urls(kind: str, d: pd.Timestamp) -> list[str]:
    mon = d.strftime("%b").upper()
    folder = "EQUITIES" if kind == "cm" else "DERIVATIVES"
    old = f"{ARCHIVE}/historical/{folder}/{d:%Y}/{mon}/{kind}{d:%d}{mon}{d:%Y}bhav.csv.zip"
    new = f"{ARCHIVE}/{kind}/BhavCopy_NSE_{kind.upper()}_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return [new, old] if d >= UDIFF_FROM else [old, new]


class Refused(Exception):
    """NSE answered 403: it will not serve that file (all of 2015, for one)."""


def _fetch(url: str, retries: int = 4) -> bytes | None:
    """The file, or None if NSE has none for that day (holiday)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read()
            return body if body[:2] == b"PK" else None     # an HTML error page is no file
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 403:
                raise Refused(url) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def _read_zip(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return pd.read_csv(z.open(z.namelist()[0]), low_memory=False)


def _day(kind: str, d: pd.Timestamp) -> pd.DataFrame | None:
    refused = None
    for url in _urls(kind, d):
        try:
            raw = _fetch(url)
        except Refused as e:
            refused = e
            continue
        if raw is not None:
            return _read_zip(raw)
    if refused:
        raise refused
    return None


def parse_cm(df: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    """Both file formats -> date, symbol, series, isin, open, close, value (Rs)."""
    if "TckrSymb" in df:
        df = df.rename(columns={"TckrSymb": "symbol", "SctySrs": "series", "ISIN": "isin",
                                "OpnPric": "open", "ClsPric": "close", "TtlTrfVal": "value"})
    else:
        df = df.rename(columns={"SYMBOL": "symbol", "SERIES": "series", "ISIN": "isin",
                                "OPEN": "open", "CLOSE": "close", "TOTTRDVAL": "value"})
    df = df[df["series"].isin(SERIES)][["symbol", "series", "isin", "open", "close", "value"]]
    df = df.astype({"open": "float32", "close": "float32", "value": "float64"})
    df.insert(0, "date", d)
    return df


def parse_fo(df: pd.DataFrame) -> list[str]:
    """Symbols with stock futures that day."""
    if "TckrSymb" in df:
        return sorted(df.loc[df["FinInstrmTp"] == "STF", "TckrSymb"].unique())
    return sorted(df.loc[df["INSTRUMENT"] == "FUTSTK", "SYMBOL"].unique())


def download(start: pd.Timestamp, end: pd.Timestamp, workers: int = 4) -> None:
    (NSE_DIR / "cm").mkdir(parents=True, exist_ok=True)
    (NSE_DIR / "fo").mkdir(parents=True, exist_ok=True)
    for m in pd.period_range(start, end, freq="M"):
        cm_path, fo_path = NSE_DIR / "cm" / f"{m}.parquet", NSE_DIR / "fo" / f"{m}.txt"
        settled = m.end_time < end - pd.Timedelta(days=5)   # older months never change
        if settled and cm_path.exists() and fo_path.exists():
            continue
        days = pd.bdate_range(max(m.start_time, start), min(m.end_time, end))
        try:
            with ThreadPoolExecutor(workers) as pool:
                raw = list(pool.map(lambda d: _day("cm", d), days))
        except Refused as e:
            print(f"{m}: NSE refused the files ({e}); skipped", file=sys.stderr)
            continue
        frames = [parse_cm(df, d) for d, df in zip(days, raw) if df is not None]
        if not frames:
            print(f"{m}: no trading days found", file=sys.stderr)
            continue
        month = pd.concat(frames, ignore_index=True)
        month.to_parquet(cm_path, compression="zstd", index=False)
        last = month["date"].max()
        fo = _day("fo", last)
        if fo is not None:
            fo_path.write_text("\n".join(parse_fo(fo)) + "\n")
        print(f"{m}: {len(frames)} trading days, {month['symbol'].nunique()} stocks, "
              f"{0 if fo is None else len(parse_fo(fo))} with futures on {last.date()}",
              file=sys.stderr)


# ==============================================================================
# Build the point-in-time list
# ==============================================================================
def load_cm(columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted((NSE_DIR / "cm").glob("*.parquet"))
    if not files:
        sys.exit("No NSE files yet: run `python nse.py download` first.")
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files), ignore_index=True)


def link_ids(rows: pd.DataFrame) -> pd.Series:
    """Map each symbol to one stock id (its latest symbol), joining symbols that
    share an ISIN (renames) and ISINs that share a symbol (splits)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = rows[["symbol", "isin"]].drop_duplicates()
    for s, i in pairs.itertuples(index=False):
        parent[find("S:" + s)] = find("I:" + i)
    last_seen = rows.groupby("symbol")["date"].max()
    groups: dict[str, list[str]] = {}
    for s in last_seen.index:
        groups.setdefault(find("S:" + s), []).append(s)
    return pd.Series({s: max(syms, key=lambda x: last_seen[x])
                      for syms in groups.values() for s in syms})


def build(top: int = 50, lookback: int = 126, min_share: float = 0.9) -> pd.DataFrame:
    rows = load_cm(["date", "symbol", "series", "isin", "value"])
    stock_id = link_ids(rows)
    eq = rows[rows["series"] == "EQ"].assign(id=lambda d: d["symbol"].map(stock_id))
    value = eq.pivot_table(index="date", columns="id", values="value", aggfunc="sum")
    med = value.fillna(0).rolling(lookback, min_periods=lookback).median()
    traded = value.notna().rolling(lookback, min_periods=lookback).mean()

    out = []
    month_ends = value.index.to_series().groupby(value.index.to_period("M")).max()
    for m, d in month_ends.items():
        fo_path = NSE_DIR / "fo" / f"{m}.txt"
        if not fo_path.exists() or pd.isna(med.loc[d]).all():
            continue
        futures = {stock_id.get(s, s) for s in fo_path.read_text().split()}
        ok = med.loc[d][(traded.loc[d] >= min_share) & med.loc[d].index.isin(futures)]
        for rank, (sid, v) in enumerate(ok.sort_values(ascending=False).head(top).items(), 1):
            out.append({"date": d.date(), "rank": rank, "stock": sid,
                        "median_value_cr": round(v / 1e7, 1)})
    uni = pd.DataFrame(out)
    uni.to_csv(UNIVERSE_CSV, index=False)
    return uni


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_CSV.exists():
        sys.exit("No point-in-time list yet: run `python nse.py download` and "
                 "`python nse.py build` first.")
    return pd.read_csv(UNIVERSE_CSV, parse_dates=["date"])


def members_by_date(uni: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    return {d: set(g["stock"]) for d, g in uni.groupby("date")}


# Price ratios a bonus or split produces overnight: 1:1 bonus or 2-for-1 split = 1/2,
# 2:1 bonus = 1/3, 1:2 bonus = 2/3, 3:2 bonus = 2/5, 1:4 split = 1/4, and so on.
SPLIT_RATIOS = (1 / 2, 1 / 3, 2 / 3, 1 / 4, 1 / 5, 2 / 5, 1 / 10)


@lru_cache(maxsize=1)
def _price_rows() -> tuple[pd.DataFrame, pd.Series]:
    rows = load_cm(["date", "symbol", "isin", "series", "open", "close"])
    return rows, link_ids(rows)


def price_series(stock: str, tol: float = 0.02) -> tuple[pd.DataFrame, list[str]]:
    """Daily Open/Close for a stock Yahoo no longer carries, from NSE's files,
    adjusted for bonuses and splits, plus a note for every adjustment made.

    NSE's prices are not adjusted, so a 1:1 bonus looks like a 50% crash. An
    overnight drop is treated as a bonus or split when it is within `tol` of one
    of SPLIT_RATIOS and either the ISIN changed that day (splits change it) or
    the price held at the new level for the next 5 days; anything else is left as
    a real move. Dividends are not added back, so returns are slightly too low."""
    rows, ids = _price_rows()
    mine = rows[rows["symbol"].map(ids) == stock].sort_values("date")
    mine = mine.drop_duplicates("date", keep="first").set_index("date")
    if mine.empty:
        raise ValueError(f"{stock} is not in NSE's files")
    o, c = mine["open"].astype(float), mine["close"].astype(float)
    factor = pd.Series(1.0, mine.index)
    notes = []
    gap = (o / c.shift()).to_numpy()
    for i in range(1, len(mine)):
        near = [r for r in SPLIT_RATIOS if abs(gap[i] / r - 1) < tol]
        if not near:
            continue
        new_isin = mine["isin"].iloc[i] != mine["isin"].iloc[i - 1]
        held = (c.iloc[i:i + 5] / c.iloc[i - 1] < near[0] * 1.25).all()
        if new_isin or held:
            factor.iloc[:i] *= near[0]
            notes.append(f"{stock} {mine.index[i].date()}: price x{near[0]:.3f} overnight "
                         f"({'ISIN changed' if new_isin else 'held at the new level'}) "
                         f"-> treated as a bonus/split")
    out = pd.DataFrame({"Open": o * factor, "Close": c * factor})
    return out, notes


def check() -> None:
    uni = load_universe()
    latest = uni[uni["date"] == uni["date"].max()]
    raw = _fetch(f"{ARCHIVE}/indices/ind_nifty50list.csv")
    nifty = set(pd.read_csv(io.BytesIO(raw))["Symbol"]) if raw else None
    if nifty is None:   # the list is a plain CSV, not a zip
        req = urllib.request.Request(f"{ARCHIVE}/indices/ind_nifty50list.csv",
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=40) as r:
            nifty = set(pd.read_csv(r)["Symbol"])
    ours = set(latest["stock"])
    print(f"Proxy list on {latest['date'].iloc[0].date()} vs today's Nifty 50: "
          f"{len(ours & nifty)} of {len(nifty)} in common")
    print("  in the Nifty, not in the proxy:", ", ".join(sorted(nifty - ours)))
    print("  in the proxy, not in the Nifty:", ", ".join(sorted(ours - nifty)))
    ever = uni.groupby("stock")["date"].agg(["min", "max", "count"])
    print(f"\n{len(ever)} different stocks were in the proxy at some month end "
          f"({uni['date'].min().date()} -> {uni['date'].max().date()}).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--start", default="2016-01-01")
    dl.add_argument("--end", default=str(pd.Timestamp.now(tz="Asia/Kolkata").date()))
    dl.add_argument("--workers", type=int, default=4)
    b = sub.add_parser("build")
    b.add_argument("--top", type=int, default=50)
    sub.add_parser("check")
    args = ap.parse_args()
    if args.cmd == "download":
        download(pd.Timestamp(args.start), pd.Timestamp(args.end), args.workers)
    elif args.cmd == "build":
        uni = build(args.top)
        print(f"{uni['date'].nunique()} month ends, {uni['stock'].nunique()} different "
              f"stocks -> {UNIVERSE_CSV}")
    else:
        check()


if __name__ == "__main__":
    main()
