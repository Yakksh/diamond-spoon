"""Offline tests for nse.py and the point-in-time parts of rotation.py."""

import numpy as np
import pandas as pd
import pytest

import nse
import rotation as rot
from test_engine import frames, walk


def test_renames_and_splits_link_to_the_latest_symbol():
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-02", "2020-01-02", "2017-01-02", "2020-01-02",
                                "2020-01-02"]),
        "symbol": ["OLDNAME", "NEWNAME", "SPLITCO", "SPLITCO", "OTHER"],
        "isin": ["INE1", "INE1", "INE2", "INE3", "INE4"],      # rename keeps ISIN; split keeps symbol
    })
    ids = nse.link_ids(rows)
    assert ids["OLDNAME"] == ids["NEWNAME"] == "NEWNAME"
    assert ids["SPLITCO"] == "SPLITCO"
    assert ids["OTHER"] == "OTHER"


def test_both_file_formats_parse_the_same():
    d = pd.Timestamp("2024-10-28")
    old = pd.DataFrame({"SYMBOL": ["A", "B"], "SERIES": ["EQ", "N1"], "OPEN": [10, 1],
                        "HIGH": [11, 1], "LOW": [9, 1], "CLOSE": [10.5, 1], "LAST": [10.5, 1],
                        "PREVCLOSE": [10, 1], "TOTTRDQTY": [5, 1], "TOTTRDVAL": [52.5, 1],
                        "TIMESTAMP": ["x", "x"], "TOTALTRADES": [1, 1], "ISIN": ["I1", "I2"]})
    new = pd.DataFrame({"TckrSymb": ["A", "B"], "SctySrs": ["EQ", "N1"], "ISIN": ["I1", "I2"],
                        "OpnPric": [10, 1], "ClsPric": [10.5, 1], "TtlTrfVal": [52.5, 1],
                        "HghPric": [11, 1], "LwPric": [9, 1], "PrvsClsgPric": [10, 1]})
    a, b = nse.parse_cm(old, d), nse.parse_cm(new, d)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert a["symbol"].tolist() == ["A"]                  # bond-like series dropped


def test_build_ranks_by_liquidity_among_stocks_with_futures(tmp_path, monkeypatch):
    monkeypatch.setattr(nse, "NSE_DIR", tmp_path)
    monkeypatch.setattr(nse, "UNIVERSE_CSV", tmp_path / "universe.csv")
    (tmp_path / "cm").mkdir()
    (tmp_path / "fo").mkdir()
    days = pd.bdate_range("2020-01-01", periods=150)
    values = {"BIG": 9e9, "MID": 5e9, "SMALL": 1e9, "NOFUT": 20e9}
    rows = pd.DataFrame([{"date": d, "symbol": s, "series": "EQ", "isin": "I" + s,
                          "open": 1.0, "close": 1.0, "value": v}
                         for d in days for s, v in values.items()])
    for m, g in rows.groupby(rows["date"].dt.to_period("M")):
        g.to_parquet(tmp_path / "cm" / f"{m}.parquet", index=False)
        (tmp_path / "fo" / f"{m}.txt").write_text("BIG\nMID\nSMALL\n")
    uni = nse.build(top=2)
    last = uni[uni["date"] == uni["date"].max()]
    assert last["stock"].tolist() == ["BIG", "MID"]       # NOFUT has no futures
    assert uni["date"].min() >= days[125].date()          # needs 6 months of history


def test_a_stock_that_stops_trading_is_sold_at_its_last_close():
    a, b = walk(300, seed=1), walk(300, seed=2)
    b.iloc[200:] = np.nan                                 # delisted after row 199
    opens, closes = frames({"A": a, "B": b})
    res = rot.run(opens, closes, 0, 299, {1}, rot.buy_once(), 100_000, 0.0)

    book = rot.Book(100_000, 0.0)                         # the same trades by hand
    book.buy("A", 50_000, a["Open"].iloc[1], a.index[1])
    book.buy("B", 50_000, b["Open"].iloc[1], b.index[1])
    book.sell("B", b["Close"].iloc[199], b.index[199])
    book.sell("A", a["Close"].iloc[299], a.index[299])
    book.pay_tax(rot.sb.fy_of(a.index[299]) + 1, a.index[299], closes.iloc[299])
    years = (a.index[299] - a.index[0]).days / 365.25
    expected = 100 * ((book.cash / 100_000) ** (1 / years) - 1)
    assert res["CAGR % (post-tax)"] == pytest.approx(expected, abs=1e-9)


def test_list_changes_are_followed_each_month():
    dfs = {n: walk(100, seed=i) for i, n in enumerate("ABC")}
    opens, closes = frames(dfs)
    switch = closes.index[50]

    def members(row):
        return {"A", "B"} if closes.index[row] < switch else {"B", "C"}

    decide = rot.hold_members(members)
    assert decide(10, [], {"A", "B", "C"}) == ["A", "B"]
    assert decide(60, ["A", "B"], {"A", "B", "C"}) == ["B", "C"]
    mom, _ = rot.momentum_decider(closes, top=3, exit_rank=3, lookback=20, skip=0,
                                  members=members)
    assert set(mom(60, ["A"], {"A", "B", "C"})) == {"B", "C"}   # A left the list: sold


def test_bonus_and_split_are_adjusted_but_a_crash_that_rebounds_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(nse, "NSE_DIR", tmp_path)
    nse._price_rows.cache_clear()
    (tmp_path / "cm").mkdir()
    days = pd.bdate_range("2020-01-01", periods=40)
    rows = []
    for i, d in enumerate(days):
        after = i >= 20
        rows += [  # 1:1 bonus: halves, ISIN unchanged, stays down
            {"symbol": "BONUSCO", "isin": "I1", "px": 50.0 if after else 100.0},
            # 1-for-5 split: a fifth, new ISIN
            {"symbol": "SPLITCO", "isin": "I3" if after else "I2", "px": 20.0 if after else 100.0},
            # halves for a day, then back near 90: a real move, not a bonus
            {"symbol": "CRASHCO", "isin": "I4", "px": 50.0 if i == 20 else (90.0 if after else 100.0)},
            # demerger: opens 45% lower and stays there
            {"symbol": "DEMERGCO", "isin": "I5", "px": 55.0 if after else 100.0},
        ]
        for r in rows[-4:]:
            r.update(date=d, series="EQ", open=r["px"], close=r["px"])
    df = pd.DataFrame(rows).drop(columns="px")
    df.to_parquet(tmp_path / "cm" / "2020-01.parquet", index=False)

    bonus, notes = nse.price_series("BONUSCO")
    assert bonus["Close"].nunique() == 1 and len(notes) == 1          # one flat line at 50
    split, notes = nse.price_series("SPLITCO")
    assert split["Close"].nunique() == 1 and "ISIN changed" in notes[0]
    crash, notes = nse.price_series("CRASHCO")
    assert notes == [] and crash["Close"].min() == 50.0
    demerged, notes = nse.price_series("DEMERGCO")
    assert np.allclose(demerged["Close"], 55.0) and "demerger" in notes[0]
    nse._price_rows.cache_clear()
