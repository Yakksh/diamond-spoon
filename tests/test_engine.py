"""Offline tests: synthetic prices only, no network."""

import numpy as np
import pandas as pd
import pytest

import rotation as rot
import swing_backtest as sb


def walk(n=600, seed=0, drift=0.0004, vol=0.015, start="2018-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    opn = close * np.exp(rng.normal(0, vol / 3, n))
    hi = np.maximum(opn, close) * (1 + rng.uniform(0, vol, n))
    lo = np.minimum(opn, close) * (1 - rng.uniform(0, vol, n))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close}, idx)


# --- data hygiene ----------------------------------------------------------------
def test_live_bar_dropped_only_while_market_is_open():
    df = walk(5, start="2026-10-26")                      # last bar: Fri 2026-10-30
    ist = "Asia/Kolkata"
    assert len(sb.drop_live_bar(df, pd.Timestamp("2026-10-30 13:15", tz=ist))) == 4
    assert len(sb.drop_live_bar(df, pd.Timestamp("2026-10-30 16:05", tz=ist))) == 5
    assert len(sb.drop_live_bar(df, pd.Timestamp("2026-10-31 10:00", tz=ist))) == 5


def test_bad_ticks_removed_but_real_crash_kept():
    df = walk(60)
    glitch = df.copy()
    glitch.iloc[20:22] /= 10                              # NIFTYBEES-style tenth-price bars
    out = sb.drop_bad_ticks(glitch)
    assert len(out) == 58 and df.index[20] not in out.index
    crash = df.copy()
    crash.iloc[30:] /= 2                                  # halves and stays down
    assert len(sb.drop_bad_ticks(crash)) == 60


# --- tax -------------------------------------------------------------------------
D1, D2 = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-06-01")   # FY 2025, FY 2026
TAX = 1 + sb.CESS


def test_short_term_loss_offsets_long_term_gain():
    owed = sb.tax_by_fy([(D1, -50_000, False), (D1, 300_000, True)])
    assert owed[2025] == pytest.approx(0.125 * (250_000 - 125_000) * TAX)


def test_long_term_loss_does_not_offset_short_term_gain_but_carries_forward():
    owed = sb.tax_by_fy([(D1, -100_000, True), (D1, 100_000, False), (D2, 300_000, True)])
    assert owed[2025] == pytest.approx(0.20 * 100_000 * TAX)
    assert owed[2026] == pytest.approx(0.125 * (200_000 - 125_000) * TAX)


def test_short_term_loss_carried_forward_can_offset_long_term_gain():
    owed = sb.tax_by_fy([(D1, -50_000, False), (D2, 200_000, True)])
    assert owed[2025] == 0
    assert owed[2026] == pytest.approx(0.125 * (150_000 - 125_000) * TAX)


# --- engine ----------------------------------------------------------------------
def test_idle_cash_earns_the_yield():
    df = walk(253)
    r = sb.backtest(df, pd.Series(0.0, df.index), "flat", capital=100_000, cash_yield=0.06)
    assert r.equity.iloc[-1] == pytest.approx(106_000, rel=1e-9)


@pytest.mark.parametrize("name", list(sb.STRATS))
def test_strategies_never_look_ahead(name):
    fn, knob, _grid, default = sb.STRATS[name]
    df = walk(700, seed=3)
    full = fn(df, **{knob: default})
    for cut in (250, 400, 699):
        part = fn(df.iloc[:cut], **{knob: default})
        pd.testing.assert_series_equal(full.iloc[:cut], part)


# --- rotation ----------------------------------------------------------------------
def test_book_sells_oldest_lot_first():
    book = rot.Book(1_000, cost=0.0)
    book.buy("X", 100, 10.0, pd.Timestamp("2024-01-01"))   # 10 shares at 10
    book.buy("X", 200, 20.0, pd.Timestamp("2025-03-01"))   # 10 shares at 20
    book.sell("X", 30.0, pd.Timestamp("2025-06-01"), shares=15)
    (d1, g1, lt1), (d2, g2, lt2) = book.realized
    assert (g1, lt1) == (pytest.approx(200), True)          # 10 x (30 - 10), held > 1 year
    assert (g2, lt2) == (pytest.approx(50), False)          # 5 x (30 - 20), held 3 months
    assert book.shares("X") == pytest.approx(5)


def frames(dfs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = pd.DataFrame({n: d["Close"] for n, d in dfs.items()})
    opens = pd.DataFrame({n: d["Open"] for n, d in dfs.items()})
    return opens, closes


def test_buy_and_hold_agrees_with_single_stock_engine():
    df = walk(800, seed=5)
    opens, closes = frames({"X": df})
    res = rot.run(opens, closes, 0, len(df) - 1, {1}, rot.buy_once(), 100_000, 0.0015)
    single = sb.backtest(df, sb.strat_buyhold(df), "bh", capital=100_000).summary()
    assert res["CAGR % (post-tax)"] == pytest.approx(single["CAGR % (post-tax)"], abs=1e-9)


def test_momentum_holds_the_leader_and_ignores_the_future():
    dfs = {f"S{i}": walk(600, seed=10 + i, drift=0.0) for i in range(6)}
    dfs["LEADER"] = walk(600, seed=99, drift=0.003)
    opens, closes = frames(dfs)
    decide, _ = rot.momentum_decider(closes, top=2, exit_rank=2, lookback=252, skip=21)
    assert "LEADER" in decide(400, [], set(closes.columns))
    # The decision on row 400 must not change if everything after it is different.
    cut_decide, _ = rot.momentum_decider(closes.iloc[:401], top=2, exit_rank=2,
                                         lookback=252, skip=21)
    assert decide(400, ["S1"], set(closes.columns)) == cut_decide(400, ["S1"], set(closes.columns))
