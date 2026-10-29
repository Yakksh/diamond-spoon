"""Offline tests for odds.py."""

import numpy as np
import pandas as pd
import pytest

import odds
import swing_backtest as sb
from test_engine import walk


def arr(*rows):
    return np.array(rows, dtype=float)


def test_limit_fills_at_the_limit_or_a_gap_below_it_else_buys_at_the_end():
    o = arr([100, 98, 94, 97], [100, 90, 94, 97], [100, 99, 99, 99])
    lo = arr([99, 95, 93, 96], [99, 89, 93, 96], [97, 97, 97, 97])
    c = arr([99.5, 96, 95, 98], [99.5, 96, 95, 98], [99, 99, 99, 102])
    filled, ratio = odds.round_trip(100.0, o, lo, c, 0.05, cost=0.0, tax_frac=0.0)
    assert filled.tolist() == [True, True, False]
    assert ratio == pytest.approx([100 / 95, 100 / 90, 100 / 102])


def test_costs_and_tax_shrink_the_shares_bought_back():
    o, lo, c = arr([100, 100]), arr([100, 100]), arr([100, 100])
    _, ratio = odds.round_trip(100.0, o, lo, c, 0.05, cost=0.001, tax_frac=0.02)
    assert ratio[0] == pytest.approx(0.999 * 0.98 / 1.001)


def test_history_windows_line_up_with_the_data():
    df = walk(300, seed=2)
    start, o, lo, c, t = odds.history_windows(df, 10, df.index[0])
    i = 37
    k = t[i]
    assert start[i] == df["Close"].iloc[k]
    assert o[i, 0] == df["Open"].iloc[k + 1]
    assert c[i, -1] == df["Close"].iloc[k + 10]
    assert t[-1] == len(df) - 11            # the last start still has 10 days after it


def test_simulation_paths_are_continuous_and_no_drift_is_flat():
    df = walk(2000, seed=4, drift=0.002)                  # strongly rising history
    ratios = odds.day_ratios(df)
    rng = np.random.default_rng(0)
    o, lo, c = odds.simulate(ratios, 100.0, 20, 4000, 10, rng, drift=False)
    assert o.shape == c.shape == (4000, 20)
    assert (lo <= np.maximum(o, c) + 1e-9).all()
    assert np.log(c[:, -1] / 100).mean() == pytest.approx(0, abs=0.01)
    _, _, c2 = odds.simulate(ratios, 100.0, 20, 4000, 10, rng, drift=True)
    assert np.log(c2[:, -1] / 100).mean() == pytest.approx(20 * 0.002, abs=0.01)


def test_tax_fraction_long_term_with_exemption_short_term_and_loss():
    today = pd.Timestamp("2026-10-29")
    frac, _ = odds.tax_fraction(1000, 500, pd.Timestamp("2020-01-01"), 1000,
                                sb.LTCG_EXEMPT, 0.0, today)
    per_share = sb.LTCG_RATE * (500 * 1000 - sb.LTCG_EXEMPT) / 1000 * (1 + sb.CESS)
    assert frac == pytest.approx(per_share / 1000)
    frac, _ = odds.tax_fraction(1000, 500, pd.Timestamp("2026-06-01"), None,
                                sb.LTCG_EXEMPT, 0.0, today)
    assert frac == pytest.approx(sb.STCG_RATE * 500 * (1 + sb.CESS) / 1000)
    frac, _ = odds.tax_fraction(1000, 1200, pd.Timestamp("2020-01-01"), 10,
                                sb.LTCG_EXEMPT, 0.0, today)
    assert frac == 0
