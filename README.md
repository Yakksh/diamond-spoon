# diamond-spoon

Honest backtests for Indian stocks (NSE): can a rule tell you when to buy, when
to sell, or which stocks to hold, after costs and Indian capital-gains tax?

```bash
pip install -r requirements.txt
python swing_backtest.py RELIANCE.NS --years 10 --walkforward   # one stock, every rule
python evaluate.py --cash-yield 0.06                            # every rule, 60 stocks
python rotation.py --top 10 --exit-rank 20                      # momentum rotation
python -m pytest                                                # offline tests
```

| file | what it answers |
| --- | --- |
| `swing_backtest.py` | One stock: how did each rule do against buy & hold, and does it survive a walk-forward test? Holds the engine, the rules and the tax model. |
| `evaluate.py` | Many stocks: how often does each rule beat buy & hold, does a buy or sell signal tell you anything about the price 1-6 months later, and when a rule sold, was the buyback cheaper? |
| `rotation.py` | Instead of timing one stock, hold the strongest N each month (12-1 momentum), against equal-weight buy & hold of the same list and the NIFTYBEES ETF. |
| `universe.py` | The stock lists and a disk cache of Yahoo prices (`data/`, not committed). |

## What the tests have shown so far (data to 28 Oct 2026)

- **Timing a single stock lost.** Across 60 large caps over 10 years, no rule
  beat buy & hold on a typical stock after tax: the best (200-day trend) did so
  on 12 of 60 and cost a median 5.5 points of CAGR a year, in return for a
  smaller worst fall.
- **The signals carried no information.** Three months after a sell signal the
  price was lower 39-41% of the time, the same as after any random day (39.7%).
- **Selling to buy back cheaper mostly failed.** Of ~4,000 sell-then-rebuy
  cycles, the buyback was cheaper 15-31% of the time for the trend and swing
  rules; the median buyback was about 5% above the sale.
- **Momentum rotation was mixed.** Every version beat the NIFTYBEES ETF
  (10.4% a year after tax) by a wide margin, but so did plain equal-weight buy
  & hold of the same list (16.2%), so most of that gap is the list, not
  momentum. Against the list, results ranged from slightly worse to about
  2-3 points better depending on settings. The best-looking version (top 10,
  NIFTYBEES 200-day market filter) beat it in both 5-year halves with half the
  worst fall, but that rests on a few market slides and was chosen after
  seeing results.

## Known limits

- **Survivorship bias.** The stock lists are today's Nifty names plus a
  hand-picked set of fallen ones, not the index as it stood on each date.
- **Yahoo data is free and unofficial.** Bars that jump 1.8x or more and snap
  back within 3 days are dropped (NIFTYBEES had two at a tenth of the price
  in Dec 2019). Pre-2006 history has more junk than the filter catches.
- **Tax** uses today's rates (FY 2026-27) for the whole history, ignores the
  8-year carry-forward limit, and the single-stock engine charges it all at
  the end rather than every year. Interest on idle cash is not taxed.
- Not investment advice. A rule that worked in the past can stop working.
