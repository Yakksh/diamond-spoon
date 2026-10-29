# diamond-spoon

Honest backtests for Indian stocks (NSE): can a rule tell you when to buy, when
to sell, or which stocks to hold, after costs and Indian capital-gains tax?

```bash
pip install -r requirements.txt
python swing_backtest.py RELIANCE.NS --years 10 --walkforward   # one stock, every rule
python evaluate.py --cash-yield 0.06                            # every rule, 60 stocks
python rotation.py --top 10 --exit-rank 20                      # momentum rotation
python odds.py RELIANCE --cost-basis 900 --bought 2021-03-15 --qty 100   # sell now, buy back lower?
python nse.py download && python nse.py build                   # point-in-time stock list
python rotation.py --pit --top 10 --exit-rank 20                # rotation on that list
python -m pytest                                                # offline tests
```

| file | what it answers |
| --- | --- |
| `swing_backtest.py` | One stock: how did each rule do against buy & hold, and does it survive a walk-forward test? Holds the engine, the rules and the tax model. |
| `evaluate.py` | Many stocks: how often does each rule beat buy & hold, does a buy or sell signal tell you anything about the price 1-6 months later, and when a rule sold, was the buyback cheaper? |
| `rotation.py` | Instead of timing one stock, hold the strongest N each month (12-1 momentum), against equal-weight buy & hold of the same list and the NIFTYBEES ETF. |
| `odds.py` | If I sell now, what are the odds I can buy back cheaper? Tests "sell at the next open, leave a limit order X% lower for N days, else buy back on day N" four ways: simulated from the stock's own past days (with and without its past drift), on every real past day, and on past days that looked like today. Includes the tax on selling. |
| `nse.py` | Downloads NSE's daily files (every listed stock, including later delisted ones) and builds a point-in-time list: each month, the 50 most traded stocks among those with futures. Removes the survivorship bias of a hand-made list. Rebuilds prices from NSE's files for stocks Yahoo no longer carries. |
| `universe.py` | The hand-made stock lists and a disk cache of Yahoo prices (`data/`, not committed). |

## What the tests have shown so far (data to 28 Oct 2026)

- **Timing a single stock lost.** Across 60 large caps over 10 years, no rule
  beat buy & hold on a typical stock after tax: the best (200-day trend) did so
  on 12 of 60 and cost a median 5.7 points of CAGR a year, in return for a
  smaller worst fall.
- **The signals carried no information.** Three months after a sell signal the
  price was lower 39-41% of the time, the same as after any random day (39.7%).
- **Selling to buy back cheaper mostly failed.** Of ~4,000 sell-then-rebuy
  cycles, the buyback was cheaper 15-31% of the time for the trend and swing
  rules; the median buyback was about 5% above the sale. `odds.py` agrees:
  "sell now, buy back 5-20% lower within 3 months" lost shares on average on
  50-53 of the 60 stocks, and on RELIANCE in three of its four views.
- **The hand-made list was flattering by about 6 points a year.** Equal-weight
  buy & hold of the 60 hand-picked stocks earned 16.6% a year after tax; of the
  point-in-time top 100 (`nse.py`, 234 different stocks over the decade,
  delisted ones included), 10.7%; NIFTYBEES, 10.4%.
- **Momentum rotation showed no reliable edge once that bias was gone.** On the
  point-in-time top 100, 10 years after tax:

  | rule | CAGR | worst fall | 2016-21 | 2021-26 |
  | --- | --- | --- | --- | --- |
  | top 10 (fixed in advance) | 9.5% | -58% | 3.1% | 14.7% |
  | top 10, keep while in top 20 | 12.4% | -50% | 6.1% | 17.1% |
  | same + NIFTYBEES 200-day filter | 11.5% | -31% | 4.8% | 17.3% |
  | top 20, keep while in top 40 | 10.8% | -55% | 7.2% | 14.7% |
  | equal-weight top 100 | 10.7% | -48% | 11.4% | 10.8% |
  | NIFTYBEES | 10.4% | -36% | 15.5% | 4.1% |

  Momentum lost badly in the first five years and won in the last five, with
  deeper falls throughout; on the top-50 list the fixed-in-advance version
  made 5.9%. Small changes in settings moved results by 4-5 points, which is
  the sign of luck, not an edge.

## Known limits

- **The point-in-time list is a proxy.** NSE does not publish past Nifty
  memberships in a usable form, so the list ranks stocks with futures by
  6-month median traded value. That favours heavily traded names over quiet
  large caps: today's top 100 shares 72 names with the Nifty 100. NSE refuses
  files before 2016.
- **Corporate actions are inferred from prices.** Overnight drops of 30%+ that
  stay down are treated as demergers or unadjusted bonuses (TMPV, STAR, ABREL,
  VEDL, TRENT, and PEL, SINTEX, MINDTREE in NSE's files) and printed when it
  happens. Smaller ones (CONCOR's 1:4 bonus, -20%) are missed, and the 14
  stocks rebuilt from NSE's files have no dividends.
- **Yahoo data is free and unofficial.** Bars that jump 1.8x or more and snap
  back within 3 days are dropped (NIFTYBEES had two at a tenth of the price
  in Dec 2019). Pre-2006 history is full of junk; no test uses it.
- **Tax** uses today's rates (FY 2026-27) for the whole history, ignores the
  8-year carry-forward limit, and the single-stock engine charges it all at
  the end rather than every year. Interest on idle cash is not taxed.
- Not investment advice. A rule that worked in the past can stop working.
