# config.py
from datetime import date

USE_REAL_DATA    = True   # True = real data on your machine

PORTFOLIO_SIZE   = 25
NIFTY_TICKER     = "^NSEI"
BACKTEST_START   = date(2020, 1, 1)
BACKTEST_END     = date(2026, 4, 1)
REBALANCE_MONTHS = [6, 12]    # June + December

WEIGHTS = {
    "Momentum": 0.30, "Quality": 0.20, "Value": 0.15,
    "Size":     0.12, "Beta":   0.10, "Invest": 0.07, "Yield": 0.06,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

LOWER_IS_BETTER  = {"Size", "Invest"}
MOM_W_A          = 0.40
MOM_W_B          = 0.60
MOM_SKIP         = 1
MOM_Q1           = 3
MOM_Q24          = 8

TRANSACTION_COST_PCT = 0.003
RISK_FREE_RATE       = 0.065

PRICE_CSV = "prices.csv"
FUND_CSV  = "fundamentals.csv"
