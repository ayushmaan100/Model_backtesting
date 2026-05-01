# config.py
from datetime import date

USE_REAL_DATA    = True   # True = real data on your machine

PORTFOLIO_SIZE   = 25
NIFTY_TICKER     = "^NSEI"
# Backtest aligned with PDF analysis window ("performance since February 2020").
# First rebalance is INCEPTION_DATE; subsequent rebalances follow REBALANCE_MONTHS.
INCEPTION_DATE   = date(2020, 2, 1)
BACKTEST_START   = INCEPTION_DATE
BACKTEST_END     = date(2026, 4, 1)
REBALANCE_MONTHS = [3, 9]    # March + September — aligned with NSE 200 reconstitution
# Skip any scheduled rebalance closer than this many months to inception.
# Prevents the back-to-back Feb 1 + Mar 1 trade observed in the previous run
# (₹2,174 wasted on inception trickery).
MIN_MONTHS_AFTER_INCEPTION = 4

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

# Warn if the cached price file is older than this many days.
CACHE_MAX_AGE_DAYS = 30

# ── DYNAMIC UNIVERSE (Option 3: size-based eligibility) ────────────────────
# Instead of relying on a fixed NSE 200 list (survivorship bias),
# at each rebalance date we include any stock with:
#   1. Total Assets >= MIN_TOTAL_ASSETS_CR (size filter)
#   2. Fundamental data not older than MAX_FUND_AGE_YEARS (freshness filter)
#   3. At least 13 months of price history (momentum filter — always required)
#
# This means a stock that shrinks below the threshold naturally exits,
# and a stock that grows above it naturally enters — no fixed list needed.
MIN_TOTAL_ASSETS_CR  = 500     # ₹500 Cr minimum total assets (covers mid+large caps)
MAX_FUND_AGE_YEARS   = 2       # Reject fundamentals older than 2 years

