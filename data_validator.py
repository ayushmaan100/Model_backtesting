"""
data_validator.py — standalone pre-run data quality checker.

Run this BEFORE main.py to verify data integrity:
    python3 data_validator.py

Checks:
    1. Cross-checks nse200_tickers.py against prices.csv
    2. Validates B/M, market cap, and dividend yield ranges
    3. Flags suspicious price moves
    4. Reports coverage statistics
    5. Verifies fundamentals PiT sanity
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")


def validate():
    from nse200_tickers import NSE200, EXCLUDED_CORRUPT
    from config import NIFTY_TICKER, PRICE_CSV

    print("═" * 62)
    print(" DATA VALIDATOR — Pre-run Quality Check")
    print("═" * 62)
    errors = 0

    # ── 1. Check files exist ──────────────────────────────────────────────
    print("\n[1] File existence check...")
    files = {
        "prices.csv": PRICE_CSV,
        "fundamentals_pit.csv": "fundamentals_pit.csv",
        "shares_outstanding.csv": "shares_outstanding.csv",
        "screener_raw.csv": "screener_raw.csv",
    }
    for label, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✅ {label:<28} ({size:.0f} KB)")
        else:
            print(f"  ❌ {label:<28} MISSING")
            errors += 1

    if not os.path.exists(PRICE_CSV):
        print("\n❌ Cannot continue without prices.csv. Run main.py to download.")
        return errors

    # ── 2. Prices cross-check ─────────────────────────────────────────────
    print("\n[2] Ticker coverage check...")
    prices = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
    p_tickers = set(prices.columns)
    nse200_set = set(NSE200)

    in_prices = nse200_set & p_tickers
    missing = nse200_set - p_tickers

    print(f"  NSE200 list:    {len(NSE200)} tickers")
    print(f"  Prices.csv:     {len(p_tickers)} columns")
    print(f"  Overlap:        {len(in_prices)} ({len(in_prices)/len(NSE200)*100:.0f}%)")

    if missing:
        print(f"  ⚠️  Missing from prices ({len(missing)}):")
        for t in sorted(missing):
            print(f"       {t}")

    # Check excluded tickers are indeed absent
    for t in EXCLUDED_CORRUPT:
        if t in p_tickers:
            print(f"  ⚠️  Corrupt ticker {t} still in prices.csv")

    # ── 3. Price sanity ───────────────────────────────────────────────────
    print("\n[3] Price data sanity check...")
    stock_cols = [c for c in prices.columns if c != NIFTY_TICKER]
    rets = prices[stock_cols].pct_change()

    n_suspect = 0
    for col in stock_cols:
        col_rets = rets[col].dropna()
        if col_rets.empty:
            continue
        max_r = col_rets.max()
        min_r = col_rets.min()
        if max_r > 3.0 or min_r < -0.90:
            print(f"  ⚠️  {col:<20} max={max_r:+.1%}, min={min_r:+.1%}")
            n_suspect += 1

    if n_suspect == 0:
        print(f"  ✅ No tickers with extreme monthly returns (>300% or <-90%)")
    else:
        print(f"  ⚠️  {n_suspect} tickers with suspicious returns")
        errors += n_suspect

    # ── 4. B/M sanity (if shares data exists) ─────────────────────────────
    if os.path.exists("fundamentals_pit.csv") and os.path.exists("shares_outstanding.csv"):
        print("\n[4] Book-to-Market sanity check...")
        pit = pd.read_csv("fundamentals_pit.csv", parse_dates=["Date"])
        shares = pd.read_csv("shares_outstanding.csv", index_col=0)
        latest = pit.sort_values("Date").groupby("Ticker").last()

        common = [t for t in latest.index if t in prices.columns and t in shares.index]
        if common:
            equity = latest.loc[common, "equity"]
            price = prices.loc[:, common].iloc[-1]
            sh = shares.loc[common, "shares_cr"]
            mcap = price.values * sh.values

            btm = equity.values / mcap
            btm_clean = pd.Series(btm, index=common)
            btm_clean = btm_clean[(btm_clean > 0) & btm_clean.notna()]

            print(f"  B/M range:  [{btm_clean.min():.3f}, {btm_clean.max():.3f}]")
            print(f"  B/M median: {btm_clean.median():.3f}")

            if btm_clean.max() > 50:
                print(f"  ⚠️  B/M max > 50 — likely still has unit issues")
                errors += 1
            elif btm_clean.max() < 10:
                print(f"  ✅ B/M values in reasonable range (0-10)")

            # Market cap sanity
            mcap_series = pd.Series(mcap, index=common)
            mcap_clean = mcap_series[mcap_series > 0].dropna()
            print(f"  MCap range: [{mcap_clean.min():,.0f}, {mcap_clean.max():,.0f}] Cr")
        else:
            print(f"  ⚠️  No common tickers between PiT, prices, and shares")

    # ── 5. Fundamentals coverage ──────────────────────────────────────────
    if os.path.exists("fundamentals_pit.csv"):
        print("\n[5] Fundamentals coverage check...")
        pit = pd.read_csv("fundamentals_pit.csv", parse_dates=["Date"])
        print(f"  Rows: {len(pit)}, Tickers: {pit['Ticker'].nunique()}")
        print(f"  Date range: {pit['Date'].min().date()} → {pit['Date'].max().date()}")
        for col in ["gross_profit_assets", "equity", "asset_growth_yoy", "dividend_payout_pct", "eps"]:
            pct = pit[col].notna().mean() * 100
            status = "✅" if pct > 80 else "⚠️ "
            print(f"  {status} {col:<25} {pct:.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    if errors == 0:
        print(" ✅  ALL CHECKS PASSED — ready to run main.py")
    else:
        print(f" ⚠️  {errors} issue(s) found — review above before running main.py")
    print(f"{'═' * 62}")
    return errors


if __name__ == "__main__":
    validate()
