"""
diagnose.py
───────────────────────────────────────────────────────────────────────────────
Run this BEFORE main.py to test exactly which data endpoints work on your
machine. Takes ~3 minutes. Saves you from a 20-minute failed download.

Usage:
    python3 diagnose.py

What it tests (for 5 sample stocks):
    ✅ yf.download()       → prices (expected: always works)
    ✅ ticker.income_stmt  → Gross Profit, EBIT, Net Income
    ✅ ticker.balance_sheet→ Total Assets, Stockholders Equity
    ✅ ticker.fast_info    → Market Cap, Current Price
    ✅ ticker.dividends    → Dividend payment history

Output: a clear table of what worked and what didn't.
        Shows expected % coverage for full 169-stock run.

If all 5 endpoints work for 3+ of the test stocks → you are ready.
Run: python3 main.py
───────────────────────────────────────────────────────────────────────────────
"""

import time
import sys
import pandas as pd
import numpy as np

TEST_TICKERS = [
    "RELIANCE.NS",   # large cap, Oil & Gas
    "HDFCBANK.NS",   # large cap, Banking (no grossProfits — tests fallback)
    "TCS.NS",        # large cap, IT
    "SUNPHARMA.NS",  # mid-large cap, Pharma
    "TATASTEEL.NS",  # large cap, Metals
]


def diagnose_one(ticker_str: str) -> dict:
    """Test all endpoints for one ticker. Returns result dict."""
    import yfinance as yf
    t = yf.Ticker(ticker_str)
    result = {
        "ticker"      : ticker_str,
        "prices_ok"   : False,
        "income_ok"   : False,
        "balance_ok"  : False,
        "fast_info_ok": False,
        "dividends_ok": False,
        "gpa_value"   : None,   # Quality metric found
        "btm_value"   : None,   # Value metric found
        "mcap_cr"     : None,   # Size metric found
        "errors"      : [],
    }

    # ── 1. Prices ─────────────────────────────────────────────────────────────
    try:
        prices = yf.download(ticker_str, period="3mo", interval="1mo",
                             auto_adjust=True, progress=False)
        if not prices.empty and prices["Close"].notna().any():
            result["prices_ok"] = True
    except Exception as e:
        result["errors"].append(f"prices: {e!s:.40s}")

    # ── 2. Income Statement ───────────────────────────────────────────────────
    gpa = None
    try:
        is_ = t.income_stmt
        if is_ is not None and not is_.empty:
            result["income_ok"] = True
            # Try to get a quality metric
            for row in ["Gross Profit","GrossProfit","EBIT",
                        "Operating Income","Net Income","NetIncome"]:
                if row in is_.index:
                    val = is_.loc[row].dropna()
                    if len(val) > 0:
                        result["gpa_raw_numerator"] = float(val.iloc[0])
                        result["gpa_row_found"] = row
                        break
    except Exception as e:
        result["errors"].append(f"income_stmt: {e!s:.40s}")

    # ── 3. Balance Sheet ──────────────────────────────────────────────────────
    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            result["balance_ok"] = True
            # Total Assets
            for row in ["Total Assets","TotalAssets"]:
                if row in bs.index:
                    val = bs.loc[row].dropna()
                    if len(val) > 0:
                        ta = float(val.iloc[0])
                        if result.get("gpa_raw_numerator") and ta > 0:
                            result["gpa_value"] = round(
                                result["gpa_raw_numerator"] / ta, 4
                            )
                        break
            # Total Equity for B/M calculation
            for row in ["Stockholders Equity","StockholdersEquity",
                        "Total Equity Gross Minority Interest",
                        "Common Stock Equity"]:
                if row in bs.index:
                    val = bs.loc[row].dropna()
                    if len(val) > 0:
                        result["equity"] = float(val.iloc[0])
                        break
    except Exception as e:
        result["errors"].append(f"balance_sheet: {e!s:.40s}")

    # ── 4. fast_info (market cap + price) ─────────────────────────────────────
    try:
        fi = t.fast_info
        mc = fi.market_cap
        px = fi.last_price
        if mc and not np.isnan(float(mc)):
            result["fast_info_ok"] = True
            result["mcap_cr"] = round(float(mc) / 1e7, 0)
            # Book-to-Market = Equity / Market Cap
            if result.get("equity") and mc > 0:
                result["btm_value"] = round(result["equity"] / float(mc), 4)
        if px and not np.isnan(float(px)):
            result["price"] = round(float(px), 2)
    except Exception as e:
        result["errors"].append(f"fast_info: {e!s:.40s}")

    # ── 5. Dividends ──────────────────────────────────────────────────────────
    try:
        div = t.dividends
        if div is not None:
            result["dividends_ok"] = True
            cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
            annual = float(div[div.index >= cutoff].sum())
            px = result.get("price", 0)
            if px and px > 0 and annual > 0:
                result["yield_pct"] = round(annual / px * 100, 2)
            else:
                result["yield_pct"] = 0.0
    except Exception as e:
        result["errors"].append(f"dividends: {e!s:.40s}")

    return result


def run_diagnosis():
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print("═" * 65)
    print(" DATA SOURCE DIAGNOSTIC")
    print(f" Testing {len(TEST_TICKERS)} sample stocks across all endpoints")
    print("═" * 65)
    print()

    results = []
    for i, ticker in enumerate(TEST_TICKERS):
        print(f"  [{i+1}/{len(TEST_TICKERS)}] Testing {ticker}...", end="", flush=True)
        r = diagnose_one(ticker)
        results.append(r)
        ok_count = sum([r["prices_ok"], r["income_ok"], r["balance_ok"],
                        r["fast_info_ok"], r["dividends_ok"]])
        print(f" {ok_count}/5 endpoints OK")
        time.sleep(1.0)

    # ── Results table ─────────────────────────────────────────────────────────
    print()
    print(f" {'Ticker':<20} {'Prices':>7} {'Income':>7} {'Balance':>8} "
          f"{'FastInfo':>9} {'Divs':>6}  {'GPA':>7} {'B/M':>7} {'MCap':>10}")
    print(f" {'─'*85}")

    ok_counts = {ep: 0 for ep in ["prices","income","balance","fast_info","dividends"]}

    for r in results:
        def yesno(b): return "  ✅" if b else "  ❌"
        def fmt(v): return f"{v:.4f}" if v is not None else "  —"
        def fmtm(v): return f"₹{v:>8,.0f}" if v is not None else "         —"

        print(f" {r['ticker']:<20}"
              f"{yesno(r['prices_ok'])}"
              f"{yesno(r['income_ok'])}"
              f"{yesno(r['balance_ok'])}"
              f"{yesno(r['fast_info_ok'])}"
              f"{yesno(r['dividends_ok'])}"
              f"  {fmt(r.get('gpa_value'))}"
              f"  {fmt(r.get('btm_value'))}"
              f"  {fmtm(r.get('mcap_cr'))}")

        if r["prices_ok"]   : ok_counts["prices"]    += 1
        if r["income_ok"]   : ok_counts["income"]     += 1
        if r["balance_ok"]  : ok_counts["balance"]    += 1
        if r["fast_info_ok"]: ok_counts["fast_info"]  += 1
        if r["dividends_ok"]: ok_counts["dividends"]  += 1

    n = len(results)
    print()
    print(f" Endpoint success rate across {n} test stocks:")
    ep_labels = {
        "prices"    : "Prices (yf.download)",
        "income"    : "Income Statement",
        "balance"   : "Balance Sheet",
        "fast_info" : "Fast Info (market cap)",
        "dividends" : "Dividends",
    }
    factor_use = {
        "prices"    : "→ Momentum + Beta",
        "income"    : "→ Quality factor",
        "balance"   : "→ Value + Invest factors",
        "fast_info" : "→ Size factor",
        "dividends" : "→ Yield factor",
    }
    all_ok = True
    for ep, label in ep_labels.items():
        pct = ok_counts[ep] / n * 100
        bar = "█" * ok_counts[ep] + "░" * (n - ok_counts[ep])
        status = "✅" if ok_counts[ep] >= 3 else "⚠️ " if ok_counts[ep] >= 1 else "❌"
        print(f"   {label:<28} {bar} {ok_counts[ep]}/{n} ({pct:.0f}%)  "
              f"{factor_use[ep]}  {status}")
        if ok_counts[ep] < 2:
            all_ok = False

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = [(r["ticker"], r["errors"]) for r in results if r["errors"]]
    if errors:
        print(f"\n  Errors encountered:")
        for ticker, errs in errors:
            for err in errs:
                print(f"    {ticker}: {err}")

    # ── Recommendation ────────────────────────────────────────────────────────
    print()
    print(f" {'═'*63}")
    if all_ok:
        print(f" ✅  ALL ENDPOINTS WORKING")
        print(f"    Expected data coverage for full 169-stock run:")
        print(f"    Quality (income_stmt):   ~70-80% real, rest = median fill")
        print(f"    Value (balance_sheet):   ~70-80% real, rest = median fill")
        print(f"    Size (fast_info):        ~90-95% real, rest = median fill")
        print(f"    Yield (dividends):       ~80-90% real, rest = 0%")
        print(f"    Momentum + Beta (prices): ~95% real")
        print(f"\n  → Ready to run: python3 main.py")
    else:
        print(f" ⚠️   SOME ENDPOINTS FAILING")
        print(f"    This is normal if you are behind a proxy or VPN.")
        print(f"    Try:")
        print(f"      1. Disable VPN / proxy")
        print(f"      2. Try a different network")
        print(f"      3. Update yfinance: pip install yfinance --upgrade")
        print(f"    The system will still work — failing factors get")
        print(f"    neutral Rank 3 (median fill) which is acceptable.")
        print(f"\n  → You can still run: python3 main.py")
    print(f" {'═'*63}")
    return all_ok


if __name__ == "__main__":
    run_diagnosis()
