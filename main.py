"""
main.py — single entry point for the entire system

USAGE:
  python3 diagnose.py          ← run THIS first (tests data endpoints ~3 min)
  python3 main.py              ← run everything  (~25 min real, ~30 sec mock)
  python3 main.py --test       ← unit tests only (fast)
  python3 main.py --score-only ← score stocks today, skip backtest

REAL DATA SETUP (one time):
  1. Run: python3 diagnose.py    ← confirms endpoints work on your machine
  2. In config.py: USE_REAL_DATA = True
  3. Run: python3 main.py
     ├── Downloads prices for ~170 stocks via yf.download()  (~8 min)
     ├── Computes beta vs Nifty 50 from our own prices       (instant)
     ├── Fetches fundamentals via income_stmt/balance_sheet   (~15 min)
     └── Generates dashboard.html                            (instant)

IF INTERRUPTED HALFWAY:
  Just re-run. Fund download resumes from fund_progress.csv automatically.

TO RE-DOWNLOAD (clear cache):
  rm prices.csv fundamentals.csv fund_progress.csv
  python3 main.py
"""

import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from config import USE_REAL_DATA, PORTFOLIO_SIZE
from data_layer import load_data
from factor_engine import run_factor_engine, run_all_tests
from backtester import run_backtest
from dashboard import build_dashboard


def main():
    test_only  = "--test"       in sys.argv
    score_only = "--score-only" in sys.argv

    print("═"*62)
    print(" NSE 200 · 7-FACTOR PORTFOLIO SYSTEM")
    print(f" Mode: {'REAL DATA' if USE_REAL_DATA else 'MOCK DATA (set USE_REAL_DATA=True for real)'}")
    if USE_REAL_DATA:
        print(" Tip: Run python3 diagnose.py first to check your data endpoints")
    print("═"*62)

    # ── [1] Load data ──────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    prices, fund = load_data()
    print(f"      Prices: {prices.shape} ({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"      Fund:   {fund.shape}   NaN={fund.isna().sum().sum()}")

    if test_only:
        print("\n[2/4] Running unit tests...")
        run_all_tests(prices, fund)
        print("\nTests complete.")
        return

    # ── [2] Unit tests ─────────────────────────────────────────────────────
    print("\n[2/4] Running unit tests...")
    run_all_tests(prices, fund)

    # ── [3] Score all stocks ───────────────────────────────────────────────
    print("\n[3/4] Scoring all stocks...")
    scored, portfolio = run_factor_engine(prices, fund, verbose=True)
    scored.to_csv("all_scores.csv")
    portfolio.to_csv("portfolio.csv")
    print(f"\n      Saved: all_scores.csv ({len(scored)} stocks)")
    print(f"      Saved: portfolio.csv  (top {PORTFOLIO_SIZE} stocks)")

    if score_only:
        return

    # ── [4] Backtest ────────────────────────────────────────────────────────
    print("\n[4/4] Running backtest (Jan 2020 – Apr 2026)...")
    results = run_backtest(prices, fund, initial_capital=500_000)

    # Save outputs
    pd.DataFrame({
        "Portfolio": results["portfolio_equity"],
        "Nifty50":   results["nifty_equity"],
    }).to_csv("equity_curves.csv")
    pd.DataFrame(results["rebalance_log"]).to_csv("rebalance_log.csv", index=False)

    # Dashboard
    print("\n[Dashboard] Building...")
    build_dashboard(scored, portfolio, results,
                    output_path="dashboard.html", initial_capital=500_000)

    # Validation
    pm = results["portfolio_metrics"]
    print(f"\n{'═'*62}")
    print(f" VALIDATION AGAINST PDF TARGETS")
    print(f"{'─'*62}")
    print(f" {'Metric':<20} {'PDF Target':>12} {'This Run':>12}  Status")
    print(f" {'─'*58}")

    checks = [
        ("CAGR",         "~17.2%",  pm["cagr"],         0.10, 0.35),
        ("Max Drawdown", "~-18.0%", pm["max_drawdown"], -0.35, -0.05),
        ("Sharpe Ratio", "~0.78",   pm["sharpe"],        0.30,  1.80),
    ]
    for name, tgt, val, lo, hi in checks:
        ok   = lo <= val <= hi
        flag = "✅" if ok else ("⚠️  mock" if not USE_REAL_DATA else "❌ check")
        fmt  = f"{val:+.1%}" if name != "Sharpe Ratio" else f"{val:.2f}"
        print(f" {name:<20} {tgt:>12} {fmt:>12}  {flag}")

    print(f"\n{'═'*62}")
    print(f" OUTPUT FILES")
    print(f"{'─'*62}")
    print(f"   🌐  dashboard.html      ← Open in your browser")
    print(f"   📊  all_scores.csv      ← All {len(scored)} stocks ranked")
    print(f"   📋  portfolio.csv       ← Top {PORTFOLIO_SIZE} stocks today")
    print(f"   📈  equity_curves.csv   ← Portfolio + Nifty 50 over time")
    print(f"   🔄  rebalance_log.csv   ← Every rebalance event")
    if not USE_REAL_DATA:
        print(f"\n  ⚠️  Mock data — set USE_REAL_DATA=True for real NSE results")
    print(f"\n  👉  open dashboard.html")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
