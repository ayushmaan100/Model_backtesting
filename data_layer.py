"""
data_layer.py
─────────────────────────────────────────────────────────────────────────────
POINT-IN-TIME DATA STRATEGY:

  OLD approach (WRONG — look-ahead bias):
    Fetch current fundamentals from yfinance → apply to all historical dates
    Result: June 2020 rebalance uses April 2026 financials!

  NEW approach (CORRECT):
    1. Screener.in historical P&L + Balance Sheet (via screener_scraper.py)
    2. build_pit.py applies 3-month institutional lag
    3. At each rebalance date, we look up the LATEST KNOWN fundamental
       data for each stock — never using future data.

  Price-derived factors (Momentum, Beta) are always point-in-time because
  we slice prices_df to <= rebal_date in the backtester.

FACTOR → DATA SOURCE MAPPING:
  Momentum  ← yf.download() price data               [PiT via slicing]
  Beta      ← OLS on our own monthly prices           [PiT via slicing]
  Quality   ← PiT: Operating Profit / Total Assets    [PiT from Screener]
  Value     ← PiT: Equity / (Price × Shares)          [PiT hybrid]
  Size      ← PiT: Price × Shares from prices.csv     [PiT via slicing]
  Invest    ← PiT: YoY Total Assets growth            [PiT from Screener]
  Yield     ← PiT: (EPS × Payout%) / Price            [PiT hybrid]

CACHING:
  Prices saved to: prices.csv            (delete to re-download)
  PiT Fund:        fundamentals_pit.csv  (rebuild via build_pit.py)
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import time
import os
import logging

from config import NIFTY_TICKER, PRICE_CSV, BACKTEST_END, CACHE_MAX_AGE_DAYS
from nse200_tickers import NSE200

log = logging.getLogger(__name__)

MARKET_RETURNS = {
    2019: 0.12,  2020: 0.12,  2021: 0.475,
    2022: -0.195, 2023: 0.368, 2024: 0.310,
    2025: 0.375,  2026: 0.08,
}
PIT_CSV    = "fundamentals_pit.csv"
SHARES_CSV = "shares_outstanding.csv"   # Current shares outstanding (Cr units)


# ─────────────────────────────────────────────────────────────────────────────
# SHARES OUTSTANDING (for true Market Cap) — fetched once, cached.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_shares_outstanding(tickers: list, force: bool = False) -> pd.Series:
    """
    Fetch current shares-outstanding for each ticker via yfinance.info.

    Cached to SHARES_CSV. Re-run with force=True to refresh.

    Returns a Series indexed by ticker, in CRORES (1 crore = 10M).
    Tickers where yfinance has no `sharesOutstanding` are absent from the result.

    NOTE: This is *current* shares only. yfinance's auto_adjust=True ensures
    historical prices are split/bonus-adjusted, so MCap_t = price_t × shares_now
    is correct under those events. It does NOT correct for buybacks/issuances
    that materially changed the share count over the lookback. Documented
    limitation; tracked for Phase 2+.
    """
    if os.path.exists(SHARES_CSV) and not force:
        print(f"[Shares] Loading from cache {SHARES_CSV}")
        s = pd.read_csv(SHARES_CSV, index_col=0)["shares_cr"]
        return s

    import yfinance as yf
    print(f"[Shares] Fetching from yfinance for {len(tickers)} tickers...")
    rows = {}
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).info
            so   = info.get("sharesOutstanding")
            if so and so > 0:
                rows[t] = so / 1e7    # → crores
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} done")
        time.sleep(0.4)

    s = pd.Series(rows, name="shares_cr")
    s.index.name = "ticker"
    s.to_frame().to_csv(SHARES_CSV)
    print(f"[Shares] ✅ {len(s)}/{len(tickers)} tickers cached → {SHARES_CSV}")
    return s


def load_shares_outstanding() -> pd.Series:
    """Load shares-outstanding from cache; empty Series if not yet built."""
    if os.path.exists(SHARES_CSV):
        return pd.read_csv(SHARES_CSV, index_col=0)["shares_cr"]
    return pd.Series(dtype=float, name="shares_cr")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA (no internet needed)
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_data(seed: int = 42):
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start="2019-01-01", end=BACKTEST_END, freq="ME")

    nifty = []
    for dt in dates:
        ann = MARKET_RETURNS.get(dt.year, 0.10)
        nifty.append(float(rng.normal(ann/12, (0.22 if ann > 0 else 0.30)/np.sqrt(12))))
    nifty_ret = pd.Series(nifty, index=dates)

    stock_rets, true_betas = {}, {}
    for ticker in NSE200:
        beta  = float(rng.uniform(0.4, 1.8))
        alpha = float(rng.normal(0.002, 0.008))
        idio  = float(rng.uniform(0.08, 0.25)) / np.sqrt(12)
        true_betas[ticker] = round(beta, 4)
        rets = [float(beta*(1.3 if r<-0.04 else 1.0)*r + alpha + rng.normal(0, idio))
                for r in nifty_ret]
        stock_rets[ticker] = rets

    prices = (1 + pd.DataFrame(stock_rets, index=dates)).cumprod() * 100
    prices[NIFTY_TICKER] = (1 + nifty_ret).cumprod() * 10000

    records = {}
    for ticker in NSE200:
        nse = ticker.replace(".NS","")
        is_bank  = nse in {"HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK",
                           "PNB","CANBK","BANKBARODA","UNIONBANK","INDIANB"}
        is_it    = nse in {"TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM"}
        is_fmcg  = nse in {"HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR"}

        if is_bank:
            gpa = float(rng.uniform(0.008, 0.025))
            pb  = float(rng.uniform(0.8, 3.5))
            mc  = float(rng.uniform(50000, 1500000))
            ag  = float(rng.uniform(0.08, 0.20))
            dy  = float(rng.uniform(0.3, 1.5))
        elif is_it:
            gpa = float(rng.uniform(0.18, 0.30))
            pb  = float(rng.uniform(5, 15))
            mc  = float(rng.uniform(100000, 1400000))
            ag  = float(rng.uniform(0.05, 0.12))
            dy  = float(rng.uniform(1.0, 3.0))
        elif is_fmcg:
            gpa = float(rng.uniform(0.12, 0.25))
            pb  = float(rng.uniform(8, 70))
            mc  = float(rng.uniform(30000, 700000))
            ag  = float(rng.uniform(0.04, 0.10))
            dy  = float(rng.uniform(1.0, 3.0))
        else:
            gpa = float(rng.beta(2,5)*0.40 + 0.02)
            pb  = float(rng.lognormal(-0.5, 0.7))
            mc  = float(rng.lognormal(11.0, 1.2))
            ag  = float(rng.normal(0.10, 0.10))
            dy  = float(max(0, rng.normal(1.5, 1.2)))

        records[ticker] = {
            "gross_profit_assets": round(gpa, 4),
            "book_to_market"     : round(1.0/max(pb, 0.5), 4),
            "market_cap_cr"      : round(mc, 0),
            "asset_growth_yoy"   : round(ag, 4),
            "dividend_yield_pct" : round(dy, 4),
            "beta_nifty50"       : true_betas[ticker],
        }

    fund = pd.DataFrame(records).T
    fund.index.name = "ticker"
    print(f"[Mock] Prices: {prices.shape} ({dates[0].date()} → {dates[-1].date()})")
    print(f"[Mock] Fundamentals: {fund.shape}  NaN={fund.isna().sum().sum()}")
    return prices, fund


# ─────────────────────────────────────────────────────────────────────────────
# BETA (from our own prices — 100% accurate vs Nifty 50)
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: Beta is computed point-in-time inside factor_engine.compute_rolling_betas.
# The historical compute_betas() that lived here was unused and has been removed.


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate_price_data(prices_df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Run sanity checks on the price matrix and return a structured report.

    Checks:
      - Cache age (warn if stale)
      - Per-ticker coverage (% of months with a price)
      - Mid-history gaps (NaN months between first and last valid)
      - Zero / negative prices (illegal)
      - Suspicious one-month moves (|return| > 50%) cross-checked vs Nifty —
        flags possible split / bonus mis-adjustments from yfinance.
      - Nifty presence

    Returns a dict; prints a human-readable report when verbose=True.
    """
    issues = {
        "cache_age_days":        None,
        "date_range":            (prices_df.index.min(), prices_df.index.max()),
        "n_months":              len(prices_df),
        "n_tickers":             0,
        "nifty_present":         NIFTY_TICKER in prices_df.columns,
        "tickers_low_coverage":  [],   # (ticker, coverage)
        "tickers_with_gaps":     [],   # (ticker, n_internal_nan)
        "tickers_zero_or_neg":   [],   # ticker
        "suspicious_moves":      [],   # (ticker, date, ret, nifty_ret)
    }

    if os.path.exists(PRICE_CSV):
        age_days = (time.time() - os.path.getmtime(PRICE_CSV)) / 86400.0
        issues["cache_age_days"] = round(age_days, 1)

    stock_cols = [c for c in prices_df.columns if c != NIFTY_TICKER]
    issues["n_tickers"] = len(stock_cols)

    nifty_rets = (prices_df[NIFTY_TICKER].pct_change()
                  if issues["nifty_present"] else None)

    for ticker in stock_cols:
        s = prices_df[ticker]
        coverage = float(s.notna().mean())
        if coverage < 0.85:
            issues["tickers_low_coverage"].append((ticker, round(coverage, 3)))

        first_valid = s.first_valid_index()
        last_valid  = s.last_valid_index()
        if first_valid is not None and last_valid is not None:
            mid_nan = int(s.loc[first_valid:last_valid].isna().sum())
            if mid_nan > 0:
                issues["tickers_with_gaps"].append((ticker, mid_nan))

        # Zero or negative (illegal — adj-close should be positive)
        if (s.dropna() <= 0).any():
            issues["tickers_zero_or_neg"].append(ticker)

        # Suspicious moves: > 50% in one month with Nifty broadly flat
        rets = s.pct_change()
        big = rets[(rets.abs() > 0.50) & rets.notna()]
        for dt, r in big.items():
            n = float(nifty_rets.loc[dt]) if (nifty_rets is not None
                                              and dt in nifty_rets.index
                                              and pd.notna(nifty_rets.loc[dt])) else None
            # Only flag if Nifty did NOT move more than 15% — otherwise it's a real market event
            if n is None or abs(n) < 0.15:
                issues["suspicious_moves"].append(
                    (ticker, str(dt.date()), round(float(r), 3),
                     round(n, 3) if n is not None else None)
                )

    if verbose:
        _print_validation_report(issues)
    return issues


def _print_validation_report(issues: dict):
    sep = "─" * 62
    print(f"\n{sep}\n DATA QUALITY VALIDATOR\n{sep}")

    age = issues["cache_age_days"]
    if age is None:
        print(" Cache age:        (no cached file)")
    elif age > CACHE_MAX_AGE_DAYS:
        print(f" Cache age:        {age} days  ⚠️  STALE (> {CACHE_MAX_AGE_DAYS}d)")
        print(f"                   delete {PRICE_CSV} to force refresh")
    else:
        print(f" Cache age:        {age} days  ✅")

    d0, d1 = issues["date_range"]
    print(f" Date range:       {pd.Timestamp(d0).date()} → {pd.Timestamp(d1).date()}  "
          f"({issues['n_months']} months)")
    print(f" Universe size:    {issues['n_tickers']} tickers + "
          f"{'Nifty 50 ✅' if issues['nifty_present'] else 'NO NIFTY ❌'}")

    n_low = len(issues["tickers_low_coverage"])
    n_gap = len(issues["tickers_with_gaps"])
    n_neg = len(issues["tickers_zero_or_neg"])
    n_sus = len(issues["suspicious_moves"])

    flag = lambda ok: "✅" if ok else "⚠️ "
    print(f" Low coverage:     {n_low}  {flag(n_low == 0)}  (< 85% months populated)")
    print(f" Internal gaps:    {n_gap}  {flag(n_gap == 0)}  (NaN between first/last)")
    print(f" Zero/neg prices:  {n_neg}  {flag(n_neg == 0)}")
    print(f" Suspicious >50%:  {n_sus}  {flag(n_sus == 0)}  (likely split/bonus issues)")

    if n_low:
        print("\n  Low-coverage tickers (top 10):")
        for t, c in sorted(issues["tickers_low_coverage"], key=lambda x: x[1])[:10]:
            print(f"    {t:<18} {c*100:.0f}%")

    if n_sus:
        print("\n  Suspicious moves (top 10) — verify on screener.in:")
        for t, dt, r, nr in issues["suspicious_moves"][:10]:
            nstr = f"Nifty {nr:+.1%}" if nr is not None else ""
            print(f"    {t:<18} {dt}  {r:+.1%}  {nstr}")

    if n_neg:
        print(f"\n  Zero/negative tickers: {issues['tickers_zero_or_neg']}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH PRICES (yf.download — most reliable endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list, start: str = "2019-01-01") -> pd.DataFrame:
    import yfinance as yf

    all_tickers = tickers + [NIFTY_TICKER]
    prices, failed = {}, []

    print(f"\n[Prices] {len(all_tickers)} tickers in batches of 10...")
    for i in range(0, len(all_tickers), 10):
        batch = all_tickers[i: i+10]
        b_num = i//10 + 1
        b_tot = (len(all_tickers)-1)//10 + 1
        print(f"  Batch {b_num}/{b_tot}: {batch[0]} … {batch[-1]}", end="", flush=True)
        try:
            raw = yf.download(tickers=batch, start=start, interval="1mo",
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                raise ValueError("empty")
            close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex)
                     else raw[["Close"]].rename(columns={"Close": batch[0]}))
            for t in batch:
                if t in close.columns and close[t].notna().sum() >= 12:
                    prices[t] = close[t]
                else:
                    failed.append(t)
            ok = len([t for t in batch if t in prices])
            print(f" ✓ {ok}/{len(batch)}")
        except Exception as e:
            print(f" ✗ {str(e)[:55]}")
            failed.extend(batch)
        time.sleep(1.5)

    if not prices:
        raise RuntimeError("Zero tickers downloaded. Check internet connection.")

    df = pd.DataFrame(prices).sort_index()
    df.index = pd.DatetimeIndex(df.index)
    n_stocks = len([c for c in df.columns if c != NIFTY_TICKER])
    print(f"\n[Prices] ✅ {n_stocks} stocks + Nifty 50 | "
          f"{len(df)} months | {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
    df.to_csv(PRICE_CSV)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# POINT-IN-TIME FUNDAMENTAL SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def get_pit_snapshot(pit_df: pd.DataFrame, prices_df: pd.DataFrame,
                     as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build a flat fundamentals DataFrame (indexed by ticker) using ONLY data
    known as of `as_of_date`. This is the core PiT function.

    DYNAMIC UNIVERSE FILTERING (Option 3):
      Instead of trusting a fixed NSE 200 list, we dynamically determine
      which stocks are eligible at each rebalance date based on:
        1. Data availability (must have both prices + fundamentals)
        2. Data freshness (fundamentals not older than MAX_FUND_AGE_YEARS)
        3. Size threshold (total_assets >= MIN_TOTAL_ASSETS_CR)

      This eliminates survivorship bias because eligibility is determined
      by the stock's actual characteristics at each point in time.

    Returns:
        DataFrame indexed by ticker with columns:
            gross_profit_assets, book_to_market, market_cap_cr,
            asset_growth_yoy, dividend_yield_pct
    """
    from config import MIN_TOTAL_ASSETS_CR, MAX_FUND_AGE_YEARS

    as_of = pd.Timestamp(as_of_date)

    # ── 1. Get latest known fundamentals per ticker ───────────────────────
    available = pit_df[pit_df['Date'] <= as_of].copy()
    if available.empty:
        return pd.DataFrame()

    # For each ticker, take the most recent report
    latest = (available
              .sort_values('Date')
              .groupby('Ticker')
              .last()
              .reset_index()
              .set_index('Ticker'))

    n_before_filter = len(latest)

    # ── 2. FRESHNESS FILTER: reject stale fundamentals ────────────────────
    # If a stock's latest report is older than MAX_FUND_AGE_YEARS, it means
    # the company stopped filing or Screener stopped tracking it.
    # We don't want to rank stocks based on ancient data.
    cutoff_date = as_of - pd.DateOffset(years=MAX_FUND_AGE_YEARS)
    fresh_mask = latest['Date'] >= cutoff_date
    n_stale = (~fresh_mask).sum()
    latest = latest[fresh_mask]

    # ── 3. SIZE FILTER: dynamic universe eligibility ──────────────────────
    # Only include stocks with total_assets >= threshold.
    # This replaces the fixed NSE 200 list with date-dependent eligibility.
    ta = pd.to_numeric(latest['total_assets'], errors='coerce')
    size_mask = ta >= MIN_TOTAL_ASSETS_CR
    n_small = (~size_mask).sum()
    latest = latest[size_mask]

    n_after_filter = len(latest)

    # ── 4. Get latest price at as_of_date ─────────────────────────────────
    prices_to_date = prices_df[prices_df.index <= as_of]
    if prices_to_date.empty:
        return pd.DataFrame()
    latest_prices = prices_to_date.iloc[-1]  # Most recent month's prices

    # ── 5. Build the snapshot ─────────────────────────────────────────────
    # Get tickers that exist in both fundamentals and prices
    stock_cols = [c for c in prices_df.columns if c != NIFTY_TICKER]
    common_tickers = [t for t in latest.index if t in stock_cols]

    if not common_tickers:
        return pd.DataFrame()

    snapshot = pd.DataFrame(index=common_tickers)
    snapshot.index.name = "ticker"

    # Quality: already computed in PiT
    snapshot['gross_profit_assets'] = latest.loc[common_tickers, 'gross_profit_assets']

    # Value: Book-to-Market = Equity / Price (cross-sectional rank proxy)
    equity = latest.loc[common_tickers, 'equity']
    price = latest_prices.reindex(common_tickers)
    snapshot['book_to_market'] = equity.values / price.values
    snapshot.loc[price.values <= 0, 'book_to_market'] = np.nan

    # Size: TRUE market cap when shares-outstanding cache is present;
    # otherwise fall back to total_assets (the old proxy) and warn once.
    shares = load_shares_outstanding()
    if not shares.empty:
        sh_aligned = shares.reindex(common_tickers)
        mcap = price.values * sh_aligned.values   # both crores; price in ₹
        # Where shares are missing, fall back to total_assets so the row isn't dropped.
        ta_fallback = latest.loc[common_tickers, 'total_assets'].values
        snapshot['market_cap_cr'] = np.where(
            np.isnan(mcap) | (sh_aligned.values <= 0), ta_fallback, mcap
        )
        snapshot.attrs['_size_source'] = (
            f"true_mcap ({sh_aligned.notna().sum()}/{len(common_tickers)}) "
            f"+ total_assets fallback"
        )
    else:
        snapshot['market_cap_cr'] = latest.loc[common_tickers, 'total_assets'].values
        snapshot.attrs['_size_source'] = "total_assets (no shares cache)"

    # Invest: YoY Asset Growth (already computed in PiT)
    snapshot['asset_growth_yoy'] = latest.loc[common_tickers, 'asset_growth_yoy'].values

    # Yield: Dividend Yield = (EPS × Payout%) / Price
    eps = latest.loc[common_tickers, 'eps'].values if 'eps' in latest.columns else np.nan
    payout = latest.loc[common_tickers, 'dividend_payout_pct'].values if 'dividend_payout_pct' in latest.columns else np.nan
    eps = pd.to_numeric(pd.Series(eps), errors='coerce').values
    payout = pd.to_numeric(pd.Series(payout), errors='coerce').values
    price_vals = price.values

    with np.errstate(divide='ignore', invalid='ignore'):
        dps = eps * (payout / 100.0)
        div_yield = np.where(price_vals > 0, dps / price_vals * 100, np.nan)
    # Preserve NaN for missing payout/EPS — rank_quintiles() will assign Rank 3 (neutral).
    # Previously we filled with 0, which incorrectly punished data-incomplete names
    # by sorting them into Rank 1 alongside genuine zero-yield stocks.
    snapshot['dividend_yield_pct'] = pd.Series(div_yield, index=common_tickers).clip(0, 20)

    # ── 6. Clean up ───────────────────────────────────────────────────────
    snapshot['gross_profit_assets'] = snapshot['gross_profit_assets'].clip(-1.0, 2.0)
    snapshot['book_to_market'] = snapshot['book_to_market'].clip(0.001, 1e6)
    snapshot['market_cap_cr'] = snapshot['market_cap_cr'].clip(10.0, 2e7)
    snapshot['asset_growth_yoy'] = snapshot['asset_growth_yoy'].clip(-0.80, 3.0)

    # ── 7. Attach filter stats for logging ────────────────────────────────
    snapshot.attrs['_filter_stats'] = {
        'as_of_date':    as_of.date(),
        'total_in_pit':  n_before_filter,
        'stale_removed': n_stale,
        'small_removed': n_small,
        'eligible':      n_after_filter,
        'with_prices':   len(common_tickers),
        'min_ta_cr':     MIN_TOTAL_ASSETS_CR,
        'max_age_yrs':   MAX_FUND_AGE_YEARS,
    }

    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def fetch_real_data():
    """
    Returns (prices_df, pit_df).
    prices_df: monthly prices indexed by date, columns = tickers
    pit_df: tall DataFrame with Date, Ticker, and fundamental columns
    """
    # 1. Prices
    if os.path.exists(PRICE_CSV):
        age_days = (time.time() - os.path.getmtime(PRICE_CSV)) / 86400.0
        stale = age_days > CACHE_MAX_AGE_DAYS
        flag = " ⚠️  STALE" if stale else ""
        print(f"[Cache] Loading prices from {PRICE_CSV} (age {age_days:.1f}d){flag}")
        if stale:
            print(f"        delete {PRICE_CSV} to force a fresh download")
        prices = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
    else:
        tickers = NSE200
        prices = fetch_prices(tickers)

    # 2. PiT Fundamentals
    if os.path.exists(PIT_CSV):
        print(f"[PiT] Loading fundamentals from {PIT_CSV}...")
        pit_df = pd.read_csv(PIT_CSV, parse_dates=['Date'])
        n_tickers = pit_df['Ticker'].nunique()
        date_range = f"{pit_df['Date'].min().date()} → {pit_df['Date'].max().date()}"
        print(f"  {len(pit_df)} rows × {n_tickers} tickers | {date_range}")
    else:
        raise FileNotFoundError(
            f"❌ {PIT_CSV} not found. Run:\n"
            f"   python3 screener_scraper.py  (scrape Screener.in)\n"
            f"   python3 build_pit.py         (build PiT database)"
        )

    print(f"  Prices: {prices.shape}")
    print(f"  PiT Fund: {pit_df.shape}")
    return prices, pit_df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    from config import USE_REAL_DATA
    return fetch_real_data() if USE_REAL_DATA else generate_mock_data()


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DATA LAYER SELF-TEST (mock) ===\n")
    prices, fund = generate_mock_data()

    print("\n[Test 1] Zero NaN in mock data...")
    assert fund.isna().sum().sum() == 0, "NaN found in mock data"
    print("  ✅ PASS")

    print("\n[Test 2] Beta computation (PiT, via factor_engine)...")
    from factor_engine import compute_rolling_betas
    betas = compute_rolling_betas(prices)
    assert betas.notna().all()
    assert (betas >= 0.1).all() and (betas <= 4.0).all()
    print(f"  ✅ PASS — {len(betas)} betas in range [0.1, 4.0]")

    print("\n[Test 5] Price validator...")
    report = validate_price_data(prices, verbose=False)
    assert report["nifty_present"]
    assert report["n_tickers"] >= 1
    print(f"  ✅ PASS — validator returned {len(report)} keys")

    print("\n[Test 3] Price data sanity...")
    assert NIFTY_TICKER in prices.columns
    assert len(prices) >= 85
    assert (prices > 0).all().all()
    print(f"  ✅ PASS — {prices.shape}, all positive")

    print("\n[Test 4] Fundamental ranges...")
    assert (fund["dividend_yield_pct"] >= 0).all()
    assert (fund["market_cap_cr"] > 0).all()
    assert (fund["beta_nifty50"] >= 0.1).all()
    assert fund["book_to_market"].between(0.001, 20).all()
    print("  ✅ PASS")

    print("\n=== ALL TESTS PASSED ===")
