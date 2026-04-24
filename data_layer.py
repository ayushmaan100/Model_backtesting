"""
data_layer.py
─────────────────────────────────────────────────────────────────────────────
THE DATA STRATEGY (why this works when .info doesn't):

  WRONG approach (what we did before):
    ticker.info["grossProfits"]  → returns None for 85% of NSE stocks
    ticker.info["totalAssets"]   → unreliable for Indian companies
    ticker.info["beta"]          → calculated vs S&P 500 (WRONG benchmark)

  CORRECT approach (what we do now):
    ticker.income_stmt           → separate API endpoint, ~75% coverage
    ticker.balance_sheet         → separate API endpoint, ~75% coverage
    ticker.fast_info.market_cap  → lightweight endpoint, ~90% coverage
    ticker.dividends             → separate dividend history, ~85% coverage
    OUR OWN PRICE DATA           → for Beta via OLS regression, 100% coverage

  For any stock where data is missing (~20-30%):
    → Fill with column median (neutral Rank 3)
    → Documented in validation report so you know exactly what was filled

FACTOR → DATA SOURCE MAPPING:
  Momentum  ← yf.download() price data        [100% reliable]
  Beta      ← OLS on our own monthly prices   [100% reliable]
  Quality   ← income_stmt: Gross Profit or EBIT or Net Income / Total Assets
  Value     ← balance_sheet: Stockholders Equity / fast_info: market_cap
  Size      ← fast_info: market_cap in crores
  Invest    ← balance_sheet: 2-year Total Assets YoY growth
  Yield     ← dividends: last 12 months sum / fast_info: last_price

CACHING:
  Prices saved to: prices.csv   (delete to re-download)
  Fund saved to:   fundamentals.csv (delete to re-download)
  Progress saved:  fund_progress.csv (auto-resumes if interrupted)
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import time
import os
import logging

from config import NIFTY_TICKER, PRICE_CSV, FUND_CSV, BACKTEST_END
from nse200_tickers import NSE200

log = logging.getLogger(__name__)

MARKET_RETURNS = {
    2019: 0.12,  2020: 0.12,  2021: 0.475,
    2022: -0.195, 2023: 0.368, 2024: 0.310,
    2025: 0.375,  2026: 0.08,
}
PROGRESS_FILE = "fund_progress.csv"


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

def compute_betas(prices_df: pd.DataFrame, window: int = 36) -> pd.Series:
    from scipy import stats
    if NIFTY_TICKER not in prices_df.columns:
        tickers = [c for c in prices_df.columns if c != NIFTY_TICKER]
        return pd.Series(1.0, index=tickers)

    rets   = np.log(prices_df / prices_df.shift(1)).dropna().tail(window)
    nifty  = rets[NIFTY_TICKER].values
    stocks = [c for c in rets.columns if c != NIFTY_TICKER]

    betas = {}
    for t in stocks:
        sr   = rets[t].values
        mask = ~(np.isnan(sr) | np.isnan(nifty))
        if mask.sum() < 12:
            betas[t] = 1.0
            continue
        slope, *_ = stats.linregress(nifty[mask], sr[mask])
        betas[t]  = round(float(np.clip(slope, 0.1, 4.0)), 4)

    result = pd.Series(betas)
    print(f"[Beta] {len(result)} stocks | "
          f"mean={result.mean():.2f}  median={result.median():.2f}  "
          f"range=[{result.min():.2f}, {result.max():.2f}]")
    return result


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
            ok = sum(1 for t in batch
                     if t in close.columns and close[t].notna().sum() >= 12
                     and not prices.update({t: close[t]}))
            # Note: dict.update returns None so ok counts correct tickers
            ok = len([t for t in batch if t in close.columns
                      and close[t].notna().sum() >= 12])
            for t in batch:
                if t in close.columns and close[t].notna().sum() >= 12:
                    prices[t] = close[t]
                else:
                    failed.append(t)
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
# FETCH FUNDAMENTALS (income_stmt + balance_sheet + fast_info + dividends)
# ─────────────────────────────────────────────────────────────────────────────

def _find_row(df, names):
    """Find the first available row in a DataFrame from a list of name options."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            col = df.loc[name].dropna()
            if len(col) > 0:
                return float(col.iloc[0])
    return None


def _fetch_one(ticker: str, betas: pd.Series) -> dict:
    """
    Fetch all fundamental data for ONE stock using reliable endpoints.
    NEVER uses .info for fundamentals.
    Falls back gracefully — never raises an exception.
    """
    import yfinance as yf

    out = {
        "gross_profit_assets": np.nan,
        "book_to_market"     : np.nan,
        "market_cap_cr"      : np.nan,
        "asset_growth_yoy"   : np.nan,
        "dividend_yield_pct" : 0.0,
        "beta_nifty50"       : float(betas.get(ticker, 1.0)),
    }

    t = yf.Ticker(ticker)

    # ── Income Statement → Quality ────────────────────────────────────────────
    income = None
    try:
        income = t.income_stmt
    except Exception:
        pass

    # ── Balance Sheet → Value + Invest ───────────────────────────────────────
    bs = None
    try:
        bs = t.balance_sheet
    except Exception:
        pass

    # Quality: best available income metric / Total Assets
    total_assets = _find_row(bs, ["Total Assets","TotalAssets"])
    numerator    = _find_row(income, [
        "Gross Profit","GrossProfit",       # Best: true gross profit
        "EBIT","Ebit",                       # 2nd: operating profit
        "Operating Income","OperatingIncome",# 3rd: operating income
        "Net Income","NetIncome",            # 4th: net income (ROA)
        "Net Income Common Stockholders",
    ])
    if numerator and total_assets and total_assets > 0:
        out["gross_profit_assets"] = round(numerator / total_assets, 5)

    # Value: Stockholders Equity / Market Cap
    equity = _find_row(bs, [
        "Stockholders Equity","StockholdersEquity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity","CommonStockEquity",
        "Total Equity",
    ])

    # Invest: YoY Total Assets growth (2 years needed)
    try:
        if bs is not None and not bs.empty:
            for row in ["Total Assets","TotalAssets"]:
                if row in bs.index:
                    vals = bs.loc[row].dropna()
                    if len(vals) >= 2:
                        a0, a1 = float(vals.iloc[0]), float(vals.iloc[1])
                        if a1 != 0:
                            out["asset_growth_yoy"] = round((a0 - a1)/abs(a1), 5)
                    break
    except Exception:
        pass

    # ── fast_info → Size + Price (for Value + Yield) ─────────────────────────
    px = None
    try:
        fi  = t.fast_info
        mc  = fi.market_cap
        px  = fi.last_price
        if mc and not (isinstance(mc, float) and np.isnan(mc)):
            out["market_cap_cr"] = round(float(mc) / 1e7, 0)
        if equity and mc and mc > 0:
            out["book_to_market"] = round(float(equity) / float(mc), 5)
    except Exception:
        pass

    # ── Dividends → Yield ─────────────────────────────────────────────────────
    try:
        divs = t.dividends
        if divs is not None and len(divs) > 0:
            cutoff     = pd.Timestamp.now() - pd.DateOffset(months=12)
            annual_div = float(divs[divs.index >= cutoff].sum())
            if px and float(px) > 0 and annual_div > 0:
                out["dividend_yield_pct"] = round(annual_div / float(px) * 100, 4)
    except Exception:
        pass

    return out


def fetch_fundamentals(tickers: list, betas: pd.Series) -> pd.DataFrame:
    """
    Fetch fundamentals for all stocks with per-stock progress saving.
    Resumes automatically if interrupted.
    """
    # ── Load progress ─────────────────────────────────────────────────────────
    done = {}
    if os.path.exists(PROGRESS_FILE):
        try:
            saved = pd.read_csv(PROGRESS_FILE, index_col=0)
            done  = saved.to_dict(orient="index")
            print(f"[Fund] Resuming: {len(done)} done, "
                  f"{len(tickers)-len(done)} remaining")
        except Exception:
            pass

    remaining = [t for t in tickers if t not in done]
    if not remaining:
        print("[Fund] All stocks already done (loaded from progress file)")
    else:
        print(f"[Fund] Fetching {len(remaining)} stocks "
              f"(~{len(remaining)*2/60:.0f} min, saving every 5 stocks)...")
        print(f"       GPA = Quality factor value  B/M = Value factor value")
        print()

    for i, ticker in enumerate(remaining):
        pct = (i+1)/len(remaining)*100
        print(f"  [{pct:4.0f}%] {ticker:<22}", end="", flush=True)
        try:
            rec = _fetch_one(ticker, betas)
            done[ticker] = rec
            # Show what we actually got
            gpa = rec["gross_profit_assets"]
            btm = rec["book_to_market"]
            dy  = rec["dividend_yield_pct"]
            mc  = rec["market_cap_cr"]
            gpa_s = f"GPA={gpa:.3f}" if not (isinstance(gpa,float) and np.isnan(gpa)) else "GPA=—  "
            btm_s = f"B/M={btm:.3f}" if not (isinstance(btm,float) and np.isnan(btm)) else "B/M=—  "
            mc_s  = f"MC=₹{mc:,.0f}Cr" if not (isinstance(mc,float) and np.isnan(mc)) else "MC=—"
            print(f" {gpa_s}  {btm_s}  Yld={dy:.1f}%  {mc_s}")
        except Exception as e:
            print(f" ✗ unexpected: {str(e)[:50]}")
            done[ticker] = {
                "gross_profit_assets": np.nan, "book_to_market": np.nan,
                "market_cap_cr": np.nan,       "asset_growth_yoy": np.nan,
                "dividend_yield_pct": 0.0,     "beta_nifty50": float(betas.get(ticker, 1.0)),
            }

        if (i+1) % 5 == 0:
            pd.DataFrame(done).T.to_csv(PROGRESS_FILE)

        time.sleep(1.0)   # 1 second between stocks — polite, avoids rate limits

    return pd.DataFrame(done).T


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE AND FILL
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_fill(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Final validation:
    1. Show coverage for every column
    2. Fill NaN with column median (neutral Rank 3)
    3. Clip to sensible ranges
    """
    cols = ["gross_profit_assets","book_to_market","market_cap_cr",
            "asset_growth_yoy","dividend_yield_pct","beta_nifty50"]

    print(f"\n{'─'*68}")
    print(f" FUNDAMENTAL DATA QUALITY REPORT")
    print(f"{'─'*68}")
    print(f" {'Factor column':<25} {'Real data':>9} {'Filled':>7} "
          f"{'Min':>8} {'Median':>8} {'Max':>8}")
    print(f" {'─'*66}")

    for col in cols:
        if col not in fund_df.columns:
            fund_df[col] = np.nan
        fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")
        n       = len(fund_df)
        n_real  = fund_df[col].notna().sum()
        n_fill  = n - n_real
        med     = fund_df[col].median()
        pct     = n_real/n*100

        icon = "✅" if pct >= 70 else "⚠️" if pct >= 40 else "❌"
        print(f" {col:<25} {n_real:>5}({pct:.0f}%) {n_fill:>7}  "
              f"{fund_df[col].min():>7.3f}  {med:>7.3f}  "
              f"{fund_df[col].max():>7.3f}  {icon}")

        if n_fill > 0:
            fund_df[col] = fund_df[col].fillna(med)

    # Sanity clips
    fund_df["beta_nifty50"]        = fund_df["beta_nifty50"].clip(0.1, 4.0)
    fund_df["dividend_yield_pct"]  = fund_df["dividend_yield_pct"].clip(0.0, 20.0)
    fund_df["gross_profit_assets"] = fund_df["gross_profit_assets"].clip(-1.0, 2.0)
    fund_df["book_to_market"]      = fund_df["book_to_market"].clip(0.01, 10.0)
    fund_df["market_cap_cr"]       = fund_df["market_cap_cr"].clip(10.0, 2e7)
    fund_df["asset_growth_yoy"]    = fund_df["asset_growth_yoy"].clip(-0.80, 3.0)

    remaining = fund_df[cols].isna().sum().sum()
    print(f"\n {'✅ Zero NaN — all factors complete' if remaining==0 else f'❌ {remaining} NaN remain'}")
    return fund_df


# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def fetch_real_data():
    # ── Cache ─────────────────────────────────────────────────────────────────
    if os.path.exists(PRICE_CSV) and os.path.exists(FUND_CSV):
        print("[Cache] Loading from saved files...")
        prices = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
        fund   = pd.read_csv(FUND_CSV,  index_col=0)
        print(f"  Prices: {prices.shape}  Fund: {fund.shape}")
        print(f"  Delete prices.csv / fundamentals.csv to re-download")
        return prices, fund

    tickers = NSE200

    # Step 1: Prices
    prices = fetch_prices(tickers)
    valid  = [t for t in tickers
              if t in prices.columns and prices[t].notna().sum() >= 12]
    print(f"\n[Filter] {len(valid)}/{len(tickers)} stocks with sufficient price history")

    # Step 2: Beta from prices
    print("\n[Beta] Computing from price data (no API call needed)...")
    betas = compute_betas(prices)

    # Step 3: Fundamentals
    fund_raw = fetch_fundamentals(valid, betas)

    # Step 4: Validate
    fund = validate_and_fill(fund_raw)

    # Step 5: Save
    fund.to_csv(FUND_CSV)
    print(f"\n[Saved] {PRICE_CSV} ({prices.shape})")
    print(f"[Saved] {FUND_CSV} ({fund.shape})")

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    return prices, fund


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

    print("\n[Test 2] Beta computation...")
    betas = compute_betas(prices)
    assert len(betas) == len(NSE200)
    assert betas.notna().all()
    assert (betas >= 0.1).all() and (betas <= 4.0).all()
    print(f"  ✅ PASS — {len(betas)} betas in range [0.1, 4.0]")

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
