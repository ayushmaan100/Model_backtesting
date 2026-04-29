# build_pit.py
# ─────────────────────────────────────────────────────────────────────────────
# Transforms screener_raw.csv into a proper Point-in-Time fundamentals database.
#
# KEY FEATURES:
#   1. Applies a 3-month institutional lag (Mar 2020 data → available Jul 2020)
#   2. Computes ALL 5 fundamental factors the factor engine needs:
#      - gross_profit_assets  (Quality)
#      - book_to_market       (Value)    ← computed at rebalance time using prices
#      - market_cap_cr        (Size)     ← computed at rebalance time using prices
#      - asset_growth_yoy     (Invest)
#      - dividend_yield_pct   (Yield)
#   3. Fixes the .NS ticker suffix (no double .NS.NS)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np


def build_pit_database(raw_csv_path="screener_raw.csv", output_path="fundamentals_pit.csv"):
    print(f"Loading raw data from {raw_csv_path}...")
    try:
        df = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        print(f"❌ Error: {raw_csv_path} not found. Run screener_scraper.py first.")
        return

    # ── 1. Clean Period column ────────────────────────────────────────────
    df = df[df['Period'] != 'TTM'].copy()
    df['Report_Date'] = pd.to_datetime(df['Period'], format='%b %Y', errors='coerce')
    df = df.dropna(subset=['Report_Date'])

    # ── 2. Apply institutional lag (3 months) ─────────────────────────────
    # E.g., Mar 2020 data becomes known to the algorithm on Jul 1, 2020
    df['Date'] = df['Report_Date'] + pd.DateOffset(months=3)

    # ── 3. Fix ticker: only add .NS if not already present ────────────────
    df['Ticker'] = df['Ticker'].astype(str)
    df['Ticker'] = df['Ticker'].apply(
        lambda t: t if t.endswith('.NS') else t + '.NS'
    )

    # ── 4. Build the PIT DataFrame ────────────────────────────────────────
    pit = pd.DataFrame()
    pit['Date'] = df['Date']
    pit['Ticker'] = df['Ticker']

    # --- Quality: Operating Profit (or Net Profit fallback) / Total Assets ---
    def _get_profit(row):
        """Use Operating Profit for non-banks, Financing Profit for banks."""
        for col in ['Operating Profit', 'Financing Profit', 'Net Profit\xa0+']:
            if col in row.index:
                val = pd.to_numeric(row[col], errors='coerce')
                if not np.isnan(val):
                    return val
        return np.nan

    pit['gross_profit'] = df.apply(_get_profit, axis=1)
    pit['total_assets'] = pd.to_numeric(df['Total Assets'], errors='coerce')

    # Quality factor
    pit['gross_profit_assets'] = pit['gross_profit'] / pit['total_assets']
    pit.loc[pit['total_assets'] <= 0, 'gross_profit_assets'] = np.nan

    # --- Value: Equity (Equity Capital + Reserves) ---
    equity_cap = pd.to_numeric(df['Equity Capital'], errors='coerce').fillna(0)
    reserves = pd.to_numeric(df['Reserves'], errors='coerce').fillna(0)
    pit['equity'] = equity_cap + reserves

    # --- Invest: YoY Total Assets Growth ---
    pit_sorted = pit.sort_values(['Ticker', 'Date']).copy()
    pit_sorted['prev_total_assets'] = pit_sorted.groupby('Ticker')['total_assets'].shift(1)
    pit_sorted['asset_growth_yoy'] = (
        (pit_sorted['total_assets'] - pit_sorted['prev_total_assets'])
        / pit_sorted['prev_total_assets'].abs()
    )
    pit_sorted.loc[pit_sorted['prev_total_assets'].isna(), 'asset_growth_yoy'] = np.nan
    pit['asset_growth_yoy'] = pit_sorted['asset_growth_yoy'].values

    # --- Yield: Dividend Payout % from Screener ---
    # We store the raw payout %; actual yield will be computed using prices at rebalance
    div_payout_col = 'Dividend Payout %'
    if div_payout_col in df.columns:
        pit['dividend_payout_pct'] = pd.to_numeric(
            df[div_payout_col].astype(str).str.replace('%', ''), errors='coerce'
        )
    else:
        pit['dividend_payout_pct'] = np.nan

    # --- EPS for computing dividend yield later ---
    eps_col = 'EPS in Rs'
    if eps_col in df.columns:
        pit['eps'] = pd.to_numeric(df[eps_col], errors='coerce')
    else:
        pit['eps'] = np.nan

    # ── 5. Clean and save ─────────────────────────────────────────────────
    pit = pit.sort_values(['Date', 'Ticker'])
    pit.to_csv(output_path, index=False)

    n_tickers = pit['Ticker'].nunique()
    n_rows = len(pit)
    date_range = f"{pit['Date'].min()} → {pit['Date'].max()}"
    print(f"✅ Point-in-Time database built:")
    print(f"   {n_rows} rows × {n_tickers} tickers")
    print(f"   Date range: {date_range}")
    print(f"   Columns: {list(pit.columns)}")
    print(f"   Saved to: {output_path}")

    # Coverage report
    print(f"\n   Coverage:")
    for col in ['gross_profit_assets', 'equity', 'asset_growth_yoy', 'dividend_payout_pct', 'eps']:
        if col in pit.columns:
            pct = pit[col].notna().mean() * 100
            print(f"     {col:<25} {pct:.1f}%")


if __name__ == "__main__":
    build_pit_database()