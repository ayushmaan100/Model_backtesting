import requests
import pandas as pd
import time
import os
import io

import requests
import pandas as pd
import time
import io  # <-- New Import

def clean_ticker(ticker):
    return ticker.replace('.NS', '')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def _find_pl_bs(tables):
    """
    Detect (pl_table, bs_table) from a list of pandas-parsed tables.
    Broadened from the original to handle:
      - Banks (Financing Profit instead of Operating Profit)
      - NBFCs / insurers (different BS labels — Deposits, Borrowings)
      - Standalone-only companies whose BS uses "Equity Share Capital"
    Returns (pl_or_None, bs_or_None).
    """
    pl_table = None
    bs_table = None
    for df in tables:
        if df.shape[1] < 3:
            continue
        col0 = df.iloc[:, 0].astype(str).str.lower()

        # P&L: any of these income/profit tags + net profit on a row.
        if (col0.str.contains(
                r'operating profit|financing profit|gross profit|revenue|sales',
                regex=True).any()
            and col0.str.contains(r'net profit', regex=True).any()):
            pl_table = df

        # BS: "total assets" plus ANY equity-side or liability-side marker.
        # Old criterion required share/equity capital — too strict for banks
        # whose Screener BS leads with "Deposits" / "Borrowings".
        has_total_assets = col0.str.contains(r'total assets', regex=True).any()
        has_equity_or_liab = col0.str.contains(
            r'share capital|equity capital|equity share capital'
            r'|reserves|borrowings|deposits|total liab',
            regex=True).any()
        if has_total_assets and has_equity_or_liab:
            bs_table = df

    return pl_table, bs_table


def _try_url(url):
    """Fetch and parse tables. Returns (tables_list_or_None, status_code)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None, r.status_code
        return pd.read_html(io.StringIO(r.text)), 200
    except Exception:
        return None, -1


def scrape_screener_fundamentals(ticker):
    clean_sym = clean_ticker(ticker)

    # Try the consolidated page first (most companies); fall back to the
    # standalone page when consolidated is missing or its tables are partial.
    sources = [
        f"https://www.screener.in/company/{clean_sym}/consolidated/",
        f"https://www.screener.in/company/{clean_sym}/",
    ]

    last_status = None
    for url in sources:
        tables, status = _try_url(url)
        last_status = status
        if tables is None:
            continue
        pl_table, bs_table = _find_pl_bs(tables)
        if pl_table is not None and bs_table is not None:
            try:
                pl_table = pl_table.set_index(pl_table.columns[0]).T
                pl_table.index.name = 'Period'
                bs_table = bs_table.set_index(bs_table.columns[0]).T
                bs_table.index.name = 'Period'
                merged = pl_table.join(bs_table, how='inner')
                merged['Ticker'] = ticker
                tag = "consolidated" if "consolidated" in url else "standalone"
                print(f" ✅ Success ({tag})")
                return merged.reset_index()
            except Exception as e:
                print(f" ❌ Merge error: {str(e)[:50]}")
                return None

    # Both URLs failed.
    if last_status and last_status != 200:
        print(f" ❌ Blocked/Not Found (HTTP {last_status})")
    else:
        print(f" ❌ Missing Tables (couldn't find P&L+BS in either consolidated or standalone)")
    return None

if __name__ == "__main__":
    from nse200_tickers import NSE200
    
    all_data = []
    print("Initiating DOM extraction pipeline...")
    
    # We will test on just 5 stocks first to ensure it works before running all 200
    test_batch = NSE200
    
    for i, t in enumerate(test_batch): 
        print(f"Scraping [{i+1}/{len(test_batch)}]: {t:<15}", end="")
        df = scrape_screener_fundamentals(t)
        if df is not None:
            all_data.append(df)
        time.sleep(2.0) # Critical: Polite delay to prevent Screener from banning your IP
        
    if len(all_data) == 0:
        print("\n🚨 CRITICAL FAILURE: No data was extracted. Screener might be blocking your IP.")
    else:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv("screener_raw.csv", index=False)
        print(f"\n✅ Pipeline complete! Saved {len(master_df)} rows to screener_raw.csv")