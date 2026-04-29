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

def scrape_screener_fundamentals(ticker):
    clean_sym = clean_ticker(ticker)
    url = f"https://www.screener.in/company/{clean_sym}/consolidated/"
    
    # Robust User-Agent to prevent 403 Forbidden blocks
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f" ❌ Blocked/Not Found (HTTP {response.status_code})")
            return None
            
        # THE FIX: Wrap the raw HTML in io.StringIO()
        tables = pd.read_html(io.StringIO(response.text))
        
        pl_table = None
        bs_table = None
        
        for df in tables:
            # Convert first column to lowercase string for fuzzy matching
            col0 = df.iloc[:, 0].astype(str).str.lower()
            
            # Identify P&L Table (Matches Manufacturing, IT, and Banking)
            if col0.str.contains('operating profit|financing profit|gross profit|revenue', regex=True).any() and col0.str.contains('net profit').any():
                pl_table = df
                
            # Identify Balance Sheet Table
            if col0.str.contains('total assets').any() and col0.str.contains('share capital|equity capital').any():
                bs_table = df
                
        if pl_table is None or bs_table is None:
            print(f" ❌ Missing Tables (P&L Found: {pl_table is not None}, BS Found: {bs_table is not None})")
            return None
            
        # Clean and Transpose P&L
        pl_table = pl_table.set_index(pl_table.columns[0]).T
        pl_table.index.name = 'Period'
        
        # Clean and Transpose BS
        bs_table = bs_table.set_index(bs_table.columns[0]).T
        bs_table.index.name = 'Period'
        
        # Merge them
        merged = pl_table.join(bs_table, how='inner')
        merged['Ticker'] = ticker
        
        print(" ✅ Success")
        return merged.reset_index()
        
    except Exception as e:
        print(f" ❌ Error: {str(e)[:50]}")
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