"""
universe_builder.py — Build historical Nifty 200 constituent universe

STRATEGY (hybrid):
  1. Mar 2020 – Mar 2022:  Parse NIFTY_200_*.pdf from monthly ZIPs (full 200-stock list)
  2. Sep 2022 – Mar 2026:  Parse NIFTY_50 + NIFTY_Next_50 PDFs → get Nifty 100 (~50% of universe)
                           then combine with current CSV list to fill in the Midcap 100 portion
  3. Current (latest):     Download live CSV from niftyindices.com (full 200-stock list)

OUTPUT:
  data/universe/universe_history.csv — all constituent snapshots
  data/universe/superset_tickers.txt — union of all stocks ever in Nifty 200

USAGE:
  python3 universe_builder.py              ← download + parse + build
  python3 universe_builder.py --skip-download  ← use cached files
"""

import os
import re
import sys
import time
import zipfile
import requests
import pandas as pd
import PyPDF2

# ── Configuration ─────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "universe")
OUTPUT_CSV = os.path.join(DATA_DIR, "universe_history.csv")

RECON_MONTHS = [
    "Mar2020", "Sep2020", "Mar2021", "Sep2021",
    "Mar2022", "Sep2022", "Mar2023", "Sep2023",
    "Mar2024", "Sep2024", "Mar2025", "Sep2025",
    "Mar2026",
]

EFFECTIVE_DATES = {
    "Mar2020": "2020-03-31", "Sep2020": "2020-09-30",
    "Mar2021": "2021-03-31", "Sep2021": "2021-09-30",
    "Mar2022": "2022-03-31", "Sep2022": "2022-09-30",
    "Mar2023": "2023-03-31", "Sep2023": "2023-09-29",
    "Mar2024": "2024-03-28", "Sep2024": "2024-09-30",
    "Mar2025": "2025-03-31", "Sep2025": "2025-09-30",
    "Mar2026": "2026-03-31",
}

ZIP_URL = (
    "https://www.niftyindices.com/"
    "Indices_-_Market_Capitalisation_and_Weightage/"
    "indices_data{}.zip"
)
CURRENT_CSV_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

# Words that look like symbols but aren't
NOISE_WORDS = {
    "NIFTY", "CONSTITUENTS", "INDEX", "MARCH", "SEPTEMBER", "SYMBOL",
    "SECURITY", "INDUSTRY", "CLOSE", "PRICE", "WEIGHTAGE", "MCAP", "OIL",
    "GAS", "FINANCIAL", "SERVICES", "IT", "PHARMA", "METALS", "CONSUMER",
    "GOODS", "POWER", "TELECOM", "CONSTRUCTION", "AUTOMOBILE", "CEMENT",
    "PRODUCTS", "MEDIA", "ENTERTAINMENT", "HEALTHCARE", "INDUSTRIAL",
    "MANUFACTURING", "FERTILISERS", "PESTICIDES", "TEXTILES", "DISCLAIMER",
    "NSE", "INDICES", "LIMITED", "IISL", "THE", "LTD", "AND", "ALL",
    "INFORMATION", "CONTAINED", "HEREWITH", "PROVIDED", "REFERENCE",
    "PURPOSE", "ONLY", "FORMERLY", "KNOWN", "INDIA", "MAKES",
    "WARRANTY", "REPRESENTATION", "ACCURACY", "COMPLETENESS",
    "RELIABILITY", "DISCLAIM", "LIABILITY", "WHATSOEVER", "PERSON",
    "DAMAGE", "LOSS", "NATURE", "ARISING", "USE", "SUCH", "DATA",
    "CAPITAL", "TOTAL", "CAPITAL",
}


# ── Download ──────────────────────────────────────────────────────────────
def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[1/3] Downloading {len(RECON_MONTHS)} ZIPs + current CSV...")

    for my in RECON_MONTHS:
        path = os.path.join(DATA_DIR, f"{my}.zip")
        if os.path.exists(path):
            print(f"  {my}.zip — cached ✓")
            continue
        url = ZIP_URL.format(my)
        print(f"  {my}.zip — downloading...", end="", flush=True)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f" ✓ ({len(r.content)//1024} KB)")
            else:
                print(f" ✗ HTTP {r.status_code}")
        except Exception as e:
            print(f" ✗ {e}")
        time.sleep(1.0)

    # Current live CSV
    csv_path = os.path.join(DATA_DIR, "nifty200_current.csv")
    print(f"  Current CSV...", end="", flush=True)
    try:
        r = requests.get(CURRENT_CSV_URL, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            with open(csv_path, "wb") as f:
                f.write(r.content)
            print(f" ✓ ({r.content.count(b',')//5} rows)")
        else:
            print(f" ✗ HTTP {r.status_code}")
    except Exception as e:
        print(f" ✗ {e}")


# ── PDF Parser ────────────────────────────────────────────────────────────
def _parse_nifty_pdf(pdf_path: str) -> list[str]:
    """Extract NSE stock symbols from an NIFTY index weightage PDF."""
    symbols = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

    lines = full_text.split("\n")
    logical_lines = []
    current_line = ""
    for line in lines:
        line = line.strip()
        if not line or "Disclaimer" in line or "Constituents" in line or "Symbol" in line or "(Rs" in line or "(%)" in line:
            continue
        current_line += " " + line
        if re.search(r'\d[\d,.]+\s+\d+\.\d+\s+\d[\d,.]+\s*$', current_line):
            logical_lines.append(current_line.strip())
            current_line = ""

    for line in logical_lines:
        line = re.sub(r'^(HDFCAMC)HDFC', r'\1 HDFC', line)
        line = re.sub(r'^(HDFCLIFE)HDFC', r'\1 HDFC', line)
        line = re.sub(r'^(ICICIGI)ICICI', r'\1 ICICI', line)
        line = re.sub(r'^(ICICIPRULI)ICICI', r'\1 ICICI', line)
        line = re.sub(r'^(SBILIFE)SBI', r'\1 SBI', line)
        line = re.sub(r'^(NAM-INDIA)Nippon', r'\1 Nippon', line)
        
        m = re.match(r'^([A-Z0-9&\-]{2,20}?)([A-Z][a-z]|\s)', line)
        if m:
            sym = m.group(1).strip()
            if sym not in NOISE_WORDS:
                symbols.append(sym)
                continue
                
        m2 = re.match(r'^([A-Z0-9&\-]+)', line)
        if m2:
            sym = m2.group(1).strip()
            if sym not in NOISE_WORDS:
                symbols.append(sym)

    return symbols


def _extract_pdf_from_zip(zip_path: str, pdf_pattern: str) -> str | None:
    """Extract a PDF matching pattern from a ZIP, return path or None."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if re.match(pdf_pattern, name, re.IGNORECASE):
                    z.extract(name, DATA_DIR)
                    return os.path.join(DATA_DIR, name)
    except (zipfile.BadZipFile, KeyError):
        pass
    return None


# ── Parse All Periods ─────────────────────────────────────────────────────
def parse_all() -> pd.DataFrame:
    print(f"\n[2/3] Parsing PDFs from each period...")
    all_records = []

    for my in RECON_MONTHS:
        zip_path = os.path.join(DATA_DIR, f"{my}.zip")
        if not os.path.exists(zip_path):
            print(f"  {my}: ZIP missing, skipping")
            continue

        effective_date = EFFECTIVE_DATES[my]
        symbols = []

        # Try full NIFTY_200 PDF first
        pdf = _extract_pdf_from_zip(zip_path, rf"NIFTY_200_{my}\.pdf")
        if pdf:
            symbols = _parse_nifty_pdf(pdf)
            os.remove(pdf)
            source = "NIFTY_200"
        else:
            # Fallback: combine NIFTY_50 + NIFTY_Next_50 (= Nifty 100)
            n50_pdf = _extract_pdf_from_zip(zip_path, rf"NIFTY_50_{my}\.pdf")
            nn50_pdf = _extract_pdf_from_zip(zip_path, rf"NIFTY_Next_50_{my}\.pdf")

            if n50_pdf:
                symbols.extend(_parse_nifty_pdf(n50_pdf))
                os.remove(n50_pdf)
            if nn50_pdf:
                symbols.extend(_parse_nifty_pdf(nn50_pdf))
                os.remove(nn50_pdf)
            source = "N50+NN50"

        # Deduplicate
        symbols = list(dict.fromkeys(symbols))

        for sym in symbols:
            all_records.append({
                "effective_date": effective_date,
                "period": my,
                "symbol": sym,
                "ticker": sym + ".NS",
                "source": source,
            })

        print(f"  {my}: {len(symbols):>3} stocks ({source})")

    # Add current live CSV data
    csv_path = os.path.join(DATA_DIR, "nifty200_current.csv")
    if os.path.exists(csv_path):
        try:
            current = pd.read_csv(csv_path)
            sym_col = "Symbol" if "Symbol" in current.columns else current.columns[2]
            current_symbols = current[sym_col].dropna().tolist()
            for sym in current_symbols:
                sym = str(sym).strip()
                if sym and sym not in NOISE_WORDS:
                    all_records.append({
                        "effective_date": "2026-04-28",  # Latest available
                        "period": "Apr2026",
                        "symbol": sym,
                        "ticker": sym + ".NS",
                        "source": "live_csv",
                    })
            print(f"  Apr2026: {len(current_symbols):>3} stocks (live_csv)")
        except Exception as e:
            print(f"  Current CSV error: {e}")

    df = pd.DataFrame(all_records)
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    return df


# ── Build Universe ────────────────────────────────────────────────────────
def build_universe(df: pd.DataFrame):
    print(f"\n[3/3] Building universe history...")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV} ({len(df)} rows)")

    superset = set(df["ticker"].unique())
    print(f"  Superset: {len(superset)} unique stocks across all periods")

    # Show timeline
    dates = sorted(df["effective_date"].unique())
    prev = set()

    print(f"\n  {'Period':<12} {'Count':>5} {'Source':<10} {'Added':>7} {'Removed':>7}")
    print(f"  {'─'*52}")

    for date in dates:
        rows = df[df["effective_date"] == date]
        current = set(rows["ticker"])
        source = rows["source"].iloc[0]
        added = current - prev if prev else set()
        removed = prev - current if prev else set()

        period = rows["period"].iloc[0]
        print(f"  {period:<12} {len(current):>5} {source:<10} "
              f"{'+'+ str(len(added)):>7} "
              f"{'-'+ str(len(removed)):>7}")
        prev = current

    # Save superset
    superset_path = os.path.join(DATA_DIR, "superset_tickers.txt")
    with open(superset_path, "w") as f:
        for t in sorted(superset):
            f.write(t + "\n")
    print(f"\n  Saved: {superset_path} ({len(superset)} tickers)")

    # Compare with our current nse200_tickers.py
    try:
        sys.path.insert(0, ".")
        from nse200_tickers import NSE200
        our_set = set(NSE200)
        missing_from_us = superset - our_set
        extra_in_us = our_set - superset
        print(f"\n  vs current nse200_tickers.py ({len(NSE200)} tickers):")
        print(f"    In superset but NOT in our list: {len(missing_from_us)}")
        if missing_from_us:
            print(f"      {sorted(missing_from_us)[:10]}...")
        print(f"    In our list but NOT in superset: {len(extra_in_us)}")
        if extra_in_us:
            print(f"      {sorted(extra_in_us)[:10]}...")
    except ImportError:
        pass

    return superset


def main():
    skip = "--skip-download" in sys.argv

    print("═" * 62)
    print(" UNIVERSE BUILDER — Historical Nifty 200 Constituents")
    print("═" * 62)

    if not skip:
        download_all()

    df = parse_all()
    superset = build_universe(df)

    print(f"\n{'═'*62}")
    print(f" RESULT: {len(superset)} unique stocks in superset")
    print(f"{'─'*62}")
    print(f" Data coverage:")
    print(f"   Full Nifty 200: Mar 2020 – Mar 2022 (5 periods)")
    print(f"   Nifty 100 only: Sep 2022 – Mar 2026 (8 periods)")
    print(f"   Current live:   Apr 2026 (full 200)")
    print(f"{'─'*62}")
    print(f" Next steps:")
    print(f"   1. Update nse200_tickers.py with superset")
    print(f"   2. Re-run: python3 screener_scraper.py")
    print(f"   3. Re-run: python3 build_pit.py")
    print(f"   4. Delete prices.csv and re-run: python3 main.py")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
