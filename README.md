# NSE 200 Multi-Factor Portfolio Research Engine

A quantitative research backtesting system that constructs and backtests a 7-factor portfolio using Nifty 200 constituents. The factors are weighted as follows:

| Factor | Weight | Direction | Source |
|--------|--------|-----------|--------|
| Momentum (12-1 month) | 30% | ↑ high is better | yfinance prices |
| Quality (Gross Profit / Assets) | 20% | ↑ high is better | Screener.in |
| Value (Book-to-Market) | 15% | ↑ high is better | Screener.in + yfinance |
| Size (Market Cap) | 12% | ↓ small is better | yfinance |
| Beta (vs Nifty 50) | 10% | ↑ high is better | yfinance prices |
| Investment (Asset Growth YoY) | 7% | ↓ low is better | Screener.in |
| Yield (Dividend Yield) | 6% | ↑ high is better | Screener.in |

---

## Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.13)
- **Java JRE 8+** — required by `tabula-py` for PDF parsing in `universe_builder.py`
  - macOS: `brew install java`
  - Ubuntu: `sudo apt install default-jre`

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd Research-Investing

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Activate venv (if not already active)
source venv/bin/activate

# (Optional) Validate data quality before running
python3 data_validator.py

# Run the full pipeline: download data → score → backtest → dashboard
python3 main.py
```

On the **first run**, `main.py` will automatically:
1. Download monthly prices from yfinance (~5 min)
2. Scrape fundamentals from Screener.in (~10 min)
3. Build the Point-in-Time database
4. Score all stocks and run the backtest
5. Generate `output/dashboard.html`

Subsequent runs use cached data files and complete in seconds.

---

## Project Structure

```
Research-Investing/
├── main.py                  # Entry point — runs the full pipeline
├── config.py                # Backtest parameters (dates, weights, capital)
├── nse200_tickers.py        # Cleaned NSE 200 ticker list
├── requirements.txt         # Python dependencies
│
├── data_layer.py            # Price download, PiT snapshot, shares outstanding
├── factor_engine.py         # 7-factor scoring (quintile ranks 1–5)
├── backtester.py            # Backtest execution with transaction costs
├── analytics.py             # Performance metrics (CAGR, Sharpe, drawdowns)
├── dashboard.py             # HTML dashboard generator
│
├── screener_scraper.py      # Scrapes Screener.in for fundamentals
├── build_pit.py             # Builds Point-in-Time fundamental database
├── universe_builder.py      # Parses NSE 200 historical constituent lists
├── data_validator.py        # Pre-run data quality checker
├── diagnose.py              # Data endpoint diagnostics
│
├── data/universe/           # Historical NSE 200 constituent PDFs/CSVs
├── output/                  # Generated outputs
│   ├── dashboard.html       # Interactive HTML dashboard
│   ├── all_scores.csv       # All stocks with factor scores
│   ├── portfolio.csv        # Top 25 portfolio holdings
│   ├── equity_curves.csv    # Portfolio vs Nifty 50 over time
│   └── rebalance_log.csv    # Rebalance events with costs
│
├── prices.csv               # Cached monthly prices (auto-generated)
├── screener_raw.csv         # Cached Screener data (auto-generated)
├── fundamentals_pit.csv     # Cached PiT database (auto-generated)
└── shares_outstanding.csv   # Cached shares data (auto-generated)
```

---

## Pipeline Steps

### Step 1: Build Universe (one-time)
```bash
python3 universe_builder.py
```
Parses historical NSE 200 PDFs (in `data/universe/`) to build `universe_history_interpolated.csv` — the list of which stocks were in the index at each semi-annual review.

### Step 2: Run Full Backtest
```bash
python3 main.py
```
This runs all steps in sequence: data download → scoring → backtest → dashboard.

### Step 3: View Results
```bash
open output/dashboard.html       # macOS
xdg-open output/dashboard.html   # Linux
```

---

## Data Refresh

Cached data files (`prices.csv`, `screener_raw.csv`, etc.) are auto-generated and reused on subsequent runs. To force a fresh download:

```bash
# Delete cached files and re-run
rm prices.csv screener_raw.csv fundamentals_pit.csv shares_outstanding.csv
python3 main.py
```

Or to refresh individual components:

```bash
# Re-scrape Screener.in only
python3 screener_scraper.py

# Rebuild PiT database from screener_raw.csv
python3 build_pit.py

# Validate data quality
python3 data_validator.py
```

---

## Configuration

Edit [config.py](config.py) to change:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INCEPTION_DATE` | `2020-02-01` | Backtest start date |
| `BACKTEST_END` | `2026-04-01` | Backtest end date |
| `INITIAL_CAPITAL` | `500000` | Starting capital (₹) |
| `PORTFOLIO_SIZE` | `25` | Number of stocks in portfolio |
| `REBALANCE_MONTHS` | `[3, 9]` | Months to rebalance (Mar, Sep) |
| `COST_PCT` | `0.003` | Transaction cost per side (0.3%) |
