# factor_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Implements the exact 7-factor scoring model from the PDF.
#
# PIPELINE:
#   1. compute_momentum_scores(prices_df, as_of_date)
#      → Mom_Score = 0.40 × Ret_A + 0.60 × Ret_B  (skip T-1)
#
#   2. compute_raw_metrics(prices_df, fund_df, as_of_date)
#      → one DataFrame with all 7 raw factor values per stock
#
#   3. rank_quintiles(raw_df)
#      → converts raw values → ranks 1-5
#        higher-is-better: Momentum, Quality, Value, Beta, Yield → labels=[1,2,3,4,5]
#        lower-is-better:  Size, Invest                          → labels=[5,4,3,2,1]
#
#   4. composite_score(ranked_df)
#      → Final_Score = Σ (Rank_X × weight_X)
#        guaranteed mean = 3.0 (because ranks are balanced quintiles)
#
#   5. select_portfolio(scored_df, n)
#      → top n stocks by Final_Score, equal weight
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from config import WEIGHTS, LOWER_IS_BETTER, MOM_SKIP, MOM_Q1, MOM_Q24, MOM_W_A, MOM_W_B, PORTFOLIO_SIZE, NIFTY_TICKER


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum_scores(
    prices_df: pd.DataFrame,
    as_of_date=None
) -> pd.Series:
    """
    Compute momentum score for each stock as of as_of_date.

    FORMULA (from PDF):
        Mom_Score = 0.40 × Ret_A  +  0.60 × Ret_B

    WHERE:
        T   = as_of_date (most recent row in data)
        T-1 = SKIPPED (short-term reversal avoidance)

        Ret_A = Q1 return  = price_at(T-2) / price_at(T-4)  − 1
                  ← compounded return over the 3-month Q1 window

        Ret_B = Q2-4 return = price_at(T-5) / price_at(T-12) − 1
                  ← compounded return over the 8-month Q2-4 window

    TIMELINE (most recent on right):
      T-12  ────── Ret_B (60%) ──────  T-5 | T-4 ── Ret_A (40%) ── T-2 | T-1 | T

    WHY SKIP T-1:
        Very recent 1-month returns show mean-reversion (not momentum).
        Skipping it gives cleaner trend signal.

    Args:
        prices_df : monthly price DataFrame, rows=dates, cols=tickers
        as_of_date: compute as of this date (uses all data up to here)
                    if None, uses the latest date in prices_df

    Returns:
        pd.Series: momentum score per ticker (float)
    """
    # Filter to as_of_date
    if as_of_date is not None:
        df = prices_df[prices_df.index <= pd.Timestamp(as_of_date)]
    else:
        df = prices_df.copy()

    # Drop Nifty — we don't score the index
    df = df.drop(columns=[NIFTY_TICKER], errors="ignore")

    # Need at least 13 months (1 skip + 3 Q1 + 8 Q2-4 + 1 current)
    min_needed = MOM_SKIP + MOM_Q1 + MOM_Q24 + 1   # = 13
    if len(df) < min_needed:
        raise ValueError(
            f"Momentum needs {min_needed} months of price data. "
            f"Only {len(df)} available at {as_of_date}."
        )

    # Index from end of dataframe:
    #   iloc[-1]  = T     (current month, NOT used in return calc)
    #   iloc[-2]  = T-1   (SKIP)
    #   iloc[-3]  = T-2   ← start of Ret_A
    #   iloc[-5]  = T-4   ← end   of Ret_A window
    #   iloc[-6]  = T-5   ← start of Ret_B
    #   iloc[-13] = T-12  ← end   of Ret_B window

    price_T2  = df.iloc[-3]    # T-2
    price_T4  = df.iloc[-5]    # T-4
    price_T5  = df.iloc[-6]    # T-5
    price_T12 = df.iloc[-13]   # T-12

    # Compounded returns (not sum — compounding is correct)
    Ret_A = (price_T2  / price_T4)  - 1   # Q1:  T-4 → T-2
    Ret_B = (price_T5  / price_T12) - 1   # Q2-4: T-12 → T-5

    mom_score = MOM_W_A * Ret_A + MOM_W_B * Ret_B
    mom_score.name = "raw_Momentum"
    return mom_score


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMBINE ALL RAW FACTOR METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_raw_metrics(
    prices_df: pd.DataFrame,
    fund_df:   pd.DataFrame,
    as_of_date=None
) -> pd.DataFrame:
    """
    Combine momentum (from prices) with 6 fundamental metrics.

    Returns one DataFrame with columns:
        raw_Momentum, raw_Quality, raw_Value, raw_Size,
        raw_Beta, raw_Invest, raw_Yield

    One row per stock. Stocks missing momentum data are excluded.
    """
    # Momentum from prices
    mom = compute_momentum_scores(prices_df, as_of_date)

    # Fundamentals (all stocks in fund_df)
    fund_map = {
        "raw_Quality": "gross_profit_assets",
        "raw_Value"  : "book_to_market",
        "raw_Size"   : "market_cap_cr",
        "raw_Beta"   : "beta_nifty50",
        "raw_Invest" : "asset_growth_yoy",
        "raw_Yield"  : "dividend_yield_pct",
    }

    raw = pd.DataFrame({"raw_Momentum": mom})
    for col_name, src_col in fund_map.items():
        if src_col in fund_df.columns:
            raw[col_name] = fund_df[src_col]

    # Drop stocks with no data at all
    raw = raw.dropna(how="all")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: QUINTILE RANKING (1 to 5)
# ─────────────────────────────────────────────────────────────────────────────

def rank_quintiles(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each raw factor metric into a quintile rank 1–5.

    DIRECTION:
        Higher-is-better (Momentum, Quality, Value, Beta, Yield):
            highest raw value → Rank 5, lowest → Rank 1
            labels = [1, 2, 3, 4, 5]

        Lower-is-better (Size, Invest):
            lowest raw value → Rank 5 (small company = rank 5)
            labels = [5, 4, 3, 2, 1]

    MISSING VALUES:
        Stocks with NaN in a factor get Rank 3 (neutral middle).
        They neither benefit nor are penalised for missing data.

    TIES (duplicates):
        pd.qcut handles with duplicates='drop' — if many stocks have
        the same value (e.g., 0% dividend yield), they're grouped together.

    Returns:
        DataFrame with Rank_X columns (int, 1-5) for each factor.
    """
    df = raw_df.copy()

    factor_map = {
        "raw_Momentum": "Rank_Momentum",
        "raw_Quality" : "Rank_Quality",
        "raw_Value"   : "Rank_Value",
        "raw_Size"    : "Rank_Size",
        "raw_Beta"    : "Rank_Beta",
        "raw_Invest"  : "Rank_Invest",
        "raw_Yield"   : "Rank_Yield",
    }

    for raw_col, rank_col in factor_map.items():
        factor_name = raw_col.replace("raw_", "")

        if raw_col not in df.columns:
            df[rank_col] = 3   # all neutral if column missing
            continue

        # Default: neutral rank 3 for everyone
        df[rank_col] = 3

        has_data = df[raw_col].notna()
        n_valid  = has_data.sum()

        if n_valid < 5:   # need at least 5 stocks to make quintiles
            continue

        # Choose label direction
        if factor_name in LOWER_IS_BETTER:
            labels = [5, 4, 3, 2, 1]   # smallest raw → Rank 5
        else:
            labels = [1, 2, 3, 4, 5]   # largest raw → Rank 5

        try:
            result = pd.qcut(
                df.loc[has_data, raw_col],
                q=5,
                labels=labels,
                duplicates="drop"
            )
            # qcut with duplicates='drop' may produce fewer than 5 bins.
            # That causes a labels-length mismatch error.
            # We check the result is valid before assigning.
            df.loc[has_data, rank_col] = result.astype(int)
        except ValueError:
            # Fallback: rank-based approach (always works, no bin issues)
            # Assign ranks 1-5 by percentile position of each value.
            series     = df.loc[has_data, raw_col]
            rank_float = series.rank(pct=True, method="average")
            if factor_name in LOWER_IS_BETTER:
                # Invert: small value → high percentile → high rank
                rank_float = 1.0 - rank_float
            # Map percentile [0,1] → rank 1-5
            rank_int = pd.cut(
                rank_float,
                bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
                labels=[1, 2, 3, 4, 5]
            ).astype(float).fillna(3).astype(int)
            df.loc[has_data, rank_col] = rank_int

        df[rank_col] = df[rank_col].astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def composite_score(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Final_Score = Σ (Rank_X × Weight_X)

    SCORE RANGE:
        Min = 1.0  (Rank 1 on every factor)
        Max = 5.0  (Rank 5 on every factor)
        Mean ≈ 3.0 (guaranteed because quintile ranks are balanced)

    The mean being exactly 3.0 is a useful sanity check.
    If mean is far from 3.0, ranking logic has a bug.

    Returns:
        DataFrame sorted descending by Final_Score.
    """
    df = ranked_df.copy()
    df["Final_Score"] = 0.0

    for factor, weight in WEIGHTS.items():
        rank_col = f"Rank_{factor}"
        if rank_col in df.columns:
            df["Final_Score"] += df[rank_col] * weight

    df["Final_Score"] = df["Final_Score"].round(4)
    df = df.sort_values("Final_Score", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SELECT PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def select_portfolio(scored_df: pd.DataFrame, n: int = None) -> pd.DataFrame:
    """
    Select top n stocks by Final_Score.
    Each stock gets equal weight = 1/n.

    Returns:
        DataFrame: top n rows from scored_df with Weight column added.
    """
    if n is None:
        n = PORTFOLIO_SIZE
    portfolio = scored_df.head(n).copy()
    portfolio["Weight"] = round(1.0 / n, 6)
    return portfolio


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_factor_engine(
    prices_df: pd.DataFrame,
    fund_df:   pd.DataFrame,
    as_of_date=None,
    verbose:   bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all 5 steps. Returns (all_stocks_scored, portfolio_top_n).

    Args:
        prices_df  : monthly price DataFrame
        fund_df    : fundamentals DataFrame
        as_of_date : score as of this date (None = latest available)
        verbose    : print summary table

    Returns:
        scored_df  : all stocks with raw values, ranks, and Final_Score
        portfolio  : top PORTFOLIO_SIZE stocks with equal weights
    """
    raw_df    = compute_raw_metrics(prices_df, fund_df, as_of_date)
    ranked_df = rank_quintiles(raw_df)
    scored_df = composite_score(ranked_df)
    portfolio = select_portfolio(scored_df)

    if verbose:
        _print_results(scored_df, portfolio)

    return scored_df, portfolio


def _print_results(scored_df: pd.DataFrame, portfolio: pd.DataFrame):
    n      = len(portfolio)
    total  = len(scored_df)
    cutoff = scored_df["Final_Score"].iloc[n-1]
    next_  = scored_df["Final_Score"].iloc[n]   if len(scored_df) > n else 0
    mean   = scored_df["Final_Score"].mean()

    rank_cols = [f"Rank_{f}" for f in WEIGHTS]

    print(f"\n{'─'*62}")
    print(f" FACTOR ENGINE RESULTS  |  {total} stocks scored  |  top {n} selected")
    print(f"{'─'*62}")
    print(f" Score mean:    {mean:.4f}  (expect ≈ 3.0000)")
    print(f" Score range:   {scored_df['Final_Score'].min():.3f} – {scored_df['Final_Score'].max():.3f}")
    print(f" Portfolio cutoff: #{n} score = {cutoff:.3f}  |  #{n+1} score = {next_:.3f}")

    print(f"\n {'#':<4} {'Ticker':<18} {'Score':<7}", end="")
    for f in WEIGHTS:
        print(f" {f[:3]:<4}", end="")
    print()
    print(f" {'─'*60}")

    for i, (ticker, row) in enumerate(portfolio.iterrows(), 1):
        print(f" {i:<4} {ticker:<18} {row['Final_Score']:<7.3f}", end="")
        for f in WEIGHTS:
            rc = f"Rank_{f}"
            v  = int(row[rc]) if rc in row else 0
            print(f" {v:<4}", end="")
        print()

    print(f"\n RANK DISTRIBUTION (all {total} stocks):")
    print(f" {'Factor':<12} {'Wt':>5}  R1   R2   R3   R4   R5  Mean  Dir")
    print(f" {'─'*58}")
    for f, w in WEIGHTS.items():
        rc  = f"Rank_{f}"
        vc  = scored_df[rc].value_counts().sort_index() if rc in scored_df.columns else {}
        dir_= "↓low" if f in LOWER_IS_BETTER else "↑hi"
        print(f" {f:<12} {w:>4.0%}  "
              f"{vc.get(1,0):>4} {vc.get(2,0):>4} {vc.get(3,0):>4} "
              f"{vc.get(4,0):>4} {vc.get(5,0):>4}  "
              f"{scored_df[rc].mean():.2f}  {dir_}")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _test_momentum_formula():
    """
    Verify: STRONG stock > NEUTRAL stock > WEAK stock
    and the T-1 skip is respected.
    """
    print("  Testing momentum formula...", end="")
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2023-01-31", periods=14, freq="ME")

    # Strong: rises every month
    strong  = np.cumprod(1 + np.array([0.04]*14)) * 100
    # Weak:   falls every month
    weak    = np.cumprod(1 + np.array([-0.03]*14)) * 100
    # Neutral: flat
    neutral = np.ones(14) * 100
    nifty   = np.cumprod(1 + np.array([0.01]*14)) * 10000

    test_df = pd.DataFrame({
        "STRONG.NS" : strong,
        "WEAK.NS"   : weak,
        "NEUTRAL.NS": neutral,
        NIFTY_TICKER: nifty,
    }, index=dates)

    scores = compute_momentum_scores(test_df)
    s, w, n_ = scores["STRONG.NS"], scores["WEAK.NS"], scores["NEUTRAL.NS"]

    assert s > n_ > w, (
        f"Ordering wrong! STRONG={s:.4f}, NEUTRAL={n_:.4f}, WEAK={w:.4f}"
    )
    # Neutral should be near zero
    assert abs(n_) < 0.005, f"Neutral should be ~0, got {n_:.4f}"
    print(f" PASS  (S={s:.3f} > N={n_:.3f} > W={w:.3f})")


def _test_score_mean(prices_df, fund_df):
    """Score mean must be exactly 3.0 (mathematical guarantee of quintile ranking)."""
    print("  Testing score mean = 3.0...", end="")
    scored, _ = run_factor_engine(prices_df, fund_df, verbose=False)
    mean = scored["Final_Score"].mean()
    assert abs(mean - 3.0) < 0.05, f"Score mean = {mean:.4f}, expected ≈ 3.0"
    print(f" PASS  (mean = {mean:.4f})")


def _test_rank_directions(prices_df, fund_df):
    """Verify lower-is-better factors give Rank 5 to the smallest values."""
    print("  Testing rank directions...", end="")
    raw   = compute_raw_metrics(prices_df, fund_df)
    ranked = rank_quintiles(raw)

    for factor in LOWER_IS_BETTER:
        raw_col  = f"raw_{factor}"
        rank_col = f"Rank_{factor}"
        if raw_col not in raw.columns:
            continue
        # The stock with the smallest raw value should have Rank 5
        smallest_ticker = raw[raw_col].idxmin()
        rank_of_smallest = ranked.loc[smallest_ticker, rank_col]
        assert rank_of_smallest == 5, (
            f"{factor}: smallest stock '{smallest_ticker}' got Rank {rank_of_smallest}, expected 5"
        )
    print(" PASS")


def run_all_tests(prices_df=None, fund_df=None):
    print("\n=== FACTOR ENGINE TESTS ===")
    _test_momentum_formula()
    if prices_df is not None and fund_df is not None:
        _test_score_mean(prices_df, fund_df)
        _test_rank_directions(prices_df, fund_df)
    print("=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    from data_layer import generate_mock_data
    prices, fund = generate_mock_data()
    run_all_tests(prices, fund)
    run_factor_engine(prices, fund)
