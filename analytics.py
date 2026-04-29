"""
analytics.py — derived analytics for the Phase-2 dashboard.

Everything in here is COMPUTED from the saved per-rebalance state in
backtest_results['scored_history'] and ['portfolio_history']. No hardcoded
percentages, correlations, or regime tables.

Provides:
  - run_single_factor_backtest()  : isolate one factor's contribution
  - compute_factor_correlations() : cross-sectional rank correlation matrix
  - compute_factor_ic()           : Information Coefficient per factor
  - compute_quintile_spreads()    : Q5 minus Q1 forward-return spread
  - compute_rolling_metrics()     : rolling 12M Sharpe / alpha / beta
  - compute_per_rebalance_attribution() : winners/losers each period
  - compute_sector_exposure()     : sector weights over time
  - compute_factor_regime()       : per-year factor performance (Q5 - Q1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    WEIGHTS, LOWER_IS_BETTER, PORTFOLIO_SIZE, NIFTY_TICKER,
    TRANSACTION_COST_PCT, RISK_FREE_RATE, BACKTEST_END,
)
from sectors import sector_of


# ─────────────────────────────────────────────────────────────────────────────
# 1. SINGLE-FACTOR BACKTEST  (replaces the hardcoded 47.5% attribution)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_factor_backtest(
    factor: str,
    scored_history: dict,
    prices_df: pd.DataFrame,
    rebalance_dates: list,
    initial_capital: float = 500_000,
    n_select: int = PORTFOLIO_SIZE,
) -> pd.Series:
    """
    Build the equity curve of a portfolio that ranks ONLY by `factor`.

    For each rebalance date in scored_history, take the top-n stocks by that
    factor's rank (or bottom-n if factor ∈ LOWER_IS_BETTER), equal-weight, then
    walk monthly until the next rebalance using the same monthly logic as the
    main backtester. Cost charged at TRANSACTION_COST_PCT × turnover × 2.

    Returns a monthly equity Series — directly comparable to the main portfolio.
    """
    rank_col = f"Rank_{factor}"
    raw_col  = f"raw_{factor}"
    ascending = factor in LOWER_IS_BETTER   # lower-is-better → take smallest

    # We rank on the RAW value (not the bucketed Rank_) for finer granularity.
    # Falls back to Rank_ if raw is unavailable for some reason.
    sort_col = raw_col

    all_dates = pd.DatetimeIndex(prices_df.index).sort_values()
    equity = []
    value  = float(initial_capital)
    held: set = set()

    for i, rd in enumerate(rebalance_dates):
        if rd not in scored_history:
            continue
        sdf = scored_history[rd]
        if sort_col not in sdf.columns:
            sort_col_used = rank_col
        else:
            sort_col_used = sort_col

        sub = sdf[sdf[sort_col_used].notna()].sort_values(
            sort_col_used, ascending=ascending
        )
        new_held = set(sub.head(n_select).index.tolist())

        # Cost
        if held:
            old_w = pd.Series({t: 1.0/n_select for t in held})
            new_w = pd.Series({t: 1.0/n_select for t in new_held})
            all_t = old_w.index.union(new_w.index)
            turnover = float(0.5 * (new_w.reindex(all_t, fill_value=0)
                                    - old_w.reindex(all_t, fill_value=0)).abs().sum())
            cost = value * turnover * TRANSACTION_COST_PCT * 2
        else:
            cost = value * TRANSACTION_COST_PCT * 1.0
        value -= cost

        if not equity:
            equity.append((rd, value))
        else:
            equity[-1] = (rd, value)

        # Walk monthly to next rebal
        next_rd = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else pd.Timestamp(BACKTEST_END)
        period_months = all_dates[(all_dates > rd) & (all_dates <= next_rd)]

        held_cols = [t for t in new_held if t in prices_df.columns]
        prev_dt = rd
        for m_dt in period_months:
            try:
                p_prev = prices_df.loc[prev_dt, held_cols]
                p_curr = prices_df.loc[m_dt,    held_cols]
            except KeyError:
                prev_dt = m_dt
                continue
            mret = (p_curr / p_prev - 1).where(p_prev > 0).fillna(0.0)
            value *= (1 + float(mret.sum() / n_select))
            equity.append((m_dt, value))
            prev_dt = m_dt

        held = new_held

    if not equity:
        return pd.Series(dtype=float)
    s = pd.Series(dict(equity)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = factor
    return s


def run_all_single_factor_backtests(
    scored_history: dict, prices_df: pd.DataFrame,
    rebalance_dates: list, initial_capital: float = 500_000,
) -> pd.DataFrame:
    """Run one single-factor backtest per factor in WEIGHTS. Returns a DataFrame
    of monthly equity curves, columns = factor names."""
    out = {}
    for f in WEIGHTS:
        out[f] = run_single_factor_backtest(
            f, scored_history, prices_df, rebalance_dates, initial_capital
        )
    df = pd.DataFrame(out)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FACTOR CORRELATIONS  (replaces the hardcoded 7×7 matrix)
# ─────────────────────────────────────────────────────────────────────────────

def compute_factor_correlations(scored_history: dict) -> pd.DataFrame:
    """
    Average cross-sectional Spearman correlation of factor *raw* values across
    all rebalance dates. Output is the symmetric 7×7 matrix used in the
    Diagnostics tab.
    """
    if not scored_history:
        return pd.DataFrame()
    factors = list(WEIGHTS.keys())
    raw_cols = [f"raw_{f}" for f in factors]

    mats = []
    for sdf in scored_history.values():
        cols_present = [c for c in raw_cols if c in sdf.columns]
        if len(cols_present) < 2:
            continue
        m = sdf[cols_present].corr(method="spearman")
        # reindex to full factor set so we can average
        m = m.reindex(index=raw_cols, columns=raw_cols)
        mats.append(m)
    if not mats:
        return pd.DataFrame(index=factors, columns=factors)
    avg = sum(mats) / len(mats)
    avg.index   = factors
    avg.columns = factors
    return avg.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. INFORMATION COEFFICIENT (IC) — gold-standard factor diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def compute_factor_ic(
    scored_history: dict, prices_df: pd.DataFrame, rebalance_dates: list,
) -> pd.DataFrame:
    """
    For each rebalance date, compute Spearman rank correlation between every
    factor's raw value (at t) and the forward period return (t → t+1).

    Returns a DataFrame indexed by rebalance date, columns = factor names,
    plus 'mean_ic' and 'ic_std' summary rows on the resulting frame's .attrs.

    A factor is "predictive" if its mean IC is meaningfully > 0 (a value of
    0.05 is decent for a quant factor; 0.10+ is strong).
    """
    factors = list(WEIGHTS.keys())
    rows = {}
    for i, rd in enumerate(rebalance_dates[:-1]):
        next_rd = rebalance_dates[i+1]
        if rd not in scored_history:
            continue
        sdf = scored_history[rd]

        # Forward return for each ticker over (rd, next_rd]
        try:
            p0 = prices_df.loc[rd]
            p1 = prices_df.loc[next_rd]
        except KeyError:
            continue
        fwd = (p1 / p0 - 1).where(p0 > 0)

        ic_row = {}
        for f in factors:
            raw_col = f"raw_{f}"
            if raw_col not in sdf.columns:
                ic_row[f] = np.nan
                continue
            x = sdf[raw_col]
            common = x.index.intersection(fwd.index)
            if len(common) < 10:
                ic_row[f] = np.nan
                continue
            xv  = x.loc[common]
            yv  = fwd.loc[common]
            mask = xv.notna() & yv.notna()
            if mask.sum() < 10:
                ic_row[f] = np.nan
                continue
            corr = xv[mask].rank().corr(yv[mask].rank())   # Spearman
            # Flip sign for lower-is-better factors so a positive IC means the
            # factor is *correctly* ordered for the strategy.
            if f in LOWER_IS_BETTER:
                corr = -corr
            ic_row[f] = float(corr)
        rows[rd] = ic_row

    df = pd.DataFrame(rows).T
    df.index.name = "rebalance_date"
    df.attrs["mean_ic"] = df.mean().to_dict()
    df.attrs["ic_std"]  = df.std().to_dict()
    df.attrs["hit_rate"] = (df > 0).mean().to_dict()   # % of periods with positive IC
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUINTILE SPREAD — does the model actually rank?
# ─────────────────────────────────────────────────────────────────────────────

def compute_quintile_spreads(
    scored_history: dict, prices_df: pd.DataFrame, rebalance_dates: list,
) -> pd.DataFrame:
    """
    For each rebalance date and each factor, compute the equal-weighted forward
    return of the top quintile (Rank 5) minus the bottom quintile (Rank 1).

    A factor that genuinely ranks should produce consistently positive spreads.
    """
    factors = list(WEIGHTS.keys())
    rows = {}
    for i, rd in enumerate(rebalance_dates[:-1]):
        next_rd = rebalance_dates[i+1]
        if rd not in scored_history:
            continue
        sdf = scored_history[rd]
        try:
            p0 = prices_df.loc[rd]
            p1 = prices_df.loc[next_rd]
        except KeyError:
            continue
        fwd = (p1 / p0 - 1).where(p0 > 0)

        sp_row = {}
        for f in factors:
            rcol = f"Rank_{f}"
            if rcol not in sdf.columns:
                sp_row[f] = np.nan
                continue
            top    = sdf[sdf[rcol] == 5].index
            bottom = sdf[sdf[rcol] == 1].index
            top    = top.intersection(fwd.index)
            bottom = bottom.intersection(fwd.index)
            if len(top) < 3 or len(bottom) < 3:
                sp_row[f] = np.nan
                continue
            top_ret    = float(fwd.loc[top].mean())
            bottom_ret = float(fwd.loc[bottom].mean())
            sp_row[f]  = top_ret - bottom_ret
        rows[rd] = sp_row

    df = pd.DataFrame(rows).T
    df.index.name = "rebalance_date"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROLLING METRICS — 12M trailing Sharpe / Alpha / Beta
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_metrics(
    portfolio_curve: pd.Series,
    nifty_curve:     pd.Series,
    window_months:   int = 12,
) -> pd.DataFrame:
    """
    Trailing rolling-window metrics computed off MONTHLY equity curves.

    Returns columns: rolling_sharpe, rolling_alpha (annualised excess vs Nifty),
    rolling_beta (OLS slope of portfolio rets on Nifty rets).
    """
    pr = portfolio_curve.pct_change().dropna()
    nr = nifty_curve.pct_change().dropna() if not nifty_curve.empty else None
    if nr is not None:
        common = pr.index.intersection(nr.index)
        pr = pr.loc[common]
        nr = nr.loc[common]

    rf_m = (1 + RISK_FREE_RATE) ** (1/12) - 1

    sharpe  = ((pr - rf_m).rolling(window_months).mean()
               / pr.rolling(window_months).std()) * np.sqrt(12)

    if nr is not None and len(nr):
        # Rolling alpha: portfolio ann_ret minus nifty ann_ret over the window
        port_ann  = (1 + pr).rolling(window_months).apply(np.prod, raw=True) ** (12/window_months) - 1
        nifty_ann = (1 + nr).rolling(window_months).apply(np.prod, raw=True) ** (12/window_months) - 1
        alpha = port_ann - nifty_ann

        # Rolling beta via covariance / variance
        cov  = pr.rolling(window_months).cov(nr)
        var  = nr.rolling(window_months).var()
        beta = cov / var
    else:
        alpha = pd.Series(np.nan, index=pr.index)
        beta  = pd.Series(np.nan, index=pr.index)

    out = pd.DataFrame({
        "rolling_sharpe": sharpe,
        "rolling_alpha":  alpha,
        "rolling_beta":   beta,
    })
    return out.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PER-REBALANCE ATTRIBUTION — winners and losers each period
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_rebalance_attribution(
    portfolio_history: dict, prices_df: pd.DataFrame,
    rebalance_dates: list, n_top: int = 5,
) -> dict:
    """
    For each rebalance period (rd → next_rd), compute the per-stock return of
    every held name and tag the top-n winners and bottom-n losers.

    Returns dict[rebalance_date] -> dict with keys:
        held_returns    : Series(ticker -> return)
        top_winners     : list of (ticker, return)
        top_losers      : list of (ticker, return)
        period_return   : equal-weight period return
    """
    out = {}
    for i, rd in enumerate(rebalance_dates):
        if rd not in portfolio_history:
            continue
        next_rd = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else pd.Timestamp(BACKTEST_END)
        held = portfolio_history[rd]
        held_in_px = [t for t in held if t in prices_df.columns]
        if not held_in_px:
            continue
        try:
            p0 = prices_df.loc[rd, held_in_px]
            p1 = prices_df.loc[next_rd, held_in_px]
        except KeyError:
            continue
        rets = (p1 / p0 - 1).where(p0 > 0).dropna().sort_values(ascending=False)
        out[rd] = {
            "held_returns":  rets,
            "top_winners":   list(zip(rets.head(n_top).index.tolist(),
                                      rets.head(n_top).round(4).tolist())),
            "top_losers":    list(zip(rets.tail(n_top).index.tolist(),
                                      rets.tail(n_top).round(4).tolist())),
            "period_return": float(rets.mean()),
            "next_date":     next_rd,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. SECTOR EXPOSURE OVER TIME
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_exposure(portfolio_history: dict) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by rebalance date, columns = sector,
    values = portfolio weight in that sector (fraction in [0, 1]).
    """
    rows = {}
    for rd, held in portfolio_history.items():
        n = len(held) if held else 1
        sec_counts: dict[str, int] = {}
        for t in held:
            s = sector_of(t)
            sec_counts[s] = sec_counts.get(s, 0) + 1
        rows[rd] = {s: c / n for s, c in sec_counts.items()}
    df = pd.DataFrame(rows).T.fillna(0.0).sort_index()
    df.index.name = "rebalance_date"
    return df.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FACTOR REGIME — per-year quintile-spread (computed, not hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

def compute_factor_regime(quintile_spreads: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate quintile spreads into per-year factor performance. Replaces the
    hardcoded 'regime' table on the old dashboard with real numbers.

    Per-year value = mean spread observed across rebalances in that year × 100
    (so units are %). Positive means the factor's top quintile beat its bottom.
    """
    if quintile_spreads.empty:
        return pd.DataFrame()
    df = quintile_spreads.copy()
    df["year"] = pd.DatetimeIndex(df.index).year
    out = df.groupby("year").mean(numeric_only=True) * 100
    return out.round(2)


# ─────────────────────────────────────────────────────────────────────────────
# 9. PER-STOCK SCORE HISTORY — for the Stocks drill-down tab
# ─────────────────────────────────────────────────────────────────────────────

def compute_stock_score_history(scored_history: dict) -> dict:
    """
    Pivot scored_history into per-ticker time series.

    Returns dict[ticker] -> DataFrame indexed by rebalance date with columns:
        Final_Score, Rank_Momentum, Rank_Quality, ..., raw_Momentum, ...
    Used by the dashboard's Stocks drill-down.
    """
    if not scored_history:
        return {}
    factors = list(WEIGHTS.keys())
    cols = ["Final_Score"] + [f"Rank_{f}" for f in factors] + [f"raw_{f}" for f in factors]

    per_ticker: dict[str, list[dict]] = {}
    for rd, sdf in scored_history.items():
        for tk in sdf.index:
            row = {"date": rd}
            for c in cols:
                if c in sdf.columns:
                    v = sdf.loc[tk, c]
                    row[c] = float(v) if pd.notna(v) else None
            per_ticker.setdefault(tk, []).append(row)

    out = {}
    for tk, rows in per_ticker.items():
        df = pd.DataFrame(rows).set_index("date").sort_index()
        out[tk] = df
    return out
