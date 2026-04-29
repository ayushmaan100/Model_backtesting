# backtester.py
# ─────────────────────────────────────────────────────────────────────────────
# Simulates the portfolio's historical performance.
#
# HOW IT WORKS:
#   1. Identify all rebalance dates (June and December month-ends)
#   2. At each rebalance date:
#        a. Score all stocks using ONLY data available AT THAT DATE
#           (no look-ahead bias)
#        b. Select top 25 portfolio
#        c. Calculate transaction costs for stocks that changed
#        d. Hold the portfolio until the next rebalance date
#        e. Record the return earned during the holding period
#   3. Compound all returns into an equity curve
#   4. Compute performance metrics from the equity curve
#
# LOOK-AHEAD BIAS PREVENTION:
#   At the June 2021 rebalance, we only use prices up to June 2021.
#   We never peek at July 2021 prices to make June 2021 decisions.
#   This is enforced by slicing prices_df to each rebalance date.
#
# EQUAL-WEIGHT PORTFOLIO:
#   Each of the 25 stocks gets 1/25 of capital at each rebalance.
#   Period return = simple average of all 25 stock returns over the period.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from datetime import date

from config import (
    BACKTEST_START, BACKTEST_END, REBALANCE_MONTHS,
    INCEPTION_DATE, PORTFOLIO_SIZE, TRANSACTION_COST_PCT,
    RISK_FREE_RATE, NIFTY_TICKER,
)
from factor_engine import run_factor_engine


# ─────────────────────────────────────────────────────────────────────────────
# REBALANCE DATES
# ─────────────────────────────────────────────────────────────────────────────

def get_rebalance_dates(prices_df: pd.DataFrame) -> list:
    """
    Build the rebalance schedule:

      1. INCEPTION_DATE  (one-off, snapped to the nearest available month-end
         in prices_df). Aligns the backtest with the PDF's "since Feb 2020"
         analysis window so the COVID drawdown is included.
      2. Every month-end in prices_df whose month is in REBALANCE_MONTHS
         and which falls strictly *after* inception and within the window.
    """
    available = pd.DatetimeIndex(prices_df.index)
    if available.empty:
        return []

    inception_target = pd.Timestamp(INCEPTION_DATE)
    end_target       = pd.Timestamp(BACKTEST_END)

    # Snap inception to the closest available month-end on or after the target.
    candidates = available[available >= inception_target]
    if candidates.empty:
        return []
    inception_actual = candidates[0]

    rebalance = [inception_actual]
    for dt in available:
        if (dt > inception_actual
                and dt <= end_target
                and dt.month in REBALANCE_MONTHS):
            rebalance.append(dt)

    # Dedup (in case inception happens to land on a Jun/Dec) and sort
    return sorted(set(rebalance))


# ─────────────────────────────────────────────────────────────────────────────
# PERIOD RETURN
# ─────────────────────────────────────────────────────────────────────────────

def compute_period_return(
    portfolio_tickers: list,
    prices_df:         pd.DataFrame,
    period_start:      pd.Timestamp,
    period_end:        pd.Timestamp,
) -> tuple[float, dict]:
    """
    Compute equal-weighted portfolio return between two dates.

    For each stock:
        return = price_at(period_end) / price_at(period_start) − 1

    Portfolio return = simple mean of all stock returns.

    Args:
        portfolio_tickers: list of ticker strings
        prices_df:         full price DataFrame
        period_start:      date of previous rebalance (buy price)
        period_end:        date of next rebalance (sell price)

    Returns:
        portfolio_return (float), stock_returns (dict ticker→float)
    """
    stock_rets = {}
    n_portfolio = len(portfolio_tickers)

    for ticker in portfolio_tickers:
        if ticker not in prices_df.columns:
            stock_rets[ticker] = 0.0  # Missing stock = 0% return (not excluded)
            continue

        # Price at period_start
        prices_before = prices_df.loc[
            prices_df.index <= period_start, ticker
        ].dropna()
        if prices_before.empty:
            stock_rets[ticker] = 0.0
            continue

        # Price at period_end
        prices_at_end = prices_df.loc[
            prices_df.index <= period_end, ticker
        ].dropna()
        if prices_at_end.empty:
            stock_rets[ticker] = 0.0
            continue

        p_start = prices_before.iloc[-1]
        p_end   = prices_at_end.iloc[-1]

        if p_start > 0:
            stock_rets[ticker] = (p_end / p_start) - 1
        else:
            stock_rets[ticker] = 0.0

    if not stock_rets:
        return 0.0, {}

    # Equal-weight: always divide by portfolio size, not by valid count
    portfolio_return = float(sum(stock_rets.values()) / n_portfolio)
    return portfolio_return, stock_rets


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(equity_curve: pd.Series, label: str = "Portfolio") -> dict:
    """
    Performance metrics from a (now monthly) equity curve.

    Sharpe / Sortino use the proper definition:
        sharpe = mean(period_excess) / stdev(period_excess) * sqrt(periods_per_yr)
    instead of the ad-hoc (CAGR - rf)/ann_vol mix used previously.
    """
    period_rets = equity_curve.pct_change().dropna()
    if len(period_rets) == 0:
        return {}

    n_days  = (equity_curve.index[-1] - equity_curve.index[0]).days
    n_years = max(n_days / 365.25, 0.01)
    cagr    = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1

    # Detect period frequency dynamically. With a monthly curve this is ~12.
    periods_per_yr = len(period_rets) / n_years if n_years > 0 else 12

    running_max = equity_curve.cummax()
    max_dd      = float(((equity_curve - running_max) / running_max).min())

    rf_periodic = (1 + RISK_FREE_RATE) ** (1 / periods_per_yr) - 1
    excess_rets = period_rets - rf_periodic

    ann_vol = float(period_rets.std() * np.sqrt(periods_per_yr))

    if excess_rets.std() > 0:
        sharpe = float(excess_rets.mean() / excess_rets.std() * np.sqrt(periods_per_yr))
    else:
        sharpe = 0.0

    down = excess_rets[excess_rets < 0]
    if len(down) > 0 and down.std() > 0:
        sortino = float(excess_rets.mean() / down.std() * np.sqrt(periods_per_yr))
    else:
        sortino = 0.0

    # Real annual returns (Jan→Dec compounded). Requires monthly granularity to be honest.
    annual_returns = {}
    for yr in sorted(equity_curve.index.year.unique()):
        yr_rets = period_rets[period_rets.index.year == yr]
        if len(yr_rets) > 0:
            annual_returns[yr] = float((1 + yr_rets).prod() - 1)

    return {
        "label":          label,
        "cagr":           round(cagr, 4),
        "max_drawdown":   round(max_dd, 4),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "volatility":     round(ann_vol, 4),
        "win_rate":       round(float((period_rets > 0).mean()), 3),
        "total_return":   round(float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1), 4),
        "annual_returns": annual_returns,
        "final_value":    round(float(equity_curve.iloc[-1]), 2),
        "initial_value":  round(float(equity_curve.iloc[0]), 2),
        "periods_per_yr": round(float(periods_per_yr), 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    prices_df:       pd.DataFrame,
    fund_df:         pd.DataFrame,
    initial_capital: float = 500_000,
) -> dict:
    """
    Run the full historical backtest.

    Args:
        prices_df:       monthly prices for all stocks + Nifty
        fund_df:         fundamental metrics for all stocks
        initial_capital: starting portfolio value in ₹

    Returns:
        dict containing:
            portfolio_equity  : pd.Series — portfolio equity curve
            nifty_equity      : pd.Series — Nifty 50 equity curve
            portfolio_metrics : dict
            nifty_metrics     : dict
            rebalance_log     : list of dicts (one per rebalance event)
    """
    from config import MIN_TOTAL_ASSETS_CR, MAX_FUND_AGE_YEARS

    print(f"\n{'═'*62}")
    print(f" BACKTEST  |  {BACKTEST_START} → {BACKTEST_END}")
    print(f" Capital: ₹{initial_capital:,.0f}  |  "
          f"Rebalance: {REBALANCE_MONTHS} (months)  |  "
          f"Portfolio: top {PORTFOLIO_SIZE}")
    print(f" Universe: Dynamic (TA ≥ ₹{MIN_TOTAL_ASSETS_CR} Cr, "
          f"fund age ≤ {MAX_FUND_AGE_YEARS}yr)")
    print(f"{'═'*62}")

    rebalance_dates = get_rebalance_dates(prices_df)
    if not rebalance_dates:
        raise RuntimeError("No rebalance dates found in price data.")

    print(f" Rebalance dates: {len(rebalance_dates)}  "
          f"({rebalance_dates[0].date()} → {rebalance_dates[-1].date()})")

    # ── State ──────────────────────────────────────────────────────────────
    portfolio_value   = float(initial_capital)
    current_portfolio: set[str] = set()
    equity_records: list[tuple] = []   # (date, value) — appended monthly
    rebalance_log: list[dict]   = []
    total_cost_paid   = 0.0
    # NEW: per-rebalance state for analytics. scored_history maps date → full
    # scored_df (every stock, every rank, raw + composite). portfolio_history
    # maps date → list of held tickers. Both are point-in-time correct.
    scored_history: dict   = {}
    portfolio_history: dict = {}

    # Load universe history mapping
    import os
    universe_path = os.path.join("data", "universe", "universe_history_interpolated.csv")
    universe_df = pd.read_csv(universe_path, parse_dates=['effective_date'])
    # Pre-compute active universe for each valid date
    universe_map = {}
    for date in universe_df['effective_date'].unique():
        universe_map[pd.Timestamp(date)] = set(universe_df[universe_df['effective_date'] == date]['ticker'].tolist())

    # All month-end dates available in the price matrix (used to walk monthly
    # between rebalance dates so the equity curve has true monthly granularity).
    all_dates = pd.DatetimeIndex(prices_df.index).sort_values()

    # ── Main Loop ──────────────────────────────────────────────────────────
    print(f"\n {'Date':<14} {'Univ':>5} {'Value (₹)':>12} {'Period Ret':>11} "
          f"{'Turnover':>10} {'Cost (₹)':>10}")
    print(f" {'─'*66}")

    for i, rebal_date in enumerate(rebalance_dates):

        # ── A. Score stocks using only data available at rebal_date ─────
        prices_to_date = prices_df[prices_df.index <= rebal_date]
        
        # Determine active universe map date
        available_univ_dates = [d for d in universe_map.keys() if d <= rebal_date]
        active_universe = universe_map[max(available_univ_dates)] if available_univ_dates else None

        # PiT slicing is handled inside factor_engine.compute_raw_metrics()
        # We pass the full fund_df; factor_engine uses get_pit_snapshot()

        if len(prices_to_date) < 13:
            print(f" {str(rebal_date.date()):<14}  skipped — insufficient history")
            continue

        # Anchor the equity curve at the first valid rebalance.
        if not equity_records:
            equity_records.append((rebal_date, portfolio_value))

        try:
            scored_df, new_port_df = run_factor_engine(
                prices_to_date, fund_df, as_of_date=rebal_date, verbose=False, active_universe=active_universe
            )
            new_portfolio = set(new_port_df.index.tolist())
            universe_size = len(scored_df)
            # Persist for downstream analytics (factor IC, attribution, drill-down).
            scored_history[rebal_date]    = scored_df.copy()
            portfolio_history[rebal_date] = list(new_portfolio)
        except Exception as e:
            print(f" {str(rebal_date.date()):<14}  scoring error: {e}")
            new_portfolio = current_portfolio
            universe_size = 0

        # ── B. Transaction costs ────────────────────────────────────────
        # Cost model: cost_pct is charged on each side (buy AND sell).
        # One-way turnover = ½·Σ|Δw|; round-trip cost = turnover × pct × 2.
        if current_portfolio:
            old_w = pd.Series({t: 1.0/PORTFOLIO_SIZE for t in current_portfolio})
            new_w = pd.Series({t: 1.0/PORTFOLIO_SIZE for t in new_portfolio})
            all_t = old_w.index.union(new_w.index)
            old_w = old_w.reindex(all_t, fill_value=0.0)
            new_w = new_w.reindex(all_t, fill_value=0.0)
            turnover = float(0.5 * (new_w - old_w).abs().sum())
            cost     = portfolio_value * turnover * TRANSACTION_COST_PCT * 2
        else:
            # Initial deployment is a one-way buy of the full portfolio.
            # B7 fix: was 0.5×, should be 1.0× — full-portfolio purchase.
            turnover = 1.0
            cost     = portfolio_value * TRANSACTION_COST_PCT * 1.0

        portfolio_value -= cost
        total_cost_paid += cost

        # Update the (just-anchored or last) equity record after costs so the
        # curve reflects the post-cost value the portfolio actually starts with.
        equity_records[-1] = (rebal_date, portfolio_value)

        # ── C. Walk MONTHLY until the next rebalance ────────────────────
        # This is the B1 fix: instead of one giant period return, compute
        # equal-weighted portfolio return month by month using monthly prices.
        next_date = (rebalance_dates[i+1] if i+1 < len(rebalance_dates)
                     else pd.Timestamp(BACKTEST_END))

        # Monthly grid strictly after rebal_date and on/before next_date.
        period_months = all_dates[(all_dates > rebal_date)
                                  & (all_dates <= next_date)]

        # Slice prices for the held portfolio; missing tickers contribute 0%
        # (cash drag — same convention as the prior simple-mean approach).
        held = [t for t in new_portfolio if t in prices_df.columns]
        held_prices = prices_df[held] if held else pd.DataFrame(index=prices_df.index)

        compounded = 1.0
        prev_dt = rebal_date
        for m_dt in period_months:
            try:
                p_prev = held_prices.loc[prev_dt]
                p_curr = held_prices.loc[m_dt]
            except KeyError:
                prev_dt = m_dt
                continue
            # Monthly return per stock; treat NaN/zero-prev as 0% (cash drag).
            mret = (p_curr / p_prev - 1).where(p_prev > 0).fillna(0.0)
            # Equal weight across the *target* portfolio size — preserves cash drag
            # behaviour for missing tickers (same as compute_period_return).
            month_port_ret = float(mret.sum() / PORTFOLIO_SIZE)
            portfolio_value *= (1 + month_port_ret)
            compounded     *= (1 + month_port_ret)
            equity_records.append((m_dt, portfolio_value))
            prev_dt = m_dt

        period_ret = compounded - 1

        # ── D. Nifty 50 return for same period ─────────────────────────
        if NIFTY_TICKER in prices_df.columns:
            nifty_prices = prices_df[NIFTY_TICKER]
            n_start = nifty_prices[nifty_prices.index <= rebal_date]
            n_end   = nifty_prices[nifty_prices.index <= next_date]
            nifty_ret = (n_end.iloc[-1] / n_start.iloc[-1] - 1) if (
                len(n_start) > 0 and len(n_end) > 0) else 0.0
        else:
            nifty_ret = 0.0

        # ── E. Log ─────────────────────────────────────────────────────
        rebalance_log.append({
            "date"           : rebal_date,
            "portfolio_value": round(portfolio_value, 2),
            "period_return"  : round(period_ret, 4),
            "nifty_return"   : round(nifty_ret, 4),
            "turnover"       : round(turnover, 3),
            "cost"           : round(cost, 2),
            "n_stocks_in"    : len(new_portfolio - current_portfolio),
            "n_stocks_out"   : len(current_portfolio - new_portfolio),
            "universe_size"  : universe_size,
        })
        current_portfolio = new_portfolio

        print(f" {str(rebal_date.date()):<14} "
              f"{universe_size:>5} "
              f"₹{portfolio_value:>11,.0f} "
              f"{period_ret:>+10.1%} "
              f"{turnover:>10.1%} "
              f"₹{cost:>9,.0f}")

    # ── Build equity curves ────────────────────────────────────────────────
    if not equity_records:
        raise RuntimeError("No equity records generated. Check price data coverage.")
    equity_df = (pd.DataFrame(equity_records, columns=["date","value"])
                 .set_index("date")["value"]
                 .sort_index())
    equity_df = equity_df[~equity_df.index.duplicated(keep="last")]

    # Nifty 50 equity curve (same initial capital, same dates)
    nifty_equity = _build_nifty_curve(prices_df, initial_capital, equity_df)

    # ── Metrics ────────────────────────────────────────────────────────────
    port_metrics  = compute_metrics(equity_df,   label="7-Factor Portfolio")
    nifty_metrics = compute_metrics(nifty_equity, label="Nifty 50")

    # ── Summary ────────────────────────────────────────────────────────────
    _print_summary(port_metrics, nifty_metrics, initial_capital, total_cost_paid)

    return {
        "portfolio_equity"  : equity_df,
        "nifty_equity"      : nifty_equity,
        "portfolio_metrics" : port_metrics,
        "nifty_metrics"     : nifty_metrics,
        "rebalance_log"     : rebalance_log,
        "total_cost_paid"   : total_cost_paid,
        "scored_history"    : scored_history,
        "portfolio_history" : portfolio_history,
        "rebalance_dates"   : rebalance_dates,
    }


def _build_nifty_curve(prices_df, initial_capital, portfolio_curve):
    """Build Nifty 50 equity curve aligned to portfolio curve dates."""
    if NIFTY_TICKER not in prices_df.columns:
        return pd.Series(dtype=float)
    nifty = prices_df[NIFTY_TICKER].dropna().sort_index()
    start = portfolio_curve.index[0]
    end   = portfolio_curve.index[-1]
    nifty = nifty[(nifty.index >= start) & (nifty.index <= end)]
    if nifty.empty:
        return pd.Series(dtype=float)
    # Reindex to match portfolio curve dates for fair metric comparison
    nifty_aligned = nifty.reindex(portfolio_curve.index, method='nearest')
    nifty_curve = nifty_aligned / nifty_aligned.iloc[0] * initial_capital
    nifty_curve.name = "Nifty50"
    return nifty_curve


def _print_summary(pm, nm, initial_capital, total_cost):
    w = 14
    print(f"\n{'═'*62}")
    print(f" RESULTS")
    print(f"{'═'*62}")
    print(f" {'Metric':<24} {'Portfolio':>{w}} {'Nifty 50':>{w}} {'Alpha':>{w}}")
    print(f" {'─'*60}")

    rows = [
        ("CAGR",         pm["cagr"],        nm["cagr"],        "%"),
        ("Max Drawdown",  pm["max_drawdown"],nm["max_drawdown"],"%"),
        ("Sharpe Ratio",  pm["sharpe"],      nm["sharpe"],      "x"),
        ("Sortino Ratio", pm["sortino"],     nm["sortino"],     "x"),
        ("Volatility",    pm["volatility"],  nm["volatility"],  "%"),
        ("Win Rate",      pm["win_rate"],    nm["win_rate"],    "%"),
        ("Total Return",  pm["total_return"],nm["total_return"],"%"),
    ]
    for name, pv, nv, unit in rows:
        if unit == "%":
            fmt = lambda v: f"{v:+.1%}"
        else:
            fmt = lambda v: f"{v:.2f}"
        diff = pv - nv
        diff_str = f"{diff:+.1%}" if unit == "%" else f"{diff:+.2f}"
        print(f" {name:<24} {fmt(pv):>{w}} {fmt(nv):>{w}} {diff_str:>{w}}")

    print(f"\n {'Initial Capital':24} ₹{initial_capital:>{w-1},.0f}")
    print(f" {'Portfolio Final':24} ₹{pm['final_value']:>{w-1},.0f}")
    if nm.get('final_value'):
        print(f" {'Nifty 50 Final':24} ₹{nm['final_value']:>{w-1},.0f}")
        print(f" {'Outperformance':24} ₹{pm['final_value']-nm['final_value']:>{w-1},.0f}")
    print(f" {'Total Costs Paid':24} ₹{total_cost:>{w-1},.0f}")

    print(f"\n Annual Returns:")
    print(f" {'Year':<8} {'Portfolio':>12} {'Nifty 50':>12} {'Alpha':>10}")
    print(f" {'─'*44}")
    for yr in sorted(pm["annual_returns"]):
        pv = pm["annual_returns"][yr]
        nv = nm["annual_returns"].get(yr, float("nan"))
        nv_str = f"{nv:+.1%}" if not np.isnan(nv) else "    N/A"
        al_str = f"{pv-nv:+.1%}" if not np.isnan(nv) else ""
        print(f" {yr:<8} {pv:>+12.1%} {nv_str:>12} {al_str:>10}")

    print(f"\n PDF targets:  CAGR ~17.2%  |  Max DD ~-18%  |  Sharpe ~0.78")
    print(f" (Note: mock data ≠ real data. Run with USE_REAL_DATA=True on your machine.)")
    print(f"{'═'*62}")


if __name__ == "__main__":
    from data_layer import generate_mock_data
    prices, fund = generate_mock_data()
    results = run_backtest(prices, fund, initial_capital=500_000)
