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
    PORTFOLIO_SIZE, TRANSACTION_COST_PCT, RISK_FREE_RATE, NIFTY_TICKER
)
from factor_engine import run_factor_engine


# ─────────────────────────────────────────────────────────────────────────────
# REBALANCE DATES
# ─────────────────────────────────────────────────────────────────────────────

def get_rebalance_dates(prices_df: pd.DataFrame) -> list:
    """
    Find all month-end dates in prices_df that fall in REBALANCE_MONTHS
    and within the backtest window.

    Returns:
        List of pd.Timestamps sorted ascending.
    """
    available = pd.DatetimeIndex(prices_df.index)
    rebalance = []

    for dt in available:
        if (dt.month in REBALANCE_MONTHS
                and pd.Timestamp(BACKTEST_START) <= dt <= pd.Timestamp(BACKTEST_END)):
            rebalance.append(dt)

    return sorted(rebalance)


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
    stock_cols = [c for c in prices_df.columns if c != NIFTY_TICKER]

    for ticker in portfolio_tickers:
        if ticker not in prices_df.columns:
            continue

        # Price at period_start
        prices_before = prices_df.loc[
            prices_df.index <= period_start, ticker
        ].dropna()
        if prices_before.empty:
            continue

        # Price at period_end
        prices_at_end = prices_df.loc[
            prices_df.index <= period_end, ticker
        ].dropna()
        if prices_at_end.empty:
            continue

        p_start = prices_before.iloc[-1]
        p_end   = prices_at_end.iloc[-1]

        if p_start > 0:
            stock_rets[ticker] = (p_end / p_start) - 1

    if not stock_rets:
        return 0.0, {}

    portfolio_return = float(np.mean(list(stock_rets.values())))
    return portfolio_return, stock_rets


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(equity_curve: pd.Series, label: str = "Portfolio") -> dict:
    """
    Compute standard performance metrics from an equity curve.

    CAGR:
        (Final / Initial) ^ (1 / years) − 1

    Max Drawdown:
        Worst peak-to-trough decline in the equity curve.
        max_dd = min( (equity − running_peak) / running_peak )

    Sharpe Ratio:
        (Annualised Excess Return) / (Annualised Volatility)
        Using monthly returns, annualised by × √12
        Excess return = CAGR − Risk Free Rate

    Args:
        equity_curve: pd.Series, index=dates, values=portfolio value in ₹
        label: name for printing

    Returns:
        dict with cagr, max_drawdown, sharpe, sortino, volatility,
              win_rate, total_return, annual_returns
    """
    monthly_rets = equity_curve.pct_change().dropna()
    if len(monthly_rets) == 0:
        return {}

    # CAGR using actual calendar days
    n_days  = (equity_curve.index[-1] - equity_curve.index[0]).days
    n_years = max(n_days / 365.25, 0.01)
    cagr    = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1

    # Max Drawdown
    running_max  = equity_curve.cummax()
    drawdown     = (equity_curve - running_max) / running_max
    max_dd       = float(drawdown.min())

    # Sharpe Ratio (annualised)
    rf_monthly   = (1 + RISK_FREE_RATE) ** (1/12) - 1
    excess_rets  = monthly_rets - rf_monthly
    ann_vol      = monthly_rets.std() * np.sqrt(12)
    sharpe       = ((cagr - RISK_FREE_RATE) / ann_vol) if ann_vol > 0 else 0.0

    # Sortino (only downside vol)
    down_rets    = monthly_rets[monthly_rets < rf_monthly]
    down_vol     = down_rets.std() * np.sqrt(12) if len(down_rets) > 0 else ann_vol
    sortino      = ((cagr - RISK_FREE_RATE) / down_vol) if down_vol > 0 else 0.0

    # Win rate
    win_rate     = float((monthly_rets > 0).mean())

    # Total return
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    # Annual returns
    annual_returns = {}
    for yr in sorted(equity_curve.index.year.unique()):
        yr_vals = equity_curve[equity_curve.index.year == yr]
        if len(yr_vals) >= 2:
            annual_returns[yr] = float((yr_vals.iloc[-1] / yr_vals.iloc[0]) - 1)

    return {
        "label"         : label,
        "cagr"          : round(cagr, 4),
        "max_drawdown"  : round(max_dd, 4),
        "sharpe"        : round(sharpe, 3),
        "sortino"       : round(sortino, 3),
        "volatility"    : round(ann_vol, 4),
        "win_rate"      : round(win_rate, 3),
        "total_return"  : round(total_return, 4),
        "annual_returns": annual_returns,
        "final_value"   : round(float(equity_curve.iloc[-1]), 2),
        "initial_value" : round(float(equity_curve.iloc[0]), 2),
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
    print(f"\n{'═'*62}")
    print(f" BACKTEST  |  {BACKTEST_START} → {BACKTEST_END}")
    print(f" Capital: ₹{initial_capital:,.0f}  |  "
          f"Rebalance: {REBALANCE_MONTHS} (months)  |  "
          f"Portfolio: top {PORTFOLIO_SIZE}")
    print(f"{'═'*62}")

    rebalance_dates = get_rebalance_dates(prices_df)
    if not rebalance_dates:
        raise RuntimeError("No rebalance dates found in price data.")

    print(f" Rebalance dates: {len(rebalance_dates)}  "
          f"({rebalance_dates[0].date()} → {rebalance_dates[-1].date()})")

    # ── State ──────────────────────────────────────────────────────────────
    portfolio_value   = float(initial_capital)
    current_portfolio = set()
    equity_records    = [(pd.Timestamp(BACKTEST_START), portfolio_value)]
    rebalance_log     = []
    total_cost_paid   = 0.0

    # ── Main Loop ──────────────────────────────────────────────────────────
    print(f"\n {'Date':<14} {'Value (₹)':>12} {'Period Ret':>11} "
          f"{'Turnover':>10} {'Cost (₹)':>10}")
    print(f" {'─'*60}")

    for i, rebal_date in enumerate(rebalance_dates):

        # ── A. Score stocks using only data available at rebal_date ─────
        prices_to_date = prices_df[prices_df.index <= rebal_date]

        if len(prices_to_date) < 13:
            print(f" {str(rebal_date.date()):<14}  skipped — insufficient history")
            continue

        try:
            scored_df, new_port_df = run_factor_engine(
                prices_to_date, fund_df, verbose=False
            )
            new_portfolio = set(new_port_df.index.tolist())
        except Exception as e:
            print(f" {str(rebal_date.date()):<14}  scoring error: {e}")
            new_portfolio = current_portfolio   # keep previous

        # ── B. Transaction costs ────────────────────────────────────────
        if current_portfolio:
            stocks_sold   = current_portfolio - new_portfolio
            stocks_bought = new_portfolio     - current_portfolio
            n_traded      = len(stocks_sold) + len(stocks_bought)
            n_total       = max(len(current_portfolio), PORTFOLIO_SIZE)
            turnover      = n_traded / (2 * n_total)
            cost          = portfolio_value * turnover * TRANSACTION_COST_PCT * 2
        else:
            # First rebalance — just buy costs
            turnover = 1.0
            cost     = portfolio_value * TRANSACTION_COST_PCT * 0.5

        portfolio_value -= cost
        total_cost_paid += cost

        # ── C. Calculate return until next rebalance ────────────────────
        next_date = (rebalance_dates[i+1] if i+1 < len(rebalance_dates)
                     else pd.Timestamp(BACKTEST_END))

        period_ret, stock_rets = compute_period_return(
            list(new_portfolio), prices_df, rebal_date, next_date
        )
        portfolio_value *= (1 + period_ret)

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
        equity_records.append((next_date, portfolio_value))
        rebalance_log.append({
            "date"          : rebal_date,
            "portfolio_value": round(portfolio_value, 2),
            "period_return" : round(period_ret, 4),
            "nifty_return"  : round(nifty_ret, 4),
            "turnover"      : round(turnover, 3),
            "cost"          : round(cost, 2),
            "n_stocks_in"   : len(new_portfolio - current_portfolio),
            "n_stocks_out"  : len(current_portfolio - new_portfolio),
        })
        current_portfolio = new_portfolio

        print(f" {str(rebal_date.date()):<14} "
              f"₹{portfolio_value:>11,.0f} "
              f"{period_ret:>+10.1%} "
              f"{turnover:>10.1%} "
              f"₹{cost:>9,.0f}")

    # ── Build equity curves ────────────────────────────────────────────────
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
    }


def _build_nifty_curve(prices_df, initial_capital, portfolio_curve):
    """Build Nifty 50 equity curve starting at same capital."""
    if NIFTY_TICKER not in prices_df.columns:
        return pd.Series(dtype=float)
    nifty = prices_df[NIFTY_TICKER].dropna().sort_index()
    # Align to portfolio curve dates (approximately)
    start = portfolio_curve.index[0]
    end   = portfolio_curve.index[-1]
    nifty = nifty[(nifty.index >= start) & (nifty.index <= end)]
    if nifty.empty:
        return pd.Series(dtype=float)
    nifty_curve = nifty / nifty.iloc[0] * initial_capital
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
