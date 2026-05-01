"""
dashboard.py — Vue-based analytical dashboard.

Single-file HTML dashboard rendered from real backtest + analytics outputs.

Current tabs (Vue-driven sidebar):
  1. Overview     — equity curve, drawdown, rolling 12M Sharpe/Alpha.
  2. Composition  — current top-25 heatmap, sector exposure over time.
  3. Factors      — single-factor backtests, IC, factor performance summary.
  4. Sandbox      — interactive factor-weight slider, recomputed top-25 live.
  5. Compare      — pick any two stocks, compare current factor ranks (radar).
  6. Logs         — rebalance history table.

NOTE: Several analytics blocks are computed and shipped to the JS payload
(`qs`, `regime`, `corr`, `transitions`, `ct`, per-stock score history) but
not yet wired into the Vue template. Future dashboard work will surface them.
"""


from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    PORTFOLIO_SIZE, WEIGHTS, LOWER_IS_BETTER, RISK_FREE_RATE,
    TRANSACTION_COST_PCT, MIN_TOTAL_ASSETS_CR, MAX_FUND_AGE_YEARS,
    INCEPTION_DATE, BACKTEST_END,
)
from sectors import sector_of
import analytics as A

# ─────────────────────────────────────────────────────────────────────────────
# JSON helper
# ─────────────────────────────────────────────────────────────────────────────

def _safe(o):
    if isinstance(o, pd.DataFrame): return o.to_dict(orient="records")
    if isinstance(o, pd.Series):    return o.to_dict()
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.ndarray):   return o.tolist()
    if isinstance(o, (pd.Timestamp,)): return str(o.date())
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def _to_chart_series(idx, vals):
    """Build {labels:[…], values:[…]} where NaN becomes null."""
    return {
        "labels": [str(pd.Timestamp(x).date()) for x in idx],
        "values": [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else round(float(v), 4)
                   for v in vals],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry — assembles every panel's data, then injects into HTML template.
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(
    scored_df:        pd.DataFrame,
    portfolio:        pd.DataFrame,
    backtest_results: dict, 
    output_path:      str   = "dashboard.html",
    initial_capital:  float = 500_000,
    prices_df:        pd.DataFrame | None = None,
) -> str:
    pm  = backtest_results["portfolio_metrics"]
    nm  = backtest_results["nifty_metrics"]
    rebal_log    = backtest_results["rebalance_log"]
    port_curve   = backtest_results["portfolio_equity"]
    nifty_curve  = backtest_results["nifty_equity"]
    sh           = backtest_results.get("scored_history",    {})
    ph           = backtest_results.get("portfolio_history", {})
    rdates       = backtest_results.get("rebalance_dates",   [])

    factors = list(WEIGHTS.keys())

    # ── Equity curves (normalised to 100) ─────────────────────────────────
    p0 = float(port_curve.iloc[0])
    n0 = float(nifty_curve.iloc[0]) if not nifty_curve.empty else 1.0
    nifty_aligned = nifty_curve.reindex(port_curve.index, method="nearest") if not nifty_curve.empty else pd.Series(np.nan, index=port_curve.index)

    equity_block = {
        "labels":   [str(d.date()) for d in port_curve.index],
        "port":     [round(v / p0 * 100, 2) for v in port_curve.values],
        "nifty":    [round(v / n0 * 100, 2) if pd.notna(v) else None for v in nifty_aligned.values],
        "port_inr": [round(float(v)) for v in port_curve.values],
    }

    # Drawdown (monthly, real)
    running_max = port_curve.cummax()
    dd_pct = ((port_curve - running_max) / running_max * 100).round(2)
    nifty_dd_pct = pd.Series(dtype=float)
    if not nifty_curve.empty:
        nm_max = nifty_aligned.cummax()
        nifty_dd_pct = ((nifty_aligned - nm_max) / nm_max * 100).round(2)
    drawdown_block = {
        "labels": equity_block["labels"],
        "port":   dd_pct.tolist(),
        "nifty":  nifty_dd_pct.reindex(port_curve.index).tolist() if not nifty_dd_pct.empty else [None]*len(port_curve),
    }

    # Rolling 12M metrics
    rolling = A.compute_rolling_metrics(port_curve, nifty_curve, window_months=12)
    rolling_block = {
        "labels": [str(d.date()) for d in rolling.index],
        "sharpe": [None if pd.isna(v) else round(float(v), 3) for v in rolling["rolling_sharpe"]],
        "alpha":  [None if pd.isna(v) else round(float(v), 4) for v in rolling["rolling_alpha"]],
        "beta":   [None if pd.isna(v) else round(float(v), 3) for v in rolling["rolling_beta"]],
    }

    # Annual returns
    ann_p = pm.get("annual_returns", {})
    ann_n = nm.get("annual_returns", {})
    yrs   = sorted(ann_p)
    annual_block = {
        "labels": [str(y) for y in yrs],
        "port":   [round(ann_p[y]*100, 1)        for y in yrs],
        "nifty":  [round(ann_n.get(y, 0)*100, 1) for y in yrs],
    }

    # ── Single-factor backtests ────────────────────────────────────────────
    sf_curves = (A.run_all_single_factor_backtests(sh, prices_df, rdates, initial_capital)
                 if sh and prices_df is not None else pd.DataFrame())

    sf_block = {"labels": [], "series": {}, "summary": []}
    if not sf_curves.empty:
        # Normalise to 100 from each curve's first non-NaN value
        sf_block["labels"] = [str(d.date()) for d in sf_curves.index]
        for f in factors:
            if f not in sf_curves.columns: continue
            s = sf_curves[f].dropna()
            if s.empty: continue
            base = s.iloc[0]
            sf_block["series"][f] = [round(float(v / base * 100), 2)
                                     if pd.notna(v) and v > 0 else None
                                     for v in sf_curves[f].reindex(sf_curves.index).values]

        # Add the multi-factor (main portfolio) curve for comparison
        port_aligned = port_curve.reindex(sf_curves.index, method="nearest")
        if not port_aligned.empty:
            base = port_aligned.dropna().iloc[0]
            sf_block["series"]["MultiFactor"] = [round(float(v/base*100), 2)
                                                  if pd.notna(v) and v > 0 else None
                                                  for v in port_aligned.values]

        # Summary table — final return + max drawdown per factor
        for f, s in sf_curves.items():
            ss = s.dropna()
            if ss.empty: continue
            tot = float(ss.iloc[-1] / ss.iloc[0] - 1)
            mx  = ss.cummax()
            dd  = float(((ss - mx) / mx).min())
            n_yrs = max((ss.index[-1] - ss.index[0]).days / 365.25, 0.01)
            cagr  = float((ss.iloc[-1] / ss.iloc[0]) ** (1/n_yrs) - 1)
            sf_block["summary"].append({
                "factor":       f,
                "total_return": round(tot, 4),
                "cagr":         round(cagr, 4),
                "max_dd":       round(mx.iloc[-1] and dd, 4),
                "weight":       WEIGHTS[f],
            })

    # ── IC + Quintile spreads + Regime ─────────────────────────────────────
    ic    = A.compute_factor_ic(sh, prices_df, rdates) if (sh and prices_df is not None) else pd.DataFrame()
    qs    = A.compute_quintile_spreads(sh, prices_df, rdates) if (sh and prices_df is not None) else pd.DataFrame()
    regime = A.compute_factor_regime(qs)

    ic_block = {"factors": factors, "mean_ic": [], "hit_rate": [], "ic_std": []}
    if not ic.empty:
        for f in factors:
            ic_block["mean_ic"].append(round(float(ic.attrs["mean_ic"].get(f, 0) or 0), 4))
            ic_block["hit_rate"].append(round(float(ic.attrs["hit_rate"].get(f, 0) or 0), 3))
            ic_block["ic_std"].append(round(float(ic.attrs["ic_std"].get(f, 0) or 0), 4))

    qs_block = {"factors": factors, "mean_spread": []}
    if not qs.empty:
        means = qs.mean()
        for f in factors:
            v = means.get(f, np.nan)
            qs_block["mean_spread"].append(None if pd.isna(v) else round(float(v), 4))

    regime_block = {"years": [], "factors": factors, "matrix": []}
    if not regime.empty:
        regime_block["years"] = [int(y) for y in regime.index]
        for y in regime.index:
            row = []
            for f in factors:
                v = regime.loc[y, f] if f in regime.columns else np.nan
                row.append(None if pd.isna(v) else round(float(v), 2))
            regime_block["matrix"].append(row)

    # Computed correlation matrix (Spearman, averaged over rebal dates)
    corr_df = A.compute_factor_correlations(sh)
    corr_block = {"factors": factors, "matrix": []}
    if not corr_df.empty:
        for f in factors:
            row = []
            for g in factors:
                v = corr_df.loc[f, g] if (f in corr_df.index and g in corr_df.columns) else np.nan
                row.append(None if pd.isna(v) else round(float(v), 3))
            corr_block["matrix"].append(row)

    # ── Composition: heatmap, sector area, transitions ────────────────────
    heatmap = []
    for ticker, row in portfolio.iterrows():
        ranks   = [int(row.get(f"Rank_{f}", 3)) for f in factors]
        raws    = [None if pd.isna(row.get(f"raw_{f}", np.nan)) else round(float(row[f"raw_{f}"]), 4)
                   for f in factors]
        heatmap.append({
            "ticker": ticker.replace(".NS", ""),
            "full":   ticker,
            "score":  round(float(row["Final_Score"]), 3),
            "ranks":  ranks,
            "raws":   raws,
            "sector": sector_of(ticker),
        })

    sector_exposure = A.compute_sector_exposure(ph)
    se_block = {"labels": [], "sectors": [], "matrix": []}
    if not sector_exposure.empty:
        # Order sectors by total exposure (desc) for cleaner stacking
        totals = sector_exposure.sum().sort_values(ascending=False)
        ordered = list(totals.index)
        se_block["labels"] = [str(d.date()) for d in sector_exposure.index]
        se_block["sectors"] = ordered
        for s in ordered:
            se_block["matrix"].append([round(float(v)*100, 1) for v in sector_exposure[s].tolist()])

    # Holdings transitions per rebalance
    transitions = []
    prev: set = set()
    for rd in rdates:
        if rd not in ph: continue
        cur = set(ph[rd])
        transitions.append({
            "date":  str(rd.date()),
            "in":   sorted([t.replace(".NS","") for t in cur - prev]),
            "out":  sorted([t.replace(".NS","") for t in prev - cur]),
            "kept": sorted([t.replace(".NS","") for t in cur & prev]),
            "size": len(cur),
        })
        prev = cur

    # ── Per-rebalance attribution ─────────────────────────────────────────
    attr = A.compute_per_rebalance_attribution(ph, prices_df, rdates) if (ph and prices_df is not None) else {}
    rebal_block = []
    for r in rebal_log:
        rd = pd.Timestamp(r["date"])
        a  = attr.get(rd, {})
        rebal_block.append({
            "date":     str(rd.date()),
            "value":    round(r.get("portfolio_value", 0)),
            "period_r": round(r.get("period_return", 0)*100, 2),
            "nifty_r":  round(r.get("nifty_return",  0)*100, 2),
            "turn":     round(r.get("turnover", 0)*100, 1),
            "n_in":     r.get("n_stocks_in", 0),
            "n_out":    r.get("n_stocks_out", 0),
            "cost":     round(r.get("cost", 0)),
            "winners":  [(t.replace(".NS",""), round(v*100, 1)) for t, v in a.get("top_winners", [])],
            "losers":   [(t.replace(".NS",""), round(v*100, 1)) for t, v in a.get("top_losers",  [])],
            "univ":     r.get("universe_size", 0),
        })

    # Cost & turnover trend
    ct_block = {
        "labels":   [r["date"] for r in rebal_block],
        "turnover": [r["turn"] for r in rebal_block],
        "cost":     [r["cost"] for r in rebal_block],
    }

    # ── Universe table ────────────────────────────────────────────────────
    universe_rows = []
    for i, (ticker, row) in enumerate(scored_df.iterrows()):
        universe_rows.append({
            "rank":   i + 1,
            "ticker": ticker.replace(".NS",""),
            "full":   ticker,
            "score":  round(float(row["Final_Score"]), 3),
            "in_port": ticker in portfolio.index,
            "ranks":  [int(row.get(f"Rank_{f}", 3)) for f in factors],
            "sector": sector_of(ticker),
        })

    # ── Per-stock score history (lightweight pivot) ───────────────────────
    score_hist = A.compute_stock_score_history(sh) if sh else {}
    stock_db = {}   # ticker → {dates, scores, ranks (per factor), in_port (per date)}
    in_port_by_date = {pd.Timestamp(rd): set(ph[rd]) for rd in ph}
    for tk, hdf in score_hist.items():
        dates = [str(d.date()) for d in hdf.index]
        scores = [None if pd.isna(v) else round(float(v), 3) for v in hdf.get("Final_Score", pd.Series([np.nan]*len(hdf))).values]
        ranks_per_factor = {}
        for f in factors:
            col = f"Rank_{f}"
            if col in hdf.columns:
                ranks_per_factor[f] = [None if pd.isna(v) else int(v) for v in hdf[col].values]
        in_port_flags = [tk in in_port_by_date.get(d, set()) for d in hdf.index]
        # Trim price series to dashboard's date window
        if prices_df is not None and tk in prices_df.columns:
            ps = prices_df[tk].dropna()
            ps = ps[ps.index >= pd.Timestamp(INCEPTION_DATE)]
            price_block = {
                "labels": [str(d.date()) for d in ps.index],
                "values": [round(float(v), 2) for v in ps.values],
            }
        else:
            price_block = {"labels": [], "values": []}
        stock_db[tk] = {
            "ticker":  tk,
            "name":    tk.replace(".NS",""),
            "sector":  sector_of(tk),
            "score_hist": {"labels": dates, "values": scores},
            "ranks_hist": {"labels": dates, "ranks": ranks_per_factor},
            "in_port":  in_port_flags,
            "price":    price_block,
        }

    # ── KPI block ─────────────────────────────────────────────────────────
    cagr_alpha = pm["cagr"] - nm["cagr"]
    pct = lambda v: f"{v:+.1%}" if v == v else "—"
    num = lambda v, d=2: f"{v:.{d}f}" if v == v else "—"
    inr = lambda v: f"₹{v:,.0f}"

    KPI = {
        "cagr":        pct(pm["cagr"]),
        "maxdd":       pct(pm["max_drawdown"]),
        "sharpe":      num(pm["sharpe"]),
        "sortino":     num(pm["sortino"]),
        "calmar":      num(abs(pm["cagr"]/pm["max_drawdown"]) if pm["max_drawdown"] else 0),
        "alpha":       pct(cagr_alpha),
        "vol":         pct(pm["volatility"]),
        "winrate":     f"{pm['win_rate']:.0%}",
        "total_ret":   pct(pm["total_return"]),
        "final_val":   inr(pm["final_value"]),
        "init_val":    inr(initial_capital),
        "nifty_cagr":  pct(nm.get("cagr", 0)),
        "nifty_dd":    pct(nm.get("max_drawdown", 0)),
        "nifty_final": inr(nm.get("final_value", 0)),
        "n_stocks":    PORTFOLIO_SIZE,
        "n_universe":  len(scored_df),
        "run_date":    datetime.now().strftime("%d %b %Y %H:%M"),
        "inception":   str(INCEPTION_DATE),
        "end":         str(BACKTEST_END),
        "n_rebal":     len(rebal_log),
        "tot_cost":    inr(backtest_results.get("total_cost_paid", 0)),
        "rf":          f"{RISK_FREE_RATE*100:.1f}%",
        "tcost":       f"{TRANSACTION_COST_PCT*100:.2f}%",
        "min_ta":      f"₹{MIN_TOTAL_ASSETS_CR} Cr",
        "max_age":     f"{MAX_FUND_AGE_YEARS}y",
    }

    # Pack everything into one JS payload
    JS = {
        "factors":     factors,
        "weights":     WEIGHTS,
        "lower_better": list(LOWER_IS_BETTER),
        "kpi":         KPI,
        "equity":      equity_block,
        "drawdown":    drawdown_block,
        "rolling":     rolling_block,
        "annual":      annual_block,
        "sf":          sf_block,
        "ic":          ic_block,
        "qs":          qs_block,
        "regime":      regime_block,
        "corr":        corr_block,
        "heatmap":     heatmap,
        "sector_exp":  se_block,
        "transitions": transitions,
        "rebal":       rebal_block,
        "ct":          ct_block,
        "universe":    universe_rows,
        "stocks":      stock_db,
        "n_universe":  len(scored_df),
        "initial_capital": initial_capital,
    }

    html = _render_html(KPI, JS)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[Dashboard] Saved → {output_path}  ({len(html)//1024} KB)")
    print(f"[Dashboard] Open: open {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(K: dict, JS: dict) -> str:
    import json
    js = json.dumps(JS)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Portfolio Analytics</title>
    
    <!-- Fonts & Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css" rel="stylesheet">
    
    <!-- Vue.js for Reactivity -->
    <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {{
            --bg-base: #0B0E14;
            --bg-surface: #151821;
            --bg-card: rgba(26, 30, 41, 0.6);
            --border: rgba(255, 255, 255, 0.06);
            --text-main: #E2E8F0;
            --text-muted: #8F9EB2;
            
            --accent: #3B82F6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --success: #10B981;
            --danger: #EF4444;
            --warning: #F59E0B;
            
            --font-sans: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: var(--font-sans);
            background-color: var(--bg-base);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }}
        
        /* Sidebar */
        .sidebar {{
            width: 260px;
            background: var(--bg-surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 24px 0;
            z-index: 10;
        }}
        
        .brand {{
            padding: 0 24px 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 12px;
        }}
        
        .brand-title {{
            font-size: 18px;
            font-weight: 700;
            background: linear-gradient(135deg, #E2E8F0 0%, #8F9EB2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .nav-item {{
            padding: 12px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            font-weight: 500;
        }}
        
        .nav-item i {{ font-size: 18px; }}
        
        .nav-item:hover {{ color: var(--text-main); background: rgba(255,255,255,0.02); }}
        .nav-item.active {{
            color: var(--accent);
            background: linear-gradient(90deg, rgba(59,130,246,0.1) 0%, transparent 100%);
            border-left: 3px solid var(--accent);
        }}
        
        /* Main Content */
        .main-content {{
            flex: 1;
            overflow-y: auto;
            padding: 32px 40px;
            scroll-behavior: smooth;
        }}
        
        .main-content::-webkit-scrollbar {{ width: 6px; }}
        .main-content::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
        
        .page-header {{ margin-bottom: 32px; }}
        .page-title {{ font-size: 24px; font-weight: 600; margin-bottom: 8px; }}
        .page-desc {{ color: var(--text-muted); font-size: 14px; }}
        
        /* Grid Layouts */
        .grid-cols-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }}
        .grid-cols-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        
        /* Glass Cards */
        .card {{
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .card-title {{ font-size: 14px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
        
        /* KPI Styles */
        .kpi-value {{
            font-size: 32px;
            font-weight: 600;
            font-family: var(--font-mono);
            margin-bottom: 4px;
        }}
        .kpi-sub {{ font-size: 13px; color: var(--text-muted); }}
        
        .text-success {{ color: var(--success); }}
        .text-danger {{ color: var(--danger); }}
        .text-accent {{ color: var(--accent); }}
        
        /* Charts */
        .chart-container {{ position: relative; height: 300px; width: 100%; }}
        
        /* Tables */
        .table-container {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        th {{ padding: 12px 16px; background: rgba(255,255,255,0.02); color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); }}
        td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
        tr:hover td {{ background: rgba(255,255,255,0.02); }}
        
        /* Sandbox sliders */
        .slider-group {{ margin-bottom: 16px; }}
        .slider-label {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }}
        input[type=range] {{
            -webkit-appearance: none; width: 100%; background: transparent; 
        }}
        input[type=range]::-webkit-slider-thumb {{
            -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%;
            background: var(--accent); cursor: pointer; margin-top: -6px;
        }}
        input[type=range]::-webkit-slider-runnable-track {{
            width: 100%; height: 4px; cursor: pointer; background: var(--border); border-radius: 2px;
        }}
        
        .btn {{
            background: var(--accent); color: white; border: none; padding: 10px 20px;
            border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer;
            transition: background 0.2s;
        }}
        .btn:hover {{ background: #2563EB; }}
        
        /* Comparison */
        .compare-select {{
            background: var(--bg-surface); border: 1px solid var(--border); color: var(--text-main);
            padding: 10px; border-radius: 8px; width: 100%; font-size: 14px; outline: none;
        }}
        
        .fade-enter-active, .fade-leave-active {{ transition: opacity 0.3s ease; }}
        .fade-enter-from, .fade-leave-to {{ opacity: 0; }}
        
        /* Rank badges */
        .rank-badge {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 24px; height: 24px; border-radius: 6px; font-family: var(--font-mono); font-size: 12px; font-weight: 600;
        }}
        .r5 {{ background: rgba(16, 185, 129, 0.2); color: var(--success); }}
        .r4 {{ background: rgba(16, 185, 129, 0.1); color: var(--success); }}
        .r3 {{ background: rgba(143, 158, 178, 0.1); color: var(--text-muted); }}
        .r2 {{ background: rgba(245, 158, 11, 0.1); color: var(--warning); }}
        .r1 {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        
    </style>
</head>
<body>

<div id="app" style="display: flex; width: 100%;">
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="brand">
            <div class="brand-title">QuantCore</div>
            <div style="font-size: 12px; color: var(--text-muted);">NSE 200 Factor Engine</div>
        </div>
        
        <div class="nav-item" :class="{{active: activeTab === 'overview'}}" @click="activeTab = 'overview'">
            <i class="ri-dashboard-line"></i> Dashboard
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'composition'}}" @click="activeTab = 'composition'">
            <i class="ri-pie-chart-line"></i> Portfolio Comp
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'factors'}}" @click="activeTab = 'factors'">
            <i class="ri-bar-chart-grouped-line"></i> Factor Attribution
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'diagnostics'}}" @click="activeTab = 'diagnostics'">
            <i class="ri-microscope-line"></i> Factor Diagnostics
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'sandbox'}}" @click="activeTab = 'sandbox'">
            <i class="ri-equalizer-line"></i> Model Sandbox
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'compare'}}" @click="activeTab = 'compare'">
            <i class="ri-scales-3-line"></i> Stock Comparison
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'drilldown'}}" @click="activeTab = 'drilldown'">
            <i class="ri-search-eye-line"></i> Stock Drill-Down
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'universe'}}" @click="activeTab = 'universe'">
            <i class="ri-grid-line"></i> Universe
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'logs'}}" @click="activeTab = 'logs'">
            <i class="ri-history-line"></i> Rebalance Logs
        </div>
        <div class="nav-item" :class="{{active: activeTab === 'model'}}" @click="activeTab = 'model'">
            <i class="ri-book-open-line"></i> Model
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Overview Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'overview'">
            <div class="page-header">
                <h1 class="page-title">Performance Overview</h1>
                <div class="page-desc">{K['inception']} to {K['end']} • Capital: {K['init_val']} → {K['final_val']}</div>
            </div>
            
            <div class="grid-cols-4" style="margin-bottom: 24px;">
                <div class="card">
                    <div class="card-title">CAGR</div>
                    <div class="kpi-value text-success">{K['cagr']}</div>
                    <div class="kpi-sub">Nifty 50: {K['nifty_cagr']}</div>
                </div>
                <div class="card">
                    <div class="card-title">Max Drawdown</div>
                    <div class="kpi-value text-danger">{K['maxdd']}</div>
                    <div class="kpi-sub">Nifty 50: {K['nifty_dd']}</div>
                </div>
                <div class="card">
                    <div class="card-title">Sharpe Ratio</div>
                    <div class="kpi-value text-accent">{K['sharpe']}</div>
                    <div class="kpi-sub">Sortino: {K['sortino']}</div>
                </div>
                <div class="card">
                    <div class="card-title">Alpha</div>
                    <div class="kpi-value text-success">{K['alpha']}</div>
                    <div class="kpi-sub">Annualised Excess</div>
                </div>
            </div>
            
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header">
                    <div class="card-title">Equity Curve vs Benchmark</div>
                </div>
                <div class="chart-container" style="height: 400px;">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>
            
            <div class="grid-cols-2">
                <div class="card">
                    <div class="card-header"><div class="card-title">Drawdown Profile</div></div>
                    <div class="chart-container"><canvas id="drawdownChart"></canvas></div>
                </div>
                <div class="card">
                    <div class="card-header"><div class="card-title">Rolling 12M Metrics</div></div>
                    <div class="chart-container"><canvas id="rollingChart"></canvas></div>
                </div>
            </div>
        </div>
        </transition>

        <!-- Composition Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'composition'">
            <div class="page-header">
                <h1 class="page-title">Portfolio Composition</h1>
                <div class="page-desc">Current Holdings and Sector Exposures</div>
            </div>
            
            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header"><div class="card-title">Current Top {K['n_stocks']} Holdings</div></div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Stock</th>
                                <th>Sector</th>
                                <th>Final Score</th>
                                <th v-for="f in data.factors" :key="f">{{{{ f.substring(0,3) }}}}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(h, i) in data.heatmap" :key="i">
                                <td style="font-weight: 600;">{{{{ h.ticker }}}}</td>
                                <td style="color: var(--text-muted)">{{{{ h.sector }}}}</td>
                                <td style="font-family: var(--font-mono)">{{{{ h.score.toFixed(3) }}}}</td>
                                <td v-for="(rank, idx) in h.ranks" :key="idx">
                                    <span :class="['rank-badge', 'r'+rank]">{{{{ rank }}}}</span>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header"><div class="card-title">Sector Exposure Over Time</div></div>
                <div class="chart-container" style="height: 400px;"><canvas id="sectorChart"></canvas></div>
            </div>
        </div>
        </transition>
        
        <!-- Factor Attribution Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'factors'">
            <div class="page-header">
                <h1 class="page-title">Factor Attribution</h1>
                <div class="page-desc">Isolated single-factor performance and Information Coefficients</div>
            </div>
            
            <div class="grid-cols-2" style="margin-bottom: 24px;">
                <div class="card">
                    <div class="card-header"><div class="card-title">Single Factor Returns (Base 100)</div></div>
                    <div class="chart-container"><canvas id="sfChart"></canvas></div>
                </div>
                <div class="card">
                    <div class="card-header"><div class="card-title">Factor IC (Information Coefficient)</div></div>
                    <div class="chart-container"><canvas id="icChart"></canvas></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header"><div class="card-title">Factor Performance Summary</div></div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr><th>Factor</th><th>Model Weight</th><th>CAGR</th><th>Max Drawdown</th></tr>
                        </thead>
                        <tbody>
                            <tr v-for="f in data.sf.summary" :key="f.factor">
                                <td>{{{{ f.factor }}}}</td>
                                <td>{{{{ (f.weight * 100).toFixed(0) }}}}%</td>
                                <td class="text-success">{{{{ (f.cagr * 100).toFixed(2) }}}}%</td>
                                <td class="text-danger">{{{{ (f.max_dd * 100).toFixed(2) }}}}%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        </transition>

        <!-- Sandbox Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'sandbox'">
            <div class="page-header">
                <h1 class="page-title">Model Sandbox</h1>
                <div class="page-desc">Adjust factor weights and see hypothetical current portfolio</div>
            </div>
            
            <div class="grid-cols-2">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Factor Weights (%)</div>
                        <div style="font-size:12px; color:var(--text-muted)">Total: {{{{ sandboxTotalWeight }}}}%</div>
                    </div>
                    
                    <div class="slider-group" v-for="f in data.factors" :key="f">
                        <div class="slider-label">
                            <span>{{{{ f }}}}</span>
                            <span class="text-accent">{{{{ sandboxWeights[f] }}}}%</span>
                        </div>
                        <input type="range" min="0" max="100" v-model.number="sandboxWeights[f]" @input="recalculateSandbox">
                    </div>
                    
                    <div style="margin-top: 24px; display:flex; gap: 12px;">
                        <button class="btn" @click="normalizeWeights">Normalize to 100%</button>
                        <button class="btn" style="background: transparent; border: 1px solid var(--border);" @click="resetWeights">Reset</button>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header"><div class="card-title">Hypothetical Top {K['n_stocks']}</div></div>
                    <div class="table-container" style="max-height: 500px; overflow-y: auto;">
                        <table>
                            <thead>
                                <tr><th>Rank</th><th>Stock</th><th>New Score</th></tr>
                            </thead>
                            <tbody>
                                <tr v-for="(s, i) in sandboxResults" :key="s.ticker" :style="s.isNew ? 'background: rgba(16, 185, 129, 0.05)' : ''">
                                    <td>{{{{ i + 1 }}}}</td>
                                    <td>{{{{ s.ticker }}}} <span v-if="s.isNew" style="color:var(--success); font-size:10px; margin-left:8px;">★ NEW</span></td>
                                    <td style="font-family: var(--font-mono)">{{{{ s.score.toFixed(3) }}}}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        </transition>

        <!-- Compare Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'compare'">
            <div class="page-header">
                <h1 class="page-title">Stock Comparison</h1>
                <div class="page-desc">Compare factor ranks and scores between any two stocks</div>
            </div>
            
            <div class="grid-cols-2" style="margin-bottom: 24px;">
                <div class="card">
                    <div class="card-header"><div class="card-title">Stock A</div></div>
                    <select class="compare-select" v-model="compareA" @change="updateCompareChart">
                        <option v-for="t in allTickers" :value="t" :key="'A'+t">{{{{ t }}}}</option>
                    </select>
                    
                    <div v-if="compareA && data.stocks[compareA]" style="margin-top: 20px;">
                        <div style="font-size: 20px; font-weight: 600;">{{{{ data.stocks[compareA].name }}}}</div>
                        <div style="color: var(--text-muted); font-size: 13px; margin-bottom: 16px;">{{{{ data.stocks[compareA].sector }}}}</div>
                        <div class="kpi-value text-accent">{{{{ data.stocks[compareA].score_hist.values.slice(-1)[0] }}}}</div>
                        <div class="kpi-sub">Current Final Score</div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header"><div class="card-title">Stock B</div></div>
                    <select class="compare-select" v-model="compareB" @change="updateCompareChart">
                        <option v-for="t in allTickers" :value="t" :key="'B'+t">{{{{ t }}}}</option>
                    </select>
                    
                    <div v-if="compareB && data.stocks[compareB]" style="margin-top: 20px;">
                        <div style="font-size: 20px; font-weight: 600;">{{{{ data.stocks[compareB].name }}}}</div>
                        <div style="color: var(--text-muted); font-size: 13px; margin-bottom: 16px;">{{{{ data.stocks[compareB].sector }}}}</div>
                        <div class="kpi-value text-success">{{{{ data.stocks[compareB].score_hist.values.slice(-1)[0] }}}}</div>
                        <div class="kpi-sub">Current Final Score</div>
                    </div>
                </div>
            </div>
            
            <div class="card" v-show="compareA && compareB">
                <div class="card-header"><div class="card-title">Current Factor Ranks Comparison</div></div>
                <div class="chart-container" style="height: 400px; display:flex; justify-content:center;">
                    <canvas id="compareRadarChart"></canvas>
                </div>
            </div>
        </div>
        </transition>

        <!-- Diagnostics Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'diagnostics'">
            <div class="page-header">
                <h1 class="page-title">Factor Diagnostics</h1>
                <div class="page-desc">Information Coefficient · Quintile Spread · Cross-Factor Correlation — every number computed, none hardcoded.</div>
            </div>

            <div class="grid-cols-2" style="margin-bottom: 24px;">
                <div class="card">
                    <div class="card-header"><div class="card-title">Mean Information Coefficient</div></div>
                    <div class="chart-container"><canvas id="diagICChart"></canvas></div>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 12px;">
                        Spearman rank-correlation between factor value at <em>t</em> and forward return (<em>t→t+1</em>).
                        Sign-flipped for lower-is-better factors. ≥0.05 is decent; ≥0.10 is strong.
                    </div>
                </div>
                <div class="card">
                    <div class="card-header"><div class="card-title">Hit Rate · % of Periods Positive</div></div>
                    <div class="chart-container"><canvas id="diagHitChart"></canvas></div>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 12px;">
                        How consistently the factor was on the right side. 50% = coin flip.
                    </div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header"><div class="card-title">Quintile Spread (Q5 − Q1) · Mean Forward Return per Period</div></div>
                <div class="chart-container"><canvas id="diagQSChart"></canvas></div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 12px;">
                    Equal-weighted top quintile return minus bottom quintile return, averaged across all rebalances.
                    Larger spread = factor genuinely separates winners from losers.
                </div>
            </div>

            <div class="card">
                <div class="card-header"><div class="card-title">Cross-Factor Rank Correlation (Spearman, averaged)</div></div>
                <div class="table-container" style="border:none;">
                    <table style="font-family: var(--font-mono); font-size:12px;">
                        <thead>
                            <tr>
                                <th></th>
                                <th v-for="g in data.corr.factors" :key="'corr-h-'+g" style="text-align:center;">{{{{ g.substring(0,3) }}}}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(row, i) in data.corr.matrix" :key="'corr-r-'+i">
                                <td style="font-weight: 600;">{{{{ data.corr.factors[i] }}}}</td>
                                <td v-for="(v, j) in row" :key="'corr-c-'+i+'-'+j"
                                    :style="corrCellStyle(v, i, j)"
                                    style="text-align:center;">
                                    {{{{ v === null ? 'NaN' : (i === j ? '—' : v.toFixed(2)) }}}}
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 12px;">
                    Negative entries = genuine diversification. Strong positive (&gt;0.3) means the two factors are picking
                    overlapping stocks and one of them is partially redundant.
                </div>
            </div>
        </div>
        </transition>

        <!-- Stock Drill-Down Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'drilldown'">
            <div class="page-header">
                <h1 class="page-title">Stock Drill-Down</h1>
                <div class="page-desc">Pick a ticker to see its price, score history, factor-rank trajectory, and in-portfolio timeline.</div>
            </div>

            <div class="card" style="margin-bottom: 24px; padding: 16px 24px;">
                <div style="display:flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                    <select class="compare-select" v-model="drillTicker" @change="renderDrill"
                            style="max-width: 320px;">
                        <option v-for="t in allTickers" :value="t" :key="'dd-'+t">{{{{ data.stocks[t] ? data.stocks[t].name : t }}}}</option>
                    </select>
                    <div v-if="drillTicker && data.stocks[drillTicker]" style="font-size: 13px; color: var(--text-muted);">
                        <strong style="color: var(--text-main); font-size: 16px;">{{{{ data.stocks[drillTicker].name }}}}</strong>
                        &nbsp;·&nbsp; {{{{ data.stocks[drillTicker].sector }}}}
                        &nbsp;·&nbsp; latest score: <span style="color: var(--accent); font-family: var(--font-mono);">{{{{ drillLatestScore }}}}</span>
                        &nbsp;·&nbsp; in-portfolio at: <span :style="{{color: drillInPortNow ? 'var(--success)' : 'var(--text-muted)'}}">{{{{ drillInPortNow ? 'YES' : 'no' }}}}</span>
                    </div>
                </div>
            </div>

            <div class="grid-cols-2" style="margin-bottom: 24px;">
                <div class="card">
                    <div class="card-header"><div class="card-title">Price · since inception</div></div>
                    <div class="chart-container"><canvas id="drillPxChart"></canvas></div>
                </div>
                <div class="card">
                    <div class="card-header"><div class="card-title">Composite Final Score Over Time</div></div>
                    <div class="chart-container"><canvas id="drillScoreChart"></canvas></div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header"><div class="card-title">Factor Rank Trajectory</div></div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th v-for="f in data.factors" :key="'drh-'+f" style="text-align:center;">{{{{ f.substring(0,3) }}}}</th>
                                <th style="text-align:center;">Held</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(d, i) in drillRows" :key="'drr-'+i">
                                <td style="font-family: var(--font-mono); color: var(--text-muted);">{{{{ d.date }}}}</td>
                                <td v-for="f in data.factors" :key="'drv-'+i+'-'+f" style="text-align:center;">
                                    <span v-if="d.ranks[f] !== null && d.ranks[f] !== undefined" :class="['rank-badge', 'r'+d.ranks[f]]">{{{{ d.ranks[f] }}}}</span>
                                    <span v-else style="color: var(--text-muted);">—</span>
                                </td>
                                <td style="text-align:center;">
                                    <i v-if="d.held" class="ri-check-line" style="color: var(--success);"></i>
                                    <span v-else style="color: var(--text-muted);">·</span>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        </transition>

        <!-- Universe Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'universe'">
            <div class="page-header">
                <h1 class="page-title">Universe</h1>
                <div class="page-desc">Every ranked stock at the latest score date. Search by ticker / sector, filter to portfolio only, sort by any column.</div>
            </div>

            <div class="card" style="margin-bottom: 16px; padding: 16px 24px;">
                <div style="display:flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                    <input class="compare-select" type="text" v-model="univSearch" placeholder="Search ticker or sector…" style="max-width: 280px;">
                    <select class="compare-select" v-model="univSector" style="max-width: 240px;">
                        <option value="">All sectors</option>
                        <option v-for="s in univSectors" :value="s" :key="'us-'+s">{{{{ s }}}}</option>
                    </select>
                    <label style="font-size: 13px; color: var(--text-muted); display: flex; align-items: center; gap: 6px;">
                        <input type="checkbox" v-model="univPortOnly"> portfolio only
                    </label>
                    <span style="margin-left: auto; font-size: 12px; color: var(--text-muted);">
                        {{{{ univFiltered.length }}}} of {{{{ data.universe.length }}}} stocks
                    </span>
                </div>
            </div>

            <div class="card">
                <div class="table-container" style="max-height: 65vh; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th @click="univSortBy('rank')" style="cursor: pointer;">#</th>
                                <th @click="univSortBy('ticker')" style="cursor: pointer;">Stock</th>
                                <th>Sector</th>
                                <th @click="univSortBy('score')" style="cursor: pointer;">Score</th>
                                <th v-for="f in data.factors" :key="'uh-'+f" style="text-align:center;">{{{{ f.substring(0,3) }}}}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="r in univFiltered" :key="r.full"
                                :style="r.in_port ? 'background: rgba(16, 185, 129, 0.05)' : ''">
                                <td style="color: var(--text-muted);">{{{{ r.rank }}}}</td>
                                <td>
                                    <span style="font-weight: 600; cursor: pointer;" @click="goToDrill(r.full)">{{{{ r.ticker }}}}</span>
                                    <span v-if="r.in_port" class="rank-badge r5" style="margin-left:8px; font-size:9px; padding: 0 6px; width:auto; height:auto;">IN</span>
                                </td>
                                <td style="color: var(--text-muted); font-size: 12px;">{{{{ r.sector }}}}</td>
                                <td style="font-family: var(--font-mono); font-weight: 600;">{{{{ r.score.toFixed(3) }}}}</td>
                                <td v-for="(rk, idx) in r.ranks" :key="'urk-'+r.full+'-'+idx" style="text-align:center;">
                                    <span :class="['rank-badge', 'r'+rk]">{{{{ rk }}}}</span>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        </transition>

        <!-- Logs Tab (enhanced) -->
        <transition name="fade">
        <div v-show="activeTab === 'logs'">
            <div class="page-header">
                <h1 class="page-title">Rebalance History</h1>
                <div class="page-desc">Click any row to see the top winners and losers from that period. The chart shows turnover & cost over time.</div>
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header"><div class="card-title">Cost &amp; Turnover Trend</div></div>
                <div class="chart-container" style="height: 280px;"><canvas id="logsCTChart"></canvas></div>
            </div>

            <div class="grid-cols-2">
                <div class="card">
                    <div class="card-header"><div class="card-title">Rebalance Log</div></div>
                    <div class="table-container" style="max-height: 60vh; overflow-y: auto;">
                        <table>
                            <thead>
                                <tr><th>Date</th><th>Univ</th><th>Port Ret</th><th>Nifty</th><th>Turn</th><th>Cost</th></tr>
                            </thead>
                            <tbody>
                                <tr v-for="(r, i) in data.rebal" :key="'lg-'+i"
                                    @click="logsSelected = i"
                                    :style="logsSelected === i ? 'background: rgba(59, 130, 246, 0.08); cursor:pointer;' : 'cursor:pointer;'">
                                    <td style="font-weight: 500;">{{{{ r.date }}}}</td>
                                    <td>{{{{ r.univ }}}}</td>
                                    <td :class="r.period_r >= 0 ? 'text-success' : 'text-danger'">{{{{ r.period_r >= 0 ? '+' : '' }}}}{{{{ r.period_r }}}}%</td>
                                    <td :class="r.nifty_r >= 0 ? 'text-success' : 'text-danger'">{{{{ r.nifty_r >= 0 ? '+' : '' }}}}{{{{ r.nifty_r }}}}%</td>
                                    <td>{{{{ r.turn }}}}%</td>
                                    <td class="text-danger" style="font-family: var(--font-mono);">₹{{{{ r.cost.toLocaleString() }}}}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Period Detail</div>
                        <div v-if="logsSelectedRow" style="font-size: 12px; color: var(--text-muted);">{{{{ logsSelectedRow.date }}}} · {{{{ logsSelectedRow.period_r >= 0 ? '+' : '' }}}}{{{{ logsSelectedRow.period_r }}}}% return</div>
                    </div>
                    <div v-if="logsSelectedRow">
                        <div style="margin-bottom: 16px;">
                            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px;">Top Winners</div>
                            <table style="font-family: var(--font-mono); font-size: 12px;">
                                <tr v-for="w in logsSelectedRow.winners" :key="'win-'+w[0]">
                                    <td style="color: var(--text-main); font-weight: 500;">{{{{ w[0] }}}}</td>
                                    <td class="text-success" style="text-align: right;">+{{{{ w[1] }}}}%</td>
                                </tr>
                                <tr v-if="!logsSelectedRow.winners.length"><td style="color:var(--text-muted)">—</td></tr>
                            </table>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px;">Top Losers</div>
                            <table style="font-family: var(--font-mono); font-size: 12px;">
                                <tr v-for="l in logsSelectedRow.losers" :key="'los-'+l[0]">
                                    <td style="color: var(--text-main); font-weight: 500;">{{{{ l[0] }}}}</td>
                                    <td class="text-danger" style="text-align: right;">{{{{ l[1] }}}}%</td>
                                </tr>
                                <tr v-if="!logsSelectedRow.losers.length"><td style="color:var(--text-muted)">—</td></tr>
                            </table>
                        </div>
                    </div>
                    <div v-else style="color: var(--text-muted); font-size: 13px;">Click a row in the log to see winners and losers from that period.</div>
                </div>
            </div>
        </div>
        </transition>

        <!-- Model Tab -->
        <transition name="fade">
        <div v-show="activeTab === 'model'">
            <div class="page-header">
                <h1 class="page-title">Model Reference</h1>
                <div class="page-desc">What the 7 factors measure, how the pipeline runs, and the limitations to keep in mind.</div>
            </div>

            <div class="grid-cols-2" style="margin-bottom: 24px;">
                <div class="card">
                    <div class="card-header"><div class="card-title">The 7 Factors</div></div>
                    <div v-for="(f, i) in data.factors" :key="'fact-'+f" style="background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 600; color: var(--text-main);">{{{{ f }}}}</span>
                            <span style="font-family: var(--font-mono); color: var(--accent); font-weight: 600;">{{{{ (data.weights[f]*100).toFixed(0) }}}}%</span>
                        </div>
                        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">
                            {{{{ factorDesc[f].metric }}}}
                            <span v-if="data.lower_better.includes(f)" class="rank-badge r2" style="font-size: 9px; padding: 1px 6px; width:auto; height:auto; margin-left: 6px;">LOWER better</span>
                        </div>
                        <div style="font-size: 13px; color: var(--text-main); line-height: 1.6;">{{{{ factorDesc[f].desc }}}}</div>
                    </div>
                </div>
                <div>
                    <div class="card" style="margin-bottom: 16px;">
                        <div class="card-header"><div class="card-title">Factor Weights</div></div>
                        <div class="chart-container" style="height: 240px;"><canvas id="modelWeightsChart"></canvas></div>
                    </div>
                    <div class="card">
                        <div class="card-header"><div class="card-title">Pipeline (data → score → portfolio)</div></div>
                        <ol style="font-size: 13px; line-height: 2; color: var(--text-main); padding-left: 20px;">
                            <li><code>screener_scraper.py</code> → <code>screener_raw.csv</code></li>
                            <li><code>build_pit.py</code> → <code>fundamentals_pit.csv</code> (3-month institutional lag)</li>
                            <li><code>universe_builder.py</code> → date-aware Nifty 200 constituents</li>
                            <li><code>data_layer.py</code> · yfinance → <code>prices.csv</code> + <code>shares_outstanding.csv</code></li>
                            <li><code>factor_engine.py</code> · 7 raw → quintile rank → composite</li>
                            <li><code>backtester.py</code> · semi-annual rebalance + monthly equity walk</li>
                            <li><code>analytics.py</code> · single-factor BTs, IC, attribution, sector tilt</li>
                            <li><code>dashboard.py</code> · this page</li>
                        </ol>
                    </div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <div class="card-header"><div class="card-title">Methodology</div></div>
                <div style="font-size: 13px; line-height: 1.8; color: var(--text-main);">
                    <p><strong>Momentum:</strong> 0.4·Ret(T-4 → T-2) + 0.6·Ret(T-12 → T-5). T-1 is skipped to avoid short-term reversal.</p>
                    <p><strong>Beta:</strong> 36-month rolling OLS slope vs Nifty 50, clipped to [0.1, 4.0].</p>
                    <p><strong>Quality:</strong> Operating Profit / Total Assets, taken from the latest report visible at the rebal date with a 3-month institutional lag.</p>
                    <p><strong>Value:</strong> Book-to-Market = Equity / Price (computed at the rebal date — Equity is PiT, Price is the rebal close).</p>
                    <p><strong>Size:</strong> Market Cap = Price × shares-outstanding (yfinance current snapshot, split/bonus-corrected via auto_adjust prices). Falls back to Total Assets if shares unavailable.</p>
                    <p><strong>Investment:</strong> YoY total-assets growth.</p>
                    <p><strong>Yield:</strong> EPS × Dividend Payout %, ÷ Price. Missing payout/EPS → Rank 3 (neutral, never zero — that would punish data-incomplete names).</p>
                    <p><strong>Universe gating (Phase 3):</strong> at each rebal date, a stock must (a) be in the Nifty 200 historical constituent list, (b) have ≥13 months of price history, (c) have ≥3 of 5 fundamental factors populated. Stocks failing (c) are dropped — previously they were silently propped up by Rank-3 NaN fills.</p>
                    <p><strong>Ranking:</strong> Each raw value bucketed into quintiles 1–5. Lower-is-better factors (Size, Invest) inverted.</p>
                    <p><strong>Composite:</strong> Final Score = Σ Rank · Weight. Mean is mathematically guaranteed ≈ 3.0. Top {K['n_stocks']} by Final Score, equal-weight.</p>
                </div>
            </div>

            <div class="card">
                <div class="card-header"><div class="card-title">Known Limitations</div></div>
                <div style="font-size: 13px; line-height: 1.8; color: var(--text-main);">
                    <p><span class="rank-badge r2" style="font-size: 9px; padding: 1px 6px; width:auto; height:auto; margin-right: 6px;">DATA</span><strong>Buyback drift in MCap.</strong> Size uses <em>current</em> shares-outstanding. Splits/bonuses are corrected by yfinance auto-adjusted prices, but companies with major buybacks/issuances since 2020 will have biased historical Size ranks.</p>
                    <p><span class="rank-badge r2" style="font-size: 9px; padding: 1px 6px; width:auto; height:auto; margin-right: 6px;">DATA</span><strong>Bank fundamentals.</strong> ~15 banks/financial-services entities (BANDHANBNK, SBICARD, SBILIFE, ICICIGI, etc.) fail the screener Balance-Sheet parser and are excluded from the eligible universe via the data-completeness gate.</p>
                    <p><span class="rank-badge r3" style="font-size: 9px; padding: 1px 6px; width:auto; height:auto; margin-right: 6px;">UNIV</span><strong>Nifty 200 carry-forward.</strong> For Sep 2022 → Mar 2024 the NSE source ZIPs only contain Nifty 50 + Next 50 PDFs (no full Nifty 200 list). The universe builder carries forward the previous full list and merges in the available top-100 — approximate, but strictly better than capping at 50–100 names.</p>
                    <p><span class="rank-badge r3" style="font-size: 9px; padding: 1px 6px; width:auto; height:auto; margin-right: 6px;">UNIV</span><strong>Forward-fill at inception.</strong> Feb 1 2020 inception predates the earliest reconstitution snapshot (Mar 31 2020). The backtester forward-fills the earliest snapshot for that one rebal — minor look-ahead, ~95% of constituents are stable across one quarter.</p>
                    <p><span class="rank-badge r4" style="font-size: 9px; padding: 1px 6px; width:auto; height:auto; margin-right: 6px;">STYLE</span><strong>Beta-as-high-rank.</strong> The model gives Rank 5 to high-beta stocks per the PDF spec. A "low-vol anomaly" model would invert this. Style choice, not a bug.</p>
                    <p><span class="rank-badge r3" style="font-size: 9px; padding: 1px 6px; width:auto; height:auto; margin-right: 6px;">UNIV</span><strong>Sector mapping is static.</strong> Labels come from <code>sectors.py</code>. NSE sector reclassifications and corporate restructurings are not tracked.</p>
                </div>
            </div>
        </div>
        </transition>
    </div>
</div>

<script>
const rawData = {js};

// Global chart defaults
Chart.defaults.color = '#8F9EB2';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 17, 23, 0.9)';
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;

const app = Vue.createApp({{
    data() {{
        return {{
            data: rawData,
            activeTab: 'overview',
            charts: {{}},

            // Sandbox state
            sandboxWeights: {{}},
            sandboxResults: [],
            currentHoldings: new Set(),

            // Compare state
            allTickers: [],
            compareA: '',
            compareB: '',

            // Drill-down state
            drillTicker: '',

            // Universe filter state
            univSearch: '',
            univSector: '',
            univPortOnly: false,
            univSortKey: 'rank',
            univSortAsc: true,

            // Logs state
            logsSelected: 0,

            // Static factor descriptions for the Model tab
            factorDesc: {{
                Momentum: {{ metric: 'Price persistence', desc: 'Compounded return T-12 → T-2 with T-1 skipped. Captures the well-documented 6–12 month price-trend anomaly.' }},
                Quality:  {{ metric: 'Operating profit per ₹ of assets', desc: 'Operating Profit ÷ Total Assets. High-quality firms generate more profit from each rupee invested in their balance sheet.' }},
                Value:    {{ metric: 'Cheapness', desc: 'Book Equity ÷ Market Price. The Fama-French value factor — buy what is cheap relative to fundamentals.' }},
                Size:     {{ metric: 'Market capitalisation', desc: 'Smaller companies historically outperform on a risk-adjusted basis (size effect). Lower values rank higher.' }},
                Beta:     {{ metric: 'Market sensitivity', desc: 'OLS slope of stock returns vs Nifty 50 returns over a rolling 36-month window. Higher = more aggressive.' }},
                Invest:   {{ metric: 'Conservative investment', desc: 'YoY total-assets growth. The investment factor: firms that don\\'t over-invest tend to outperform. Lower values rank higher.' }},
                Yield:    {{ metric: 'Cash returned to shareholders', desc: 'EPS × payout%, ÷ price. A "cash-flow discipline" filter on the high-momentum tilt.' }},
            }},
        }}
    }},
    computed: {{
        sandboxTotalWeight() {{
            return Object.values(this.sandboxWeights).reduce((a,b) => a+b, 0);
        }},

        // Drill-down derived state
        drillRows() {{
            const s = this.data.stocks[this.drillTicker];
            if (!s) return [];
            const labels = s.score_hist.labels;
            const ranks = s.ranks_hist.ranks || {{}};
            const inP   = s.in_port || [];
            return labels.map((d, i) => {{
                const r = {{}};
                for (const f of this.data.factors) {{
                    r[f] = ranks[f] ? ranks[f][i] : null;
                }}
                return {{ date: d, ranks: r, held: !!inP[i] }};
            }});
        }},
        drillLatestScore() {{
            const s = this.data.stocks[this.drillTicker];
            if (!s) return '—';
            const v = s.score_hist.values.slice(-1)[0];
            return v === null || v === undefined ? '—' : v.toFixed(3);
        }},
        drillInPortNow() {{
            const s = this.data.stocks[this.drillTicker];
            if (!s || !s.in_port) return false;
            return s.in_port.slice(-1)[0] === true;
        }},

        // Universe derived state
        univSectors() {{
            const set = new Set();
            for (const r of this.data.universe) set.add(r.sector);
            return Array.from(set).sort();
        }},
        univFiltered() {{
            const q  = (this.univSearch || '').toLowerCase();
            const sc = this.univSector;
            const po = this.univPortOnly;
            let rows = this.data.universe.filter(r => {{
                if (po && !r.in_port) return false;
                if (sc && r.sector !== sc) return false;
                if (q && !(r.ticker.toLowerCase().includes(q) || (r.sector||'').toLowerCase().includes(q))) return false;
                return true;
            }});
            const k = this.univSortKey, asc = this.univSortAsc;
            rows.sort((a, b) => {{
                const av = a[k], bv = b[k];
                if (typeof av === 'string') return asc ? av.localeCompare(bv) : bv.localeCompare(av);
                return asc ? av - bv : bv - av;
            }});
            return rows;
        }},

        // Logs derived state
        logsSelectedRow() {{
            return this.data.rebal[this.logsSelected] || null;
        }},
    }},
    mounted() {{
        // Init sandbox
        this.resetWeights();
        this.currentHoldings = new Set(this.data.heatmap.map(h => h.full));

        // Init compare + drilldown
        this.allTickers = Object.keys(this.data.stocks).sort();
        if (this.allTickers.length >= 2) {{
            this.compareA = this.data.heatmap[0].full;
            this.compareB = this.data.heatmap[1].full;
            // Drill defaults to the highest-scored portfolio name
            this.drillTicker = this.data.heatmap[0].full;
        }}

        // Logs default to the most recent rebalance
        this.logsSelected = Math.max(0, (this.data.rebal || []).length - 1);

        // Render charts slightly delayed to ensure DOM is ready
        setTimeout(() => {{
            this.renderCharts();
            this.recalculateSandbox();
            this.updateCompareChart();
            this.renderDiagnostics();
            this.renderDrill();
            this.renderLogsCT();
            this.renderModelWeights();
        }}, 100);
    }},
    methods: {{
        resetWeights() {{
            for(let f of this.data.factors) {{
                this.sandboxWeights[f] = Math.round(this.data.weights[f] * 100);
            }}
            this.recalculateSandbox();
        }},
        normalizeWeights() {{
            let total = this.sandboxTotalWeight;
            if(total === 0) return;
            for(let f of this.data.factors) {{
                this.sandboxWeights[f] = Math.round((this.sandboxWeights[f] / total) * 100);
            }}
            this.recalculateSandbox();
        }},
        recalculateSandbox() {{
            let results = [];
            // We use universe data which has the ranks for all stocks
            for(let u of this.data.universe) {{
                let score = 0;
                for(let i=0; i<this.data.factors.length; i++) {{
                    let f = this.data.factors[i];
                    let rank = u.ranks[i];
                    let w = this.sandboxWeights[f] / 100.0;
                    // Lower is better logic is already baked into ranks! (Rank 5 is always best)
                    score += rank * w;
                }}
                results.push({{
                    ticker: u.ticker,
                    full: u.full,
                    score: score,
                    isNew: !this.currentHoldings.has(u.full)
                }});
            }}
            // Sort by score desc
            results.sort((a,b) => b.score - a.score);
            this.sandboxResults = results.slice(0, {K['n_stocks']});
        }},
        
        renderCharts() {{
            // 1. Equity Chart
            this.charts.equity = new Chart(document.getElementById('equityChart'), {{
                type: 'line',
                data: {{
                    labels: this.data.equity.labels,
                    datasets: [
                        {{ label: 'Portfolio', data: this.data.equity.port, borderColor: '#3B82F6', borderWidth: 2, tension: 0.1, pointRadius: 0 }},
                        {{ label: 'Nifty 50', data: this.data.equity.nifty, borderColor: '#8F9EB2', borderWidth: 1, borderDash: [5,5], tension: 0.1, pointRadius: 0 }}
                    ]
                }},
                options: {{ responsive: true, maintainAspectRatio: false, interaction: {{ mode: 'index', intersect: false }} }}
            }});
            
            // 2. Drawdown Chart
            this.charts.dd = new Chart(document.getElementById('drawdownChart'), {{
                type: 'line',
                data: {{
                    labels: this.data.drawdown.labels,
                    datasets: [
                        {{ label: 'Portfolio DD', data: this.data.drawdown.port, borderColor: '#EF4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', fill: true, borderWidth: 1, pointRadius: 0 }}
                    ]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
            
            // 3. Rolling Metrics
            this.charts.rolling = new Chart(document.getElementById('rollingChart'), {{
                type: 'line',
                data: {{
                    labels: this.data.rolling.labels,
                    datasets: [
                        {{ label: 'Sharpe', data: this.data.rolling.sharpe, borderColor: '#10B981', borderWidth: 1, pointRadius: 0 }},
                        {{ label: 'Alpha', data: this.data.rolling.alpha, borderColor: '#F59E0B', borderWidth: 1, pointRadius: 0 }}
                    ]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
            
            // 4. Sector Exposure
            let sectorDatasets = this.data.sector_exp.sectors.map((sec, i) => ({{
                label: sec,
                data: this.data.sector_exp.matrix[i],
                fill: true,
                borderWidth: 0,
                // Assign distinct colors using HSL
                backgroundColor: `hsla(${{i * (360 / this.data.sector_exp.sectors.length)}}, 70%, 50%, 0.7)`
            }}));
            
            this.charts.sector = new Chart(document.getElementById('sectorChart'), {{
                type: 'line',
                data: {{
                    labels: this.data.sector_exp.labels,
                    datasets: sectorDatasets
                }},
                options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ stacked: true, max: 100 }} }}, elements: {{ point: {{ radius: 0 }} }} }}
            }});
            
            // 5. Single Factor Curves
            let sfDatasets = this.data.factors.map((f, i) => ({{
                label: f,
                data: this.data.sf.series[f],
                borderWidth: 1.5,
                borderColor: `hsla(${{i * (360 / this.data.factors.length)}}, 70%, 60%, 1)`,
                pointRadius: 0,
                tension: 0.1
            }}));
            sfDatasets.push({{
                label: 'MultiFactor',
                data: this.data.sf.series['MultiFactor'],
                borderWidth: 3,
                borderColor: '#E2E8F0',
                pointRadius: 0,
                tension: 0.1
            }});
            
            this.charts.sf = new Chart(document.getElementById('sfChart'), {{
                type: 'line',
                data: {{ labels: this.data.sf.labels, datasets: sfDatasets }},
                options: {{ responsive: true, maintainAspectRatio: false, interaction: {{ mode: 'index', intersect: false }} }}
            }});
            
            // 6. IC Chart
            this.charts.ic = new Chart(document.getElementById('icChart'), {{
                type: 'bar',
                data: {{
                    labels: this.data.ic.factors,
                    datasets: [{{ label: 'Mean IC', data: this.data.ic.mean_ic, backgroundColor: '#3B82F6', borderRadius: 4 }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false }}
            }});
        }},
        
        updateCompareChart() {{
            if(!this.compareA || !this.compareB) return;
            
            // Get latest ranks
            let ranksA = [], ranksB = [];
            
            for(let f of this.data.factors) {{
                let rhA = this.data.stocks[this.compareA].ranks_hist.ranks[f];
                let rhB = this.data.stocks[this.compareB].ranks_hist.ranks[f];
                ranksA.push(rhA ? rhA[rhA.length-1] : 3);
                ranksB.push(rhB ? rhB[rhB.length-1] : 3);
            }}
            
            if(this.charts.compare) this.charts.compare.destroy();
            
            this.charts.compare = new Chart(document.getElementById('compareRadarChart'), {{
                type: 'radar',
                data: {{
                    labels: this.data.factors,
                    datasets: [
                        {{
                            label: this.data.stocks[this.compareA].name,
                            data: ranksA,
                            backgroundColor: 'rgba(59, 130, 246, 0.2)',
                            borderColor: '#3B82F6',
                            pointBackgroundColor: '#3B82F6',
                        }},
                        {{
                            label: this.data.stocks[this.compareB].name,
                            data: ranksB,
                            backgroundColor: 'rgba(16, 185, 129, 0.2)',
                            borderColor: '#10B981',
                            pointBackgroundColor: '#10B981',
                        }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    scales: {{ r: {{ min: 0, max: 5, ticks: {{ stepSize: 1 }} }} }}
                }}
            }});
        }},

        // ── Diagnostics ───────────────────────────────────────────────────
        corrCellStyle(v, i, j) {{
            if (v === null || v === undefined || i === j) {{
                return 'padding:8px 4px; color: var(--text-muted);';
            }}
            const a = Math.min(Math.abs(v), 0.7);
            let bg = 'transparent';
            if (v < -0.1)      bg = `rgba(239, 68, 68, ${{a}})`;
            else if (v >  0.1) bg = `rgba(59, 130, 246, ${{a}})`;
            const c = Math.abs(v) > 0.25 ? 'white' : 'var(--text-main)';
            return `padding:8px 4px; background:${{bg}}; color:${{c}}; border-radius:4px;`;
        }},
        renderDiagnostics() {{
            // IC bar (separate from the small IC chart on the Factors tab)
            this.charts.diagIC = new Chart(document.getElementById('diagICChart'), {{
                type: 'bar',
                data: {{
                    labels: this.data.ic.factors,
                    datasets: [{{
                        label: 'Mean IC',
                        data: this.data.ic.mean_ic,
                        backgroundColor: this.data.ic.mean_ic.map(v => (v ?? 0) >= 0 ? '#10B981' : '#EF4444'),
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }},
                        tooltip: {{ callbacks: {{ label: c => `IC: ${{(c.raw ?? 0).toFixed(3)}}` }} }} }},
                    scales: {{ y: {{ ticks: {{ callback: v => v.toFixed(2) }} }} }}
                }}
            }});

            // Hit rate bar
            this.charts.diagHit = new Chart(document.getElementById('diagHitChart'), {{
                type: 'bar',
                data: {{
                    labels: this.data.ic.factors,
                    datasets: [{{
                        label: 'Hit %',
                        data: this.data.ic.hit_rate.map(v => (v ?? 0) * 100),
                        backgroundColor: this.data.ic.hit_rate.map(v => (v ?? 0) >= 0.5 ? '#10B981' : '#EF4444'),
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }},
                        tooltip: {{ callbacks: {{ label: c => `${{c.raw.toFixed(0)}}% positive` }} }} }},
                    scales: {{ y: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }} }}
                }}
            }});

            // Quintile spread bar (in %)
            this.charts.diagQS = new Chart(document.getElementById('diagQSChart'), {{
                type: 'bar',
                data: {{
                    labels: this.data.qs.factors,
                    datasets: [{{
                        label: 'Q5 − Q1',
                        data: this.data.qs.mean_spread.map(v => v === null ? null : v * 100),
                        backgroundColor: this.data.qs.mean_spread.map(v => (v ?? 0) >= 0 ? '#10B981' : '#EF4444'),
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }},
                        tooltip: {{ callbacks: {{ label: c => `${{(c.raw ?? 0).toFixed(2)}}% per period` }} }} }},
                    scales: {{ y: {{ ticks: {{ callback: v => v.toFixed(1) + '%' }} }} }}
                }}
            }});
        }},

        // ── Stock Drill-Down ──────────────────────────────────────────────
        renderDrill() {{
            const s = this.data.stocks[this.drillTicker];
            if (!s) return;
            if (this.charts.drillPx)    this.charts.drillPx.destroy();
            if (this.charts.drillScore) this.charts.drillScore.destroy();

            this.charts.drillPx = new Chart(document.getElementById('drillPxChart'), {{
                type: 'line',
                data: {{
                    labels: s.price.labels,
                    datasets: [{{
                        label: 'Price',
                        data: s.price.values,
                        borderColor: '#3B82F6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true, tension: 0.25, borderWidth: 1.8, pointRadius: 0,
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }},
                        tooltip: {{ callbacks: {{ label: c => '₹' + (c.raw ?? 0).toFixed(2) }} }} }}
                }}
            }});

            this.charts.drillScore = new Chart(document.getElementById('drillScoreChart'), {{
                type: 'line',
                data: {{
                    labels: s.score_hist.labels,
                    datasets: [{{
                        label: 'Final Score',
                        data: s.score_hist.values,
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        fill: true, tension: 0.25, borderWidth: 1.8, pointRadius: 3,
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }},
                        tooltip: {{ callbacks: {{ label: c => 'Score: ' + (c.raw ?? 0).toFixed(3) }} }} }},
                    scales: {{ y: {{ min: 1, max: 5 }} }}
                }}
            }});
        }},
        goToDrill(full) {{
            this.drillTicker = full;
            this.activeTab = 'drilldown';
            this.$nextTick(() => this.renderDrill());
        }},

        // ── Universe ──────────────────────────────────────────────────────
        univSortBy(k) {{
            if (this.univSortKey === k) {{
                this.univSortAsc = !this.univSortAsc;
            }} else {{
                this.univSortKey = k;
                this.univSortAsc = (k === 'rank');
            }}
        }},

        // ── Logs cost/turnover trend ──────────────────────────────────────
        renderLogsCT() {{
            const labels = this.data.rebal.map(r => r.date);
            const turn   = this.data.rebal.map(r => r.turn);
            const cost   = this.data.rebal.map(r => r.cost);
            this.charts.logsCT = new Chart(document.getElementById('logsCTChart'), {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ label: 'Turnover (%)', data: turn, borderColor: '#F59E0B',
                            yAxisID: 'y',  borderWidth: 1.8, pointRadius: 3, tension: 0.25 }},
                        {{ label: 'Cost (₹)',      data: cost, borderColor: '#EF4444',
                            yAxisID: 'y2', borderWidth: 1.8, pointRadius: 3, tension: 0.25 }},
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    interaction: {{ mode: 'index', intersect: false }},
                    scales: {{
                        y:  {{ position: 'left',  ticks: {{ callback: v => v + '%' }} }},
                        y2: {{ position: 'right', grid: {{ display: false }},
                              ticks: {{ callback: v => '₹' + (v/1000).toFixed(0) + 'k' }} }}
                    }}
                }}
            }});
        }},

        // ── Model · weights donut ─────────────────────────────────────────
        renderModelWeights() {{
            const F = this.data.factors;
            const colors = F.map((_, i) => `hsla(${{i * (360 / F.length)}}, 70%, 60%, 1)`);
            this.charts.modelW = new Chart(document.getElementById('modelWeightsChart'), {{
                type: 'doughnut',
                data: {{
                    labels: F,
                    datasets: [{{ data: F.map(f => this.data.weights[f] * 100),
                                  backgroundColor: colors, borderWidth: 0 }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false, cutout: '55%',
                    plugins: {{
                        legend: {{ position: 'right', labels: {{ font: {{ size: 11 }}, boxWidth: 10 }} }},
                        tooltip: {{ callbacks: {{ label: c => `${{c.label}}: ${{c.raw}}%` }} }}
                    }}
                }}
            }});
        }}
    }}
}}).mount('#app');
</script>
</body>
</html>"""
