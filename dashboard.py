"""
dashboard.py — Phase-2 analytical dashboard.

Single-file HTML dashboard rendered from real backtest + analytics outputs.
EVERY chart and table is driven from computed data (no hardcoded percentages,
correlations, or regime numbers).

Tabs:
  1. Overview        — equity curve, rolling 12M Sharpe/α/β, annual returns,
                       monthly drawdown.
  2. Attribution     — single-factor backtests (7 + multi), final values,
                       computed factor-regime heatmap.
  3. Diagnostics     — Information Coefficient, quintile spreads, computed
                       cross-factor correlations.
  4. Composition     — current top-25 heatmap, sector exposure over time,
                       holdings transition table.
  5. Rebalance       — full rebalance log, per-period winners/losers,
                       cost & turnover trend.
  6. Universe        — searchable, sortable ranking of all stocks.
  7. Stocks          — per-stock drill-down: price, score history, rank
                       trajectory, in-portfolio timeline.
  8. Model           — factor descriptions, weights, pipeline, methodology,
                       known limitations.
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
        "port_inr": [round(float(v))            for v in port_curve.values],
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
    js = json.dumps(JS, default=_safe)
    F_COLORS = ["#378ADD","#1D9E75","#EF9F27","#D85A30","#7F77DD","#5DCAA5","#D4537E"]

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSE 200 · 7-Factor Portfolio · {K['run_date']}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root{{
  --bg:#0F1117; --card:#1A1D27; --card2:#20253A;
  --border:rgba(255,255,255,0.08); --text:#E8EAF0; --sub:#8B90A0;
  --jade:#1D9E75; --amber:#EF9F27; --rust:#D85A30; --blue:#378ADD;
  --r5:#1D9E75; --r4:rgba(29,158,117,.55); --r3:#4A5568; --r2:#EF9F27; --r1:#D85A30;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:var(--bg);color:var(--text);font-size:13px;line-height:1.5}}
.mono{{font-family:'JetBrains Mono','Fira Code','Courier New',monospace}}
a{{color:var(--blue);text-decoration:none}}

.wrap{{max-width:1500px;margin:0 auto;padding:16px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.grid4{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.grid6{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
.grid8{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}}
@media(max-width:900px){{.grid2,.grid3{{grid-template-columns:1fr}}}}

.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}}
.card-sm{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px}}

.kpi-label{{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--sub);margin-bottom:4px}}
.kpi-value{{font-size:21px;font-weight:600;font-family:'JetBrains Mono',monospace;line-height:1}}
.kpi-sub{{font-size:11px;color:var(--sub);margin-top:3px}}
.pos{{color:var(--jade)}}.neg{{color:var(--rust)}}.neu{{color:var(--amber)}}

.section{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;
         color:var(--sub);margin:24px 0 10px;padding-left:2px}}

.tabs{{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:18px;overflow-x:auto}}
.tab{{padding:9px 14px;font-size:12px;font-weight:500;cursor:pointer;
     border-radius:8px 8px 0 0;border:1px solid transparent;border-bottom:none;
     color:var(--sub);background:transparent;transition:all .15s;white-space:nowrap}}
.tab.active{{background:var(--card);color:var(--text);border-color:var(--border);
            border-bottom-color:var(--card)}}
.tab-content{{display:none}}.tab-content.active{{display:block}}

.pip{{display:inline-flex;align-items:center;justify-content:center;
     width:22px;height:22px;border-radius:4px;font-size:11px;font-weight:700;
     color:white;font-family:'JetBrains Mono',monospace}}
.r5{{background:var(--r5)}}.r4{{background:var(--r4)}}
.r3{{background:var(--r3)}}.r2{{background:var(--r2)}}.r1{{background:var(--r1)}}

.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th{{color:var(--sub);font-weight:500;text-align:left;padding:7px 10px;
        border-bottom:1px solid var(--border);font-size:11px;
        cursor:pointer;user-select:none;white-space:nowrap}}
.tbl th:hover{{color:var(--text)}}
.tbl td{{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle}}
.tbl tr:hover td{{background:rgba(255,255,255,.03)}}
.tbl tr.in-port td{{background:rgba(29,158,117,.06)}}
.sticky-head th{{position:sticky;top:0;background:var(--card);z-index:1}}

.search-row{{display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap}}
.search-input{{flex:1;max-width:280px;padding:6px 12px;background:var(--card2);
              border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:12px}}
.search-input::placeholder{{color:var(--sub)}}
.badge{{display:inline-block;padding:2px 7px;border-radius:5px;font-size:10px;font-weight:600}}
.badge-in{{background:rgba(29,158,117,.2);color:var(--jade)}}
.badge-warn{{background:rgba(239,159,39,.2);color:var(--amber)}}
.badge-info{{background:rgba(55,138,221,.2);color:var(--blue)}}
.badge-sec{{background:rgba(255,255,255,.04);color:var(--sub);font-weight:400}}

.chart-wrap{{position:relative}}
.tbl-scroll{{overflow-x:auto;max-height:520px;overflow-y:auto}}
.tbl-scroll::-webkit-scrollbar{{width:4px;height:4px}}
.tbl-scroll::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}

.heatmap-cell{{width:30px;height:24px;border-radius:4px;display:inline-flex;
              align-items:center;justify-content:center;font-size:10px;
              font-weight:600;color:white}}

.legend-row{{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:10px;align-items:center;font-size:11px}}
.legend-dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}}
.legend-sq{{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px}}

.prog-bar{{height:6px;border-radius:3px;overflow:hidden;background:rgba(255,255,255,.08)}}
.prog-fill{{height:100%;border-radius:3px;transition:width .5s}}

.alert{{padding:10px 14px;border-radius:8px;font-size:12px;margin-bottom:12px;border-left:3px solid}}
.alert-ok{{background:rgba(29,158,117,.1);border-color:var(--jade);color:var(--jade)}}
.alert-warn{{background:rgba(239,159,39,.1);border-color:var(--amber);color:var(--amber)}}
.alert-info{{background:rgba(55,138,221,.08);border-color:var(--blue);color:var(--text)}}

.stock-pill{{display:inline-block;padding:4px 9px;background:var(--card2);
             border:1px solid var(--border);border-radius:14px;font-size:11px;
             margin:2px;cursor:pointer;color:var(--sub)}}
.stock-pill:hover{{color:var(--text);border-color:var(--blue)}}
.stock-pill.active{{background:var(--blue);color:white;border-color:var(--blue)}}

.fact-card{{background:var(--card2);border:1px solid var(--border);border-radius:8px;
            padding:12px;margin-bottom:8px}}
.fact-name{{font-weight:600;color:var(--text);font-size:13px}}
.fact-meta{{font-size:11px;color:var(--sub);margin-top:2px}}
.fact-desc{{font-size:12px;color:var(--text);margin-top:6px;line-height:1.55}}
</style>
</head><body>
<div class="wrap">

<!-- HEADER -->
<div style="display:flex;align-items:flex-start;justify-content:space-between;
            margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--border)">
  <div>
    <div style="font-size:11px;color:var(--sub);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px">
      NSE 200 · Quantitative Factor Investing · Phase 2
    </div>
    <h1 style="font-size:26px;font-weight:700;letter-spacing:-.5px">
      7-Factor Portfolio · Analytical Dashboard
    </h1>
    <div style="font-size:12px;color:var(--sub);margin-top:6px">
      {K['inception']} → {K['end']}  ·  Universe {K['n_universe']} stocks  ·
      Portfolio top {K['n_stocks']} (equal-weight)  ·  {K['n_rebal']} rebalances  ·
      Run {K['run_date']}
    </div>
  </div>
  <div style="text-align:right">
    <div class="badge badge-info" style="font-size:11px;padding:4px 10px">
      Capital {K['init_val']} → {K['final_val']}
    </div>
    <div style="font-size:10px;color:var(--sub);margin-top:6px">
      Cost {K['tcost']}/side · RFR {K['rf']}
    </div>
  </div>
</div>

<!-- KPI -->
<div class="section">Performance Summary · vs Nifty 50</div>
<div class="grid6" style="margin-bottom:20px">
  <div class="card-sm">
    <div class="kpi-label">CAGR</div>
    <div class="kpi-value mono {('pos' if '+' in K['cagr'] else 'neg')}">{K['cagr']}</div>
    <div class="kpi-sub">Nifty: {K['nifty_cagr']}</div>
  </div>
  <div class="card-sm">
    <div class="kpi-label">Alpha vs Nifty</div>
    <div class="kpi-value mono {('pos' if '+' in K['alpha'] else 'neg')}">{K['alpha']}</div>
    <div class="kpi-sub">annualised excess</div>
  </div>
  <div class="card-sm">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value mono neg">{K['maxdd']}</div>
    <div class="kpi-sub">Nifty: {K['nifty_dd']}</div>
  </div>
  <div class="card-sm">
    <div class="kpi-label">Sharpe</div>
    <div class="kpi-value mono">{K['sharpe']}</div>
    <div class="kpi-sub">Sortino: {K['sortino']}</div>
  </div>
  <div class="card-sm">
    <div class="kpi-label">Calmar</div>
    <div class="kpi-value mono pos">{K['calmar']}</div>
    <div class="kpi-sub">CAGR / |MaxDD|</div>
  </div>
  <div class="card-sm">
    <div class="kpi-label">Total Return</div>
    <div class="kpi-value mono pos" style="font-size:18px">{K['total_ret']}</div>
    <div class="kpi-sub">Win rate: {K['winrate']}  ·  Vol {K['vol']}</div>
  </div>
</div>

<!-- TABS -->
<div class="tabs">
  <div class="tab active" onclick="goTab(event,'t-overview')">📈 Overview</div>
  <div class="tab" onclick="goTab(event,'t-attr')">🔬 Attribution</div>
  <div class="tab" onclick="goTab(event,'t-diag')">🧪 Factor Diagnostics</div>
  <div class="tab" onclick="goTab(event,'t-comp')">🧬 Composition</div>
  <div class="tab" onclick="goTab(event,'t-rebal')">🔄 Rebalance</div>
  <div class="tab" onclick="goTab(event,'t-univ')">🌐 Universe</div>
  <div class="tab" onclick="goTab(event,'t-stocks')">📋 Stocks</div>
  <div class="tab" onclick="goTab(event,'t-model')">📚 Model</div>
</div>

<!-- ── TAB: OVERVIEW ─────────────────────────────────────────────────── -->
<div id="t-overview" class="tab-content active">
  <div class="card">
    <div class="section" style="margin-top:0">Equity Curve · Base = 100 · Monthly</div>
    <div class="legend-row">
      <span><span class="legend-dot" style="background:#378ADD"></span>7-Factor Portfolio</span>
      <span><span class="legend-dot" style="background:#888;opacity:.6"></span>Nifty 50</span>
    </div>
    <div class="chart-wrap" style="height:340px"><canvas id="cEquity"></canvas></div>
  </div>

  <div class="grid2" style="margin-top:16px">
    <div class="card">
      <div class="section" style="margin-top:0">Monthly Drawdown · % below peak</div>
      <div class="chart-wrap" style="height:240px"><canvas id="cDD"></canvas></div>
      <div style="font-size:11px;color:var(--sub);margin-top:8px">
        Real monthly drawdown, computed off the monthly equity curve.
      </div>
    </div>
    <div class="card">
      <div class="section" style="margin-top:0">Annual Returns · Real Jan→Dec</div>
      <div class="chart-wrap" style="height:240px"><canvas id="cAnnual"></canvas></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="section" style="margin-top:0">Rolling 12-Month Metrics — when did it work?</div>
    <div class="legend-row">
      <span><span class="legend-dot" style="background:#1D9E75"></span>Rolling Sharpe</span>
      <span><span class="legend-dot" style="background:#EF9F27"></span>Rolling Alpha (vs Nifty)</span>
      <span><span class="legend-dot" style="background:#D4537E"></span>Rolling Beta</span>
    </div>
    <div class="chart-wrap" style="height:280px"><canvas id="cRolling"></canvas></div>
    <div style="font-size:11px;color:var(--sub);margin-top:8px">
      Trailing 12-month windows. A flat-or-falling rolling Sharpe is your earliest
      warning that the model is regime-shifting.
    </div>
  </div>
</div>

<!-- ── TAB: ATTRIBUTION ───────────────────────────────────────────────── -->
<div id="t-attr" class="tab-content">
  <div class="alert alert-info">
    <strong>What this replaces:</strong> the old dashboard hardcoded factor attribution
    as fixed percentages. This page <em>computes</em> a single-factor backtest for each factor —
    same universe, same rebalances, weight=1.0 on that factor only. The terminal CAGR
    and drawdown of each line is the honest contribution.
  </div>
  <div class="card">
    <div class="section" style="margin-top:0">Single-Factor vs Multi-Factor · Equity Curves (Base=100)</div>
    <div class="chart-wrap" style="height:380px"><canvas id="cSF"></canvas></div>
    <div id="sfLegend" class="legend-row" style="margin-top:10px"></div>
  </div>

  <div class="grid2" style="margin-top:16px">
    <div class="card">
      <div class="section" style="margin-top:0">Per-Factor Performance</div>
      <table class="tbl"><thead><tr>
        <th>Factor</th><th>Weight</th><th>CAGR</th><th>Total Ret</th><th>Max DD</th>
      </tr></thead><tbody id="sfTable"></tbody></table>
    </div>
    <div class="card">
      <div class="section" style="margin-top:0">Factor Regime · Per-Year Quintile Spread (Q5−Q1, %)</div>
      <div id="regimeTable" style="overflow-x:auto"></div>
      <div style="font-size:11px;color:var(--sub);margin-top:8px">
        Positive = top quintile beat bottom quintile that year. Computed, not hardcoded.
      </div>
    </div>
  </div>
</div>

<!-- ── TAB: DIAGNOSTICS ───────────────────────────────────────────────── -->
<div id="t-diag" class="tab-content">
  <div class="alert alert-info">
    Two gold-standard factor tests: <strong>Information Coefficient</strong> (rank
    correlation between a factor's value and the next period's stock return) and
    <strong>Quintile Spread</strong> (top quintile mean return minus bottom). A factor
    that doesn't deliver consistently positive numbers here probably shouldn't be in your model.
  </div>

  <div class="grid2">
    <div class="card">
      <div class="section" style="margin-top:0">Information Coefficient · Mean per Factor</div>
      <div class="chart-wrap" style="height:260px"><canvas id="cIC"></canvas></div>
      <div style="font-size:11px;color:var(--sub);margin-top:8px">
        Spearman rank-correlation between factor value at t and forward return (t→t+1).
        Sign-flipped for lower-is-better factors. ≥0.05 is decent; ≥0.10 is strong.
      </div>
    </div>
    <div class="card">
      <div class="section" style="margin-top:0">Hit Rate · % of Periods with Positive IC</div>
      <div class="chart-wrap" style="height:260px"><canvas id="cHit"></canvas></div>
      <div style="font-size:11px;color:var(--sub);margin-top:8px">
        How consistently the factor was on the right side. 50% = coin flip.
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="section" style="margin-top:0">Quintile Spread · Q5 minus Q1 forward return (mean)</div>
    <div class="chart-wrap" style="height:240px"><canvas id="cQS"></canvas></div>
    <div style="font-size:11px;color:var(--sub);margin-top:8px">
      Equal-weighted Q5 forward return minus equal-weighted Q1 forward return,
      averaged across rebalances. Bigger spread = factor genuinely separates winners from losers.
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="section" style="margin-top:0">Cross-Factor Correlation (computed, Spearman, averaged)</div>
    <div id="corrTable" style="overflow-x:auto"></div>
    <div style="font-size:11px;color:var(--sub);margin-top:8px">
      Average cross-sectional rank correlation across all rebalance dates.
      Negative entries = genuine diversification.
    </div>
  </div>
</div>

<!-- ── TAB: COMPOSITION ──────────────────────────────────────────────── -->
<div id="t-comp" class="tab-content">
  <div class="card">
    <div class="section" style="margin-top:0">Current Top {K['n_stocks']} · Factor Heatmap</div>
    <div class="legend-row">
      <span><span class="pip r5">5</span>Top 20%</span>
      <span><span class="pip r4">4</span>Top 40%</span>
      <span><span class="pip r3">3</span>Mid</span>
      <span><span class="pip r2">2</span>Bot 40%</span>
      <span><span class="pip r1">1</span>Bot 20%</span>
    </div>
    <div style="overflow-x:auto">
      <table class="tbl"><thead><tr>
        <th>#</th><th>Stock</th><th>Sector</th><th>Score</th>
        <th>Mom</th><th>Qua</th><th>Val</th><th>Siz</th><th>Bet</th><th>Inv</th><th>Yld</th>
        <th>Bar</th>
      </tr></thead><tbody id="heatBody"></tbody></table>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="section" style="margin-top:0">Sector Exposure Over Time · % of portfolio</div>
    <div class="chart-wrap" style="height:340px"><canvas id="cSector"></canvas></div>
    <div style="font-size:11px;color:var(--sub);margin-top:8px">
      Where the model concentrated risk by sector at each rebalance.
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="section" style="margin-top:0">Holdings Transitions · Per Rebalance</div>
    <div class="tbl-scroll">
      <table class="tbl sticky-head"><thead><tr>
        <th>Date</th><th>Size</th><th>New (in)</th><th>Removed (out)</th>
      </tr></thead><tbody id="transBody"></tbody></table>
    </div>
  </div>
</div>

<!-- ── TAB: REBALANCE ─────────────────────────────────────────────────── -->
<div id="t-rebal" class="tab-content">
  <div class="card">
    <div class="section" style="margin-top:0">Rebalance Log · Click row for winners/losers</div>
    <div class="tbl-scroll">
      <table class="tbl sticky-head"><thead><tr>
        <th>Date</th><th>Value</th><th>Period Ret</th><th>Nifty Ret</th><th>Alpha</th>
        <th>Turn</th><th>In</th><th>Out</th><th>Cost</th>
      </tr></thead><tbody id="rebalBody"></tbody></table>
    </div>
  </div>

  <div class="grid2" style="margin-top:16px">
    <div class="card">
      <div class="section" style="margin-top:0">Cost & Turnover Trend</div>
      <div class="chart-wrap" style="height:240px"><canvas id="cCT"></canvas></div>
    </div>
    <div class="card" id="rebalDetail">
      <div class="section" style="margin-top:0">Period Detail · Click a row above</div>
      <div style="font-size:12px;color:var(--sub)">Click any row in the rebalance log to see the top winners and losers from that period.</div>
    </div>
  </div>
</div>

<!-- ── TAB: UNIVERSE ──────────────────────────────────────────────────── -->
<div id="t-univ" class="tab-content">
  <div class="card">
    <div class="section" style="margin-top:0">Universe · {K['n_universe']} stocks</div>
    <div class="search-row">
      <input class="search-input" type="text" id="univSearch" placeholder="Search ticker or sector…" oninput="filterUniv()">
      <select id="univSector" onchange="filterUniv()" class="search-input" style="max-width:180px"></select>
      <label style="font-size:11px;color:var(--sub);display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="univPortOnly" onchange="filterUniv()"> Portfolio only
      </label>
      <span id="univCount" style="font-size:11px;color:var(--sub);margin-left:auto"></span>
    </div>
    <div class="tbl-scroll">
      <table class="tbl sticky-head"><thead><tr>
        <th onclick="sortUniv('rank')"># ▼</th>
        <th onclick="sortUniv('ticker')">Ticker</th>
        <th>Sector</th>
        <th onclick="sortUniv('score')">Score</th>
        <th>Mom</th><th>Qua</th><th>Val</th><th>Siz</th><th>Bet</th><th>Inv</th><th>Yld</th>
      </tr></thead><tbody id="univBody"></tbody></table>
    </div>
  </div>
</div>

<!-- ── TAB: STOCKS (DRILL-DOWN) ───────────────────────────────────────── -->
<div id="t-stocks" class="tab-content">
  <div class="card">
    <div class="section" style="margin-top:0">Per-Stock Drill-Down</div>
    <div class="search-row">
      <input class="search-input" type="text" id="stockSearch" placeholder="Type to search a ticker (e.g. INFY, HDFCBANK)…" oninput="filterStockPills()">
      <span id="stockMeta" style="font-size:11px;color:var(--sub);margin-left:auto"></span>
    </div>
    <div id="stockPills" style="margin-bottom:12px;max-height:90px;overflow-y:auto"></div>
    <div id="stockBody" style="display:none">
      <div class="grid2">
        <div>
          <div class="section" style="margin-top:0">Price · since inception</div>
          <div class="chart-wrap" style="height:230px"><canvas id="cStockPx"></canvas></div>
        </div>
        <div>
          <div class="section" style="margin-top:0">Composite Score Over Time</div>
          <div class="chart-wrap" style="height:230px"><canvas id="cStockScore"></canvas></div>
        </div>
      </div>
      <div style="margin-top:14px">
        <div class="section">Factor Rank Trajectory</div>
        <div id="stockRankTable" style="overflow-x:auto"></div>
      </div>
      <div style="margin-top:14px">
        <div class="section">In-Portfolio Timeline</div>
        <div id="stockPortTimeline" style="overflow-x:auto"></div>
      </div>
    </div>
  </div>
</div>

<!-- ── TAB: MODEL ─────────────────────────────────────────────────────── -->
<div id="t-model" class="tab-content">
  <div class="grid2">
    <div class="card">
      <div class="section" style="margin-top:0">The 7 Factors</div>
      <div id="factorList"></div>
    </div>
    <div>
      <div class="card">
        <div class="section" style="margin-top:0">Factor Weights</div>
        <div class="chart-wrap" style="height:220px"><canvas id="cWeights"></canvas></div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="section" style="margin-top:0">Pipeline (Daily → Dashboard)</div>
        <div style="font-size:12px;line-height:2">
          <div><strong>1.</strong> screener_scraper.py → screener_raw.csv</div>
          <div><strong>2.</strong> build_pit.py → fundamentals_pit.csv (3-month lag)</div>
          <div><strong>3.</strong> data_layer · yfinance → prices.csv  + shares_outstanding.csv</div>
          <div><strong>4.</strong> factor_engine · momentum + 5 fundamentals + beta → 7 raw factors → quintile rank → composite</div>
          <div><strong>5.</strong> backtester · semi-annual rebalance + monthly equity walk</div>
          <div><strong>6.</strong> analytics · single-factor BTs, IC, attribution, sector tilt</div>
          <div><strong>7.</strong> dashboard · this page</div>
        </div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="section" style="margin-top:0">Methodology</div>
    <div style="font-size:12px;line-height:1.7">
      <p><strong>Momentum:</strong> 0.4·Ret(T-4 → T-2) + 0.6·Ret(T-12 → T-5). T-1 is skipped to avoid short-term reversal.</p>
      <p><strong>Beta:</strong> 36-month rolling OLS slope vs Nifty 50, clipped to [0.1, 4.0].</p>
      <p><strong>Quality:</strong> Operating Profit / Total Assets (latest available with 3-month institutional lag).</p>
      <p><strong>Value:</strong> Book-to-Market = Equity / Price (computed at rebalance).</p>
      <p><strong>Size:</strong> Market Cap = Price × shares-outstanding (yfinance current snapshot, split/bonus-corrected via auto_adjust prices). Falls back to Total Assets when shares unavailable.</p>
      <p><strong>Investment:</strong> YoY Total-Assets growth.</p>
      <p><strong>Yield:</strong> EPS × Dividend Payout % / Price. Missing payout/EPS → Rank 3 (neutral).</p>
      <p><strong>Ranking:</strong> Each raw value bucketed into quintiles (1–5). Lower-is-better factors ({", ".join(LOWER_IS_BETTER)}) inverted.</p>
      <p><strong>Composite:</strong> Final Score = Σ Rank · Weight. Mean is mathematically guaranteed ≈ 3.0.</p>
      <p><strong>Selection:</strong> Top {K['n_stocks']} by Final Score, equal-weight = 1/{K['n_stocks']} per name.</p>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="section" style="margin-top:0">Known Limitations</div>
    <div style="font-size:12px;line-height:1.7">
      <p><span class="badge badge-warn">Survivorship</span> The NSE 200 ticker list is a snapshot. Stocks delisted or dropped from NSE 200 since 2020 are absent — backtest is biased toward winners. Truly historical NSE 200 constituent lists would be needed to fix this.</p>
      <p><span class="badge badge-warn">Buyback drift</span> Market Cap uses current shares-outstanding. Splits/bonuses are corrected by yfinance auto-adjusted prices, but companies that did large buybacks/issuances since 2020 will have biased historical MCap.</p>
      <p><span class="badge badge-info">Beta-as-high-rank</span> The model gives Rank 5 to high-beta stocks. This is a deliberate aggressive-tilt choice from the PDF spec. A "low-vol" anomaly model would invert this.</p>
      <p><span class="badge badge-info">Sector mapping</span> Sector labels come from a static dictionary in <code>sectors.py</code>. Reclassifications by NSE or company restructurings are not tracked.</p>
    </div>
  </div>
</div>

</div><!-- /wrap -->

<script>
const D = {js};
const F = D.factors, F_COLORS = {json.dumps(F_COLORS)};
const RC = {{1:'#D85A30',2:'#EF9F27',3:'#4A5568',4:'rgba(29,158,117,.55)',5:'#1D9E75'}};
const grid='rgba(255,255,255,0.06)', txt='rgba(232,234,240,0.55)';
Chart.defaults.color = txt;
Chart.defaults.borderColor = grid;

// ─── Tab switching ────────────────────────────────────────────────────────
function goTab(e, id) {{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById(id).classList.add('active');
  if (id==='t-overview' && !window._init.overview) initOverview();
  if (id==='t-attr'     && !window._init.attr)     initAttribution();
  if (id==='t-diag'     && !window._init.diag)     initDiagnostics();
  if (id==='t-comp'     && !window._init.comp)     initComposition();
  if (id==='t-rebal'    && !window._init.rebal)    initRebalance();
  if (id==='t-univ'     && !window._init.univ)     initUniverse();
  if (id==='t-stocks'   && !window._init.stocks)   initStocks();
  if (id==='t-model'    && !window._init.model)    initModel();
}}
window._init = {{}};

// ─── Overview ─────────────────────────────────────────────────────────────
function initOverview() {{
  window._init.overview = true;

  new Chart(document.getElementById('cEquity'), {{
    type:'line', data:{{
      labels: D.equity.labels,
      datasets:[
        {{label:'Portfolio', data:D.equity.port, borderColor:'#378ADD',
          backgroundColor:'rgba(55,138,221,.1)', fill:true, tension:.25, pointRadius:0, borderWidth:2.5}},
        {{label:'Nifty 50', data:D.equity.nifty, borderColor:'rgba(136,135,128,.7)',
          borderDash:[5,4], fill:false, tension:.25, pointRadius:0, borderWidth:1.5}},
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{mode:'index',intersect:false}}}},
      scales:{{x:{{ticks:{{maxTicksLimit:14,font:{{size:10}}}},grid:{{display:false}}}},
              y:{{grid:{{color:grid}},ticks:{{font:{{size:10}}}}}}}}
    }}
  }});

  new Chart(document.getElementById('cDD'), {{
    type:'line', data:{{
      labels:D.drawdown.labels,
      datasets:[
        {{label:'Portfolio',data:D.drawdown.port,borderColor:'#D85A30',
          backgroundColor:'rgba(216,90,48,.18)',fill:true,tension:0,pointRadius:0,borderWidth:1.5}},
        {{label:'Nifty 50',data:D.drawdown.nifty,borderColor:'rgba(136,135,128,.6)',
          borderDash:[3,3],fill:false,tension:0,pointRadius:0,borderWidth:1}},
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.raw?.toFixed(1)}}%`}}}}}},
      scales:{{x:{{ticks:{{maxTicksLimit:12,font:{{size:9}}}},grid:{{display:false}}}},
              y:{{max:0,ticks:{{font:{{size:9}},callback:v=>v+'%'}},grid:{{color:grid}}}}}}}}
  }});

  new Chart(document.getElementById('cAnnual'), {{
    type:'bar', data:{{labels:D.annual.labels,datasets:[
      {{label:'Portfolio',data:D.annual.port,backgroundColor:'#378ADD',borderRadius:3}},
      {{label:'Nifty 50',data:D.annual.nifty,backgroundColor:'rgba(136,135,128,.4)',borderRadius:3}}
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{labels:{{font:{{size:11}},boxWidth:10}}}},
        tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.raw>0?'+':''}}${{c.raw}}%`}}}}}},
      scales:{{x:{{grid:{{display:false}}}},y:{{grid:{{color:grid}},ticks:{{callback:v=>v+'%',font:{{size:10}}}}}}}}}}
  }});

  new Chart(document.getElementById('cRolling'), {{
    type:'line', data:{{labels:D.rolling.labels,datasets:[
      {{label:'Sharpe',data:D.rolling.sharpe,borderColor:'#1D9E75',fill:false,tension:.25,pointRadius:0,borderWidth:1.8,yAxisID:'y'}},
      {{label:'Alpha', data:D.rolling.alpha,borderColor:'#EF9F27',fill:false,tension:.25,pointRadius:0,borderWidth:1.5,yAxisID:'y2'}},
      {{label:'Beta',  data:D.rolling.beta, borderColor:'#D4537E',fill:false,tension:.25,pointRadius:0,borderWidth:1.5,yAxisID:'y3'}},
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{ticks:{{maxTicksLimit:12,font:{{size:9}}}},grid:{{display:false}}}},
        y:{{position:'left',grid:{{color:grid}},ticks:{{font:{{size:9}}}}, title:{{display:true,text:'Sharpe',font:{{size:10}}}}}},
        y2:{{position:'right',grid:{{display:false}},ticks:{{font:{{size:9}},callback:v=>(v*100).toFixed(0)+'%'}}, title:{{display:true,text:'Alpha',font:{{size:10}}}}}},
        y3:{{position:'right',grid:{{display:false}},ticks:{{font:{{size:9}}}}, offset:true, title:{{display:true,text:'Beta',font:{{size:10}}}}}}
      }}}}
  }});
}}

// ─── Attribution ──────────────────────────────────────────────────────────
function initAttribution() {{
  window._init.attr = true;
  const colors = {{}};
  F.forEach((f,i) => colors[f] = F_COLORS[i]);
  colors['MultiFactor'] = '#FFFFFF';
  const datasets = [];
  Object.entries(D.sf.series).forEach(([f,vals]) => {{
    datasets.push({{
      label:f, data:vals,
      borderColor: f==='MultiFactor' ? '#FFFFFF' : colors[f],
      borderWidth: f==='MultiFactor' ? 2.5 : 1.4,
      borderDash:  f==='MultiFactor' ? [] : [],
      fill:false, tension:.2, pointRadius:0
    }});
  }});
  new Chart(document.getElementById('cSF'), {{
    type:'line', data:{{labels:D.sf.labels,datasets}},
    options:{{responsive:true,maintainAspectRatio:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.raw?.toFixed(1)}}`}}}}}},
      scales:{{x:{{ticks:{{maxTicksLimit:14,font:{{size:9}}}},grid:{{display:false}}}},
              y:{{grid:{{color:grid}},ticks:{{font:{{size:9}}}}}}}}}}
  }});

  // Custom legend
  const lg = document.getElementById('sfLegend');
  Object.keys(D.sf.series).forEach(f => {{
    const c = f==='MultiFactor'?'#FFFFFF':colors[f];
    lg.innerHTML += `<span><span class="legend-dot" style="background:${{c}}"></span>${{f}}</span>`;
  }});

  // Per-factor table
  const tb = document.getElementById('sfTable');
  D.sf.summary.sort((a,b) => b.cagr - a.cagr).forEach(r => {{
    const cls = (r.cagr>=0)?'pos':'neg';
    tb.innerHTML += `<tr>
      <td style="font-weight:600">${{r.factor}}</td>
      <td class="mono">${{(r.weight*100).toFixed(0)}}%</td>
      <td class="mono ${{cls}}">${{(r.cagr*100).toFixed(1)}}%</td>
      <td class="mono ${{cls}}">${{(r.total_return*100).toFixed(1)}}%</td>
      <td class="mono neg">${{(r.max_dd*100).toFixed(1)}}%</td>
    </tr>`;
  }});

  // Regime table
  const r = D.regime;
  if (!r.years.length) {{
    document.getElementById('regimeTable').innerHTML = '<div style="color:var(--sub);font-size:11px">No regime data (need ≥2 rebalances per year).</div>';
  }} else {{
    let h = '<table class="tbl" style="font-size:11px"><thead><tr><th>Year</th>';
    r.factors.forEach(f => h += `<th>${{f}}</th>`);
    h += '</tr></thead><tbody>';
    r.matrix.forEach((row,i) => {{
      h += `<tr><td class="mono" style="font-weight:600">${{r.years[i]}}</td>`;
      row.forEach(v => {{
        const bg = v===null ? 'transparent'
                 : v >  3 ? `rgba(29,158,117,${{Math.min(Math.abs(v)/30, .6)}})`
                 : v < -3 ? `rgba(216,90,48,${{Math.min(Math.abs(v)/30, .6)}})`
                 : 'transparent';
        const txt = v===null ? '—' : (v>0?'+':'')+v.toFixed(1);
        h += `<td class="mono" style="text-align:center;background:${{bg}};border-radius:3px">${{txt}}</td>`;
      }});
      h += '</tr>';
    }});
    h += '</tbody></table>';
    document.getElementById('regimeTable').innerHTML = h;
  }}
}}

// ─── Diagnostics ──────────────────────────────────────────────────────────
function initDiagnostics() {{
  window._init.diag = true;

  new Chart(document.getElementById('cIC'), {{
    type:'bar', data:{{labels:D.ic.factors,datasets:[
      {{label:'Mean IC',data:D.ic.mean_ic,
        backgroundColor: D.ic.mean_ic.map(v => v>=0?'#1D9E75':'#D85A30'),
        borderRadius:3}},
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`IC: ${{c.raw?.toFixed(3)}}`}}}}}},
      scales:{{x:{{grid:{{display:false}}}},y:{{grid:{{color:grid}},ticks:{{callback:v=>v.toFixed(2)}}}}}}}}
  }});

  new Chart(document.getElementById('cHit'), {{
    type:'bar', data:{{labels:D.ic.factors,datasets:[
      {{label:'Hit Rate',data:D.ic.hit_rate.map(v=>v*100),
        backgroundColor: D.ic.hit_rate.map(v => v>=0.5?'#1D9E75':'#D85A30'),
        borderRadius:3}},
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`${{c.raw.toFixed(0)}}% of periods positive`}}}}}},
      scales:{{x:{{grid:{{display:false}}}},
              y:{{min:0,max:100,grid:{{color:grid}},ticks:{{callback:v=>v+'%'}}}}}}}}
  }});

  new Chart(document.getElementById('cQS'), {{
    type:'bar', data:{{labels:D.qs.factors,datasets:[
      {{label:'Q5−Q1 spread',data:D.qs.mean_spread.map(v=>v===null?null:v*100),
        backgroundColor: D.qs.mean_spread.map(v => (v??0)>=0?'#1D9E75':'#D85A30'),
        borderRadius:3}}
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`${{c.raw?.toFixed(2)}}% per period`}}}}}},
      scales:{{x:{{grid:{{display:false}}}},
              y:{{grid:{{color:grid}},ticks:{{callback:v=>v.toFixed(1)+'%'}}}}}}}}
  }});

  // Correlation matrix
  const c = D.corr;
  let h = '<table class="tbl" style="font-size:11px"><thead><tr><th></th>';
  c.factors.forEach(f => h += `<th style="text-align:center">${{f.slice(0,3)}}</th>`);
  h += '</tr></thead><tbody>';
  c.matrix.forEach((row,i) => {{
    h += `<tr><td class="mono" style="font-weight:600">${{c.factors[i]}}</td>`;
    row.forEach((v,j) => {{
      let bg='transparent', col='var(--text)';
      if (v!==null && i!==j) {{
        if (v < -0.1) {{ bg = `rgba(216,90,48,${{Math.min(Math.abs(v),.7)}})`; col='white'; }}
        else if (v > 0.1) {{ bg = `rgba(55,138,221,${{Math.min(v*1.5,.7)}})`; col='white'; }}
      }}
      const txt = v===null ? 'NaN' : (i===j ? '—' : v.toFixed(2));
      h += `<td class="mono" style="text-align:center;background:${{bg}};color:${{col}};border-radius:3px">${{txt}}</td>`;
    }});
    h += '</tr>';
  }});
  h += '</tbody></table>';
  document.getElementById('corrTable').innerHTML = h;
}}

// ─── Composition ──────────────────────────────────────────────────────────
function initComposition() {{
  window._init.comp = true;

  // Heatmap
  const tb = document.getElementById('heatBody');
  D.heatmap.forEach((s,i) => {{
    const bar = ((s.score - 1) / 4 * 100).toFixed(1);
    let row = `<tr>
      <td style="color:var(--sub)">${{i+1}}</td>
      <td style="font-weight:600">${{s.ticker}}</td>
      <td><span class="badge badge-sec">${{s.sector}}</span></td>
      <td class="mono" style="font-weight:700">${{s.score.toFixed(3)}}</td>`;
    s.ranks.forEach(r => row += `<td><span class="pip r${{r}}">${{r}}</span></td>`);
    row += `<td><div class="prog-bar" style="width:90px"><div class="prog-fill" style="width:${{bar}}%;background:#378ADD"></div></div></td></tr>`;
    tb.innerHTML += row;
  }});

  // Sector area
  const se = D.sector_exp;
  const datasets = se.sectors.map((s,i) => ({{
    label:s, data:se.matrix[i],
    backgroundColor: F_COLORS[i % F_COLORS.length],
    borderColor: 'transparent',
    fill:true, tension:0, pointRadius:0,
  }}));
  new Chart(document.getElementById('cSector'), {{
    type:'line', data:{{labels:se.labels,datasets}},
    options:{{responsive:true,maintainAspectRatio:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{labels:{{font:{{size:9}},boxWidth:8,padding:4}},position:'right'}},
        tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.raw?.toFixed(1)}}%`}}}}}},
      scales:{{x:{{ticks:{{font:{{size:9}}}},grid:{{display:false}}}},
              y:{{stacked:true,max:100,grid:{{color:grid}},ticks:{{callback:v=>v+'%',font:{{size:9}}}}}}}}}}
  }});

  // Transitions
  const tr = document.getElementById('transBody');
  D.transitions.forEach(t => {{
    const inHtml  = t.in.length  ? t.in.slice(0, 12).map(x=>`<span class="badge badge-in" style="margin:1px">${{x}}</span>`).join('')   + (t.in.length>12?` <span style="color:var(--sub)">+${{t.in.length-12}}</span>`:'') : '<span style="color:var(--sub)">—</span>';
    const outHtml = t.out.length ? t.out.slice(0, 12).map(x=>`<span class="badge badge-warn" style="margin:1px">${{x}}</span>`).join('') + (t.out.length>12?` <span style="color:var(--sub)">+${{t.out.length-12}}</span>`:'') : '<span style="color:var(--sub)">—</span>';
    tr.innerHTML += `<tr>
      <td class="mono" style="color:var(--sub)">${{t.date}}</td>
      <td>${{t.size}}</td>
      <td>${{inHtml}}</td>
      <td>${{outHtml}}</td>
    </tr>`;
  }});
}}

// ─── Rebalance ────────────────────────────────────────────────────────────
function initRebalance() {{
  window._init.rebal = true;
  const tb = document.getElementById('rebalBody');
  D.rebal.forEach((r,i) => {{
    const a = (r.period_r - r.nifty_r);
    tb.innerHTML += `<tr onclick="showRebalDetail(${{i}})" style="cursor:pointer">
      <td class="mono" style="color:var(--sub)">${{r.date}}</td>
      <td class="mono">₹${{r.value.toLocaleString('en-IN')}}</td>
      <td class="mono ${{r.period_r>=0?'pos':'neg'}}">${{r.period_r>=0?'+':''}}${{r.period_r}}%</td>
      <td class="mono ${{r.nifty_r>=0?'pos':'neg'}}">${{r.nifty_r>=0?'+':''}}${{r.nifty_r}}%</td>
      <td class="mono ${{a>=0?'pos':'neg'}}">${{a>=0?'+':''}}${{a.toFixed(1)}}%</td>
      <td class="mono">${{r.turn}}%</td>
      <td class="pos">+${{r.n_in}}</td>
      <td class="neg">-${{r.n_out}}</td>
      <td class="mono" style="color:var(--sub)">₹${{r.cost.toLocaleString('en-IN')}}</td>
    </tr>`;
  }});

  new Chart(document.getElementById('cCT'), {{
    type:'line', data:{{labels:D.ct.labels,datasets:[
      {{label:'Turnover (%)',data:D.ct.turnover,borderColor:'#EF9F27',
        fill:false,tension:.25,pointRadius:3,borderWidth:1.8,yAxisID:'y'}},
      {{label:'Cost (₹)',data:D.ct.cost,borderColor:'#D85A30',
        fill:false,tension:.25,pointRadius:3,borderWidth:1.8,yAxisID:'y2'}},
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{labels:{{font:{{size:11}},boxWidth:10}}}}}},
      scales:{{x:{{grid:{{display:false}},ticks:{{font:{{size:9}}}}}},
              y:{{position:'left',grid:{{color:grid}},ticks:{{callback:v=>v+'%',font:{{size:9}}}}}},
              y2:{{position:'right',grid:{{display:false}},ticks:{{callback:v=>'₹'+(v/1000).toFixed(0)+'k',font:{{size:9}}}}}}}}}}
  }});
}}

function showRebalDetail(i) {{
  const r = D.rebal[i];
  const winners = r.winners.map(([t,v]) => `<tr><td>${{t}}</td><td class="mono pos">+${{v}}%</td></tr>`).join('');
  const losers  = r.losers.map(([t,v])  => `<tr><td>${{t}}</td><td class="mono neg">${{v}}%</td></tr>`).join('');
  document.getElementById('rebalDetail').innerHTML = `
    <div class="section" style="margin-top:0">${{r.date}} · period return ${{r.period_r}}%</div>
    <div style="font-size:12px;color:var(--sub);margin-bottom:10px">
      Universe ${{r.univ}}  ·  Turnover ${{r.turn}}%  ·  Cost ₹${{r.cost.toLocaleString('en-IN')}}
    </div>
    <div class="grid2">
      <div>
        <div class="section">Top Winners</div>
        <table class="tbl">${{winners||'<tr><td style="color:var(--sub)">—</td></tr>'}}</table>
      </div>
      <div>
        <div class="section">Top Losers</div>
        <table class="tbl">${{losers||'<tr><td style="color:var(--sub)">—</td></tr>'}}</table>
      </div>
    </div>`;
}}

// ─── Universe ─────────────────────────────────────────────────────────────
let univSortKey='rank', univSortAsc=true;
function initUniverse() {{
  window._init.univ = true;
  const sel = document.getElementById('univSector');
  const sectors = [...new Set(D.universe.map(r=>r.sector))].sort();
  sel.innerHTML = '<option value="">All sectors</option>' + sectors.map(s=>`<option>${{s}}</option>`).join('');
  filterUniv();
}}

function filterUniv() {{
  const q  = document.getElementById('univSearch').value.toLowerCase();
  const sec= document.getElementById('univSector').value;
  const po = document.getElementById('univPortOnly').checked;
  let rows = D.universe.filter(r => {{
    if (po && !r.in_port) return false;
    if (sec && r.sector !== sec) return false;
    if (q && !(r.ticker.toLowerCase().includes(q) || r.sector.toLowerCase().includes(q))) return false;
    return true;
  }});
  rows.sort((a,b) => {{
    const av=a[univSortKey], bv=b[univSortKey];
    if (typeof av==='string') return univSortAsc?av.localeCompare(bv):bv.localeCompare(av);
    return univSortAsc ? av-bv : bv-av;
  }});
  const tb = document.getElementById('univBody');
  tb.innerHTML = rows.map(r => `<tr class="${{r.in_port?'in-port':''}}">
    <td style="color:var(--sub)">${{r.rank}}</td>
    <td style="font-weight:600;cursor:pointer" onclick="goToStock('${{r.full}}')">
      ${{r.ticker}} ${{r.in_port?'<span class=\\'badge badge-in\\'>IN</span>':''}}
    </td>
    <td><span class="badge badge-sec">${{r.sector}}</span></td>
    <td class="mono" style="font-weight:700">${{r.score.toFixed(3)}}</td>
    ${{r.ranks.map(x=>`<td><span class="pip r${{x}}">${{x}}</span></td>`).join('')}}
  </tr>`).join('');
  document.getElementById('univCount').textContent = `${{rows.length}} of ${{D.universe.length}}`;
}}
function sortUniv(k) {{
  univSortAsc = (univSortKey===k) ? !univSortAsc : (k==='rank');
  univSortKey = k;
  filterUniv();
}}

// ─── Stocks (drill-down) ──────────────────────────────────────────────────
let _stockChart1, _stockChart2;
function initStocks() {{
  window._init.stocks = true;
  renderStockPills();
  // Pre-select first portfolio stock if any
  const first = Object.keys(D.stocks).find(t => D.stocks[t].in_port?.some(x=>x)) || Object.keys(D.stocks)[0];
  if (first) showStock(first);
}}

function renderStockPills() {{
  const q = (document.getElementById('stockSearch')?.value || '').toLowerCase();
  const tickers = Object.keys(D.stocks).filter(t =>
    !q || D.stocks[t].name.toLowerCase().includes(q) || (D.stocks[t].sector||'').toLowerCase().includes(q)
  ).sort();
  document.getElementById('stockPills').innerHTML =
    tickers.map(t => `<span class="stock-pill" data-t="${{t}}" onclick="showStock('${{t}}')">${{D.stocks[t].name}}</span>`).join('');
  document.getElementById('stockMeta').textContent = `${{tickers.length}} of ${{Object.keys(D.stocks).length}}`;
}}
function filterStockPills() {{ renderStockPills(); }}

function showStock(tk) {{
  const s = D.stocks[tk];
  if (!s) return;
  document.getElementById('stockBody').style.display='block';
  document.querySelectorAll('.stock-pill').forEach(p =>
    p.classList.toggle('active', p.dataset.t===tk));

  if (_stockChart1) _stockChart1.destroy();
  if (_stockChart2) _stockChart2.destroy();

  _stockChart1 = new Chart(document.getElementById('cStockPx'), {{
    type:'line', data:{{labels:s.price.labels,datasets:[
      {{label:'Price',data:s.price.values,borderColor:'#378ADD',
        backgroundColor:'rgba(55,138,221,.1)',fill:true,tension:.3,pointRadius:0,borderWidth:1.8}}
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`₹${{c.raw?.toFixed(2)}}`}}}}}},
      scales:{{x:{{ticks:{{maxTicksLimit:12,font:{{size:9}}}},grid:{{display:false}}}},
              y:{{grid:{{color:grid}},ticks:{{font:{{size:9}}}}}}}}}}
  }});

  _stockChart2 = new Chart(document.getElementById('cStockScore'), {{
    type:'line', data:{{labels:s.score_hist.labels,datasets:[
      {{label:'Final Score',data:s.score_hist.values,borderColor:'#1D9E75',
        backgroundColor:'rgba(29,158,117,.15)',fill:true,tension:.3,pointRadius:3,borderWidth:1.8}}
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:c=>`Score: ${{c.raw?.toFixed(3)}}`}}}}}},
      scales:{{x:{{ticks:{{maxTicksLimit:8,font:{{size:9}}}},grid:{{display:false}}}},
              y:{{min:1,max:5,grid:{{color:grid}},ticks:{{font:{{size:9}}}}}}}}}}
  }});

  // Rank trajectory table
  const rh = s.ranks_hist;
  let rt = '<table class="tbl" style="font-size:11px"><thead><tr><th>Date</th>';
  F.forEach(f => rt += `<th>${{f.slice(0,3)}}</th>`);
  rt += '</tr></thead><tbody>';
  rh.labels.forEach((d,i) => {{
    rt += `<tr><td class="mono" style="color:var(--sub)">${{d}}</td>`;
    F.forEach(f => {{
      const v = (rh.ranks[f]||[])[i];
      rt += `<td>${{v?`<span class="pip r${{v}}">${{v}}</span>`:'<span style="color:var(--sub)">—</span>'}}</td>`;
    }});
    rt += '</tr>';
  }});
  rt += '</tbody></table>';
  document.getElementById('stockRankTable').innerHTML =
    `<div style="font-size:12px;color:var(--sub);margin-bottom:6px">${{s.name}} · ${{s.sector}}</div>` + rt;

  // In-portfolio timeline
  let pt = '<div style="display:flex;gap:4px;flex-wrap:wrap">';
  s.score_hist.labels.forEach((d,i) => {{
    const inP = s.in_port[i];
    const c = inP ? '#1D9E75' : '#3a3f50';
    pt += `<div style="background:${{c}};color:white;font-size:10px;padding:3px 7px;border-radius:3px" title="${{d}}">${{d.slice(0,7)}}</div>`;
  }});
  pt += '</div>';
  document.getElementById('stockPortTimeline').innerHTML = pt;
}}

function goToStock(full) {{
  // Switch to Stocks tab and focus the chosen ticker
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelector('.tab:nth-child(7)').classList.add('active');
  document.getElementById('t-stocks').classList.add('active');
  if (!window._init.stocks) initStocks();
  showStock(full);
}}

// ─── Model tab ────────────────────────────────────────────────────────────
function initModel() {{
  window._init.model = true;
  const FACT_DESC = {{
    Momentum: ['Price persistence', 'Compounded return T-12 → T-2 with T-1 skipped. Captures the well-documented 6–12 month price-trend anomaly.'],
    Quality:  ['Operating profit per ₹ of assets', 'Operating Profit ÷ Total Assets. High-quality firms generate more profit from each rupee invested in their balance sheet.'],
    Value:    ['Cheapness', 'Book Equity ÷ Market Price. The Fama-French value factor — buy what is cheap relative to fundamentals.'],
    Size:     ['Market capitalisation', 'Smaller companies historically outperform on a risk-adjusted basis (size effect). LOWER values = Rank 5.'],
    Beta:     ['Market sensitivity', 'OLS slope of stock returns vs Nifty 50 returns over a rolling 36-month window. Higher = more aggressive.'],
    Invest:   ['Conservative investment', 'YoY total-assets growth. The investment factor: firms that *don\\'t* over-invest tend to outperform. LOWER values = Rank 5.'],
    Yield:    ['Cash returned to shareholders', 'EPS × payout%, ÷ price. A "cash-flow discipline" filter on the high-momentum tilt.'],
  }};
  let h='';
  F.forEach((f,i) => {{
    const [tag, desc] = FACT_DESC[f];
    const w = (D.weights[f]*100).toFixed(0);
    const lower = D.lower_better.includes(f);
    h += `<div class="fact-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <span class="fact-name">${{f}}</span>
          <span class="badge badge-info" style="margin-left:8px">${{w}}%</span>
          ${{lower?'<span class="badge badge-warn" style="margin-left:4px">LOWER better</span>':''}}
        </div>
        <span style="color:${{F_COLORS[i]}};font-size:18px;font-weight:700;font-family:JetBrains Mono,monospace">${{w}}%</span>
      </div>
      <div class="fact-meta">${{tag}}</div>
      <div class="fact-desc">${{desc}}</div>
    </div>`;
  }});
  document.getElementById('factorList').innerHTML = h;

  new Chart(document.getElementById('cWeights'), {{
    type:'doughnut', data:{{labels:F,datasets:[
      {{data:F.map(f=>D.weights[f]*100),backgroundColor:F_COLORS,borderWidth:0}}
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,cutout:'55%',
      plugins:{{legend:{{position:'right',labels:{{font:{{size:11}},boxWidth:10}}}},
        tooltip:{{callbacks:{{label:c=>`${{c.label}}: ${{c.raw}}%`}}}}}}}}
  }});
}}

// Init the default tab
initOverview();
</script>
</body></html>
"""
