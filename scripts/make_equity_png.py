"""Generate the README hero chart: equity curve + rule-ablation Sharpe bars.

Run from the repo root::

    python scripts/make_equity_png.py

Writes ``docs/equity_curve.png``. Needs the ``research`` extra (matplotlib) and
internet (the backtest fetches data).
"""

from __future__ import annotations

import os
from datetime import date

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

from quantwheel.services.backtest_service import run_backtest

TICKERS = ["F", "SOFI", "NIO", "AAL"]
OUT = os.path.join("docs", "equity_curve.png")


def main() -> None:
    res = run_backtest(TICKERS, date(2024, 1, 1), date(2026, 6, 1))
    eq = pd.Series({pd.Timestamp(d): v for d, v in res.equity_curve}).sort_index()

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    # Equity curve
    ax1.plot(eq.index, eq.values, color="#1f77b4", lw=2)
    ax1.axhline(1.0, color="grey", ls="--", lw=0.8)
    m = res.metrics
    ax1.set_title(
        f"Equal-weight wheel — Sharpe {m['sharpe']:.2f} · "
        f"CAGR {m['cagr']:.1%} · max DD {m['max_drawdown']:.1%}",
        fontsize=11, fontweight="bold",
    )
    ax1.set_ylabel("equity (start = 1.0)")
    ax1.tick_params(labelsize=8)

    # Ablation: Sharpe of baseline vs each rule removed
    rows = {"all rules\non": res.metrics}
    label_map = {"no_earnings_gate": "no\nearnings", "no_regime_gate": "no\nregime",
                 "no_profit_take": "no profit\n-take"}
    for k, v in res.ablation.items():
        rows[label_map.get(k, k)] = v
    labels = list(rows.keys())
    sharpes = [rows[k]["sharpe"] for k in labels]
    colors = ["#2ca02c"] + ["#d62728"] * (len(labels) - 1)
    ax2.bar(labels, sharpes, color=colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Rule ablation (Sharpe)", fontsize=11, fontweight="bold")
    ax2.tick_params(labelsize=8)

    fig.suptitle("quant-wheel — backtest 2024→2026 (F · SOFI · NIO · AAL)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "Premiums modelled on realized vol (no historical IV). Educational, not advice.",
             ha="center", fontsize=7, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    os.makedirs("docs", exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}  ({res.n_trades} trades)")


if __name__ == "__main__":
    main()
