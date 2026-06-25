"""Command-line entry point: a quant-only weekly scan (no LLM, no UI).

    quantwheel-scan --tickers F,SOFI,NIO --account 1000

Prints a ranked table to the terminal. This is the fastest way to sanity-check
the engine and is handy for cron/CI. The council and dashboard live elsewhere.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from quantwheel.core import portfolio
from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import ScanResult, Verdict
from quantwheel.services.scan_service import run_weekly_scan
from quantwheel.universe import DEFAULT_UNIVERSE, load_shortlist

# ANSI-free verdict tags so output is clean in any terminal / log file.
_TAG = {
    Verdict.ENTER_STRONG: "ENTER+", Verdict.ENTER: "ENTER ", Verdict.WATCH: "WATCH ",
    Verdict.SKIP: "skip  ", Verdict.SKIP_REGIME: "skip-R", Verdict.SKIP_EARNINGS: "skip-E",
    Verdict.SKIP_LIQUIDITY: "skip-L", Verdict.SKIP_CAPITAL: "skip-$",
}


def _print_portfolio(result: ScanResult, config: WheelConfig) -> None:
    """Print the sized, multi-position allocation (one line per fitted name)."""
    positions = portfolio.allocate(result, config)
    if not positions:
        return
    deployed = sum(p.collateral for p in positions)
    credit = sum(p.credit for p in positions)
    print(f"\nSized portfolio (up to {config.max_positions} position(s), "
          f"${config.deployable_cash:,.0f} deployable):")
    for p in positions:
        q = p.candidate.quote
        print(f"  {p.candidate.ticker:5} {p.contracts}x ${q.strike:.2f}p "
              f"exp {q.expiry}  collateral ${p.collateral:,.0f}  credit ${p.credit:,.0f}")
    print(f"  -> ${deployed:,.0f} collateral, ${credit:,.0f} premium, "
          f"${config.account - deployed:,.0f} cash reserved.")


def _print_report(result: ScanResult) -> None:
    print(f"\nWheel scan — {result.as_of}  |  account ${result.account:,.0f}")
    print(f"universe: {', '.join(result.universe)}\n")
    header = (f"{'TICK':5} {'VERDICT':7} {'SCORE':>5} {'STK':>6} {'DTE':>3} "
             f"{'CR$':>5} {'POP':>5} {'EV$':>6} {'pAsn':>5} {'DLT':>6} {'FIT':>3} {'EST':>3}")
    print(header)
    print("-" * len(header))
    for c in result.candidates:
        p, q = c.pricing, c.quote
        print(f"{c.ticker:5} {_TAG.get(c.verdict, '?'):7} {c.score:5.2f} "
              f"{q.strike:6.2f} {q.dte:3d} {p.credit:5.0f} {p.pop:5.0%} "
              f"{p.expected_value:6.1f} {c.monte_carlo.p_assign:5.0%} "
              f"{(q.delta or 0):6.3f} {c.sizing.contracts:3d} "
              f"{'Y' if q.is_estimated else 'n':>3}")

    enter = result.enterable
    print(f"\n{len(enter)} enterable / {len(result.candidates)} scored.")
    if enter:
        top = enter[0]
        print(f"Top pick: {top.ticker} — sell {top.sizing.contracts}x "
              f"${top.quote.strike:.2f}p exp {top.quote.expiry} for "
              f"~${top.pricing.credit:.0f} (EV ${top.pricing.expected_value:.1f}).")
    print("\nLegend: skip-R regime · skip-E earnings · skip-L liquidity · skip-$ capital")
    print("EST=Y means the quote is a theoretical estimate — verify the live chain.\n")


def scan_cli() -> None:
    """Parse args and print a scan report."""
    # Windows consoles default to cp1252; force UTF-8 so any glyph is safe.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Quant-only weekly wheel scan.")
    ap.add_argument("--tickers", help="Comma-separated tickers (default: shortlist).")
    ap.add_argument("--csv", help="Path to a shortlist CSV with a 'ticker' column.")
    ap.add_argument("--account", type=float, default=DEFAULT.account)
    ap.add_argument("--buffer", type=float, default=DEFAULT.cash_buffer)
    ap.add_argument("--max-positions", type=int, default=DEFAULT.max_positions,
                    help="Concurrent positions to allow (sizing spreads across them).")
    args = ap.parse_args()

    if args.tickers:
        universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.csv:
        universe = load_shortlist(args.csv)
    else:
        universe = DEFAULT_UNIVERSE

    config = replace(DEFAULT, account=args.account, cash_buffer=args.buffer,
                     max_positions=args.max_positions)
    print("Scanning (live market data)...")
    result = run_weekly_scan(universe, config)
    _print_report(result)
    _print_portfolio(result, config)


if __name__ == "__main__":
    scan_cli()
