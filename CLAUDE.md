# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A quantitative + agentic engine for running the **Wheel options strategy** on a small ($1,000) account. It produces a ranked, gated trade sheet — it **never places orders**. Three honest constraints drive the design: one position at a time, real option-chain data (with an honest theoretical fallback), and a backtest that proves the rules add edge.

## Commands

```bash
# Install (editable). Pick extras: [all], or compose [ui]/[agents]/[tui]/[dev]/[research] + a provider.
pip install -e ".[all]"

# Quant-only weekly scan — no LLM, no API key, no UI:
quantwheel-scan --tickers F,SOFI,NIO --account 1000

# Streamlit dashboard (default UI, 7 tabs + tracker):
streamlit run src/quantwheel/ui/streamlit_app/app.py

# Textual TUI (same backend — proves the UI is swappable):
python -m quantwheel.ui.tui.app

# Tests (pyproject sets pythonpath=src, testpaths=tests):
pytest
pytest tests/test_pricing.py                 # one file
pytest tests/test_pricing.py::test_name      # one test
pytest -k council                            # by keyword
```

## The one architectural rule (do not break it)

```
ui/  →  services/  →  core/ + agents/        (strictly one-way)
```

- The **UI only ever** imports from `services/` and reads the dataclasses in `core/types.py`. It never imports `core/` models or `agents/` directly.
- The **backend** (`core/`, `services/`, `agents/`) **never** imports a UI library. `tests/test_architecture.py` enforces this by scanning for `import streamlit/textual/plotly/...` and failing the build if any leak in. Both UIs (Streamlit, Textual) are thin adapters over the *same* `services/` facade.
- When adding backend behavior, expose it through a function in `services/` that returns/consumes `core/types.py` dataclasses — keep the math and the UI on opposite sides of that boundary.

## How a scan flows

`services/scan_service.run_weekly_scan` → `core/scoring.score_universe` → per ticker `score_candidate`:

```
bars → indicators → regime → chain quote → Monte Carlo → pricing
     → earnings/liquidity/capital gates → composite score → verdict
```

Key files: `core/scoring.py` is the orchestration heart that combines every model layer into one `Verdict`. Models live in `core/models/` (`indicators`, `regime` Markov, `montecarlo` GBM, `pricing` Black-Scholes). Data fetchers in `core/data/` (`bars`, `chains`, `earnings`) — all pull free Yahoo data via yfinance.

**Verdicts are deliberately honest:** hard gates (capital, earnings, regime, liquidity) and non-positive expected value override a pretty technical score. A name can look great and still be `SKIP_CAPITAL` because it doesn't fit the account. A high Monte-Carlo assignment probability caps the verdict at `WATCH`.

**Account-aware, multi-position output.** The engine is no longer hardwired to $1,000/one-position. `core/portfolio.py:allocate(scan, config, verdicts)` greedily sizes up to `config.max_positions` enterable names within deployable cash (for `max_positions == 1` it's identical to single-position sizing). `services/report_service.py:build_report(...)` turns a scan (+ optional council) into a `PortfolioReport` of specific trades, each paired with plain-English narrative from `core/narrative.py` (beginner-friendly, defines jargon inline). The Streamlit **Report tab** (`ui/.../tabs/report.py`) is the readable, headline surface; the **Approve tab** remains the single-best human-in-the-loop card. The backtest and LangGraph weekly agent stay single-position by design.

## Configuration

**Every tunable number lives in `core/config.py`** as a frozen `WheelConfig` dataclass (account size, DTE window, target delta, gate thresholds, MC paths, regime/scoring params). `DEFAULT` is the module-level instance everyone imports. To change behavior, construct a `WheelConfig` (or `dataclasses.replace(DEFAULT, ...)`) and thread it through the services layer — do not scatter magic numbers back into the models.

## The agentic layer (LLM-optional)

The quant core, backtest, scan, and tracker all work **with no LLM and no API key**. Only the council and weekly agent call an LLM.

- **Provider-agnostic LLM factory** in `agents/llm.py`. Pick the provider with env vars (`.env`): `QUANTWHEEL_PROVIDER` = `anthropic` | `ollama` | `openai` | `google`, plus `QUANTWHEEL_MODEL`, `QUANTWHEEL_TEMPERATURE`. Adding a provider = one branch in `make_chat_model`. The council uses **structured output**, so a local model must support tool-calling / JSON-schema (llama3.1/3.2, qwen2.5 work; tiny models don't).
- **Council** (`agents/council.py`, `agents/personas/`): five anonymized **archetype** personas (the Statistician, Value Investor, Risk Manager, Macro Strategist, Systematic Quant — no real-person names; `tests/test_personas.py` guards this) vote on the same evidence; aggregation is **deterministic** (conviction-weighted mean, no extra LLM call) so it's reproducible and testable. The per-persona rubric is account-aware (`personas/base.py:_rubric(account, max_positions)`), threaded from `WheelConfig`.
- **Weekly agent** (`agents/graph.py`): a LangGraph `StateGraph` with branching (sits out a week when nothing qualifies) and a **human-in-the-loop `interrupt()`** at the approval gate. Code never auto-trades — the graph resumes only after a human yes/no. Tests/cron pass `auto_approve` to answer the gate non-interactively. State lives in `agents/nodes.py:WheelState`.

## Persistence

The trade tracker (`services/tracker_service.py`) is a single local JSON file at `data/trades.json` (override with `QUANTWHEEL_TRACKER`). The LangGraph checkpointer is in-memory (`MemorySaver`) — a paused agent resumes within the same process by `thread_id`.

## Known limitations (keep in mind when changing model code)

- Free Yahoo option data is noisy: synthetic IV / zero bid-ask after hours. `chains.py` detects this and falls back to a Black-Scholes theoretical quote flagged `is_estimated` — gates treat estimated quotes and zero open-interest as "unknown," not failures.
- The backtest uses **realized** vol (no free historical implied vol) and synthesizes quarterly earnings dates when the feed lacks them. The *relative* rule ablation is robust; absolute premium income is understated.

## Learning project

This repo doubles as a LangChain/LangGraph course — see `LEARNING.md` and `notebooks/` (`01_learn_langchain` = the council, `02_learn_langgraph` = the weekly agent, `research.ipynb` = backtest plots).
