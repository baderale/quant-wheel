# quant-wheel

**A quantitative + agentic engine for running the Wheel options strategy on a small account.**

Feed it a watchlist → get a ranked, *gated* trade sheet (real implied vol, assignment risk,
expected value, a position size that actually fits a $1,000 account), a five-persona **investor
council** that debates each candidate, and a **LangGraph weekly agent** that assembles a
human-approved trade card and logs it to a built-in tracker. A walk-forward **backtest** measures
whether the rules actually improve the odds.

> ⚠️ **Educational, not financial advice.** The Wheel has real downside; on a one-position account,
> assignment in a falling stock can erase weeks of premium. **This tool never places orders** — it
> produces suggestions you approve and execute manually.

---

## Why this exists

The Wheel (sell cash-secured puts → if assigned, sell covered calls → repeat) is simple to state
and tedious to run well: read the chart, estimate assignment risk, price the premium, check
earnings, size it to your account — across a dozen tickers, every week. This automates the analysis
while keeping a human at the trigger, and it's built around three honest constraints of a **$1,000
Robinhood account**:

1. **One position at a time** — collateral ties up the whole account, so *name selection and
   earnings-avoidance are the risk management.*
2. **Real option data** — prices against the live chain's implied vol and bid/ask (with an honest
   theoretical fallback when the feed is closed), so the numbers track what you'll actually get.
3. **Proof, not vibes** — a backtest with calibration and rule-ablation says whether the edge is
   real.

## How it works (three quant models + a council)

| Layer | Question | Method |
|---|---|---|
| **Technical** | Is the setup right *now*? | Bollinger %, SMA50 trend, RSI, gaps |
| **Markov regime** | Will the next ~week stay favourable? | 4-state transition matrix, projected `Pⁿ` |
| **Monte Carlo** | Assignment risk at this strike? | GBM + bootstrap paths over the DTE window |
| **Black-Scholes** | What's it worth, and the Greeks? | Priced with the chain's **real IV** |
| **Council** | Does it make qualitative sense? | 5 LLM personas vote, an aggregator ranks |

The council: **the Statistician** (trusts the math), **Warren Buffett** (would I be happy owning the
shares?), **Michael Burry** (tail-risk skeptic), **Ray Dalio** (macro/regime), and **Jim Simons**
(pure systematic — distrusts narrative).

## What the backtest shows (rule ablation)

Each rule is turned **off** to measure its contribution. On a 2024→2026 run over F/SOFI/NIO, every
rule improved risk-adjusted return (baseline Sharpe beat all ablated variants):

| Variant | Sharpe | Max DD |
|---|---|---|
| **baseline (all rules on)** | **~0.79** | **~−6%** |
| no profit-take | −0.36 | −34% |
| no regime gate | 0.67 | −29% |
| no earnings gate | 0.59 | −9% |

Takeaway: **profit-taking is essential** (Sharpe flips negative without it), the **regime gate**
quarters the drawdown, and the **earnings gate** adds risk-adjusted return. The model is *conservative*
on assignment (it over-predicts, erring toward caution). *(Premiums are modelled on realized vol —
see Limitations.)*

## Architecture — the UI is swappable by design

```
ui/  →  services/  →  core/ + agents/        (one-way dependency)
```

The UI only ever calls the **services** facade and reads the dataclasses in `core/types.py`; the math
and agents never import a UI library. So the **Streamlit** dashboard and the **Textual TUI** are two
thin adapters over the *same* backend — proven by `tests/test_architecture.py` (no UI import leaks)
and `tests/test_tui.py` (a second UI on the same services).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate          # Windows
pip install -e ".[all]"                                   # or just ".[ui]" / ".[agents]"

# 1) Quant-only scan — no LLM, no API key, no UI:
quantwheel-scan --tickers F,SOFI,NIO --account 1000

# 2) Full dashboard (7 tabs + tracker):
streamlit run src/quantwheel/ui/streamlit_app/app.py

# 3) Terminal UI (same backend):
python -m quantwheel.ui.tui.app
```

The quant core, backtest, scan, and tracker work **with no LLM and no key**. Only the council and
weekly agent call an LLM — and you choose which.

### Choose your LLM (local or cloud)

The council is **provider-agnostic** — pick yours with environment variables (copy
`.env.example` → `.env`):

```bash
# Fully local, no API key (recommended) — needs Ollama running:
QUANTWHEEL_PROVIDER=ollama
QUANTWHEEL_MODEL=llama3.2            # or qwen2.5, mistral-nemo — any tool/JSON-capable model
pip install -e ".[agents,ui,llm-ollama]"

# Or a cloud provider:
QUANTWHEEL_PROVIDER=anthropic   # + ANTHROPIC_API_KEY   (pip install … llm-anthropic)
QUANTWHEEL_PROVIDER=openai      # + OPENAI_API_KEY       (pip install … llm-openai)
#   ↳ also works with OpenAI-compatible local servers via QUANTWHEEL_BASE_URL
QUANTWHEEL_PROVIDER=google      # + GOOGLE_API_KEY       (pip install … llm-google)
```

Adding a new provider is one branch in `src/quantwheel/agents/llm.py`. Note: the council uses
*structured output*, so a local model must support tool-calling / JSON-schema (llama3.1/3.2, qwen2.5
do; tiny models like `gemma3:270m` don't).

## Learning project

This repo doubles as a structured LangChain/LangGraph course — see **[LEARNING.md](LEARNING.md)** and
the notebooks: `01_learn_langchain` (the council), `02_learn_langgraph` (the weekly agent),
`research.ipynb` (the backtest plots).

## Limitations (read these)

- **Free option data is noisy.** Yahoo returns synthetic IV / zero bid-ask after hours; the tool
  detects this and falls back to a Black-Scholes theoretical quote (flagged `Est`). **Always verify
  the live chain in Robinhood before trading.** A real IV source (e.g. Tradier sandbox) is the
  intended upgrade.
- **The backtest uses realized vol**, not historical implied vol (no free historical chains), and
  synthesizes quarterly earnings dates when the feed lacks them. The *relative* ablation is robust;
  absolute income is understated.
- **At exactly $1,000**, a true cash-secured put only fits underlyings with a strike below ~$7 — the
  tool says so explicitly rather than returning a silent "0 contracts."

## Tests

```bash
pytest            # 81 tests: pricing, regime, MC, sizing, council, agent, dashboard, architecture
```

## License

MIT
