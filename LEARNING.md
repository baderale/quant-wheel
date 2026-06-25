# LEARNING.md — building this as a LangChain / LangGraph course

You already know **pandas, scipy, scikit-learn**. This project uses that comfort zone for
the quant core and treats the **agent layer as the thing to learn**. Work the phases in
order; each one introduces concepts exactly where the code needs them.

## Mental model first

- **LangChain** = the plumbing for *one* LLM call: a prompt template + a model + an output
  parser, composed with the `|` pipe into a *chain*. A council persona is one chain.
- **LangGraph** = a state machine *around* many such calls: nodes (functions) connected by
  edges, sharing a typed `State`. Our weekly routine is one graph. It also gives us
  **human-in-the-loop interrupts** (the approval step) for free.
- Rule of thumb: *reach for LangChain when you have one step; reach for LangGraph the
  moment you have branching, loops, or a human in the middle.*

## Curriculum (mapped to build phases)

### Phase 1 — Quant core *(your home turf; no new frameworks)*
Port the Markov/Monte-Carlo/Black-Scholes math to numpy/scipy. Goal: get totally fluent
in the **data contract** (`core/types.py`) — every later phase just moves these objects
around. *Concept:* clean boundaries make agents and UIs trivial later.

### Phase 2 — Backtest *(pandas/scipy)*
Walk-forward simulation + metrics. *Concept:* never trust a strategy you haven't
validated; calibration answers "are my probabilities honest?".

### Phase 3 — LangChain council ← **first new framework**
- `ChatPromptTemplate` — system + human messages with variables.
- `ChatAnthropic` — the model wrapper (`langchain-anthropic`).
- **Structured output** — `.with_structured_output(CouncilVote)` so the LLM returns a
  typed object, not free text. This is the single most useful LangChain feature here.
- LCEL pipe: `prompt | model.with_structured_output(...)`.
- Running personas concurrently with `.batch()` / async.
- 📓 `notebooks/01_learn_langchain.ipynb` walks through each on a real persona.

### Phase 4 — LangGraph weekly agent ← **second new framework**
- `StateGraph`, `TypedDict` state, `add_node` / `add_edge` / conditional edges.
- `interrupt()` + checkpointer for the human approval gate.
- Why a graph beats a for-loop here: branching (sit out vs. trade), resumability, and a
  visualisable flow recruiters can read.
- 📓 `notebooks/02_learn_langgraph.ipynb`.

### Phase 5 — Services + dashboard
- *Concept:* the **facade pattern**. The UI depends on `services/`, not internals — the
  same idea that makes microservices and clean architecture testable.

### Phase 6 — Polish
- Swap a tab to a Textual TUI to *prove* the decoupling. *Concept:* if the swap is hard,
  the boundary was wrong.

## LLM provider notes (bring your own model)
- The council/agent are **provider-agnostic** — `agents/llm.py` builds the chat model from
  `QUANTWHEEL_PROVIDER` (`ollama` | `anthropic` | `openai` | `google`) and `QUANTWHEEL_MODEL`.
- **Local (Ollama):** `QUANTWHEEL_PROVIDER=ollama`, run `ollama serve`, pull a tool-capable model
  (`llama3.2`, `qwen2.5`). No API key, fully offline, free.
- **Cloud:** set the matching key (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY`).
- *LangChain concept:* `.with_structured_output()` works across providers — the persona code is
  identical whether it's Claude or a local Llama. Swapping models changes one env var.
- The quant core and backtest need **no** LLM at all, so you can learn the math offline for free.

## Glossary (options + quant terms used in the code)
- **CSP** cash-secured put · **CC** covered call · **DTE** days to expiry · **POP**
  probability of profit · **IV** implied volatility · **assignment** being forced to buy
  the shares · **theta** time decay · **delta** ~ probability-of-ITM proxy · **Kelly**
  growth-optimal bet size.
