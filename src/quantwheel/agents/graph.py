"""The weekly wheel agent as a LangGraph ``StateGraph``.

Flow::

    load_universe -> screen -> council -> rank_pick --(sit out)--> END
                                              |
                                          (a pick) -> draft_card -> human_approval
                                                                        |
                                              (approved) -> write_record -> END
                                              (declined) ------------------> END

Why a graph instead of a function? Two reasons a recruiter will recognise:
* **Branching** — it cleanly "sits out" a week when nothing qualifies.
* **Human-in-the-loop** — ``interrupt()`` pauses at the approval step and waits
  for a person. **Code never places an order.** The graph resumes only after a
  human says yes/no.

See ``notebooks/02_learn_langgraph.ipynb`` for a guided build.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from quantwheel.agents import nodes
from quantwheel.agents.nodes import WheelState
from quantwheel.core.config import DEFAULT, WheelConfig

# Process-wide in-memory checkpointer so a paused agent can be resumed by a
# *separate* call (e.g. the dashboard's Approve button) within the same process,
# keyed by thread_id. Swap for a SqliteSaver to persist across restarts.
_CHECKPOINTER = MemorySaver()


def _human_approval(state: WheelState) -> WheelState:
    """Pause for human review. Resumes with the caller's yes/no decision."""
    card = state.get("card")
    decision = interrupt({
        "ticker": card.ticker if card else None,
        "summary": (f"Sell {card.contracts}x ${card.strike:.2f} put on {card.ticker}, "
                    f"{card.dte} DTE, ~${card.target_credit:.0f} credit"
                    if card else "no trade"),
        "council": card.council_consensus if card else "",
    })
    return {"approved": bool(decision)}


def _route_after_pick(state: WheelState) -> str:
    return "draft_card" if state.get("pick") is not None else END


def _route_after_approval(state: WheelState) -> str:
    return "write_record" if state.get("approved") else END


def build_weekly_agent(
    llm=None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: WheelConfig = DEFAULT,
    checkpointer=None,
):
    """Compile the weekly-agent graph.

    Parameters
    ----------
    llm:
        Inject a fake runnable to run the council offline (tests/demo). ``None``
        uses the configured provider (see :mod:`quantwheel.agents.llm`).
    provider, model:
        Override the LLM provider/model (else read from the environment).
    checkpointer:
        Required for the human-in-the-loop interrupt to be resumable. Defaults to
        an in-memory saver.
    """
    g = StateGraph(WheelState)
    g.add_node("load_universe", nodes.load_universe)
    g.add_node("screen", partial(nodes.screen, wheel_config=config))
    g.add_node("council", partial(nodes.council, llm=llm, provider=provider,
                                   model=model, wheel_config=config))
    g.add_node("rank_pick", nodes.rank_pick)
    g.add_node("draft_card", partial(nodes.draft_card, wheel_config=config))
    g.add_node("human_approval", _human_approval)
    g.add_node("write_record", nodes.write_record)

    g.add_edge(START, "load_universe")
    g.add_edge("load_universe", "screen")
    g.add_edge("screen", "council")
    g.add_edge("council", "rank_pick")
    g.add_conditional_edges("rank_pick", _route_after_pick,
                            {"draft_card": "draft_card", END: END})
    g.add_edge("draft_card", "human_approval")
    g.add_conditional_edges("human_approval", _route_after_approval,
                            {"write_record": "write_record", END: END})
    g.add_edge("write_record", END)

    return g.compile(checkpointer=checkpointer or _CHECKPOINTER)


def run_weekly_agent(
    universe: Optional[list[str]] = None,
    *,
    llm=None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: WheelConfig = DEFAULT,
    auto_approve: Optional[bool] = None,
    thread_id: str = "weekly",
) -> WheelState:
    """Run the agent end-to-end, handling the approval interrupt.

    ``auto_approve`` lets non-interactive callers (tests, the notebook, a cron
    run) answer the human gate automatically:

    * ``None``  — stop at the interrupt and return the paused state (a UI then
      resumes it via :func:`resume_weekly_agent`).
    * ``True/False`` — answer the gate immediately and run to completion.
    """
    agent = build_weekly_agent(llm=llm, provider=provider, model=model, config=config)
    cfg = {"configurable": {"thread_id": thread_id}}
    initial: WheelState = {"universe": universe} if universe else {}

    result = agent.invoke(initial, cfg)
    if "__interrupt__" not in result:
        return result  # sat out — never reached the approval gate
    if auto_approve is None:
        return result  # leave paused for an interactive caller
    return agent.invoke(Command(resume=auto_approve), cfg)


def resume_weekly_agent(
    approved: bool, *, thread_id: str = "weekly",
    llm=None, provider: Optional[str] = None, model: Optional[str] = None,
    config: WheelConfig = DEFAULT,
) -> WheelState:
    """Resume a paused agent with the human's decision (same ``thread_id``)."""
    agent = build_weekly_agent(llm=llm, provider=provider, model=model, config=config)
    cfg = {"configurable": {"thread_id": thread_id}}
    return agent.invoke(Command(resume=approved), cfg)
