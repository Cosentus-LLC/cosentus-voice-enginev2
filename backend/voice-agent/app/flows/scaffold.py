"""Pipecat Flows scaffold — the integration proof for EPIC #16 (#41).

This module stands up the **wiring** between Pipecat Flows and the
existing pipeline; it does **not** introduce any real conversation
steps (that's 16b/16c). It exists to prove two things:

* a :class:`~pipecat_flows.FlowManager` can be constructed against the
  call's real collaborators — the ``PipelineTask``, the LLM service,
  the ``LLMContextAggregatorPair``, and the transport — without
  disturbing them (no frames queued at construction); and
* a trivial **2-step flow** (``start`` → ``advance`` → ``end``)
  advances correctly once initialized.

Why the scaffold flow is inert about the opener
------------------------------------------------
``FlowManager._set_node`` on the *first* node calls
``_update_llm_context``, which queues an ``LLMMessagesUpdateFrame``
(it **replaces** ``LLMContext.messages``) and — when
``respond_immediately`` is ``True`` (the Flows default) — an
``LLMRunFrame``. Either would collide with ``bot.py``'s opener: the
message-replace would clobber the opener-seeded context
(``bot.py`` LLMContext seeding) and the run frame would double-trigger
generation against ``deliver_opener_if_needed``.

So both scaffold nodes set ``respond_immediately=False``: even on the
flag-on path the flow never queues a competing ``LLMRunFrame``. And
``bot.py`` only calls :meth:`FlowManager.initialize` behind
``settings.flows_enabled`` (default ``False``), so production calls
never run the node path at all — the manager is constructed and left
untouched. See :func:`app.bot.bot.run_bot`.
"""

from __future__ import annotations

from typing import Any

from pipecat.transports.base_transport import BaseTransport
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
)
from pipecat_flows.types import ConsolidatedFunctionResult, NodeConfig

# Node identifiers — kept as constants so tests and bot.py logging can
# reference them without restating string literals.
START_NODE = "start"
END_NODE = "end"


async def _advance(
    args: FlowArgs,  # noqa: ARG001 — scaffold transition takes no real args
    flow_manager: FlowManager,  # noqa: ARG001 — unused; 16b/16c will read state
) -> ConsolidatedFunctionResult:
    """Scaffold transition handler: move from ``start`` to ``end``.

    Returns the consolidated ``(result, next_node)`` tuple Flows 1.1.x
    expects from a "direct"/consolidated function — a no-op result plus
    the terminal node config. No side effects; the real per-step logic
    arrives in 16b/16c.
    """
    return {"status": "ok"}, _end_node()


def _start_node() -> NodeConfig:
    """Initial node of the trivial 2-step flow.

    Exposes a single ``advance`` function that transitions to the
    terminal node. ``respond_immediately=False`` keeps the node from
    queueing an ``LLMRunFrame`` so it never races ``bot.py``'s opener.
    """
    return {
        "name": START_NODE,
        "task_messages": [
            {
                "role": "system",
                "content": "Scaffold start node (no-op). Call advance to continue.",
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="advance",
                description="Advance the scaffold flow to the end node.",
                properties={},
                required=[],
                handler=_advance,
            )
        ],
        "respond_immediately": False,
    }


def _end_node() -> NodeConfig:
    """Terminal node of the trivial 2-step flow.

    No functions, no immediate response — reaching it is the proof the
    flow advanced.
    """
    return {
        "name": END_NODE,
        "task_messages": [
            {
                "role": "system",
                "content": "Scaffold end node (no-op). Flow complete.",
            }
        ],
        "functions": [],
        "respond_immediately": False,
    }


def build_scaffold_flow() -> NodeConfig:
    """Return the initial node config for the trivial 2-step flow.

    Passed to :meth:`FlowManager.initialize` by ``bot.py`` only when
    ``settings.flows_enabled`` is ``True``.
    """
    return _start_node()


def build_flow_manager(
    *,
    task: Any,
    llm: Any,
    context_aggregator: Any,
    transport: BaseTransport | None,
) -> FlowManager:
    """Construct an **uninitialized** ``FlowManager`` for one call.

    Wires Flows to the call's real collaborators. Construction is
    side-effect-free: it queues no frames and does not touch the
    seeded ``LLMContext`` — initialization (which would) happens later
    and only behind the ``flows_enabled`` flag.

    Args:
        task: The call's ``PipelineTask`` (Flows queues frames onto it).
        llm: The call's LLM service — Flows registers node functions
            onto the same service ``bot.py`` already uses.
        context_aggregator: The ``LLMContextAggregatorPair`` built by
            ``bot.py`` — Flows reads/updates the shared ``LLMContext``
            through it rather than constructing its own.
        transport: The call's transport, exposed to Flows action
            handlers (unused by the scaffold; present for 16b/16c).

    Returns:
        An uninitialized :class:`~pipecat_flows.FlowManager`. Call
        :meth:`FlowManager.initialize` to start a node.
    """
    return FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )
