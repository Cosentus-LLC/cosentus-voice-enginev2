"""Per-call runtime context passed to tool executors.

Built fresh inside the LLM tool handler for every tool invocation
(see Layer 8). Carries the call-scoped state the executors need
without forcing them to close over the entire pipeline.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pipecat.processors.frame_processor import FrameDirection

# Bound to ``PipelineTask.queue_frame`` by Layer 8. Two-argument form
# (frame, direction) is what the underlying method accepts — Pipecat
# 1.1.0's ``PipelineTask.queue_frame(frame, direction=DOWNSTREAM)``.
# Tools that need to push upstream (e.g. ``end_call`` queueing an
# ``EndTaskFrame`` so the pipeline drains nicely without the
# ``EndFrame`` hang race in pipecat issue #3757) call this with
# ``direction=FrameDirection.UPSTREAM``.
QueueFrameFn = Callable[..., Awaitable[None]]


@dataclass
class ToolContext:
    """Runtime state available to a tool executor during one invocation.

    Attributes:
        call_id: Unique identifier for the call (also the structured-log
            ``call_id`` field). Same value as ``session_id`` in v2.
        session_id: Pipecat session identifier. Often equal to
            ``call_id``; kept separate for forward compatibility.
        sip_session_id: Daily participant ID for the SIP leg. Required
            by ``transfer_call`` and ``press_digit``; ``None`` for
            non-SIP calls (e.g., browser-only dev rooms).
        transport: The Pipecat transport instance — typed ``Any`` to
            avoid pulling Daily-specific symbols into Layer 4. The
            executors duck-type it (``transport.sip_call_transfer``,
            ``transport.send_dtmf``).
        queue_frame: Async callable that pushes a frame into the
            running pipeline. Wired by Layer 8 to
            ``PipelineTask.queue_frame``. Accepts an optional
            ``direction`` argument (``FrameDirection.DOWNSTREAM`` /
            ``UPSTREAM``); defaults to downstream. ``end_call`` pushes
            ``EndTaskFrame`` upstream — the documented Pipecat way for
            a tool to request graceful pipeline shutdown without the
            ``EndFrame`` hang race (pipecat issue #3757).
        tool_settings: Per-agent tool config, e.g.
            ``{"targets": {"billing_supervisor": "+13105551234"}}``
            for ``transfer_call``. Sourced from
            ``AgentConfig.tools[].settings`` at registry-build time.
        message_history: Recent conversation turns. Currently unused
            by v2's three tools; kept available for future
            deterministic responses that want context.
    """

    call_id: str
    session_id: str | None = None
    sip_session_id: str | None = None
    transport: Any = None
    queue_frame: QueueFrameFn | None = None
    tool_settings: dict[str, Any] = field(default_factory=dict)
    message_history: list[dict[str, Any]] = field(default_factory=list)


__all__ = ["FrameDirection", "QueueFrameFn", "ToolContext"]
