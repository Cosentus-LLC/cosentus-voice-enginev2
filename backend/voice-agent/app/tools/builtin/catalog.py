"""Single source of truth for the platform's tool catalog.

Names match Aurora's ``VALID_TOOL_TYPES`` (see
``cosentus-voice-api-lambda/index.mjs``) so a tool the agent
designer enables on a row in ``voice_agents.tools`` maps 1:1 to
runtime registration.

Every entry is a :class:`ToolDefinition`. Per-agent customizations
(description override, target enum) are applied at registry-build
time, not in this catalog — keep the catalog stable across calls.
"""

from __future__ import annotations

from app.tools.builtin.end_call import END_CALL
from app.tools.builtin.press_digit import PRESS_DIGIT
from app.tools.builtin.transfer_call import TRANSFER_CALL
from app.tools.schema import ToolDefinition

BUILTIN_TOOLS: dict[str, ToolDefinition] = {
    TRANSFER_CALL.name: TRANSFER_CALL,
    PRESS_DIGIT.name: PRESS_DIGIT,
    END_CALL.name: END_CALL,
}
