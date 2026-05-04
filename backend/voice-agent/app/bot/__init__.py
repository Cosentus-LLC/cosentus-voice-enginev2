"""Layer 8 — pipeline composition for one call.

Pipecat foundational two-function pattern:

* :func:`bot(runner_args)` — entry point. Builds the per-call
  transport from ``runner_args`` and delegates to :func:`run_bot`.
* :func:`run_bot(transport, runner_args, settings)` — transport-
  agnostic. Composes Layers 1-7 into a Pipecat pipeline, runs it
  to completion, and fires the end-of-call write via
  :func:`finalize_call` in a ``finally`` block.

Layer 9 (future) constructs :class:`~pipecat.runner.types.DailyRunnerArguments`
for HTTP / SQS / webhook triggers and calls ``bot(runner_args)`` via
``asyncio.create_task``. Concurrent calls run as separate asyncio
tasks in the same Fargate process. Each invocation is per-call
isolated (no module-level mutable state).
"""

from app.bot.bot import bot, run_bot
from app.bot.lifecycle import finalize_call

__all__ = ["bot", "finalize_call", "run_bot"]
