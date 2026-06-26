"""v2 process entry point. Wires Layers 1-9 together.

This is what runs when the Fargate container launches. The CMD in
the Dockerfile is ``python -m app.main`` (or ``python app/main.py``);
:func:`main` is the synchronous wrapper around :func:`amain`.

Wiring order:

1. Configure structlog (JSON renderer, contextvars, ISO timestamps).
2. Construct :class:`Settings` (env-var-backed, fails fast on
   missing required values).
3. Construct singletons:
   :class:`~app.runner.protection.TaskProtection`,
   :class:`~app.runner.daily_rooms.DailyRoomClient`,
   :class:`~app.runner.manager.PipelineManager`,
   :class:`~app.runner.metrics.MetricsEmitter`.
4. Build the aiohttp app via :func:`~app.runner.server.build_app`.
5. Register SIGTERM/SIGINT handlers — single process-wide handler
   that schedules :func:`~app.runner.server.graceful_drain`.
6. Start the HTTP server. Block on the shutdown event.
7. On signal: drain + cleanup + exit.

PipelineRunner construction in Layer 8 already passes
``handle_sigint=False`` and ``handle_sigterm=False`` so concurrent
calls don't clobber this process-level handler (closes tech debt
entry 12).
"""

from __future__ import annotations

import asyncio
import signal
import sys

import structlog
from aiohttp import web

from app.config.settings import Settings
from app.observability import init_tracing, shutdown_tracing
from app.runner import (
    DailyRoomClient,
    MetricsEmitter,
    PipelineManager,
    TaskProtection,
    build_app,
    graceful_drain,
)


def _configure_logging() -> None:
    """Set up structlog once at process start.

    JSON renderer for CloudWatch ingestion. ``merge_contextvars``
    pulls per-call ``call_id`` / ``session_id`` bindings (Layer 8
    sets these via ``structlog.contextvars.bind_contextvars``) into
    every log line emitted from inside that call's coroutine.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


async def amain() -> None:
    """Async main. Constructs everything, runs the server, drains on signal."""
    _configure_logging()
    logger = structlog.get_logger("main")
    logger.info("service_starting")

    settings = Settings()
    logger.info(
        "settings_loaded",
        environment=settings.environment,
        aws_region=settings.aws_region,
        max_concurrent_calls=settings.max_concurrent_calls,
        service_port=settings.service_port,
    )

    # Per-call tracing (#13). Fail-open: a False return (disabled / SDK
    # missing / setup error) leaves every span helper a no-op and calls
    # run identically. Initialized once here at boot.
    tracing_enabled = init_tracing(settings)
    logger.info("tracing_initialized", enabled=tracing_enabled)

    # ── Layer 9 components ────────────────────────────────────────
    protection = TaskProtection()
    daily_client = DailyRoomClient(
        api_key=settings.daily_api_key,
        recording_bucket=settings.recording_bucket,
        recording_role_arn=settings.recording_role_arn,
        recording_region=settings.recording_region,
    )
    manager = PipelineManager(settings, daily_client, protection)
    metrics = MetricsEmitter(manager, settings)

    await metrics.start()

    app = await build_app(settings, manager)

    # ── Signal handling ──────────────────────────────────────────
    # Single process-wide handler. Layer 8's PipelineRunner is
    # constructed with handle_sigint=False / handle_sigterm=False
    # so it doesn't clobber these registrations on each per-call
    # runner construction (tech debt entry 12).
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    # Strong references to outstanding shutdown tasks. Without this,
    # asyncio.create_task's returned task can be garbage-collected
    # before _drain_and_shutdown reaches its first await point —
    # which was the root cause of yesterday's missing graceful_drain
    # logs on SIGTERM. Per Python docs:
    #   "A task that isn't referenced elsewhere may get garbage
    #   collected at any time, even before it's done."
    #   https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
    # The same docs prescribe this exact pattern — collection +
    # add_done_callback for fire-and-forget background tasks.
    shutdown_tasks: set[asyncio.Task] = set()

    async def _drain_and_shutdown(sig_name: str) -> None:
        logger.info("signal_handling_started", signal=sig_name)
        # Pass the metrics emitter so graceful_drain can publish
        # DrainTimeouts (feeds Wave 3's DrainTimeoutsAboveThreshold
        # alarm). The emitter is stopped AFTER drain so any
        # timeout-path emission has a live boto3 client.
        await graceful_drain(manager, protection, metrics=metrics)
        await metrics.stop()
        await daily_client.close()
        # Flush any buffered spans before exit (fail-open no-op when
        # tracing is disabled). Last, so in-flight call spans are drained.
        shutdown_tracing()
        shutdown_event.set()

    def _signal_handler(sig_name: str) -> None:
        logger.info("signal_received", signal=sig_name)
        # Schedule the drain coroutine on the loop. Sync handler
        # returns immediately; the drain runs as a separate task.
        task = asyncio.create_task(
            _drain_and_shutdown(sig_name),
            name=f"shutdown-{sig_name}",
        )
        shutdown_tasks.add(task)
        task.add_done_callback(shutdown_tasks.discard)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler, sig.name)

    # ── HTTP server ───────────────────────────────────────────────
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", settings.service_port)
    await site.start()

    logger.info("service_started", port=settings.service_port)

    # Block until the shutdown event fires.
    await shutdown_event.wait()

    # Final cleanup of HTTP server.
    await runner.cleanup()
    logger.info("service_stopped")


def main() -> None:
    """Synchronous wrapper for the entry point."""
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
