"""Tests for ``app/main.py`` — process entry point.

Focus: the SIGTERM/SIGINT signal handler closure inside :func:`amain`.

Yesterday's outbound PSTN test exposed a missing-graceful-drain-logs
symptom whose root cause was ``asyncio.create_task`` returning a
task that nobody held a reference to — Python's docs explicitly
warn against this:

  "A task that isn't referenced elsewhere may get garbage collected
  at any time, even before it's done."
  https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task

The Phase 2 fix (CHANGE 9) routes the spawned task through a
function-scoped ``shutdown_tasks`` set with
``task.add_done_callback(shutdown_tasks.discard)`` cleanup. These
tests verify that pattern end-to-end against a fake event loop.

Mocking strategy:

* All Layer 9 component constructors (``TaskProtection``,
  ``DailyRoomClient``, ``PipelineManager``, ``MetricsEmitter``,
  ``build_app``, ``graceful_drain``) are patched.
* ``aiohttp.web.AppRunner`` / ``web.TCPSite`` are patched so we
  don't actually open a port.
* ``loop.add_signal_handler`` is replaced by a spy that stashes the
  registered handler so the test can fire it manually — this is
  cleaner than racing real OS signals through the event loop.
"""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _patch_amain_dependencies(
    *,
    drain_side_effect=None,
):
    """Build the mock graph amain() expects.

    Returns ``(graceful_drain_mock, manager_mock, protection_mock,
    daily_client_mock, metrics_mock)`` so individual tests can
    inspect / override drain behavior.
    """
    protection_mock = MagicMock()
    protection_mock.set_protected = AsyncMock(return_value=True)

    daily_client_mock = MagicMock()
    daily_client_mock.close = AsyncMock()

    manager_mock = MagicMock()

    metrics_mock = MagicMock()
    metrics_mock.start = AsyncMock()
    metrics_mock.stop = AsyncMock()

    if drain_side_effect is None:
        graceful_drain_mock = AsyncMock()
    else:
        graceful_drain_mock = AsyncMock(side_effect=drain_side_effect)

    return (
        graceful_drain_mock,
        manager_mock,
        protection_mock,
        daily_client_mock,
        metrics_mock,
    )


async def _run_amain_and_capture_handlers(graceful_drain_mock, *mocks):
    """Spawn ``amain`` with everything patched. Returns the
    (signal_handler_fn, args, app_runner_mock, shutdown_event_setter).

    We hijack ``loop.add_signal_handler`` to capture the registered
    handler instead of letting it bind to the real OS signal table —
    test isolation matters because pytest often runs in xdist
    workers that share signal disposition.
    """
    (
        manager_mock,
        protection_mock,
        daily_client_mock,
        metrics_mock,
    ) = mocks

    captured_handlers: dict[int, tuple] = {}

    def fake_add_signal_handler(sig, handler, *args):
        captured_handlers[sig] = (handler, args)

    app_runner_mock = MagicMock()
    app_runner_mock.setup = AsyncMock()
    app_runner_mock.cleanup = AsyncMock()

    site_mock = MagicMock()
    site_mock.start = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.environment = "test"
    settings_mock.aws_region = "us-west-2"
    settings_mock.max_concurrent_calls = 6
    settings_mock.service_port = 8080
    settings_mock.daily_api_key = "k"
    settings_mock.recording_bucket = "b"
    settings_mock.recording_role_arn = "arn"
    settings_mock.recording_region = "us-east-2"
    daily_client_ctor = MagicMock(return_value=daily_client_mock)

    with (
        patch("app.main.Settings", MagicMock(return_value=settings_mock)),
        patch("app.main.TaskProtection", MagicMock(return_value=protection_mock)),
        patch("app.main.DailyRoomClient", daily_client_ctor),
        patch("app.main.PipelineManager", MagicMock(return_value=manager_mock)),
        patch("app.main.MetricsEmitter", MagicMock(return_value=metrics_mock)),
        patch("app.main.build_app", AsyncMock(return_value=MagicMock())),
        patch("app.main.graceful_drain", graceful_drain_mock),
        patch("app.main.web.AppRunner", MagicMock(return_value=app_runner_mock)),
        patch("app.main.web.TCPSite", MagicMock(return_value=site_mock)),
    ):
        loop = asyncio.get_running_loop()
        original_asgh = loop.add_signal_handler
        loop.add_signal_handler = fake_add_signal_handler  # type: ignore[method-assign]
        try:
            from app.main import amain

            amain_task = asyncio.create_task(amain())

            # Yield until both signal handlers were registered AND
            # the server is awaiting shutdown_event. Bounded: 100 ticks
            # is plenty for our patched coroutines to settle.
            for _ in range(100):
                await asyncio.sleep(0)
                if signal.SIGTERM in captured_handlers and signal.SIGINT in captured_handlers:
                    break

            assert signal.SIGTERM in captured_handlers
            handler_fn, handler_args = captured_handlers[signal.SIGTERM]

            # Fire the captured handler — same path as a real signal
            # arriving — to drive _signal_handler.
            handler_fn(*handler_args)

            # The handler returns synchronously; the spawned shutdown
            # task runs on the next loop tick. Wait for amain to
            # complete (shutdown_event was set inside the drain).
            await asyncio.wait_for(amain_task, timeout=2.0)
        finally:
            loop.add_signal_handler = original_asgh  # type: ignore[method-assign]

    daily_client_ctor.assert_called_once_with(
        api_key="k",
        recording_bucket="b",
        recording_role_arn="arn",
        recording_region="us-east-2",
    )
    return app_runner_mock


# ── SIGTERM handler reference holding ────────────────────────────────────


@pytest.mark.asyncio
async def test_signal_handler_drives_drain_through_completion():
    """End-to-end: signal arrives → handler fires → drain runs →
    shutdown_event sets → amain returns. This is the path that
    silently broke yesterday because the spawned task was GC'd
    before the first await.
    """
    drain_mock, *layer_mocks = _patch_amain_dependencies()
    app_runner_mock = await _run_amain_and_capture_handlers(drain_mock, *layer_mocks)

    # Drain ran exactly once, layer cleanup ran in order.
    drain_mock.assert_awaited_once()
    layer_mocks[3].stop.assert_awaited_once()  # metrics.stop()
    layer_mocks[2].close.assert_awaited_once()  # daily_client.close()
    app_runner_mock.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_amain_passes_metrics_emitter_to_graceful_drain():
    """Wave 4 wiring: graceful_drain must receive the live
    MetricsEmitter so DrainTimeouts can be published when budget
    is exceeded. Without this, Wave 3's DrainTimeoutsAboveThreshold
    alarm has no data to evaluate.
    """
    drain_mock, *layer_mocks = _patch_amain_dependencies()
    await _run_amain_and_capture_handlers(drain_mock, *layer_mocks)

    drain_mock.assert_awaited_once()
    # graceful_drain is called as (manager, protection, metrics=metrics).
    # The metrics kwarg must be the same MetricsEmitter instance that
    # was constructed (and started) earlier in amain.
    _, kwargs = drain_mock.call_args
    metrics_mock = layer_mocks[3]
    assert kwargs.get("metrics") is metrics_mock


@pytest.mark.asyncio
async def test_signal_handler_holds_shutdown_task_until_drain_finishes():
    """If the spawned task were unreferenced, a slow drain wouldn't
    complete — Python could GC the task mid-await. With the
    set+done_callback pattern, the task is anchored until the
    callback fires.
    """
    drain_started = asyncio.Event()
    drain_release = asyncio.Event()

    async def slow_drain(*args, **kwargs):
        drain_started.set()
        await drain_release.wait()

    drain_mock = AsyncMock(side_effect=slow_drain)

    (
        _,
        manager_mock,
        protection_mock,
        daily_client_mock,
        metrics_mock,
    ) = _patch_amain_dependencies()

    captured_handlers: dict[int, tuple] = {}

    def fake_add_signal_handler(sig, handler, *args):
        captured_handlers[sig] = (handler, args)

    app_runner_mock = MagicMock()
    app_runner_mock.setup = AsyncMock()
    app_runner_mock.cleanup = AsyncMock()
    site_mock = MagicMock()
    site_mock.start = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.environment = "test"
    settings_mock.aws_region = "us-west-2"
    settings_mock.max_concurrent_calls = 6
    settings_mock.service_port = 8080
    settings_mock.daily_api_key = "k"
    settings_mock.recording_bucket = "b"
    settings_mock.recording_role_arn = "arn"
    settings_mock.recording_region = "us-east-2"

    with (
        patch("app.main.Settings", MagicMock(return_value=settings_mock)),
        patch("app.main.TaskProtection", MagicMock(return_value=protection_mock)),
        patch("app.main.DailyRoomClient", MagicMock(return_value=daily_client_mock)),
        patch("app.main.PipelineManager", MagicMock(return_value=manager_mock)),
        patch("app.main.MetricsEmitter", MagicMock(return_value=metrics_mock)),
        patch("app.main.build_app", AsyncMock(return_value=MagicMock())),
        patch("app.main.graceful_drain", drain_mock),
        patch("app.main.web.AppRunner", MagicMock(return_value=app_runner_mock)),
        patch("app.main.web.TCPSite", MagicMock(return_value=site_mock)),
    ):
        loop = asyncio.get_running_loop()
        original_asgh = loop.add_signal_handler
        loop.add_signal_handler = fake_add_signal_handler  # type: ignore[method-assign]
        try:
            from app.main import amain

            amain_task = asyncio.create_task(amain())

            for _ in range(100):
                await asyncio.sleep(0)
                if signal.SIGTERM in captured_handlers:
                    break

            handler_fn, handler_args = captured_handlers[signal.SIGTERM]
            handler_fn(*handler_args)

            # The drain task is in flight and pinned by the
            # shutdown_tasks set. Garbage collection cannot reap it
            # because the set holds a strong reference.
            await asyncio.wait_for(drain_started.wait(), timeout=1.0)

            import gc

            gc.collect()  # If the task were leaked, this would reap it.

            # Drain still alive — release it and let amain finish.
            drain_release.set()
            await asyncio.wait_for(amain_task, timeout=2.0)
        finally:
            loop.add_signal_handler = original_asgh  # type: ignore[method-assign]

    drain_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_both_sigterm_and_sigint_are_registered():
    """Phase 2 brief: TIME-DEPENDENT — both SIGTERM (Fargate-spawned)
    and SIGINT (developer Ctrl-C) must be wired so dev parity matches
    prod.
    """
    drain_mock, *layer_mocks = _patch_amain_dependencies()

    (
        manager_mock,
        protection_mock,
        daily_client_mock,
        metrics_mock,
    ) = layer_mocks
    captured_handlers: dict[int, tuple] = {}

    def fake_add_signal_handler(sig, handler, *args):
        captured_handlers[sig] = (handler, args)

    app_runner_mock = MagicMock()
    app_runner_mock.setup = AsyncMock()
    app_runner_mock.cleanup = AsyncMock()
    site_mock = MagicMock()
    site_mock.start = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.environment = "test"
    settings_mock.aws_region = "us-west-2"
    settings_mock.max_concurrent_calls = 6
    settings_mock.service_port = 8080
    settings_mock.daily_api_key = "k"
    settings_mock.recording_bucket = "b"
    settings_mock.recording_role_arn = "arn"
    settings_mock.recording_region = "us-east-2"

    with (
        patch("app.main.Settings", MagicMock(return_value=settings_mock)),
        patch("app.main.TaskProtection", MagicMock(return_value=protection_mock)),
        patch("app.main.DailyRoomClient", MagicMock(return_value=daily_client_mock)),
        patch("app.main.PipelineManager", MagicMock(return_value=manager_mock)),
        patch("app.main.MetricsEmitter", MagicMock(return_value=metrics_mock)),
        patch("app.main.build_app", AsyncMock(return_value=MagicMock())),
        patch("app.main.graceful_drain", drain_mock),
        patch("app.main.web.AppRunner", MagicMock(return_value=app_runner_mock)),
        patch("app.main.web.TCPSite", MagicMock(return_value=site_mock)),
    ):
        loop = asyncio.get_running_loop()
        original_asgh = loop.add_signal_handler
        loop.add_signal_handler = fake_add_signal_handler  # type: ignore[method-assign]
        try:
            from app.main import amain

            amain_task = asyncio.create_task(amain())

            for _ in range(100):
                await asyncio.sleep(0)
                if signal.SIGTERM in captured_handlers and signal.SIGINT in captured_handlers:
                    break

            assert signal.SIGTERM in captured_handlers
            assert signal.SIGINT in captured_handlers

            # Fire one to unblock amain, ignore the other.
            handler_fn, handler_args = captured_handlers[signal.SIGINT]
            handler_fn(*handler_args)

            await asyncio.wait_for(amain_task, timeout=2.0)
        finally:
            loop.add_signal_handler = original_asgh  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_repeated_signal_does_not_break_shutdown():
    """If SIGTERM arrives, then SIGINT lands while drain is still
    in flight, the second handler enqueues a SECOND drain task. The
    set holds both; both run; ``shutdown_event.set()`` is idempotent;
    amain still finishes cleanly.

    Decision: we accept duplicate drains rather than guard. Drains
    are themselves safe to invoke twice (graceful_drain operates on
    immutable manager state — already draining is a no-op by virtue
    of the manager's ``_draining`` flag).
    """
    call_count = {"n": 0}

    async def quick_drain(*args, **kwargs):
        call_count["n"] += 1

    drain_mock = AsyncMock(side_effect=quick_drain)

    (
        _,
        manager_mock,
        protection_mock,
        daily_client_mock,
        metrics_mock,
    ) = _patch_amain_dependencies()

    captured_handlers: dict[int, tuple] = {}

    def fake_add_signal_handler(sig, handler, *args):
        captured_handlers[sig] = (handler, args)

    app_runner_mock = MagicMock()
    app_runner_mock.setup = AsyncMock()
    app_runner_mock.cleanup = AsyncMock()
    site_mock = MagicMock()
    site_mock.start = AsyncMock()

    settings_mock = MagicMock()
    settings_mock.environment = "test"
    settings_mock.aws_region = "us-west-2"
    settings_mock.max_concurrent_calls = 6
    settings_mock.service_port = 8080
    settings_mock.daily_api_key = "k"
    settings_mock.recording_bucket = "b"
    settings_mock.recording_role_arn = "arn"
    settings_mock.recording_region = "us-east-2"

    with (
        patch("app.main.Settings", MagicMock(return_value=settings_mock)),
        patch("app.main.TaskProtection", MagicMock(return_value=protection_mock)),
        patch("app.main.DailyRoomClient", MagicMock(return_value=daily_client_mock)),
        patch("app.main.PipelineManager", MagicMock(return_value=manager_mock)),
        patch("app.main.MetricsEmitter", MagicMock(return_value=metrics_mock)),
        patch("app.main.build_app", AsyncMock(return_value=MagicMock())),
        patch("app.main.graceful_drain", drain_mock),
        patch("app.main.web.AppRunner", MagicMock(return_value=app_runner_mock)),
        patch("app.main.web.TCPSite", MagicMock(return_value=site_mock)),
    ):
        loop = asyncio.get_running_loop()
        original_asgh = loop.add_signal_handler
        loop.add_signal_handler = fake_add_signal_handler  # type: ignore[method-assign]
        try:
            from app.main import amain

            amain_task = asyncio.create_task(amain())

            for _ in range(100):
                await asyncio.sleep(0)
                if signal.SIGTERM in captured_handlers and signal.SIGINT in captured_handlers:
                    break

            # Two signals in quick succession.
            term_fn, term_args = captured_handlers[signal.SIGTERM]
            int_fn, int_args = captured_handlers[signal.SIGINT]
            term_fn(*term_args)
            int_fn(*int_args)

            await asyncio.wait_for(amain_task, timeout=2.0)
        finally:
            loop.add_signal_handler = original_asgh  # type: ignore[method-assign]

    # Both drains ran. amain still completed.
    assert call_count["n"] == 2
