"""Per-process call lifecycle manager.

Owns ``active_sessions`` (the dict of in-flight asyncio tasks),
capacity gating, draining flag, and the dict-boundary lifecycle
that triggers ECS task scale-in protection.

Layer 9 split rationale (vs v1's single ``PipelineManager``):

v1's ``PipelineManager`` mixed 10 concerns. v2 splits them across
layers. This manager keeps only:

* Active session dict + lifecycle (spawn → cleanup).
* Capacity / draining gates.
* Protection 0↔1 boundary triggers + heartbeat coroutine.

What v2 dropped (lives elsewhere now):

* Per-call agent loading + hydration → Layer 8's ``run_bot``.
* Per-call ``CallRecord`` write + PCA + auto-actions →
  Layer 6 + Layer 8's ``finalize_call``.
* Per-call collector creation → Layer 7's accumulator + state.
* Per-call structlog contextvars binding → Layer 8's ``run_bot``.
* DynamoDB session tracker → Layer 11.
* EMF logger → :class:`~app.runner.metrics.MetricsEmitter`.
* HTTP-shaped capacity rejection (``http_status`` field) →
  :class:`CapacityRejected` exception caught by the route handler.

The manager is per-process (one instance shared across all
concurrent calls). Layer 9's :func:`build_app` constructs it once
at startup and stores on ``app["manager"]``.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import structlog
from pipecat.runner.types import DailyRunnerArguments

from app.bot.bot import _PRELOADED_AGENT_CONFIG_KEY, bot
from app.config.agent_config import AgentConfig, load_agent_config
from app.config.settings import Settings
from app.runner.daily_rooms import DailyRoomClient
from app.runner.protection import TaskProtection

logger = structlog.get_logger(__name__)

# Heartbeat cadence for ECS task protection renewal. 30 s is well
# under the 30-minute ``ExpiresInMinutes`` so we have plenty of
# margin for transient failures (3 retries × backoff ≈ <1 s) and
# for the call-end coalescing window.
_HEARTBEAT_INTERVAL_SECS = 30


@dataclass(frozen=True)
class CallSpawnResult:
    """Returned by ``start_*`` methods. Layer 9's HTTP route handlers
    serialize these into 202 responses.

    Attributes:
        call_id: Engine-generated UUID. The Layer-8 bot generates
            its own ``call_id`` internally; this manager's
            ``call_id`` is a separate identifier used as the dict
            key in ``active_sessions``.
        room_name: Daily room name (= ``CallRecord.session_id``).
        room_url: Full Daily room URL.
        viewer_token: Browser-only. Test clients use this to join
            the room as a non-owner participant.
    """

    call_id: str
    room_name: str
    room_url: str
    viewer_token: str | None = None


class CapacityRejected(Exception):
    """Raised by ``start_*`` when the call cannot be accepted.

    Two reasons (in :attr:`reason`):

    * ``"draining"`` — process received SIGTERM; not accepting
      new work.
    * ``"at_capacity"`` — already at ``max_concurrent_calls``.

    HTTP route handlers catch this and return 503.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PipelineManager:
    """Per-process call lifecycle manager.

    Single instance shared across the asyncio event loop. All
    public methods are coroutines and safe to call concurrently —
    the dict-boundary checks happen in await-free regions, so
    asyncio's single-threaded scheduling guarantees the 0↔1
    transitions fire exactly once per actual transition.
    """

    def __init__(
        self,
        settings: Settings,
        daily_client: DailyRoomClient,
        protection: TaskProtection,
    ) -> None:
        self._settings = settings
        self._daily = daily_client
        self._protection = protection
        self._active_sessions: dict[str, asyncio.Task] = {}
        self._draining = False
        self._max_concurrent = settings.max_concurrent_calls
        self._heartbeat_task: asyncio.Task | None = None

    # ── Status accessors ──────────────────────────────────────────

    @property
    def is_draining(self) -> bool:
        return self._draining

    @property
    def active_session_count(self) -> int:
        return len(self._active_sessions)

    @property
    def at_capacity(self) -> bool:
        return len(self._active_sessions) >= self._max_concurrent

    @property
    def active_sessions(self) -> dict[str, asyncio.Task]:
        """Read-only-by-convention dict view. Callers MUST NOT mutate.

        Used by :func:`graceful_drain` to iterate and cancel the
        engine's spawned tasks (and only those tasks — not all
        ``asyncio.all_tasks``, which would kill the HTTP server's
        accept loop along with everything else).
        """
        return self._active_sessions

    def get_status(self) -> dict[str, Any]:
        """Public status snapshot for ``/status`` and ``/ready``."""
        return {
            "active_sessions": self.active_session_count,
            "max_concurrent": self._max_concurrent,
            "draining": self._draining,
            "protected": self._protection.is_protected,
            "protection_available": self._protection.is_available,
        }

    # ── Spawn entry points ────────────────────────────────────────

    async def start_outbound(
        self,
        *,
        agent_id: str,
        target_number: str,
        from_number: str,
        case_data: dict[str, Any] | None = None,
        batch_id: str | None = None,
        batch_row_index: int | None = None,
    ) -> CallSpawnResult:
        """Outbound PSTN call. Creates a dialout-enabled Daily room,
        mints a bot token, spawns the bot, and returns the spawn
        result. The bot itself dials out from ``on_joined``.

        ``from_number`` arrives as an E.164 string (the public
        contract for ``/start``). For Aurora storage and logs we
        keep it E.164 — Aurora's ``voice_calls.from_number`` column
        is ``VARCHAR(30)`` and humans reading transcripts want to
        see the actual phone number, not a UUID. But Daily's
        ``dialOut/start`` ``callerId`` field expects the UUID of a
        purchased-phone-number record. Empirically verified
        2026-05-07: passing the E.164 form returns
        ``Incorrect callerID! No phone number maps to: <num>``;
        passing the UUID succeeds. So we resolve E.164 → UUID here
        and only inject the UUID into ``dialout_settings.callerId``;
        everything else (``body.from_number``, ``CallRecord``,
        observers) keeps the E.164.

        On unresolved E.164 (no matching purchased number in Daily),
        we fall back to passing the E.164 through and let Daily
        return its canonical "Incorrect callerID" error. Layer 8's
        ``dialout_failed_sync`` handler then cancels the bot
        cleanly. Don't fail-fast in the manager — the engine's
        existing termination path already handles this with the
        right CallRecord shape.

        Capacity gate: the slot is reserved synchronously via
        :meth:`_reserve_slot` BEFORE any awaits, so concurrent
        ``/start`` requests can't all pass the gate while the dict
        is still empty (the bug Layer 9.5 scenario d caught). On
        any post-reservation failure (Daily REST exception, etc.)
        we release the reservation so capacity doesn't leak.
        """
        call_id = str(uuid.uuid4())
        was_empty = self._reserve_slot(call_id)
        try:
            agent = await self._load_agent_for_call(agent_id)
            start_recording = self._should_start_recording(agent)
            room = await self._daily.create_outbound_room(recording_enabled=start_recording)
            token = await self._daily.mint_token(
                room.name,
                start_recording=start_recording,
            )

            caller_id_uuid = await self._daily.get_phone_number_uuid(from_number)
            caller_id_for_dailout: str = caller_id_uuid or from_number
            if caller_id_uuid is None:
                logger.warning(
                    "outbound_caller_id_uuid_unresolved",
                    from_number=from_number,
                    hint=(
                        "from_number didn't match any purchased-phone-number "
                        "record in Daily; passing E.164 through, Daily will "
                        "reject with 'Incorrect callerID' and Phase 2's "
                        "dialout_failed_sync handler will cancel the bot."
                    ),
                )

            runner_args = DailyRunnerArguments(
                room_url=room.url,
                token=token,
                body={
                    "agent_id": agent_id,
                    "direction": "outbound",
                    "target_number": target_number,
                    # E.164 — kept for Aurora storage (VARCHAR(30)) and
                    # human-readable logs / transcripts.
                    "from_number": from_number,
                    "case_data": case_data or {},
                    "batch_id": batch_id,
                    "batch_row_index": batch_row_index,
                    _PRELOADED_AGENT_CONFIG_KEY: agent,
                    # Daily SDK key naming: camelCase. Layer 8's
                    # ``on_joined`` handler passes this verbatim to
                    # ``transport.start_dialout``. ``callerId`` MUST
                    # be the Daily phone-number-record UUID, not the
                    # E.164 — see method docstring. Only Daily-
                    # recognized keys go in this dict; the E.164
                    # stays available for logging via
                    # ``body.from_number``.
                    "dialout_settings": {
                        "phoneNumber": target_number,
                        "callerId": caller_id_for_dailout,
                    },
                },
            )

            logger.info(
                "outbound_call_dispatching",
                call_id=call_id,
                from_number=from_number,
                target_number=target_number,
                caller_id_uuid_resolved=caller_id_uuid is not None,
                caller_id_uuid=caller_id_uuid,
            )

            await self._activate_protection_if_first(was_empty)
            await self._spawn(call_id, runner_args)
        except Exception:
            # Release the reservation so capacity isn't leaked when
            # Daily REST or another await raises before _spawn.
            self._release_slot(call_id)
            raise

        return CallSpawnResult(
            call_id=call_id,
            room_name=room.name,
            room_url=room.url,
        )

    async def start_browser(
        self,
        *,
        agent_id: str,
        case_data: dict[str, Any] | None = None,
    ) -> CallSpawnResult:
        """Browser test call. Creates a WebRTC-only room, mints both
        a bot token and a viewer token, spawns the bot. The dashboard
        / Cindy widget joins with the viewer token.
        """
        call_id = str(uuid.uuid4())
        was_empty = self._reserve_slot(call_id)
        try:
            room = await self._daily.create_browser_room()
            bot_token = await self._daily.mint_token(room.name, is_owner=True)
            # Viewer token is non-owner; matches the room's 15-min
            # TTL so we don't issue a long-lived token to a short-
            # lived room.
            viewer_token = await self._daily.mint_token(
                room.name,
                is_owner=False,
                exp_secs=900,
            )

            runner_args = DailyRunnerArguments(
                room_url=room.url,
                token=bot_token,
                body={
                    "agent_id": agent_id,
                    "direction": "browser",
                    "target_number": "",
                    "from_number": "",
                    "case_data": case_data or {},
                },
            )

            await self._activate_protection_if_first(was_empty)
            await self._spawn(call_id, runner_args)
        except Exception:
            self._release_slot(call_id)
            raise

        return CallSpawnResult(
            call_id=call_id,
            room_name=room.name,
            room_url=room.url,
            viewer_token=viewer_token,
        )

    async def start_inbound(
        self,
        *,
        agent_id: str,
        from_number: str,
        to_number: str,
        call_id_external: str,
        call_domain: str,
    ) -> CallSpawnResult:
        """Inbound PSTN call. Creates a SIP-dial-in-enabled room,
        mints a bot token, spawns the bot. The dialin webhook
        handler returns the room's ``sip_uri`` to Daily so the
        caller is bridged into the room.

        The ``call_id_external`` and ``call_domain`` come from
        Daily's webhook payload and are passed through to the bot's
        ``dialin_settings`` so :class:`DailyTransport` can correlate
        the SIP leg.
        """
        call_id = str(uuid.uuid4())
        was_empty = self._reserve_slot(call_id)
        try:
            agent = await self._load_agent_for_call(agent_id)
            start_recording = self._should_start_recording(agent)
            room = await self._daily.create_inbound_room(recording_enabled=start_recording)
            token = await self._daily.mint_token(
                room.name,
                start_recording=start_recording,
            )

            runner_args = DailyRunnerArguments(
                room_url=room.url,
                token=token,
                body={
                    "agent_id": agent_id,
                    "direction": "inbound",
                    "target_number": to_number,
                    "from_number": from_number,
                    "case_data": {},
                    _PRELOADED_AGENT_CONFIG_KEY: agent,
                    "dialin_settings": {
                        "call_id": call_id_external,
                        "call_domain": call_domain,
                    },
                },
            )

            await self._activate_protection_if_first(was_empty)
            await self._spawn(call_id, runner_args)
        except Exception:
            self._release_slot(call_id)
            raise

        return CallSpawnResult(
            call_id=call_id,
            room_name=room.name,
            room_url=room.url,
        )

    # ── Internals ────────────────────────────────────────────────

    async def _load_agent_for_call(self, agent_id: str) -> AgentConfig:
        """Load the agent runtime config once before PSTN room setup."""
        return await load_agent_config(agent_id, settings=self._settings)

    def _should_start_recording(self, agent: AgentConfig) -> bool:
        """Return whether this PSTN call should start Daily cloud recording."""
        return bool(agent.recording.enabled and self._daily.recording_configured)

    def _reject_if_unavailable(self) -> None:
        """Synchronous gate. Reads only — does not reserve a slot.

        Use :meth:`_reserve_slot` instead from the public ``start_*``
        entry points; ``_reject_if_unavailable`` is kept for status
        callers that need a check-only view (e.g. tests).
        """
        if self._draining:
            raise CapacityRejected("draining")
        if self.at_capacity:
            raise CapacityRejected("at_capacity")

    def _reserve_slot(self, call_id: str) -> bool:
        """Atomically check capacity AND reserve the active-sessions slot.

        Both the read (``at_capacity`` check) and the write (dict
        insert) happen in the same await-free region. asyncio's
        single-threaded scheduling guarantees no other coroutine
        can interleave between them, so the gate IS the lock.

        This closes the check-then-act race the prior implementation
        had: ``_reject_if_unavailable`` followed by an ``await
        self._daily.create_outbound_room()`` (and other awaits) gave
        up the event loop after the check, letting concurrent
        requests all pass while the dict was still empty. Layer 9.5
        scale test scenario d (N=10 concurrent /start) reproduced
        this empirically — 10 calls were accepted past
        ``max_concurrent=6``.

        The reservation uses ``None`` as a placeholder; ``_spawn``
        replaces it with the real ``asyncio.Task`` reference once
        the task is created. If the calling ``start_*`` raises
        between reserve and spawn (e.g., Daily REST throws), the
        caller MUST call :meth:`_release_slot` to free the
        reservation — otherwise capacity leaks.

        Returns:
            ``True`` if this reservation was the 0→1 transition (the
            dict was empty before the placeholder went in). Caller
            awaits :meth:`_activate_protection_if_first` with the
            return value to handle ECS task protection acquisition.
        """
        if self._draining:
            raise CapacityRejected("draining")
        if self.at_capacity:
            raise CapacityRejected("at_capacity")
        # Atomic with the check above — no awaits between read and write.
        was_empty = len(self._active_sessions) == 0
        self._active_sessions[call_id] = None  # type: ignore[assignment]
        return was_empty

    def _release_slot(self, call_id: str) -> None:
        """Release a reservation made by :meth:`_reserve_slot`.

        Idempotent. Only removes the slot if the value is still the
        ``None`` placeholder; if ``_spawn`` already replaced it with
        a real task, this is a no-op (the wrapped bot's ``finally``
        block handles cleanup for spawned tasks).
        """
        if call_id in self._active_sessions and self._active_sessions.get(call_id) is None:
            self._active_sessions.pop(call_id, None)

    async def _activate_protection_if_first(self, was_empty: bool) -> None:
        """Fire the 0→1 protection-acquire path when this is the first
        reservation.

        Split from ``_spawn`` so the await happens between the
        synchronous reservation (``_reserve_slot``) and the actual
        task creation (``_spawn``). Failures inside
        ``set_protected`` are swallowed by the protection client's
        own retry / log policy — the call still proceeds.
        """
        if not was_empty:
            return
        await self._protection.set_protected(True)
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name="task-protection-heartbeat",
            )

    async def _spawn(self, call_id: str, runner_args: DailyRunnerArguments) -> None:
        """Create the asyncio task and replace the reservation placeholder.

        Caller MUST have already reserved the slot via
        :meth:`_reserve_slot` and run
        :meth:`_activate_protection_if_first` for the 0→1 case. This
        method only swaps the dict's ``None`` placeholder for the
        real ``asyncio.Task``; protection is no longer ``_spawn``'s
        responsibility.
        """
        task = asyncio.create_task(
            self._wrapped_bot(call_id, runner_args),
            name=f"call-{call_id}",
        )
        # Replace the None placeholder reserved by _reserve_slot with
        # the actual task. (.get returning None is fine; we still
        # write the task in.)
        self._active_sessions[call_id] = task
        logger.info(
            "call_spawned",
            call_id=call_id,
            active_sessions=len(self._active_sessions),
            max_concurrent=self._max_concurrent,
        )

    async def _wrapped_bot(self, call_id: str, runner_args: DailyRunnerArguments) -> None:
        """Wrap Layer 8's :func:`~app.bot.bot` to handle dict cleanup.

        Pop happens in a ``finally`` so it fires on success, on
        exception, and on :exc:`asyncio.CancelledError`. The 1→0
        boundary check then fires the protection release.

        The bot itself never raises out (it captures everything
        into ``CallRecord.error`` via Layer 8's finally block). This
        wrapper is defensive — if the bot somehow does raise, we
        still clean up the dict so capacity isn't permanently
        consumed.
        """
        try:
            # Pass the boot-time Settings singleton so ``bot`` /
            # ``run_bot`` don't reconstruct it per call (F3).
            await bot(runner_args, self._settings)
        finally:
            self._active_sessions.pop(call_id, None)
            if len(self._active_sessions) == 0 and self._protection.is_protected:
                await self._protection.set_protected(False)
            logger.info(
                "call_finalized",
                call_id=call_id,
                active_sessions=len(self._active_sessions),
            )

    async def _heartbeat_loop(self) -> None:
        """Renew task protection every 30 s while sessions are active.

        Stops itself when ``active_sessions`` empties. The next
        ``_spawn`` 0→1 transition restarts a fresh task.

        Per-iteration error handling: a single transient renew
        failure (network blip, throttling) logs and is retried on
        the next tick. Previously the try/except wrapped the entire
        loop, so the FIRST iteration error killed protection
        permanently for the remainder of all in-flight calls
        (potentially 30 minutes — until ECS Agent expired the
        existing protection). That's the latent bug this guard
        closes.

        Cancellation (process shutdown) returns cleanly.
        """
        while len(self._active_sessions) > 0:
            try:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_SECS)
                if len(self._active_sessions) > 0:
                    await self._protection.renew_if_protected()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "heartbeat_loop_iteration_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                # Continue — next iteration retries.

    async def shutdown(self) -> None:
        """Mark the manager as draining. Called from
        :func:`~app.runner.server.graceful_drain` on SIGTERM.

        After this returns, ``/ready`` returns 503 and ``start_*``
        raises :exc:`CapacityRejected("draining")`.
        """
        self._draining = True
        logger.info(
            "manager_shutdown_initiated",
            active_sessions=self.active_session_count,
        )
