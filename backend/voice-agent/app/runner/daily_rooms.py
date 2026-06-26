"""Daily.co REST API client — room creation + meeting-token minting.

Layer 9 absorbs Daily room creation that v1 delegated to a separate
BotRunner Lambda. Three room shapes for the three call directions
v2 supports:

* :meth:`DailyRoomClient.create_inbound_room` — SIP dial-in enabled,
  cloud recording to S3.
* :meth:`DailyRoomClient.create_outbound_room` — dialout enabled,
  cloud recording to S3.
* :meth:`DailyRoomClient.create_browser_room` — WebRTC only, no SIP,
  no recording (test calls).

Plus :meth:`DailyRoomClient.mint_token` for both bot and (browser-
only) viewer tokens.

Recording is room-level config: ``enable_recording: "cloud"`` plus
``recordings_bucket`` (bucket name + region + assume-role ARN) make
Daily upload directly to the customer's S3 on call end. SSE-KMS
encryption is enforced at the bucket level via default encryption;
Layer 9 doesn't pass SSE headers.

Reference: https://docs.daily.co/reference/rest-api/rooms/create-room
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


DAILY_API_BASE = "https://api.daily.co/v1"

# 4 hours — long enough for the longest plausible Cosentus IVR
# session plus margin. ``eject_at_room_exp=true`` kicks any
# remaining participants out at expiry to free Daily resources.
_DEFAULT_TTL_SECS = 14400

# 15 minutes for browser test calls — short enough that test rooms
# don't accumulate, long enough for a thorough QA session.
_BROWSER_TTL_SECS = 900

# Total HTTP timeout per Daily REST request. Daily's API is fast
# (typical room creation < 500 ms), so 10 s is a generous ceiling
# that still fails fast on regional incidents.
_REQUEST_TIMEOUT_SECS = 10


class DailyAPIError(Exception):
    """Raised when Daily REST API returns 4xx/5xx or response is malformed."""


@dataclass(frozen=True)
class DailyRoom:
    """A successfully-created Daily room.

    Attributes:
        url: Full room URL Daily returns
            (e.g. ``https://cosentus.daily.co/<name>``). Layer 8's
            :class:`DailyRunnerArguments.room_url` consumes this.
        name: The room name. Used as ``CallRecord.session_id`` so
            the recording webhook can locate the row to patch
            ``recording_path`` on.
        sip_uri: SIP endpoint URI for inbound rooms (``None`` for
            outbound / browser). Used by Daily's SIP gateway to
            bridge incoming PSTN calls into the room.
    """

    url: str
    name: str
    sip_uri: str | None = None


class DailyRoomClient:
    """Daily REST API client. One instance per process.

    Holds a shared :class:`aiohttp.ClientSession` for connection
    pooling. The session is lazy-initialized on first request and
    closed via :meth:`close` during shutdown.

    Recording configuration (``recording_bucket`` / ``recording_role_arn``)
    is captured at construction time. If both are empty, rooms are
    created without ``recordings_bucket`` and Daily falls back to
    its own storage. Production sets both via Layer 2 ``Settings``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        recording_bucket: str = "",
        recording_role_arn: str = "",
        recording_region: str = "us-east-1",
        api_url: str = DAILY_API_BASE,
    ) -> None:
        if not api_key:
            raise ValueError("DailyRoomClient requires a non-empty api_key")
        self._api_key = api_key
        self._recording_bucket = recording_bucket
        self._recording_role_arn = recording_role_arn
        self._recording_region = recording_region
        self._api_url = api_url
        self._session: aiohttp.ClientSession | None = None
        # E.164 → Daily phone-number-record UUID. Daily's
        # ``dialOut/start`` ``callerId`` field expects the UUID of a
        # purchased-phone-number row, NOT the E.164 string. Filled
        # lazily on first lookup, refreshed on cache miss. See
        # :meth:`get_phone_number_uuid` for rationale and the docs
        # references that established the requirement.
        self._caller_id_cache: dict[str, str] = {}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazy-init the shared HTTP session with the bearer header."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECS),
            )
        return self._session

    def _recording_config(self) -> dict | None:
        """Build the ``recordings_bucket`` property dict, or None.

        ``None`` skips the field entirely so Daily falls back to its
        own recording storage. Production sets both bucket + role.
        """
        if not self._recording_bucket or not self._recording_role_arn:
            return None
        return {
            "bucket_name": self._recording_bucket,
            "bucket_region": self._recording_region,
            "assume_role_arn": self._recording_role_arn,
            # Playback goes through the Cosentus API's S3 presign
            # endpoint; Daily only needs write access to the bucket.
            "allow_api_access": False,
        }

    async def create_inbound_room(self, *, ttl_secs: int = _DEFAULT_TTL_SECS) -> DailyRoom:
        """Create a SIP-dial-in-enabled room for an inbound PSTN call.

        Daily's SIP gateway bridges the caller into the room when
        Layer 9's webhook handler returns the ``sip_uri`` from this
        response. Recording is enabled when the client was
        constructed with bucket + role.
        """
        properties: dict = {
            "exp": int(time.time()) + ttl_secs,
            "eject_at_room_exp": True,
            "sip": {
                "display_name": "Cosentus voice assistant",
                "sip_mode": "dial-in",
                "video": False,
                "num_endpoints": 1,
            },
            "start_audio_off": False,
            "start_video_off": True,
        }
        rec = self._recording_config()
        if rec is not None:
            properties["enable_recording"] = "cloud"
            properties["recordings_bucket"] = rec
        return await self._create_room(properties)

    async def create_outbound_room(self, *, ttl_secs: int = _DEFAULT_TTL_SECS) -> DailyRoom:
        """Create a dialout-enabled room for an outbound PSTN call.

        ``allow_room_start: True`` is critical — without it, Daily
        rejects ``transport.start_dialout`` calls when the room is
        empty (until the bot joins). The bot joins first, then
        dials, so the room IS empty at dialout time.
        """
        properties: dict = {
            "exp": int(time.time()) + ttl_secs,
            "eject_at_room_exp": True,
            "enable_dialout": True,
            "dialout_config": {"allow_room_start": True},
            "start_audio_off": False,
            "start_video_off": True,
        }
        rec = self._recording_config()
        if rec is not None:
            properties["enable_recording"] = "cloud"
            properties["recordings_bucket"] = rec
        return await self._create_room(properties)

    async def create_browser_room(self, *, ttl_secs: int = _BROWSER_TTL_SECS) -> DailyRoom:
        """Create a WebRTC-only room for browser test calls.

        No SIP, no recording. Shorter TTL than PSTN rooms — test
        calls don't run for hours.
        """
        properties: dict = {
            "exp": int(time.time()) + ttl_secs,
            "eject_at_room_exp": True,
            "start_audio_off": False,
            "start_video_off": True,
        }
        return await self._create_room(properties)

    async def _create_room(self, properties: dict) -> DailyRoom:
        """POST /rooms with the supplied properties.

        Generates a UUID name so adjacent calls can't collide on
        room names. ``privacy=private`` means a meeting token is
        required to join.
        """
        session = await self._ensure_session()
        payload = {
            "name": str(uuid.uuid4()),
            "privacy": "private",
            "properties": properties,
        }
        try:
            async with session.post(f"{self._api_url}/rooms", json=payload) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise DailyAPIError(f"Room creation failed: {resp.status} {body[:500]}")
                data = await self._parse_json(body)
        except DailyAPIError:
            raise
        except aiohttp.ClientError as exc:
            raise DailyAPIError(f"Room creation network error: {exc}") from exc

        room = DailyRoom(
            url=data["url"],
            name=data["name"],
            sip_uri=data.get("sip_uri"),
        )
        logger.info(
            "daily_room_created",
            room_name=room.name,
            sip_enabled=room.sip_uri is not None,
            recording_enabled=("enable_recording" in properties),
        )
        return room

    async def mint_token(
        self,
        room_name: str,
        *,
        is_owner: bool = True,
        exp_secs: int = _DEFAULT_TTL_SECS,
    ) -> str:
        """POST /meeting-tokens for the given room.

        Args:
            room_name: The room to issue the token for.
            is_owner: Bot tokens are owners (full transport
                control). Browser viewer tokens pass ``False``.
            exp_secs: Token TTL in seconds. Match or exceed the
                room TTL so the token doesn't expire mid-call.
        """
        session = await self._ensure_session()
        payload = {
            "properties": {
                "room_name": room_name,
                "is_owner": is_owner,
                "exp": int(time.time()) + exp_secs,
            },
        }
        try:
            async with session.post(f"{self._api_url}/meeting-tokens", json=payload) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise DailyAPIError(f"Token mint failed: {resp.status} {body[:500]}")
                data = await self._parse_json(body)
        except DailyAPIError:
            raise
        except aiohttp.ClientError as exc:
            raise DailyAPIError(f"Token mint network error: {exc}") from exc

        return data["token"]

    async def get_phone_number_uuid(self, e164: str) -> str | None:
        """Resolve an E.164 phone number to its Daily purchased-number UUID.

        Daily's ``dialOut/start`` REST endpoint expects the
        ``callerId`` field to carry the UUID of a record in
        ``GET /v1/purchased-phone-numbers``, **not** the E.164
        string. Empirically verified 2026-05-07 — passing the E.164
        form (e.g. ``"+12098210846"``) returns
        ``"Incorrect callerID! No phone number maps to: +12098210846"``;
        passing the matching UUID (e.g.
        ``"c3f81341-7c92-499f-ab37-ecda466e2ab9"``) succeeds. Both
        the Pipecat
        `Daily dial-out guide <https://docs.pipecat.ai/deployment/pipecat-cloud/guides/telephony/daily-dial-out>`_
        and Daily's
        `startDialOut reference <https://docs.daily.co/reference/daily-js/instance-methods/start-dial-out>`_
        document this explicitly: "To specify the caller ID, use
        the phone number's id as the callerId."

        Cache: hit → return immediately. Miss → fetch the full
        ``/v1/purchased-phone-numbers`` page, populate the cache,
        return. Cosentus has 3 purchased numbers today; even if
        Daily's pagination grows the list, ``GET`` is cheap and the
        list rarely changes. ``None`` return signals "no record
        matches this E.164" so the caller can decide whether to
        proceed (Daily will reject the dialout) or fail-fast.

        Args:
            e164: E.164-formatted phone number (e.g. ``+12098210846``).

        Returns:
            The Daily UUID for the matching purchased-number record,
            or ``None`` if no match is found after a fresh fetch.
        """
        cached = self._caller_id_cache.get(e164)
        if cached:
            return cached

        await self._refresh_phone_number_cache()
        return self._caller_id_cache.get(e164)

    async def _refresh_phone_number_cache(self) -> None:
        """Fetch ``/v1/purchased-phone-numbers`` and rebuild the cache.

        Errors are logged but never raised — a refresh failure leaves
        the existing cache intact (graceful degradation). The caller
        of :meth:`get_phone_number_uuid` will see ``None`` for any
        E.164 that wasn't cached and Daily will return the same
        ``Incorrect callerID`` error as before; from the caller's
        perspective the experience is the unchanged E.164-passthrough
        baseline, not a regression.
        """
        session = await self._ensure_session()
        try:
            async with session.get(
                f"{self._api_url}/purchased-phone-numbers",
                params={"limit": 100},
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    logger.error(
                        "phone_number_cache_refresh_failed",
                        status=resp.status,
                        body=body[:200],
                    )
                    return
                data = await self._parse_json(body)
        except aiohttp.ClientError as exc:
            logger.error("phone_number_cache_refresh_network_error", error=str(exc))
            return

        new_cache: dict[str, str] = {}
        for entry in data.get("data", []) or []:
            number = entry.get("number")
            uuid_value = entry.get("id")
            if isinstance(number, str) and isinstance(uuid_value, str) and number and uuid_value:
                new_cache[number] = uuid_value
        self._caller_id_cache = new_cache
        logger.info(
            "phone_number_cache_refreshed",
            entry_count=len(new_cache),
        )

    @staticmethod
    async def _parse_json(body: str) -> dict:
        import json

        try:
            return json.loads(body)
        except (ValueError, json.JSONDecodeError) as exc:
            raise DailyAPIError(f"Daily API returned non-JSON: {body[:200]}") from exc

    async def close(self) -> None:
        """Close the shared HTTP session. Call during shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
