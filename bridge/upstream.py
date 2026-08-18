"""HTTP client for Mistral Conversations create / append / GET passthrough."""

import asyncio
import json

import aiohttp

from .cache import evict_conversation
from .config import (
    APPEND_CONFLICT_BACKOFF,
    APPEND_CONFLICT_RETRIES,
    CREATE_CONNECT_RETRIES,
    CREATE_RETRY_BACKOFF,
    MISTRAL_API,
    MISTRAL_BASE,
    UPSTREAM_TIMEOUT,
    log,
)
from .utils import _auth_headers, resolve_key


def strip_for_append(payload: dict) -> dict:
    """AppendConversationRequest forbids model/instructions/tools."""
    out = {"inputs": payload.get("inputs") or []}
    if payload.get("completion_args"):
        out["completion_args"] = payload["completion_args"]
    if "store" in payload:
        out["store"] = payload["store"]
    if payload.get("stream"):
        out["stream"] = True
    if payload.get("handoff_execution"):
        out["handoff_execution"] = payload["handoff_execution"]
    if payload.get("tool_confirmations"):
        out["tool_confirmations"] = payload["tool_confirmations"]
    return out


def concurrent_conversation(status: int, text: str) -> bool:
    """True when append raced another in-flight update (often also code 3000)."""
    if status == 409:
        return True
    lowered = (text or "").lower()
    return "concurrent" in lowered or "being updated" in lowered


def missing_conversation(status: int, text: str) -> bool:
    """True when append targeted an id Mistral does not have."""
    if status == 404:
        return True
    if concurrent_conversation(status, text):
        return False
    try:
        obj = json.loads(text or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        obj = None
    if isinstance(obj, dict):
        msg = str(obj.get("message") or "")
        if "was not found" in msg or "Conversation with id" in msg:
            return True
    lowered = (text or "").lower()
    return "was not found" in lowered or "conversation with id" in lowered


def upstream_history_corrupt(status: int, text: str) -> bool:
    """True when the stored conversation can no longer be replayed at all.

    A tool call whose arguments were cut mid-string stays on the Mistral
    conversation forever. Every later append replays it to the model gateway,
    which fails to json.loads it and answers 400 ThirdPartyException. Nothing
    can repair the thread from here, so the only way out is to drop it and
    create; otherwise every following turn 400s and the client sees an empty
    stream.
    """
    if status != 400:
        return False
    lowered = (text or "").lower()
    return any(marker in lowered for marker in (
        "unterminated string",
        "expecting value",
        "expecting ',' delimiter",
        "expecting ':' delimiter",
        "expecting property name",
        "invalid control character",
        "invalid \\escape",
        "extra data",
    ))


def append_state_mismatch(status: int, text: str) -> bool:
    """True when this conversation cannot accept the append payload."""
    if status != 400:
        return False
    if upstream_history_corrupt(status, text):
        return True
    lowered = (text or "").lower()
    return (
        "function results are still missing" in lowered
        or "cannot append other inputs" in lowered
        # Local prefix is stale: those tool_call_ids were already consumed
        # or never existed on this conversation. Evict and create, or the
        # client retries the same append forever.
        or "already have a result" in lowered
        or "already has a result" in lowered
        or "unknown tool_call_ids" in lowered
        or "unknown tool_call_id" in lowered
    )


def append_backoff(attempt: int) -> float:
    """Exponential: Mistral can hold a conversation for several seconds after
    our stream ends, and waiting is far cheaper than falling back to create,
    which throws away the whole prompt cache."""
    return APPEND_CONFLICT_BACKOFF * (2 ** (attempt - 1))


def append_should_fallback(status: int, text: str) -> bool:
    return missing_conversation(status, text) or append_state_mismatch(status, text)


def conversation_attempts(payload: dict, conv_id: str, append_payload: dict = None):
    """(url, body, reason) pairs. Append first when an id is present; create is fallback."""
    if conv_id:
        yield f"{MISTRAL_BASE}/{conv_id}", (append_payload or strip_for_append(payload)), "append"
    yield MISTRAL_BASE, payload, "create"


def is_transient_upstream(err: BaseException) -> bool:
    """True when Mistral dropped the socket before a usable HTTP/SSE body."""
    if isinstance(err, (
        aiohttp.ClientConnectorError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientPayloadError,
        aiohttp.ClientOSError,
        asyncio.TimeoutError,
        TimeoutError,
    )):
        return True
    msg = str(err).lower()
    return (
        "disconnected" in msg
        or "cannot connect" in msg
        or "not enough data" in msg
        or "timeout" in msg
        or "transfer encoding" in msg
    )


# ── Upstream POST (create / append / one create-after-miss) ───────────────────
async def post_conversation(session, key: str, payload: dict, conv_id: str = None, append_payload: dict = None):
    """POST create or append. On append miss / tool-state mismatch, retry as create.

    Concurrent 409 is retried as append, not treated as a missing conversation.
    Returns (resp, reason, error_text). Caller must read resp / close it.
    """
    last_text = ""
    resp = None
    for url, body, reason in conversation_attempts(payload, conv_id, append_payload):
        tries = CREATE_CONNECT_RETRIES if reason == "create" else APPEND_CONFLICT_RETRIES
        for attempt in range(1, tries + 1):
            try:
                resp = await session.post(
                    url, headers=_auth_headers(key), json=body, timeout=UPSTREAM_TIMEOUT,
                )
                if resp.status == 200:
                    if reason == "create" and conv_id:
                        # Why append gave up was already logged at that point;
                        # do not relabel every fallback as a missing conversation.
                        log.info("append %s → create", conv_id)
                    return resp, reason, ""
                last_text = await resp.text()
                log.warning("upstream %s %s: %s", reason, resp.status, last_text[:200])
                if reason == "append" and concurrent_conversation(resp.status, last_text):
                    if attempt < tries:
                        log.info(
                            "append %s concurrent, retry in %ss",
                            conv_id, append_backoff(attempt),
                        )
                        await asyncio.sleep(append_backoff(attempt))
                        continue
                    # Another turn owns this conversation. Creating costs a
                    # cache miss; returning the 409 costs the caller its turn.
                    log.info("append %s still busy after %d tries → create", conv_id, tries)
                    evict_conversation(conv_id)
                    resp.release()
                    break
                if reason == "append" and append_should_fallback(resp.status, last_text):
                    if upstream_history_corrupt(resp.status, last_text):
                        evict_conversation(conv_id)
                        log.warning("append %s history corrupt → create", conv_id)
                    elif append_state_mismatch(resp.status, last_text):
                        evict_conversation(conv_id)
                        log.info("append %s state mismatch → create", conv_id)
                    else:
                        log.info("append %s missing → create", conv_id)
                    break
                return resp, reason, last_text
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.error("connection error: %s (reason=%s attempt=%d/%d)", e, reason, attempt, tries)
                if is_transient_upstream(e) and attempt < tries:
                    await asyncio.sleep(CREATE_RETRY_BACKOFF * attempt)
                    continue
                if reason == "append":
                    # Nothing was delivered yet, so create can still serve this turn.
                    log.info("append %s unreachable → create", conv_id)
                    break
                raise
    return resp, "create", last_text


async def proxy_mistral_get(request: Request, path: str, params=None):
    """GET passthrough to api.mistral.ai. Returns (status, body) or (None, None)."""
    key = resolve_key(request)
    if not key:
        return None, None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{MISTRAL_API}{path}",
                headers={"Authorization": f"Bearer {key}"},
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return resp.status, await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("upstream GET %s: %s", path, e)
        return None, None

