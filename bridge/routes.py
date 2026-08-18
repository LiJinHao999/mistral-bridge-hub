"""HTTP routes: Chat Completions, Responses, models, health."""

import asyncio
import json

import aiohttp
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from .cache import (
    cache_conversation,
    find_append_match,
    mark_conversation_busy,
    pending_call_ids_from_outputs,
)
from .config import MODEL, MISTRAL_BASE, PORT, UPSTREAM_TIMEOUT, log
from .models import local_model_card, local_models_list
from .streaming import stream_response, stream_responses
from .translate import (
    cache_log_label,
    extract_previous_id,
    mistral_to_openai,
    mistral_to_responses,
    mistral_usage_fields,
    openai_to_mistral,
    responses_to_mistral,
    usage_from_event,
)
from .upstream import post_conversation, proxy_mistral_get, strip_for_append
from .utils import (
    _auth_headers,
    content_to_text,
    error_response,
    input_kinds,
    resolve_key,
    resolve_model,
)

router = APIRouter()


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response(400, "Invalid JSON body", "bad_request")

    model = resolve_model(body.get("model"))
    try:
        payload = openai_to_mistral(body)
    except Exception as e:
        return error_response(422, f"Payload translation error: {e}", "invalid_payload")

    upstream_key = resolve_key(request)
    if not upstream_key:
        return error_response(401, "Missing API key (client Authorization or MISTRAL_KEY)", "missing_api_key")

    raw_msgs = body.get("messages") if isinstance(body.get("messages"), list) else []
    raw_roles = []
    user_preview = []
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "?"
        extra = ""
        if role == "assistant" and m.get("tool_calls"):
            extra = f"+{len(m.get('tool_calls') or [])}tc"
        raw_roles.append(f"{role}{extra}")
        if role == "user":
            txt = content_to_text(m.get("content")).replace("\n", " ")
            user_preview.append(txt[:80] + ("…" if len(txt) > 80 else ""))
    log.info(
        "chat → %s raw=%s users=%s inputs=%s tools=%d reason=%s max_tokens=%s store=%s",
        model,
        raw_roles,
        user_preview,
        input_kinds(payload.get("inputs")),
        len(payload.get("tools") or []),
        payload.get("completion_args", {}).get("reasoning_effort"),
        payload.get("completion_args", {}).get("max_tokens"),
        payload.get("store"),
    )

    if body.get("stream"):
        payload["stream"] = True

        # Same reason as /v1/responses: the status line is committed before the
        # generator runs, so the upstream request has to happen out here for a
        # 402/429 to reach AxonHub as a real status instead of a 200.
        session = aiohttp.ClientSession()
        try:
            resp = await session.post(
                MISTRAL_BASE,
                headers=_auth_headers(upstream_key),
                json=payload,
                timeout=UPSTREAM_TIMEOUT,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.error("connection error: %s", e)
            await session.close()
            return error_response(502, str(e), "upstream_unreachable")
        if resp.status != 200:
            text = await resp.text()
            log.warning("upstream %s: %s", resp.status, text[:200])
            resp.release()
            await session.close()
            return error_response(
                resp.status if resp.status < 500 else 502, text[:500]
            )

        return StreamingResponse(
            stream_response(session, resp, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                MISTRAL_BASE,
                headers=_auth_headers(upstream_key),
                json=payload,
                timeout=UPSTREAM_TIMEOUT,
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    log.warning("upstream %s: %s", resp.status, text[:200])
                    return error_response(
                        resp.status if resp.status < 500 else 502,
                        text[:500],
                    )
                data = json.loads(text)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.error("connection error: %s", e)
            return error_response(502, str(e), "upstream_unreachable")
        except json.JSONDecodeError:
            log.error("bad upstream JSON")
            return error_response(502, "Bad upstream response")

    translated = mistral_to_openai(data, model)
    usage = translated.get("usage") or {}
    fields = mistral_usage_fields(usage_from_event(data), translated.get("id"))
    log.info(
        "chat done conv=%s usage=%s raw_usage_keys=%s",
        translated.get("id"),
        f"{usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)}={usage.get('total_tokens', 0)}"
        f" {cache_log_label(fields)}",
        sorted((usage_from_event(data) or {}).keys()),
    )
    return Response(
        status_code=200,
        media_type="application/json",
        content=json.dumps(translated),
    )


@router.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response(400, "Invalid JSON body", "bad_request")

    model = resolve_model(body.get("model"))
    try:
        payload, entries = responses_to_mistral(body)
    except Exception as e:
        return error_response(422, f"Payload translation error: {e}", "invalid_payload")

    upstream_key = resolve_key(request)
    if not upstream_key:
        return error_response(401, "Missing API key (client Authorization or MISTRAL_KEY)", "missing_api_key")

    client_prev = extract_previous_id(body, request)
    prev = None
    append_payload = None
    match = find_append_match(entries)
    if match:
        prev = match[0]
        append_payload = strip_for_append(payload)
        append_payload["inputs"] = match[1]
        # Held until the stream generator (or the non-stream branch) finishes,
        # so a parallel turn creates its own thread instead of racing this one
        # into 409 'Conversation is being updated'.
        mark_conversation_busy(prev, True)
        log.info(
            "append match conv=%s new_entries=%d total_entries=%d kinds=%s client_prev=%s",
            prev, len(match[1]), len(entries), input_kinds(match[1]), client_prev,
        )
    elif client_prev:
        # Have an id but nothing this conversation can accept next. Appending
        # the full window while tools are pending is a 400; create instead.
        log.info("append skip unmatched client_prev=%s → create", client_prev)

    raw_in = body.get("input")
    raw_kinds = []
    if isinstance(raw_in, list):
        for item in raw_in:
            if isinstance(item, dict):
                raw_kinds.append(item.get("type") or item.get("role") or "?")
            elif isinstance(item, str):
                raw_kinds.append("str")
    elif isinstance(raw_in, str):
        raw_kinds = ["str"]
    first_preview = ""
    for item in payload.get("inputs") or []:
        if isinstance(item, dict) and item.get("role") == "user":
            first_preview = (item.get("content") or "")[:80]
            break
    log.info(
        "responses → %s raw=%s inputs=%s first=%s tools=%s reason=%s max_tokens=%s prev=%s store=%s",
        model,
        raw_kinds,
        input_kinds(payload.get("inputs")),
        first_preview,
        len(payload.get("tools") or []),
        payload.get("completion_args", {}).get("reasoning_effort"),
        payload.get("completion_args", {}).get("max_tokens"),
        prev,
        payload.get("store"),
    )

    if body.get("stream"):
        payload["stream"] = True
        if append_payload:
            append_payload["stream"] = True

        # Run the upstream request here, not inside the generator: Starlette
        # sends the status line before it iterates a StreamingResponse, so an
        # upstream 402/429 discovered in there can only be a 200 carrying
        # response.failed. AxonHub holds the key pool and rotates on the HTTP
        # status, so swallowing 402 pins it to an exhausted key forever.
        session = aiohttp.ClientSession()
        opener = lambda: post_conversation(  # noqa: E731 - reopen after a mid-stream cut
            session, upstream_key, payload, prev, append_payload
        )
        try:
            resp, reason, err_text = await opener()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.error("connection error: %s", e)
            await session.close()
            mark_conversation_busy(prev, False)
            return error_response(502, str(e), "upstream_unreachable")
        if resp is None or resp.status != 200:
            status = resp.status if resp is not None else 502
            text = err_text or (await resp.text() if resp is not None else "upstream unreachable")
            if resp is not None:
                resp.release()
            await session.close()
            mark_conversation_busy(prev, False)
            return error_response(status if status < 500 else 502, text[:500])

        return StreamingResponse(
            stream_responses(session, resp, reason, opener, model, prev, entries),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with aiohttp.ClientSession() as session:
        try:
            resp, reason, err_text = await post_conversation(session, upstream_key, payload, prev, append_payload)
            try:
                if resp.status != 200:
                    return error_response(
                        resp.status if resp.status < 500 else 502,
                        (err_text or await resp.text())[:500],
                    )
                data = json.loads(await resp.text())
            finally:
                resp.release()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.error("connection error: %s", e)
            return error_response(502, str(e), "upstream_unreachable")
        except json.JSONDecodeError:
            log.error("bad upstream JSON")
            return error_response(502, "Bad upstream response")
        finally:
            mark_conversation_busy(prev, False)

    translated = mistral_to_responses(data, model)
    usage = translated.get("usage") or {}
    raw_usage = usage_from_event(data) or {}
    fields = mistral_usage_fields(raw_usage, translated.get("id"))
    cache_conversation(
        translated.get("id"),
        entries,
        pending_call_ids_from_outputs(data),
        prompt_tokens=usage.get("input_tokens") or 0,
    )
    log.info(
        "responses done conv=%s reason=%s usage=%s raw_usage=%s",
        translated.get("id"),
        reason,
        f"{usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)}={usage.get('total_tokens', 0)}"
        f" {cache_log_label(fields)}",
        json.dumps(raw_usage, ensure_ascii=False)[:400] if raw_usage else "{}",
    )
    return Response(
        status_code=200,
        media_type="application/json",
        content=json.dumps(translated),
    )


@router.get("/v1/models")
async def models(request: Request):
    """Passthrough Mistral /v1/models; fall back to the configured local model."""
    params = {}
    for key in ("provider", "model"):
        val = request.query_params.get(key)
        if val:
            params[key] = val
    status, text = await proxy_mistral_get(request, "/models", params or None)
    if status == 200 and text:
        return Response(status_code=200, media_type="application/json", content=text)
    if status and status != 200:
        log.warning("upstream /models %s: %s", status, (text or "")[:200])
    return local_models_list()


@router.get("/v1/models/{model_id:path}")
async def model_retrieve(model_id: str, request: Request):
    """Passthrough Mistral /v1/models/{id}; fall back to a local card."""
    wanted = resolve_model(model_id)
    status, text = await proxy_mistral_get(request, f"/models/{wanted}")
    if status == 200 and text:
        return Response(status_code=200, media_type="application/json", content=text)
    if wanted == MODEL or not status:
        return local_model_card(wanted)
    return error_response(status if status < 500 else 502, (text or "model not found")[:500])


@router.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "port": PORT}

