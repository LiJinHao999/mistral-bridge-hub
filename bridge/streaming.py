"""SSE proxies: Mistral conversations stream -> OpenAI Chat / Responses."""

import asyncio
import json
import time

import aiohttp

from .cache import (
    cache_conversation,
    evict_conversation,
    mark_conversation_busy,
    pending_call_ids_from_tool_state,
)
from .config import log
from .sse import iter_conversation_events
from .tools import _json_args_complete
from .translate import (
    _args_fragment,
    _delta_from_message_content,
    _openai_chunk,
    _responses_event,
    cache_log_label,
    chat_usage_from_fields,
    merge_raw_usage,
    mistral_usage_fields,
    responses_usage_from_fields,
    usage_from_event,
)
from .utils import _as_json_string


# ── Streaming ─────────────────────────────────────────────────────────────────
async def stream_response(session, resp, model: str):
    """SSE proxy: Mistral conversations stream -> OpenAI chat.completion.chunk.

    `resp` is already open and answered 200 — the caller keeps the upstream
    request outside this generator so it can still return the real HTTP status.
    Starlette sends the status line before iterating a StreamingResponse, so an
    upstream 402/429 reported from in here would reach AxonHub as a 200 and its
    key pool would never rotate off the exhausted key.
    """
    chunk_id = f"chatcmpl-{int(time.time())}"
    conv_id = None
    try:
        saw_tool_call = False
        sent_role = False
        yielded = 0
        seen_types = []
        tool_names = []
        tool_state = {}
        text_acc = ""
        think_acc = ""
        raw_usage = {}

        def emit(delta: dict):
            nonlocal yielded, sent_role
            if not sent_role:
                delta = dict(delta)
                delta["role"] = "assistant"
                if "tool_calls" in delta and "content" not in delta:
                    delta["content"] = None
                sent_role = True
            yielded += 1
            return f"data: {json.dumps(_openai_chunk(model, delta, chunk_id=chunk_id))}\n\n"

        async for etype, data in iter_conversation_events(resp):
            if etype and len(seen_types) < 8:
                seen_types.append(etype)
            if etype in ("done",):
                break
            if etype == "conversation.response.started":
                conv_id = data.get("conversation_id") or conv_id
                raw_usage = usage_from_event(data) or raw_usage
                continue
            if etype == "conversation.response.error":
                msg = data.get("message") or "upstream stream error"
                log.warning("stream error: %s", msg)
                yield f"data: {json.dumps({'error': {'message': msg, 'type': 'upstream_error'}})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if etype == "conversation.response.done":
                conv_id = data.get("conversation_id") or conv_id
                raw_usage = usage_from_event(data) or raw_usage
                break

            if etype == "message.output.delta":
                delta = _delta_from_message_content(data.get("content"))
                if not delta:
                    continue
                if delta.get("content"):
                    text_acc += delta["content"]
                if delta.get("reasoning_content"):
                    think_acc += delta["reasoning_content"]
                yield emit(delta)
            elif etype == "function.call.delta":
                tcid = data.get("tool_call_id") or data.get("id") or ""
                name = data.get("name") or ""
                incoming = data.get("arguments")
                if incoming is None:
                    incoming = ""
                elif not isinstance(incoming, str):
                    incoming = _as_json_string(incoming)
                if tcid not in tool_state:
                    tool_state[tcid] = {"index": len(tool_state), "args": ""}
                state = tool_state[tcid]
                frag = _args_fragment(state["args"], incoming)
                state["args"] += frag
                if name:
                    state["name"] = name

        ordered = sorted(tool_state.items(), key=lambda kv: kv[1]["index"])
        dropped = []
        for tcid, state in ordered:
            name = state.get("name") or ""
            args = state.get("args") or "{}"
            if not name:
                continue
            # A token limit truncates arguments without any error event.
            # Handing the client half-written JSON is worse than dropping
            # the call: a cut heredoc or patch gets executed as-is.
            if not _json_args_complete(args):
                log.warning(
                    "drop truncated tool call id=%s name=%s len=%d",
                    tcid, name, len(args),
                )
                dropped.append(name)
                continue
            saw_tool_call = True
            tool_names.append(name)
            preview = args if len(args) < 200 else args[:200] + "…"
            log.info("tool call id=%s name=%s args=%s", tcid, name, preview)
            yield emit({
                "tool_calls": [{
                    "index": state["index"],
                    "id": tcid,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }]
            })

        if dropped:
            # Keep the turn non-empty and tell the model why, so the
            # caller's loop can correct instead of replaying the turn.
            note = (
                "[bridge] Dropped truncated tool call(s): %s. The arguments were "
                "cut off by the upstream token limit, so the call was never valid "
                "JSON. Retry in smaller chunks." % ", ".join(dropped)
            )
            yield emit({"content": ("\n\n" + note) if text_acc else note})

        if not sent_role:
            yield emit({"content": ""})
        finish_reason = "tool_calls" if saw_tool_call else "stop"
        fields = mistral_usage_fields(raw_usage, conv_id)
        usage = chat_usage_from_fields(fields)
        log.info(
            "stream done conv=%s types=%s chunks=%d finish=%s tools=%s usage=%s think=%d raw_usage_keys=%s",
            conv_id, seen_types, yielded, finish_reason, tool_names,
            f"{usage['prompt_tokens']}+{usage['completion_tokens']}={usage['total_tokens']}"
            f" {cache_log_label(fields)}",
            len(think_acc),
            sorted((raw_usage or {}).keys()),
        )
        yield f"data: {json.dumps(_openai_chunk(model, {}, finish_reason, chunk_id, usage))}\n\n"
        yield "data: [DONE]\n\n"
        return
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.error("connection error mid-stream: %s", e)
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_unreachable'}})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        resp.release()
        await session.close()


async def stream_responses(session, resp, reason, opener, model: str, conv_id: str = None, entries: list = None):
    """SSE proxy: Mistral conversations stream -> OpenAI Responses events.

    Clients treat a stream without output_text.done / function_call_arguments.done
    / response.completed as incomplete and retry the same turn, so every exit
    here ends in response.completed or response.failed. A turn that already
    streamed content and then lost the socket is completed with what arrived —
    failing it mid-agent is what breaks the caller's loop.

    `resp` is already open and answered 200. The caller runs the upstream
    request itself so it can still choose the HTTP status: Starlette sends the
    status line before it iterates this generator, so a 402/429 discovered in
    here could only ever be a 200 carrying response.failed — which AxonHub
    reads as a successful turn and never rotates the exhausted key on.
    """
    terminal = False
    last_err = None

    # Stream state. created_sent / seq span attempts: a retry that has not
    # emitted anything yet continues the same logical stream, it does not
    # restart it with a second response.created.
    created_sent = False
    seq = 0
    created_at = int(time.time())
    rid = ""

    next_out = 0
    text_item_id = "msg_0"
    reason_item_id = "rs_0"
    text_index = 0
    reason_index = 0
    text_started = False
    reason_started = False
    text_acc = ""
    reason_acc = ""
    tool_state = {}
    seen_types = []
    raw_usage = {}

    def reset_turn():
        nonlocal next_out, text_index, reason_index, text_started, reason_started
        nonlocal text_acc, reason_acc, tool_state, seen_types, raw_usage
        next_out = 0
        text_index = 0
        reason_index = 0
        text_started = False
        reason_started = False
        text_acc = ""
        reason_acc = ""
        tool_state = {}
        seen_types = []
        raw_usage = {}

    def emitted_output() -> bool:
        return text_started or reason_started or bool(tool_state)

    def next_seq():
        nonlocal seq
        seq += 1
        return seq

    def take_index():
        nonlocal next_out
        idx = next_out
        next_out += 1
        return idx

    def failed(message: str, rid_: str = ""):
        return _responses_event("response.failed", {
            "type": "response.failed",
            "response": {"id": rid_ or rid, "error": {"message": message}},
        })

    def response_obj(status: str, output=None, usage=None):
        obj = {
            "id": rid or f"resp_{int(time.time())}",
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": status,
            "output": output if output is not None else [],
        }
        if usage is not None:
            obj["usage"] = usage
        return obj

    def emit_text_note(note: str):
        """Add a bridge note to the assistant text, opening the item if needed.

        A turn whose only output was a dropped tool call would otherwise stream
        nothing, and the client reads a zero-output turn as an empty response
        and retries it. The note also tells the model what went wrong, so the
        agent loop can correct itself instead of repeating the same call.
        """
        nonlocal text_started, text_index, text_acc
        chunk = note if not text_acc or text_acc.endswith("\n") else "\n\n" + note
        text_acc += chunk
        if not text_started:
            text_started = True
            text_index = take_index()
            yield _responses_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": text_index,
                "item": {
                    "type": "message",
                    "id": text_item_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [{"type": "output_text", "text": ""}],
                },
                "sequence_number": next_seq(),
            })
            yield _responses_event("response.content_part.added", {
                "type": "response.content_part.added",
                "item_id": text_item_id,
                "output_index": text_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
                "sequence_number": next_seq(),
            })
        yield _responses_event("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": text_item_id,
            "output_index": text_index,
            "content_index": 0,
            "delta": chunk,
            "sequence_number": next_seq(),
        })

    def ensure_created():
        nonlocal created_sent
        if created_sent:
            return []
        created_sent = True
        obj = response_obj("in_progress")
        return [
            _responses_event("response.created", {
                "type": "response.created",
                "response": obj,
            }),
            _responses_event("response.in_progress", {
                "type": "response.in_progress",
                "response": obj,
            }),
        ]

    def finalize(partial: bool = False):
        """Close the turn: done events, cache update, response.completed."""
        nonlocal rid, terminal
        for ev in ensure_created():
            yield ev

        # A cut stream leaves the last arguments blob half written. Upstream has
        # already stored it on this conversation, and replays it to the gateway
        # on every later append as a permanent 400, so the thread is burnt:
        # drop the call, say why, and never append here again. This holds even
        # when the stream ended cleanly — a token limit truncates arguments
        # without any error event.
        truncated = [
            state.get("name")
            for _, state in sorted(tool_state.items(), key=lambda kv: kv[1]["index"])
            if state.get("name") and not _json_args_complete(state.get("args") or "{}")
        ]
        if truncated:
            evict_conversation(rid)
            for ev in emit_text_note(
                "[bridge] Dropped truncated tool call(s): %s. The arguments were cut "
                "off by the upstream token limit, so the call was never valid JSON. "
                "Retry in smaller chunks." % ", ".join(truncated)
            ):
                yield ev

        # A reasoning item is not visible output: a turn carrying nothing else
        # reaches the client as an empty response and it reports "no visible
        # output". Say so instead, and do not promote the thinking to text —
        # that text would be stored on the conversation and replayed on every
        # later append, so a long ramble would cost tokens for the rest of the
        # thread and invite the agent to act on half-formed reasoning.
        # A cut stream is different and is failed below, where retrying helps.
        surviving_tools = [
            state for _, state in tool_state.items()
            if state.get("name") and _json_args_complete(state.get("args") or "{}")
        ]
        if (
            not partial
            and not text_acc.strip()
            and not surviving_tools
            and reason_acc.strip()
        ):
            log.warning(
                "turn produced only reasoning (%d chars) → note instead of empty turn",
                len(reason_acc),
            )
            for ev in emit_text_note(
                "[bridge] The model produced only reasoning this turn — no reply "
                "text and no tool call. Ask again to get an actual answer."
            ):
                yield ev

        indexed = []
        if reason_started:
            reason_item = {
                "type": "reasoning",
                "id": reason_item_id,
                "summary": [{"type": "summary_text", "text": reason_acc}],
            }
            indexed.append((reason_index, reason_item))
            yield _responses_event("response.reasoning_summary_text.done", {
                "type": "response.reasoning_summary_text.done",
                "item_id": reason_item_id,
                "output_index": reason_index,
                "summary_index": 0,
                "text": reason_acc,
                "sequence_number": next_seq(),
            })
            yield _responses_event("response.reasoning_summary_part.done", {
                "type": "response.reasoning_summary_part.done",
                "item_id": reason_item_id,
                "output_index": reason_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": reason_acc},
                "sequence_number": next_seq(),
            })
            yield _responses_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": reason_index,
                "item": reason_item,
                "sequence_number": next_seq(),
            })

        if text_started:
            text_item = {
                "type": "message",
                "id": text_item_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text_acc}],
            }
            indexed.append((text_index, text_item))
            yield _responses_event("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": text_item_id,
                "output_index": text_index,
                "content_index": 0,
                "text": text_acc,
                "sequence_number": next_seq(),
            })
            yield _responses_event("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": text_item_id,
                "output_index": text_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text_acc},
                "sequence_number": next_seq(),
            })
            yield _responses_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": text_index,
                "item": text_item,
                "sequence_number": next_seq(),
            })

        tool_names = []
        for tcid, state in sorted(tool_state.items(), key=lambda kv: kv[1]["index"]):
            name = state.get("name") or ""
            args = state.get("args") or "{}"
            if not name:
                continue
            # Handing the client truncated JSON is worse than dropping the call:
            # a half-written heredoc or patch is executed as-is. Already
            # reported and evicted above.
            if not _json_args_complete(args):
                log.warning(
                    "drop truncated tool call id=%s name=%s len=%d partial=%s",
                    tcid, name, len(args), partial,
                )
                continue
            tool_names.append(name)
            if state.get("output_index") is None:
                state["output_index"] = take_index()
            idx = state["output_index"]
            item = {
                "type": "function_call",
                "id": state["item_id"],
                "call_id": tcid,
                "name": name,
                "arguments": args,
            }
            indexed.append((idx, item))
            preview = args if len(args) < 200 else args[:200] + "…"
            log.info("tool call id=%s name=%s args=%s", tcid, name, preview)
            if not state.get("announced"):
                yield _responses_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": {**item, "arguments": ""},
                    "sequence_number": next_seq(),
                })
            yield _responses_event("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": state["item_id"],
                "output_index": idx,
                "name": name,
                "arguments": args,
                "sequence_number": next_seq(),
            })
            yield _responses_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": idx,
                "item": item,
                "sequence_number": next_seq(),
            })

        items = [it for _, it in sorted(indexed, key=lambda p: p[0])]
        rid = rid or f"resp_{int(time.time())}"
        fields = mistral_usage_fields(raw_usage, rid)
        usage = responses_usage_from_fields(fields)
        log.info(
            "responses done conv=%s types=%s tools=%s reason=%s usage=%s think=%d partial=%s raw_usage=%s",
            rid, seen_types, tool_names, reason,
            f"{usage['input_tokens']}+{usage['output_tokens']}={usage['total_tokens']}"
            f" {cache_log_label(fields)}",
            len(reason_acc),
            partial,
            json.dumps(raw_usage, ensure_ascii=False)[:400] if raw_usage else "{}",
        )
        terminal = True
        if partial:
            # Truncated tool calls already evicted the thread above. For a
            # thinking-only or text cut, keep the previous cache record so the
            # next turn can still append — dropping it is what made every
            # disconnect throw away the whole prompt-cache chain.
            # Reasoning alone is not something the client can show or act on,
            # so it must not count as output here: reporting completed made the
            # caller treat a cut turn as a successful empty one and never retry.
            if not text_started and not tool_names:
                yield failed("upstream stream ended before any visible output")
                return
            if entries and rid and not truncated:
                cache_conversation(
                    rid,
                    entries,
                    pending_call_ids_from_tool_state(tool_state),
                    prompt_tokens=usage.get("input_tokens") or 0,
                )
        elif entries and rid and not truncated:
            # `truncated` was evicted above — caching it back would hand the
            # next turn the same poisoned conversation.
            cache_conversation(
                rid,
                entries,
                pending_call_ids_from_tool_state(tool_state),
                prompt_tokens=usage.get("input_tokens") or 0,
            )
        yield _responses_event("response.completed", {
            "type": "response.completed",
            "response": response_obj("completed", items, usage),
        })

    reopened = False
    try:
        while True:
            got_event = False
            reset_turn()
            # Never advertise a client-local id on a create fallback.
            rid = conv_id if reason == "append" else ""
            try:
                log.info("responses stream %s", reason)
                async for etype, data in iter_conversation_events(resp):
                    got_event = True
                    if etype and len(seen_types) < 8:
                        seen_types.append(etype)
                    if etype == "conversation.response.started":
                        rid = data.get("conversation_id") or rid
                        raw_usage = merge_raw_usage(raw_usage, usage_from_event(data))
                        for ev in ensure_created():
                            yield ev
                        continue
                    if etype == "conversation.response.error":
                        msg = data.get("message") or "upstream stream error"
                        log.warning("stream error: %s", msg)
                        if emitted_output():
                            for ev in finalize(partial=True):
                                yield ev
                            return
                        terminal = True
                        yield failed(msg, rid)
                        return
                    if etype in ("conversation.response.done", "done"):
                        rid = (data or {}).get("conversation_id") or rid
                        raw_usage = merge_raw_usage(raw_usage, usage_from_event(data))
                        break

                    for ev in ensure_created():
                        yield ev

                    if etype == "message.output.delta":
                        delta = _delta_from_message_content(data.get("content"))
                        if not delta:
                            continue
                        thought = delta.get("reasoning_content") or ""
                        if thought:
                            reason_acc += thought
                            if not reason_started:
                                reason_started = True
                                reason_index = take_index()
                                item = {
                                    "type": "reasoning",
                                    "id": reason_item_id,
                                    "summary": [{"type": "summary_text", "text": ""}],
                                }
                                yield _responses_event("response.output_item.added", {
                                    "type": "response.output_item.added",
                                    "output_index": reason_index,
                                    "item": item,
                                    "sequence_number": next_seq(),
                                })
                                yield _responses_event("response.reasoning_summary_part.added", {
                                    "type": "response.reasoning_summary_part.added",
                                    "item_id": reason_item_id,
                                    "output_index": reason_index,
                                    "summary_index": 0,
                                    "part": {"type": "summary_text", "text": ""},
                                    "sequence_number": next_seq(),
                                })
                            yield _responses_event("response.reasoning_summary_text.delta", {
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": reason_item_id,
                                "output_index": reason_index,
                                "summary_index": 0,
                                "delta": thought,
                                "sequence_number": next_seq(),
                            })
                        text = delta.get("content") or ""
                        if not text:
                            continue
                        text_acc += text
                        if not text_started:
                            text_started = True
                            text_index = take_index()
                            item = {
                                "type": "message",
                                "id": text_item_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [{"type": "output_text", "text": ""}],
                            }
                            yield _responses_event("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": text_index,
                                "item": item,
                                "sequence_number": next_seq(),
                            })
                            yield _responses_event("response.content_part.added", {
                                "type": "response.content_part.added",
                                "item_id": text_item_id,
                                "output_index": text_index,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": ""},
                                "sequence_number": next_seq(),
                            })
                        yield _responses_event("response.output_text.delta", {
                            "type": "response.output_text.delta",
                            "item_id": text_item_id,
                            "output_index": text_index,
                            "content_index": 0,
                            "delta": text,
                            "sequence_number": next_seq(),
                        })
                    elif etype == "function.call.delta":
                        tcid = data.get("tool_call_id") or data.get("id") or ""
                        name = data.get("name") or ""
                        incoming = data.get("arguments")
                        if incoming is None:
                            incoming = ""
                        elif not isinstance(incoming, str):
                            incoming = _as_json_string(incoming)
                        if tcid not in tool_state:
                            tool_state[tcid] = {
                                "index": len(tool_state),
                                "args": "",
                                "item_id": tcid or f"fc_{len(tool_state)}",
                                "announced": False,
                                "output_index": None,
                            }
                        state = tool_state[tcid]
                        frag = _args_fragment(state["args"], incoming)
                        state["args"] += frag
                        if name:
                            state["name"] = name
                        if not state["announced"] and state.get("name"):
                            state["announced"] = True
                            state["output_index"] = take_index()
                            yield _responses_event("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": state["output_index"],
                                "item": {
                                    "type": "function_call",
                                    "id": state["item_id"],
                                    "call_id": tcid,
                                    "name": state["name"],
                                    "arguments": "",
                                },
                                "sequence_number": next_seq(),
                            })
                        if frag and state.get("output_index") is not None:
                            yield _responses_event("response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta",
                                "item_id": state["item_id"],
                                "output_index": state["output_index"],
                                "delta": frag,
                                "sequence_number": next_seq(),
                            })

                for ev in finalize():
                    yield ev
                return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                log.error(
                    "connection error: %s (reason=%s got_event=%s emitted=%s reopened=%s)",
                    e, reason, got_event, emitted_output(), reopened,
                )
                if emitted_output():
                    # Tokens are already on the wire; a retry would duplicate
                    # them. Close the turn with what arrived.
                    log.warning("upstream cut mid-stream → completing partial turn")
                    for ev in finalize(partial=True):
                        yield ev
                    return
                # Nothing reached the client yet, so the turn can still be
                # served by reopening once. The status line is already 200 by
                # now, so a second failure can only be reported in-band.
                if reopened:
                    break
                reopened = True
                resp.release()
                try:
                    resp, reason, err_text = await opener()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e2:
                    last_err = e2
                    resp = None
                    break
                if resp is None or resp.status != 200:
                    if resp is not None:
                        last_err = Exception(err_text[:200] or f"upstream {resp.status}")
                        resp.release()
                        resp = None
                    break
                log.warning("stream reopened after disconnect (%s)", reason)
                continue
        if not terminal:
            terminal = True
            yield failed(
                str(last_err) if last_err else "upstream stream ended without a terminal event"
            )
    finally:
        if resp is not None:
            resp.release()
        await session.close()
        if conv_id:
            mark_conversation_busy(conv_id, False)

