"""Tool-call / tool-result translation into Mistral Conversations entries."""

import json

from .config import THINKING_CREATE_MAX_CHARS, log
from .utils import _as_json_string, content_to_text


def _json_args_complete(args: str) -> bool:
    """True when a tool call's arguments blob is whole enough to hand over."""
    text = (args or "").strip()
    if not text:
        return True
    try:
        json.loads(text)
    except (TypeError, ValueError):
        return False
    return True


def sanitize_tool_args(args, tcid: str = "", name: str = ""):
    """Normalize tool arguments to something that always parses as JSON.

    Mistral stores `arguments` on the conversation forever and replays it to the
    model gateway on every later append. A blob the gateway cannot json.loads is
    a permanent 400 on that conversation (see `upstream_history_corrupt`), so
    nothing malformed may be sent upstream.
    """
    # dicts / lists are structurally valid already, and Mistral accepts them.
    if isinstance(args, (dict, list)):
        return args
    text = _as_json_string(args).strip() or "{}"
    if _json_args_complete(text):
        # Claude Code wraps arguments it could not parse itself. Unwrap it when
        # the raw blob is usable after all, so the model sees real arguments
        # instead of the wrapper.
        try:
            obj = json.loads(text)
        except (TypeError, ValueError):
            obj = None
        if isinstance(obj, dict) and "__unparsedToolInput" in obj:
            wrapper = obj.get("__unparsedToolInput")
            raw = wrapper.get("raw") if isinstance(wrapper, dict) else wrapper
            if isinstance(raw, str) and raw.strip() and _json_args_complete(raw):
                return raw.strip()
        return text
    log.warning(
        "tool args not valid JSON, replaced id=%s name=%s len=%d preview=%r",
        tcid or "?", name or "?", len(text), text[:200],
    )
    return json.dumps({"__truncated__": "arguments were cut off upstream"})


def _preview_payload(value, limit: int = 200) -> tuple:
    """(python_type, keys_or_len, preview) for tool-result diagnostics."""
    if value is None:
        return "none", 0, ""
    if isinstance(value, str):
        return "str", len(value), value[:limit]
    if isinstance(value, (int, float, bool)):
        text = str(value)
        return type(value).__name__, len(text), text
    if isinstance(value, list):
        kinds = []
        for item in value[:6]:
            if isinstance(item, dict):
                kinds.append(item.get("type") or ",".join(list(item)[:4]))
            else:
                kinds.append(type(item).__name__)
        dumped = _as_json_string(value)
        return f"list[{len(value)}:{','.join(kinds)}]", len(dumped), dumped[:limit]
    if isinstance(value, dict):
        keys = ",".join(list(value)[:12])
        dumped = _as_json_string(value)
        return f"dict[{keys}]", len(dumped), dumped[:limit]
    dumped = _as_json_string(value)
    return type(value).__name__, len(dumped), dumped[:limit]


def pick_tool_payload(item: dict) -> tuple:
    """Which field holds the tool result, and its raw value."""
    for key in ("output", "result", "content"):
        if key in item:
            return key, item.get(key)
    return "missing", ""


def tool_result_string(value) -> str:
    """Preserve strings; flatten text blocks; otherwise JSON-dump so nothing is dropped."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    text = content_to_text(value)
    if isinstance(value, list) and text.strip():
        return text
    if isinstance(value, dict) and isinstance(value.get("text"), str) and value["text"].strip():
        return value["text"]
    if text.strip() and not isinstance(value, (dict, list)):
        return text
    if isinstance(value, (dict, list)):
        return _as_json_string(value)
    return text or str(value)


def _function_result_entry(tool_call_id: str, result) -> dict:
    return {
        "type": "function.result",
        "tool_call_id": str(tool_call_id),
        "result": tool_result_string(result),
    }


def emit_function_call(name, tcid, args, call_names=None):
    """Build a Mistral function.call entry and log it. None if name is empty."""
    if not name:
        return None
    tcid = str(tcid or f"call_{name}")
    if isinstance(call_names, dict):
        call_names[tcid] = name
    # Never let a malformed blob reach the conversation: it would be replayed
    # to the gateway on every later append and 400 the thread for good.
    args = sanitize_tool_args(args, tcid, name)
    arg_preview = args if isinstance(args, str) else _as_json_string(args)
    log.info(
        "tool call in name=%s id=%s arg_len=%s preview=%r",
        name,
        tcid,
        len(arg_preview or ""),
        (arg_preview or "")[:200],
    )
    return {
        "type": "function.call",
        "tool_call_id": tcid,
        "name": str(name),
        "arguments": args,
    }


def emit_function_result(item: dict, call_names=None):
    """Build a Mistral function.result (or user fallback) and log the raw payload."""
    tcid = item.get("call_id") or item.get("tool_call_id") or item.get("id") or item.get("name") or ""
    field, result = pick_tool_payload(item)
    kind, nbytes, preview = _preview_payload(result)
    name = ""
    if isinstance(call_names, dict):
        name = call_names.get(str(tcid), "")
    name = name or item.get("name") or ""
    sent = None
    if tcid:
        sent = _function_result_entry(tcid, result)
    else:
        text = tool_result_string(result)
        if text.strip():
            sent = {"role": "user", "content": f"[tool result] {text}"}
    sent_len = len((sent or {}).get("result") or (sent or {}).get("content") or "")
    log.info(
        "tool result in name=%s id=%s field=%s raw=%s raw_len=%s sent_len=%s keys=%s preview=%r",
        name or "?",
        tcid or "?",
        field,
        kind,
        nbytes,
        sent_len,
        ",".join(item.keys()),
        preview,
    )
    return sent


def _flatten_thinking(block) -> str:
    """Pull plain thinking text out of a Responses / Chat / Mistral block."""
    if isinstance(block, str):
        return block.strip()
    if not isinstance(block, dict):
        return ""
    parts = []
    summary = block.get("summary")
    if isinstance(summary, list):
        for item in summary:
            if isinstance(item, str) and item:
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item.get("text") or "")
    nested = block.get("thinking") if block.get("thinking") is not None else block.get("reasoning")
    if isinstance(nested, str) and nested:
        parts.append(nested)
    elif isinstance(nested, list):
        for item in nested:
            if isinstance(item, str) and item:
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item.get("text") or "")
    if not parts:
        if isinstance(block.get("text"), str) and block["text"]:
            parts.append(block["text"])
        elif isinstance(block.get("content"), str) and block["content"]:
            parts.append(block["content"])
    return "".join(parts).strip()


def _thinking_signature(block: dict) -> str:
    """Mistral signature / OpenAI encrypted_content for replaying a ThinkChunk.

    Synthetic Responses ids like `rs_0` are not signatures — omitting them is
    safer than sending a value the gateway cannot verify.
    """
    if not isinstance(block, dict):
        return ""
    for key in ("encrypted_content", "signature"):
        val = block.get(key)
        if isinstance(val, str) and val.strip() and not val.startswith("rs_"):
            return val.strip()
    return ""


def _usable_thinking(text: str) -> str:
    """Drop empty traces and bridge notes; those must not be replayed as thought."""
    text = (text or "").strip()
    if not text or text.startswith("[bridge]"):
        return ""
    return text


def _think_chunk(text: str, signature: str = "") -> dict:
    chunk = {
        "type": "thinking",
        "thinking": [{"type": "text", "text": text}],
    }
    if signature:
        chunk["signature"] = signature
    return chunk


def _assistant_content(text: str, thinking: str = "", signature: str = ""):
    """String content, or [ThinkChunk, TextChunk] when a trace is being replayed."""
    if thinking:
        chunks = [_think_chunk(thinking, signature)]
        if (text or "").strip():
            chunks.append({"type": "text", "text": text})
        return chunks
    return text


def _thinking_chars(content) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        len(_flatten_thinking(block))
        for block in content
        if isinstance(block, dict) and block.get("type") == "thinking"
    )


def _drop_think_chunks(content):
    """Assistant content list -> plain text, thinking removed."""
    if not isinstance(content, list):
        return content
    return "".join(
        (block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _truncate_think_chunks(content, keep_chars: int, signature: str = ""):
    """Keep the suffix of thinking text (most recent) up to keep_chars."""
    if not isinstance(content, list) or keep_chars <= 0:
        return _drop_think_chunks(content)
    traces = []
    texts = []
    sig = signature
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "thinking":
            traces.append(_flatten_thinking(block))
            sig = sig or _thinking_signature(block)
        elif block.get("type") == "text":
            texts.append(block.get("text") or "")
    joined = "".join(traces)
    if len(joined) > keep_chars:
        joined = joined[-keep_chars:]
        sig = ""
    return _assistant_content("".join(texts), joined, sig)


def strip_thinking_for_match(entries: list) -> list:
    """Match-side view: thinking must not affect the message-prefix hash.

    A thinking-only assistant (no visible text) is dropped. Tool entries stay.
    """
    out = []
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("role") != "assistant":
            out.append(entry)
            continue
        content = entry.get("content")
        if not isinstance(content, list):
            out.append(entry)
            continue
        text = _drop_think_chunks(content)
        if text.strip():
            rewritten = dict(entry)
            rewritten["content"] = text
            out.append(rewritten)
    return out


def trim_thinking_for_create(entries: list, max_chars: int = None) -> list:
    """Keep the most recent ThinkChunks on a create payload, under a char cap.

    Older traces become plain assistant text. A single oversize newest trace
    is suffix-truncated (and loses its signature — it is no longer whole).
    A thinking-only assistant that does not fit the cap is dropped.
    """
    if not entries:
        return entries
    budget = THINKING_CREATE_MAX_CHARS if max_chars is None else max_chars
    indexed = [
        (i, _thinking_chars(entry.get("content")))
        for i, entry in enumerate(entries)
        if isinstance(entry, dict) and entry.get("role") == "assistant"
        and _thinking_chars(entry.get("content"))
    ]
    if not indexed:
        return entries
    # i -> chars of thinking to keep. Missing = unchanged (no thinking).
    keep = {}
    remaining = budget
    for i, n in reversed(indexed):
        if n <= remaining:
            keep[i] = n
            remaining -= n
        elif remaining > 0:
            keep[i] = remaining
            remaining = 0
        else:
            keep[i] = 0
    out = []
    for i, entry in enumerate(entries):
        if i not in keep:
            out.append(entry)
            continue
        content = entry.get("content")
        n_keep = keep[i]
        orig_n = _thinking_chars(content)
        rewritten = dict(entry)
        if n_keep <= 0:
            text = _drop_think_chunks(content)
            if (text or "").strip():
                rewritten["content"] = text
                out.append(rewritten)
            continue
        if n_keep >= orig_n:
            out.append(entry)
            continue
        rewritten["content"] = _truncate_think_chunks(content, n_keep)
        out.append(rewritten)
    kept_chars = sum(
        _thinking_chars(e.get("content"))
        for e in out
        if isinstance(e, dict)
    )
    log.info(
        "thinking on create: traces=%d kept_chars=%d dropped_chars=%d cap=%d",
        len(indexed), kept_chars, max(0, sum(n for _, n in indexed) - kept_chars), budget,
    )
    return out


def normalize_messages(messages: list, include_thinking: bool = False) -> tuple:
    """OpenAI/Anthropic/Responses-shaped messages -> (Mistral inputs, instructions).

    assistant.tool_calls / function_call / Anthropic tool_use  -> function.call
    role=tool / function / function_call_output / tool_result  -> function.result
    system / developer                                         -> instructions
    type=reasoning / thinking content blocks                   -> ThinkChunk on
        the next assistant entry when include_thinking=True (create replay).
        Append matching leaves these out so the prefix hash stays stable.
    """
    inputs = []
    instructions_parts = []
    call_names = {}
    pending_thinking = ""
    pending_sig = ""

    def park_thinking(block):
        nonlocal pending_thinking, pending_sig
        if not include_thinking:
            return
        text = _usable_thinking(_flatten_thinking(block))
        if not text:
            return
        pending_thinking += text
        pending_sig = pending_sig or _thinking_signature(block)

    def consume_pending() -> tuple:
        nonlocal pending_thinking, pending_sig
        text, sig = pending_thinking, pending_sig
        pending_thinking, pending_sig = "", ""
        return text, sig

    def flush_thinking():
        text, sig = consume_pending()
        if not text:
            return
        inputs.append({
            "role": "assistant",
            "content": _assistant_content("", text, sig),
        })

    for m in messages:
        if isinstance(m, str):
            flush_thinking()
            if m.strip():
                inputs.append({"role": "user", "content": m})
            continue
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        itype = m.get("type") or ""
        content = m.get("content")

        if itype in ("reasoning", "thinking"):
            park_thinking(m)
            continue

        if itype in ("function_call_output", "tool_result") or role in ("tool", "function"):
            entry = emit_function_result(m, call_names)
            if entry:
                inputs.append(entry)
            continue

        if itype == "function_call" or (
            role == "assistant" and m.get("name") and ("arguments" in m or "call_id" in m)
        ):
            flush_thinking()
            name = m.get("name") or ""
            if not name:
                fn = m.get("function") if isinstance(m.get("function"), dict) else {}
                name = fn.get("name") or ""
            args = m.get("arguments", m.get("input", "{}"))
            tcid = m.get("call_id") or m.get("id") or ""
            entry = emit_function_call(name, tcid, args, call_names)
            if entry:
                inputs.append(entry)
            continue

        if role in ("system", "developer") or itype in ("system", "developer"):
            text = content_to_text(content)
            if text.strip():
                instructions_parts.append(text)
            continue

        if role not in ("user", "assistant", ""):
            continue

        openai_tcs = m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []
        leftover = []
        inline_thinking = ""
        inline_sig = ""

        if isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    if block:
                        leftover.append(block)
                    continue
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in ("tool_result", "function_call_output"):
                    entry = emit_function_result({
                        "type": btype,
                        "tool_call_id": block.get("tool_use_id") or block.get("tool_call_id") or block.get("call_id") or "",
                        "call_id": block.get("call_id") or "",
                        "name": block.get("name") or "",
                        "content": block.get("content", block.get("output", block.get("result", ""))),
                    }, call_names)
                    if entry:
                        inputs.append(entry)
                elif btype == "tool_use" and not openai_tcs:
                    entry = emit_function_call(
                        block.get("name") or "",
                        block.get("id") or "",
                        block.get("input", {}),
                        call_names,
                    )
                    if entry:
                        inputs.append(entry)
                elif btype in ("thinking", "reasoning"):
                    if include_thinking:
                        t = _usable_thinking(_flatten_thinking(block))
                        if t:
                            inline_thinking += t
                            inline_sig = inline_sig or _thinking_signature(block)
                elif btype in ("text", "output_text", "input_text") or (
                    btype is None and "text" in block
                ):
                    leftover.append(block.get("text") or "")
            text = "".join(leftover)
        else:
            text = content_to_text(content)

        if role == "assistant":
            extra = m.get("reasoning_content")
            if include_thinking and extra:
                t = _usable_thinking(
                    extra if isinstance(extra, str) else _flatten_thinking(extra)
                )
                if t:
                    inline_thinking = t + inline_thinking
            parked, parked_sig = consume_pending()
            thinking = parked + inline_thinking
            sig = parked_sig or inline_sig
            if text.strip() or thinking:
                inputs.append({
                    "role": "assistant",
                    "content": _assistant_content(text, thinking, sig),
                })
            for tc in openai_tcs:
                name = ""
                args = "{}"
                tcid = ""
                if isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name") or tc.get("name") or ""
                    args = fn.get("arguments", tc.get("arguments", "{}"))
                    tcid = tc.get("id") or tc.get("call_id") or ""
                entry = emit_function_call(name, tcid, args, call_names)
                if entry:
                    inputs.append(entry)
            legacy = m.get("function_call")
            if isinstance(legacy, dict) and not openai_tcs:
                entry = emit_function_call(
                    legacy.get("name") or "",
                    m.get("id") or "",
                    legacy.get("arguments", "{}"),
                    call_names,
                )
                if entry:
                    inputs.append(entry)
        else:
            flush_thinking()
            if text.strip():
                inputs.append({"role": "user" if role != "assistant" else "assistant", "content": text})

    flush_thinking()
    return inputs, "\n\n".join(instructions_parts)


def normalize_tools(tools, functions=None) -> list:
    """OpenAI / Anthropic / legacy `functions` -> Mistral FunctionTool[].

    FunctionTool/Function reject unknown keys (additionalProperties: false),
    so extra client fields (cache_control, input_schema, …) are stripped.
    """
    raw = []
    if isinstance(tools, list):
        raw.extend(tools)
    if isinstance(functions, list):
        raw.extend(functions)

    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("function"), dict):
            fn = t["function"]
        elif t.get("name") and (t.get("parameters") is not None or t.get("input_schema") is not None):
            fn = t
        else:
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters")
        if params is None:
            params = fn.get("input_schema")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        elif params.get("type") is None:
            params = dict(params)
            params["type"] = "object"
        item = {
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(fn.get("description") or ""),
                "parameters": params,
            },
        }
        if isinstance(fn.get("strict"), bool):
            item["function"]["strict"] = fn["strict"]
        out.append(item)
    return out


def map_tool_choice(tool_choice):
    """OpenAI / Anthropic tool_choice -> Mistral CompletionArgs.tool_choice enum.

    OpenAI:  "auto" | "none" | "required" | {type:"function",function:{name}}
    Anthropic: {type:"auto"} | {type:"any"} | {type:"tool",name:"…"}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        v = tool_choice.lower()
        if v in ("none", "auto", "required", "any"):
            return "any" if v == "any" else v
        if v in ("off", "disabled"):
            return "none"
        return None
    if isinstance(tool_choice, dict):
        t = str(tool_choice.get("type", "")).lower()
        if t == "auto":
            return "auto"
        if t in ("any", "required"):
            return "any"
        if t in ("none", "off", "disabled"):
            return "none"
        # OpenAI {type:"function",…} or Anthropic {type:"tool",…} → force a call.
        return "any"
    return None


def _entry_kind(entry) -> str:
    if not isinstance(entry, dict):
        return "?"
    return entry.get("type") or entry.get("role") or "?"


def _preview_result_text(value, limit: int = 400) -> str:
    text = tool_result_string(value) if not isinstance(value, str) else value
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def compact_settled_tools(entries: list) -> list:
    """On create, fold completed historical tool pairs into assistant text.

    Cursor/Claude Code replay the whole window as a new conversation.
    Mistral create hangs / disconnects on hundreds of function.call +
    function.result entries. Split at the last user message: keep that
    turn native, fold earlier settled tool pairs.
    """
    if not entries:
        return entries

    kinds = [_entry_kind(e) for e in entries]
    call_n = sum(1 for k in kinds if k == "function.call")
    if call_n < 2:
        return entries

    last_user = -1
    for i, kind in enumerate(kinds):
        if kind == "user":
            last_user = i
    # Keep the last user turn (and anything after it) as native entries.
    open_start = last_user if last_user >= 0 else len(entries)

    out = []
    i = 0
    folded = 0
    while i < len(entries):
        if i >= open_start:
            out.extend(entries[i:])
            break
        entry = entries[i]
        kind = kinds[i]
        if kind != "function.call":
            out.append(entry)
            i += 1
            continue

        j = i
        names = []
        results = []
        pending = []
        while j < open_start:
            e = entries[j]
            k = kinds[j]
            if k == "function.call":
                pending.append(e.get("tool_call_id") or "")
                names.append(e.get("name") or "tool")
            elif k == "function.result":
                tcid = e.get("tool_call_id") or ""
                if tcid in pending:
                    pending.remove(tcid)
                elif pending:
                    pending.pop(0)
                results.append(_preview_result_text(e.get("result")))
            else:
                break
            j += 1
            if not pending and j > i:
                break
        if pending:
            out.append(entry)
            i += 1
            continue
        label = ", ".join(names) or "tools"
        body = " | ".join(r for r in results if r) or "(empty)"
        out.append({"role": "assistant", "content": f"[tool {label}] {body}"})
        folded += 1
        i = j

    if folded:
        log.info(
            "compact tools: %d→%d folded=%d open_from=%d calls=%d",
            len(entries),
            len(out),
            folded,
            open_start,
            call_n,
        )
    return out

