#!/usr/bin/env python3
"""
Mistral GLM Bridge — OpenAI-compatible -> Mistral /v1/conversations

Translates:
  POST /v1/chat/completions  (stateless: full messages, always create)
  POST /v1/responses         (stateful: input + previous_response_id → append)

Env vars (all optional except a key from the client or MISTRAL_KEY):
    MISTRAL_KEY     Mistral API key (no default)
    BRIDGE_MODEL    model id (default: glm-5-2)
    BRIDGE_PORT     listen port (default: 8090)
    BRIDGE_HOST     listen host (default: 0.0.0.0)
"""

import os
import time
import json
import logging

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── Config ───────────────────────────────────────────────────────────────────
MISTRAL_KEY  = os.environ.get("MISTRAL_KEY", "")
MODEL        = os.environ.get("BRIDGE_MODEL", "glm-5-2")
PORT         = int(os.environ.get("BRIDGE_PORT", 8090))
HOST         = os.environ.get("BRIDGE_HOST", "0.0.0.0")
MISTRAL_API  = "https://api.mistral.ai/v1"
MISTRAL_BASE = f"{MISTRAL_API}/conversations"
UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=300)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mistral-bridge")

app = FastAPI(title="Mistral GLM Bridge", version="1.0.0")


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_key(request: Request) -> str:
    """Client Authorization takes precedence; fall back to server MISTRAL_KEY."""
    auth_header = request.headers.get("Authorization", "")
    key = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    return key or MISTRAL_KEY


def resolve_model(requested) -> str:
    """Use the client model id when present; strip 9router 'provider/model' prefixes."""
    if not isinstance(requested, str):
        return MODEL
    name = requested.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1].strip()
    return name or MODEL


def local_model_card(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "mistral",
        "permission": [],
    }


def local_models_list() -> dict:
    return {"object": "list", "data": [local_model_card(MODEL)]}


def _as_json_string(value) -> str:
    """Normalize tool arguments / results to a JSON/text string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def content_to_text(content) -> str:
    """Flatten OpenAI / Anthropic content (str | list | dict | null) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return _as_json_string(content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype in ("text", "output_text", "input_text") or (
                    btype is None and "text" in block
                ):
                    parts.append(block.get("text") or "")
        joined = "".join(parts)
        return joined if joined else _as_json_string(content)
    return str(content)


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
    if args is None:
        args = "{}"
    elif not isinstance(args, (str, dict)):
        args = _as_json_string(args)
    tcid = str(tcid or f"call_{name}")
    if isinstance(call_names, dict):
        call_names[tcid] = name
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


def normalize_messages(messages: list) -> tuple:
    """OpenAI/Anthropic/Responses-shaped messages -> (Mistral inputs, instructions).

    assistant.tool_calls / function_call / Anthropic tool_use  -> function.call
    role=tool / function / function_call_output / tool_result  -> function.result
    system / developer                                         -> instructions
    """
    inputs = []
    instructions_parts = []
    call_names = {}

    for m in messages:
        if isinstance(m, str):
            if m.strip():
                inputs.append({"role": "user", "content": m})
            continue
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        itype = m.get("type") or ""
        content = m.get("content")

        if itype in ("function_call_output", "tool_result") or role in ("tool", "function"):
            entry = emit_function_result(m, call_names)
            if entry:
                inputs.append(entry)
            continue

        if itype == "function_call" or (
            role == "assistant" and m.get("name") and ("arguments" in m or "call_id" in m)
        ):
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
                    continue
                elif btype in ("text", "output_text", "input_text") or (
                    btype is None and "text" in block
                ):
                    leftover.append(block.get("text") or "")
            text = "".join(leftover)
        else:
            text = content_to_text(content)

        if role == "assistant":
            if text.strip():
                inputs.append({"role": "assistant", "content": text})
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
        elif text.strip():
            inputs.append({"role": "user" if role != "assistant" else "assistant", "content": text})

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
    """OpenAI tool_choice -> Mistral CompletionArgs.tool_choice enum."""
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
        return "any"
    return None


def resolve_reasoning(body: dict):
    """Client options -> Mistral reasoning_effort ('none'/'high'). Default: thinking ON."""
    if body.get("reasoning_effort"):
        effort = str(body["reasoning_effort"]).lower()
        if effort in ("none", "off", "disabled"):
            return "none"
        return "high"
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        effort = str(reasoning["effort"]).lower()
        if effort in ("none", "off", "disabled"):
            return "none"
        return "high"
    options = body.get("options")
    if isinstance(options, dict):
        th = options.get("thinking")
        if isinstance(th, dict):
            if th.get("type") == "disabled":
                return "none"
            if th.get("type") == "enabled":
                return "high"
        if options.get("enable_thinking") is False:
            return "none"
        if options.get("enable_thinking") is True:
            return "high"
    return "high"


def _num(body: dict, key: str, default, cast):
    """Read a numeric field; treat explicit null as default (avoids float(None))."""
    value = body.get(key, default)
    if value is None:
        return default
    return cast(value)


def completion_args_from_body(body: dict) -> dict:
    temperature = max(0.0, min(1.0, _num(body, "temperature", 0.7, float)))
    top_p = max(0.0, min(1.0, _num(body, "top_p", 1.0, float)))
    raw_max = body.get("max_tokens")
    if raw_max is None:
        raw_max = body.get("max_output_tokens")
    if raw_max is None:
        raw_max = body.get("max_completion_tokens")
    if raw_max is None:
        raw_max = 8192
    args = {
        "temperature": temperature,
        "max_tokens": max(1, int(raw_max)),
        "top_p": top_p,
    }
    reasoning = resolve_reasoning(body)
    if reasoning:
        args["reasoning_effort"] = reasoning
    tool_choice = map_tool_choice(body.get("tool_choice"))
    if tool_choice:
        args["tool_choice"] = tool_choice
    return args


def store_from_body(body: dict, default: bool) -> bool:
    if "store" not in body:
        return default
    val = body.get("store")
    if val is None:
        return default
    return bool(val)


def extract_previous_id(body: dict, request: Request):
    """previous_response_id / conversation / X-Conversation-Id from the client."""
    for key in ("previous_response_id", "conversation_id"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    conv = body.get("conversation")
    if isinstance(conv, str) and conv.strip():
        return conv.strip()
    if isinstance(conv, dict):
        for key in ("id", "conversation_id"):
            val = conv.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    header = request.headers.get("X-Conversation-Id") or request.headers.get("X-Previous-Response-Id")
    if header and header.strip():
        return header.strip()
    return None


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


def missing_conversation(status: int, text: str) -> bool:
    """True when append targeted an id Mistral does not have (code 3000)."""
    if status == 404:
        return True
    try:
        obj = json.loads(text or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        obj = None
    if isinstance(obj, dict):
        if obj.get("code") == 3000 or str(obj.get("code")) == "3000":
            return True
        msg = str(obj.get("message") or "")
        if "was not found" in msg or "Conversation with id" in msg:
            return True
    lowered = (text or "").lower()
    return "was not found" in lowered or "conversation with id" in lowered


def conversation_attempts(payload: dict, conv_id: str):
    """(url, body, reason) pairs. Append first when an id is present; create is fallback."""
    if conv_id:
        yield f"{MISTRAL_BASE}/{conv_id}", strip_for_append(payload), "append"
    yield MISTRAL_BASE, payload, "create"


# ── Request translation ───────────────────────────────────────────────────────
def openai_to_mistral(body: dict) -> dict:
    """Chat Completions -> Conversations create. Always a new thread (full messages)."""
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    inputs, instructions = normalize_messages(messages)
    if not inputs:
        inputs = [{"role": "user", "content": " "}]

    tools = normalize_tools(body.get("tools"), body.get("functions"))
    payload = {
        "model": resolve_model(body.get("model")),
        "inputs": inputs,
        "completion_args": completion_args_from_body(body),
        "store": store_from_body(body, False),
    }
    if instructions:
        payload["instructions"] = instructions
    if tools:
        payload["tools"] = tools
    return payload


def responses_input_to_entries(inp) -> tuple:
    """Responses `input` -> same Conversations entries as Chat `messages`."""
    if inp is None:
        return [], ""
    if isinstance(inp, str):
        text = inp.strip()
        return ([{"role": "user", "content": text}] if text else []), ""
    if isinstance(inp, dict):
        inp = [inp]
    if not isinstance(inp, list):
        return [], ""
    return normalize_messages(inp)


def responses_to_mistral(body: dict) -> dict:
    """OpenAI Responses request -> Mistral conversations payload (create shape)."""
    inp = body.get("input", body.get("messages", []))
    entries, extra_instr = responses_input_to_entries(inp)
    instructions = body.get("instructions") or ""
    if extra_instr:
        instructions = "\n\n".join(p for p in (instructions, extra_instr) if p)
    if not entries:
        entries = [{"role": "user", "content": " "}]

    payload = {
        "model": resolve_model(body.get("model")),
        "inputs": entries,
        "completion_args": completion_args_from_body(body),
        "store": store_from_body(body, True),
    }
    if instructions:
        payload["instructions"] = instructions
    tools = normalize_tools(body.get("tools"), body.get("functions"))
    if tools:
        payload["tools"] = tools
    return payload


# ── Response translation ──────────────────────────────────────────────────────
def extract_content_parts(content) -> tuple:
    """content (str | list of thinking/text/reasoning blocks) -> (text, reasoning)."""
    text, reasoning = "", ""
    if isinstance(content, str):
        return content, ""
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("text", "output_text", "input_text"):
                text += block.get("text") or ""
            elif btype in ("thinking", "reasoning"):
                reasoning += _thinking_text(block)
    return text, reasoning


def _thinking_text(block: dict) -> str:
    """Flatten a Mistral thinking / reasoning content block to text."""
    parts = []
    nested = block.get("thinking") or block.get("reasoning") or []
    if isinstance(nested, str):
        parts.append(nested)
    elif isinstance(nested, list):
        for t in nested:
            if isinstance(t, str) and t:
                parts.append(t)
            elif isinstance(t, dict):
                if t.get("type") in ("text", "reasoning_text", "summary_text", None) and t.get("text"):
                    parts.append(t.get("text") or "")
    if isinstance(block.get("text"), str) and block["text"]:
        parts.append(block["text"])
    return "".join(parts)


def _as_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def usage_from_event(data) -> dict:
    """Pull a raw Mistral usage object off a stream/non-stream payload."""
    if not isinstance(data, dict):
        return {}
    usage = data.get("usage")
    if isinstance(usage, dict) and usage:
        return usage
    resp = data.get("response")
    if isinstance(resp, dict):
        usage = resp.get("usage")
        if isinstance(usage, dict) and usage:
            return usage
    return {}


def mistral_usage_fields(usage) -> dict:
    """Normalize Mistral usage aliases to a flat dict."""
    if not isinstance(usage, dict):
        usage = {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total = usage.get("total_tokens", 0)
    prompt = _as_int(prompt)
    completion = _as_int(completion)
    total = _as_int(total)
    if not total and (prompt or completion):
        total = prompt + completion

    in_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if not isinstance(in_details, dict):
        in_details = {}
    cached = (
        in_details.get("cached_tokens")
        or in_details.get("cache_read_tokens")
        or usage.get("cached_tokens")
        or usage.get("num_cached_tokens")
        or usage.get("cache_read_tokens")
        or 0
    )
    cache_write = (
        in_details.get("cache_write_tokens")
        or in_details.get("cache_creation_tokens")
        or usage.get("cache_write_tokens")
        or usage.get("cache_creation_tokens")
        or 0
    )

    out_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    if not isinstance(out_details, dict):
        out_details = {}
    reasoning = (
        out_details.get("reasoning_tokens")
        or out_details.get("thinking_tokens")
        or usage.get("reasoning_tokens")
        or usage.get("thinking_tokens")
        or 0
    )
    return {
        "prompt": prompt,
        "completion": completion,
        "total": total,
        "cached": _as_int(cached),
        "cache_write": _as_int(cache_write),
        "reasoning": _as_int(reasoning),
    }


def responses_usage_from_mistral(usage) -> dict:
    """Mistral usage -> OpenAI Responses usage (incl. cache / reasoning details)."""
    fields = mistral_usage_fields(usage)
    details_in = {"cached_tokens": fields["cached"]}
    if fields["cache_write"]:
        details_in["cache_write_tokens"] = fields["cache_write"]
    return {
        "input_tokens": fields["prompt"],
        "output_tokens": fields["completion"],
        "total_tokens": fields["total"],
        "input_tokens_details": details_in,
        "output_tokens_details": {"reasoning_tokens": fields["reasoning"]},
    }


def chat_usage_from_mistral(usage) -> dict:
    """Mistral usage -> OpenAI Chat Completions usage."""
    fields = mistral_usage_fields(usage)
    out = {
        "prompt_tokens": fields["prompt"],
        "completion_tokens": fields["completion"],
        "total_tokens": fields["total"],
    }
    if fields["cached"] or fields["cache_write"]:
        details = {"cached_tokens": fields["cached"]}
        if fields["cache_write"]:
            details["cache_write_tokens"] = fields["cache_write"]
        out["prompt_tokens_details"] = details
    if fields["reasoning"]:
        out["completion_tokens_details"] = {"reasoning_tokens": fields["reasoning"]}
    return out


def _function_call_to_openai(entry: dict, index: int) -> dict:
    args = entry.get("arguments", "{}")
    if not isinstance(args, str):
        args = _as_json_string(args)
    return {
        "id": entry.get("tool_call_id") or entry.get("id") or f"call_{index}",
        "type": "function",
        "function": {
            "name": entry.get("name") or "",
            "arguments": args,
        },
    }


def mistral_to_openai(data: dict, model: str) -> dict:
    """Mistral conversations response -> OpenAI chat.completion."""
    text, reasoning = "", ""
    tool_calls = []
    for o in data.get("outputs", []):
        if not isinstance(o, dict):
            continue
        otype = o.get("type")
        if otype == "message.output" and o.get("role", "assistant") == "assistant":
            t, r = extract_content_parts(o.get("content", ""))
            text += t
            reasoning += r
        elif otype == "function.call":
            tool_calls.append(_function_call_to_openai(o, len(tool_calls)))

    message = {
        "role": "assistant",
        "content": text if text else (None if tool_calls else ""),
    }
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": data.get("conversation_id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": chat_usage_from_mistral(usage_from_event(data)),
    }


def mistral_to_responses(data: dict, model: str) -> dict:
    """Mistral conversations response -> OpenAI Responses object."""
    conv_id = data.get("conversation_id") or f"resp_{int(time.time())}"
    output = []
    text_acc, reason_acc = "", ""
    tool_items = []
    for o in data.get("outputs", []):
        if not isinstance(o, dict):
            continue
        otype = o.get("type")
        if otype == "message.output" and o.get("role", "assistant") == "assistant":
            t, r = extract_content_parts(o.get("content", ""))
            text_acc += t
            reason_acc += r
        elif otype == "function.call":
            args = o.get("arguments", "{}")
            if not isinstance(args, str):
                args = _as_json_string(args)
            tool_items.append({
                "type": "function_call",
                "id": o.get("id") or o.get("tool_call_id") or f"fc_{len(tool_items)}",
                "call_id": o.get("tool_call_id") or o.get("id") or f"call_{len(tool_items)}",
                "name": o.get("name") or "",
                "arguments": args,
            })
    if reason_acc:
        output.append({
            "type": "reasoning",
            "id": f"rs_{conv_id}",
            "summary": [{"type": "summary_text", "text": reason_acc}],
        })
    output.extend(tool_items)
    if text_acc or not output:
        output.append({
            "type": "message",
            "id": f"msg_{conv_id}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text_acc}],
        })
    usage = responses_usage_from_mistral(usage_from_event(data))
    return {
        "id": conv_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
        "usage": usage,
    }


def _openai_chunk(model: str, delta: dict, finish_reason=None, chunk_id: str = "", usage=None) -> dict:
    chunk = {
        "id": chunk_id or f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _args_fragment(prev: str, incoming: str) -> str:
    """Emit only the new suffix if upstream sends cumulative arguments."""
    if not incoming:
        return ""
    if prev and incoming.startswith(prev):
        return incoming[len(prev):]
    return incoming


def _sse_field(line: str):
    """Parse one SSE 'field: value' line. Returns (field, value) or None."""
    if not line or line.startswith(":"):
        return None
    if ":" not in line:
        return line, ""
    field, value = line.split(":", 1)
    if value.startswith(" "):
        value = value[1:]
    return field, value


def parse_sse_block(block: str):
    """Parse one SSE event (event: + data:). Returns (event_name, payload_dict|None)."""
    event_name, data_parts = "", []
    for raw_line in block.splitlines():
        parsed = _sse_field(raw_line.strip("\r"))
        if not parsed:
            continue
        field, value = parsed
        if field == "event":
            event_name = value
        elif field == "data":
            data_parts.append(value)
    if not data_parts:
        return event_name, None
    payload_str = "\n".join(data_parts)
    if payload_str == "[DONE]":
        return event_name or "done", {"type": "done"}
    try:
        obj = json.loads(payload_str)
    except json.JSONDecodeError:
        return event_name, None
    if not isinstance(obj, dict):
        return event_name, None
    if "data" in obj and isinstance(obj.get("data"), dict) and "content" not in obj:
        inner = obj["data"]
        event_name = event_name or obj.get("event") or inner.get("type") or ""
        if event_name and not inner.get("type"):
            inner = dict(inner)
            inner["type"] = event_name
        return event_name or inner.get("type") or "", inner
    etype = obj.get("type") or event_name
    if etype and not obj.get("type"):
        obj = dict(obj)
        obj["type"] = etype
    return etype or "", obj


async def iter_conversation_events(resp):
    """Yield (event_type, payload) from a Mistral conversations SSE response."""
    buf = ""
    while True:
        raw = await resp.content.readline()
        if not raw:
            if buf.strip():
                event_name, data = parse_sse_block(buf)
                if data is not None:
                    yield data.get("type") or event_name or "", data
            return
        line = raw.decode("utf-8", errors="replace")
        if line in ("\n", "\r\n"):
            event_name, data = parse_sse_block(buf)
            buf = ""
            if data is not None:
                yield data.get("type") or event_name or "", data
        else:
            buf += line


def _delta_from_message_content(content):
    """Mistral message.output.delta content -> OpenAI delta dict or None."""
    if content is None or content == "":
        return None
    if isinstance(content, str):
        return {"content": content}
    if isinstance(content, list):
        text, reasoning = extract_content_parts(content)
        if reasoning and not text:
            return {"reasoning_content": reasoning}
        if text and reasoning:
            return {"content": text, "reasoning_content": reasoning}
        if text:
            return {"content": text}
        return None
    if isinstance(content, dict):
        btype = content.get("type")
        if btype in ("thinking", "reasoning"):
            joined = _thinking_text(content)
            return {"reasoning_content": joined} if joined else None
        if btype in ("text", "output_text", "input_text") or "text" in content:
            text = content.get("text") or ""
            return {"content": text} if text else None
    return None


def _responses_event(etype: str, payload: dict) -> str:
    return f"event: {etype}\ndata: {json.dumps(payload)}\n\n"


def _auth_headers(key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def input_kinds(inputs) -> list:
    kinds = []
    for item in inputs or []:
        if isinstance(item, dict):
            kinds.append(item.get("type") or item.get("role") or "?")
        elif isinstance(item, str):
            kinds.append("str")
    return kinds


# ── Upstream POST (create / append / one create-after-miss) ───────────────────
async def post_conversation(session, key: str, payload: dict, conv_id: str = None):
    """POST create or append. On append 404/3000, retry once as create.

    Returns (resp, reason, error_text). Caller must read resp / close it.
    """
    last_text = ""
    for url, body, reason in conversation_attempts(payload, conv_id):
        resp = await session.post(url, headers=_auth_headers(key), json=body, timeout=UPSTREAM_TIMEOUT)
        if resp.status == 200:
            if reason == "create" and conv_id:
                log.info("append %s missing → create", conv_id)
            return resp, reason, ""
        last_text = await resp.text()
        log.warning("upstream %s %s: %s", reason, resp.status, last_text[:200])
        if reason == "append" and missing_conversation(resp.status, last_text):
            continue
        return resp, reason, last_text
    return resp, "create", last_text


# ── Streaming ─────────────────────────────────────────────────────────────────
async def stream_response(upstream_key: str, model: str, payload: dict):
    """SSE proxy: Mistral conversations stream -> OpenAI chat.completion.chunk."""
    chunk_id = f"chatcmpl-{int(time.time())}"
    conv_id = None
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                MISTRAL_BASE,
                headers=_auth_headers(upstream_key),
                json=payload,
                timeout=UPSTREAM_TIMEOUT,
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.warning("upstream %s: %s", resp.status, text[:200])
                    yield f"data: {json.dumps({'error': {'message': text[:500], 'type': 'upstream_error'}})}\n\n"
                    return

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
                for tcid, state in ordered:
                    name = state.get("name") or ""
                    args = state.get("args") or "{}"
                    if not name:
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

                if not sent_role:
                    yield emit({"content": ""})
                finish_reason = "tool_calls" if saw_tool_call else "stop"
                usage = chat_usage_from_mistral(raw_usage)
                log.info(
                    "stream done conv=%s types=%s chunks=%d finish=%s tools=%s usage=%s think=%d raw_usage_keys=%s",
                    conv_id, seen_types, yielded, finish_reason, tool_names,
                    f"{usage['prompt_tokens']}+{usage['completion_tokens']}={usage['total_tokens']}"
                    f" cache={(usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)}",
                    len(think_acc),
                    sorted((raw_usage or {}).keys()),
                )
                yield f"data: {json.dumps(_openai_chunk(model, {}, finish_reason, chunk_id, usage))}\n\n"
                yield "data: [DONE]\n\n"
        except aiohttp.ClientError as e:
            log.error("connection error: %s", e)
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_unreachable'}})}\n\n"


async def stream_responses(upstream_key: str, model: str, payload: dict, conv_id: str = None):
    """SSE proxy: Mistral conversations stream -> OpenAI Responses events.

    Clients treat a stream without output_text.done / function_call_arguments.done
    / response.completed as incomplete and retry the same turn.
    """
    terminal = False

    def failed(message: str, rid: str = ""):
        return _responses_event("response.failed", {
            "type": "response.failed",
            "response": {"id": rid, "error": {"message": message}},
        })

    async with aiohttp.ClientSession() as session:
        try:
            for url, body, reason in conversation_attempts(payload, conv_id):
                async with session.post(
                    url,
                    headers=_auth_headers(upstream_key),
                    json=body,
                    timeout=UPSTREAM_TIMEOUT,
                ) as resp:
                    if resp.status != 200:
                        last_text = await resp.text()
                        log.warning("upstream %s %s: %s", reason, resp.status, last_text[:200])
                        if reason == "append" and missing_conversation(resp.status, last_text):
                            log.info("append %s missing → create", conv_id)
                            continue
                        terminal = True
                        yield failed(last_text[:500])
                        return

                    log.info("responses stream %s", reason)
                    created_sent = False
                    seq = 0
                    next_out = 0
                    text_item_id = "msg_0"
                    text_index = 0
                    text_started = False
                    text_acc = ""
                    reason_item_id = "rs_0"
                    reason_index = 0
                    reason_started = False
                    reason_acc = ""
                    tool_state = {}
                    seen_types = []
                    raw_usage = {}
                    # Never advertise a client-local id on a create fallback.
                    rid = conv_id if reason == "append" else ""
                    created_at = int(time.time())

                    def next_seq():
                        nonlocal seq
                        seq += 1
                        return seq

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

                    def take_index():
                        nonlocal next_out
                        idx = next_out
                        next_out += 1
                        return idx

                    async for etype, data in iter_conversation_events(resp):
                        if etype and len(seen_types) < 8:
                            seen_types.append(etype)
                        if etype == "conversation.response.started":
                            rid = data.get("conversation_id") or rid
                            raw_usage = usage_from_event(data) or raw_usage
                            for ev in ensure_created():
                                yield ev
                            continue
                        if etype == "conversation.response.error":
                            msg = data.get("message") or "upstream stream error"
                            log.warning("stream error: %s", msg)
                            terminal = True
                            yield failed(msg, rid)
                            return
                        if etype in ("conversation.response.done", "done"):
                            rid = (data or {}).get("conversation_id") or rid
                            raw_usage = usage_from_event(data) or raw_usage
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

                    for ev in ensure_created():
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
                    usage = responses_usage_from_mistral(raw_usage)
                    log.info(
                        "responses done conv=%s types=%s tools=%s reason=%s usage=%s think=%d raw_usage_keys=%s",
                        rid, seen_types, tool_names, reason,
                        f"{usage['input_tokens']}+{usage['output_tokens']}={usage['total_tokens']}"
                        f" cache={usage['input_tokens_details'].get('cached_tokens', 0)}",
                        len(reason_acc),
                        sorted((raw_usage or {}).keys()),
                    )
                    terminal = True
                    yield _responses_event("response.completed", {
                        "type": "response.completed",
                        "response": response_obj("completed", items, usage),
                    })
                    return
        except aiohttp.ClientError as e:
            log.error("connection error: %s", e)
            if not terminal:
                terminal = True
                yield failed(str(e))
        if not terminal:
            yield failed("upstream stream ended without a terminal event")


def error_response(status: int, message: str, code: str = "upstream_error") -> Response:
    return Response(
        status_code=status,
        media_type="application/json",
        content=json.dumps({"error": {"message": message, "type": code}}),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
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
        return StreamingResponse(
            stream_response(upstream_key, model, payload),
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
        except aiohttp.ClientError as e:
            log.error("connection error: %s", e)
            return error_response(502, str(e), "upstream_unreachable")
        except json.JSONDecodeError:
            log.error("bad upstream JSON")
            return error_response(502, "Bad upstream response")

    translated = mistral_to_openai(data, model)
    usage = translated.get("usage") or {}
    log.info(
        "chat done conv=%s usage=%s raw_usage_keys=%s",
        translated.get("id"),
        f"{usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)}={usage.get('total_tokens', 0)}"
        f" cache={(usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)}",
        sorted((usage_from_event(data) or {}).keys()),
    )
    return Response(
        status_code=200,
        media_type="application/json",
        content=json.dumps(translated),
    )


@app.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error_response(400, "Invalid JSON body", "bad_request")

    model = resolve_model(body.get("model"))
    try:
        payload = responses_to_mistral(body)
    except Exception as e:
        return error_response(422, f"Payload translation error: {e}", "invalid_payload")

    upstream_key = resolve_key(request)
    if not upstream_key:
        return error_response(401, "Missing API key (client Authorization or MISTRAL_KEY)", "missing_api_key")

    prev = extract_previous_id(body, request)
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
        return StreamingResponse(
            stream_responses(upstream_key, model, payload, prev),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with aiohttp.ClientSession() as session:
        try:
            resp, reason, err_text = await post_conversation(session, upstream_key, payload, prev)
            try:
                if resp.status != 200:
                    return error_response(
                        resp.status if resp.status < 500 else 502,
                        (err_text or await resp.text())[:500],
                    )
                data = json.loads(await resp.text())
            finally:
                resp.release()
        except aiohttp.ClientError as e:
            log.error("connection error: %s", e)
            return error_response(502, str(e), "upstream_unreachable")
        except json.JSONDecodeError:
            log.error("bad upstream JSON")
            return error_response(502, "Bad upstream response")

    translated = mistral_to_responses(data, model)
    usage = translated.get("usage") or {}
    log.info(
        "responses done conv=%s reason=%s usage=%s raw_usage_keys=%s",
        translated.get("id"),
        reason,
        f"{usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)}={usage.get('total_tokens', 0)}"
        f" cache={(usage.get('input_tokens_details') or {}).get('cached_tokens', 0)}",
        sorted((usage_from_event(data) or {}).keys()),
    )
    return Response(
        status_code=200,
        media_type="application/json",
        content=json.dumps(translated),
    )


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
    except aiohttp.ClientError as e:
        log.warning("upstream GET %s: %s", path, e)
        return None, None


@app.get("/v1/models")
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


@app.get("/v1/models/{model_id:path}")
async def model_retrieve(model_id: str, request: Request):
    """Passthrough Mistral /v1/models/{id}; fall back to a local card."""
    wanted = resolve_model(model_id)
    status, text = await proxy_mistral_get(request, f"/models/{wanted}")
    if status == 200 and text:
        return Response(status_code=200, media_type="application/json", content=text)
    if wanted == MODEL or not status:
        return local_model_card(wanted)
    return error_response(status if status < 500 else 502, (text or "model not found")[:500])


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "port": PORT}


if __name__ == "__main__":
    log.info("Mistral bridge on %s:%d, model=%s", HOST, PORT, MODEL)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
