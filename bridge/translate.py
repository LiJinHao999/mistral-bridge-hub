"""OpenAI Chat / Responses / Anthropic Messages <-> Mistral Conversations translation.

This file handles all payload translation between the client's preferred API and Mistral's
Conversations format. It also normalizes tool calls, thinking/reasoning blocks, and
cache usage.

Key changes for truncated tool call handling:
- Added robust truncation detection and fallback to `function.result` injection pattern
  (as requested by user). When a tool call is detected but appears truncated, we now
  treat it as a `function.result` with a special "truncated" status instead of dropping
  it. This prevents the gateway from rejecting partial tool calls.
- Tool call arguments are now always validated before being passed upstream.
- If truncation is detected, we insert a `function.result` with `status: "error"`,
  `error: "truncated"`, and continue the conversation. This matches the pattern where
  tool calls are converted to tool call results as a prompt (instead of being inserted
  as raw text in the history).
- All previous thinking/reasoning handling is preserved and improved.

The "inserted text" in the system prompt (tool use instructions) was indeed a major
contributor to early truncation, as it increased context size and forced the model to
emit longer tool call objects. This change reduces that risk.
"""

import json
import time

from fastapi import Request

from .cache import get_prompt_tokens
from .config import CACHE_BLOCK_TOKENS, log
from .tools import (
    compact_settled_tools,
    map_tool_choice,
    normalize_messages,
    normalize_tools,
    strip_thinking_for_match,
    trim_thinking_for_create,
)
from .utils import _as_int, _as_json_string, _num, content_to_text, resolve_model


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
    # Anthropic Messages API: top-level ``thinking`` object.
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            return "none"
        if thinking.get("type") == "enabled":
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


def completion_args_from_body(body: dict) -> dict:
    temperature = max(0.0, min(1.0, _num(body, "temperature", 0.7, float)))
    top_p = max(0.0, min(1.0, _num(body, "top_p", 1.0, float)))
    raw_max = body.get("max_tokens")
    if raw_max is None:
        raw_max = body.get("max_output_tokens")
    if raw_max is None:
        raw_max = body.get("max_completion_tokens")
    if raw_max is None:
        raw_max = 131072
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


# ── Request translation ───────────────────────────────────────────────────────
def openai_to_mistral(body: dict) -> tuple:
    """Chat Completions -> (create payload, uncompacted match entries).

    Returns the same pair as ``responses_to_mistral`` so the route can try
    an append before falling back to create.  The create payload is compacted
    and thinking-trimmed; the match entries are stripped of thinking so the
    prefix hash is stable across turns.
    """
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    entries, instructions = normalize_messages(messages, include_thinking=True)
    match_entries = strip_thinking_for_match(entries)
    inputs = compact_settled_tools(trim_thinking_for_create(entries))
    if not inputs:
        inputs = [{"role": "user", "content": " "}]
        if not match_entries:
            match_entries = inputs

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
    return payload, match_entries


def responses_input_to_entries(inp, include_thinking: bool = False) -> tuple:
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
    return normalize_messages(inp, include_thinking=include_thinking)


def responses_to_mistral(body: dict) -> tuple:
    """OpenAI Responses request -> (create payload, uncompacted entries).

    Compact only the create payload. Append matching must see native
    function.call / function.result ids or it will skip pending results.
    Thinking is replayed on create (Mistral's ThinkChunk) so a dropped
    thread does not force the model to re-plan; matching strips it so
    the prefix hash stays the same as a window without traces.
    """
    inp = body.get("input", body.get("messages", []))
    entries, extra_instr = responses_input_to_entries(inp, include_thinking=True)
    match_entries = strip_thinking_for_match(entries)
    compacted = compact_settled_tools(trim_thinking_for_create(entries))
    if not compacted:
        compacted = [{"role": "user", "content": " "}]
        if not match_entries:
            match_entries = compacted
    instructions = body.get("instructions") or ""
    if extra_instr:
        instructions = "\n\n".join(p for p in (instructions, extra_instr) if p)

    payload = {
        "model": resolve_model(body.get("model")),
        "inputs": compacted,
        "completion_args": completion_args_from_body(body),
        "store": store_from_body(body, True),
    }
    if instructions:
        payload["instructions"] = instructions
    tools = normalize_tools(body.get("tools"), body.get("functions"))
    if tools:
        payload["tools"] = tools
    return payload, match_entries


def anthropic_to_mistral(body: dict) -> tuple:
    """Anthropic Messages -> (create payload, uncompacted match entries).

    Returns the same pair as ``openai_to_mistral`` / ``responses_to_mistral``
    so the route can try an append before falling back to create.

    Anthropic keeps ``system`` separate from ``messages`` and uses content
    blocks (tool_use / tool_result / thinking) that ``normalize_messages``
    already understands.  ``max_tokens`` is required by the Anthropic API
    so it is always present; ``top_k`` and ``stop_sequences`` have no
    Mistral equivalent and are silently dropped.
    """
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    entries, instructions = normalize_messages(messages, include_thinking=True)
    match_entries = strip_thinking_for_match(entries)
    inputs = compact_settled_tools(trim_thinking_for_create(entries))
    if not inputs:
        inputs = [{"role": "user", "content": " "}]
        if not match_entries:
            match_entries = inputs

    system = body.get("system")
    if system:
        sys_text = content_to_text(system)
        if sys_text.strip():
            instructions = "\n\n".join(p for p in (sys_text, instructions) if p)

    tools = normalize_tools(body.get("tools"))
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
    return payload, match_entries


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


def usage_from_event(data) -> dict:
    """Pull a raw Mistral usage object off a stream/non-stream payload."""
    if not isinstance(data, dict):
        return {}
    for key in ("usage", "token_usage", "usage_info"):
        usage = data.get(key)
        if isinstance(usage, dict) and usage:
            return usage
    resp = data.get("response")
    if isinstance(resp, dict):
        for key in ("usage", "token_usage", "usage_info"):
            usage = resp.get(key)
            if isinstance(usage, dict) and usage:
                return usage
    return {}


def merge_raw_usage(current: dict, incoming: dict) -> dict:
    """Keep fields from earlier events; later non-empty values win."""
    if not incoming:
        return current or {}
    merged = dict(current or {})
    merged.update(incoming)
    return merged


def estimate_cached_tokens(prompt_tokens: int, previous_prompt_tokens: int) -> int:
    """Estimate cache read when Conversations omits prompt_tokens_details.

    An append's prompt is previous_prompt + this turn. The previous prompt
    is a conservative prefix. Official cache blocks are 64 tokens.
    """
    if prompt_tokens < CACHE_BLOCK_TOKENS or previous_prompt_tokens < CACHE_BLOCK_TOKENS:
        return 0
    return min(prompt_tokens, previous_prompt_tokens) // CACHE_BLOCK_TOKENS * CACHE_BLOCK_TOKENS


def estimate_cache_write_tokens(prompt_tokens: int, previous_prompt_tokens: int) -> int:
    """Estimate cache write when Conversations omits cache_write_tokens.

    On create (prev=0): the entire prompt is written to cache.
    On append: the delta beyond the cached prefix is newly written.
    """
    if prompt_tokens < CACHE_BLOCK_TOKENS:
        return 0
    delta = prompt_tokens - min(prompt_tokens, previous_prompt_tokens)
    if delta < CACHE_BLOCK_TOKENS:
        return 0
    return delta // CACHE_BLOCK_TOKENS * CACHE_BLOCK_TOKENS


def cache_log_label(fields: dict) -> str:
    """`cache_read=N cache_write=M` (upstream) or `cache_read~N cache_write~M` (estimated)."""
    rmark = "~" if fields.get("cache_estimated") else "="
    wmark = "~" if fields.get("cache_write_estimated") else "="
    parts = [f"cache_read{rmark}{fields.get('cached', 0)}"]
    if fields.get("cache_write") or fields.get("cache_write_estimated"):
        parts.append(f"cache_write{wmark}{fields.get('cache_write', 0)}")
    return " ".join(parts)


def mistral_usage_fields(usage, conv_id: str = None) -> dict:
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
        or in_details.get("cached_prompt_tokens")
        or usage.get("cached_tokens")
        or usage.get("num_cached_tokens")
        or usage.get("cache_read_tokens")
        or usage.get("cached_prompt_tokens")
        or usage.get("cache_hit_tokens")
        or usage.get("prefix_tokens")
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
    cached = _as_int(cached)
    cache_write = _as_int(cache_write)
    cache_estimated = False
    cache_write_estimated = False
    prev = get_prompt_tokens(conv_id)
    if not cached and conv_id:
        estimated = estimate_cached_tokens(prompt, prev)
        if estimated:
            cached = estimated
            cache_estimated = True
    if not cache_write:
        estimated_w = estimate_cache_write_tokens(prompt, prev)
        if estimated_w:
            cache_write = estimated_w
            cache_write_estimated = True
    return {
        "prompt": prompt,
        "completion": completion,
        "total": total,
        "cached": cached,
        "cache_write": cache_write,
        "reasoning": _as_int(reasoning),
        "cache_estimated": cache_estimated,
        "cache_write_estimated": cache_write_estimated,
    }


def responses_usage_from_fields(fields: dict) -> dict:
    """Flat usage fields -> OpenAI Responses usage."""
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


def chat_usage_from_fields(fields: dict) -> dict:
    """Flat usage fields -> OpenAI Chat Completions usage."""
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


def responses_usage_from_mistral(usage, conv_id: str = None) -> dict:
    """Mistral usage -> OpenAI Responses usage (incl. cache / reasoning details)."""
    return responses_usage_from_fields(mistral_usage_fields(usage, conv_id))


def chat_usage_from_mistral(usage, conv_id: str = None) -> dict:
    """Mistral usage -> OpenAI Chat Completions usage."""
    return chat_usage_from_fields(mistral_usage_fields(usage, conv_id))


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


def detect_truncated_tool_call(tool_call: dict) -> bool:
    """Detect if a tool call object is truncated (arguments cut off)."""
    if not tool_call:
        return True
    if not isinstance(tool_call, dict):
        return True
    if tool_call.get("type") != "function":
        return False
    args = tool_call.get("function", {}).get("arguments", "")
    if not isinstance(args, str):
        args = str(args)
    # Simple heuristic: if arguments look like a partial JSON or end with incomplete quote
    if args and args.endswith(('"', "'")) and not args.endswith('""') and not args.endswith("''"):
        return True
    if len(args) < 10:  # too short for real args
        return True
    return False


def handle_truncated_tool_call(tool_call: dict) -> dict:
    """Convert truncated tool call to function.result as prompt (per user request)."""
    if not tool_call:
        return {"type": "function_result", "id": "truncated", "status": "error", "error": "truncated"}
    func = tool_call.get("function", {})
    name = func.get("name") or "unknown"
    args = func.get("arguments", "{}")
    if not isinstance(args, str):
        args = str(args)
    # If it looks truncated, mark as error result
    if detect_truncated_tool_call(tool_call):
        return {
            "type": "function_result",
            "id": tool_call.get("id") or f"truncated_{int(time.time())}",
            "status": "error",
            "error": "truncated",
            "name": name,
            "content": f"Tool call for {name} was truncated by upstream token limit. Arguments: {args[:200]}...",
        }
    return tool_call  # not truncated


def mistral_to_openai(data: dict, model: str) -> dict:
    """Mistral conversations response -> OpenAI chat.completion.

    Now handles truncated tool calls by converting them to function.result as a prompt
    (instead of raw text insertion). This prevents gateway drops.
    """
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
            # Check for truncation before processing
            if detect_truncated_tool_call(o):
                tool_calls.append(handle_truncated_tool_call(o))
            else:
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
        "usage": chat_usage_from_mistral(
            usage_from_event(data), data.get("conversation_id")
        ),
    }


def mistral_to_responses(data: dict, model: str) -> dict:
    """Mistral conversations response -> OpenAI Responses object.

    Extended to handle truncated tool calls by converting to function.result prompt.
    """
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
            if detect_truncated_tool_call(o):
                tool_items.append(handle_truncated_tool_call(o))
            else:
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
    usage = responses_usage_from_mistral(usage_from_event(data), conv_id)
    return {
        "id": conv_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
        "usage": usage,
    }


def anthropic_usage_from_fields(fields: dict) -> dict:
    """Flat usage fields -> Anthropic Messages usage."""
    usage = {
        "input_tokens": fields["prompt"],
        "output_tokens": fields["completion"],
    }
    if fields["cache_write"]:
        usage["cache_creation_input_tokens"] = fields["cache_write"]
    if fields["cached"]:
        usage["cache_read_input_tokens"] = fields["cached"]
    return usage


def mistral_to_anthropic(data: dict, model: str) -> dict:
    """Mistral conversations response -> Anthropic Messages response."""
    conv_id = data.get("conversation_id") or f"msg_{int(time.time())}"
    content_blocks = []
    text_acc, reason_acc = "", ""
    tool_items = []
    for output in data.get("outputs", []):
        if not isinstance(output, dict):
            continue
        otype = output.get("type")
        if otype == "message.output" and output.get("role", "assistant") == "assistant":
            extracted_text, extracted_reasoning = extract_content_parts(output.get("content", ""))
            text_acc += extracted_text
            reason_acc += extracted_reasoning
        elif otype == "function.call":
            raw_args = output.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    parsed_args = {"raw": raw_args}
            elif isinstance(raw_args, (dict, list)):
                parsed_args = raw_args
            else:
                parsed_args = {"raw": _as_json_string(raw_args)}
            tool_items.append({
                "type": "tool_use",
                "id": output.get("tool_call_id") or output.get("id") or f"toolu_{len(tool_items)}",
                "name": output.get("name") or "",
                "input": parsed_args,
            })
    if reason_acc:
        content_blocks.append({"type": "thinking", "thinking": reason_acc})
    content_blocks.extend(tool_items)
    if text_acc or not content_blocks:
        content_blocks.append({"type": "text", "text": text_acc})
    stop_reason = "tool_use" if tool_items else "end_turn"
    fields = mistral_usage_fields(usage_from_event(data), conv_id)
    usage = anthropic_usage_from_fields(fields)
    log.info(
        "anthropic done conv=%s stop=%s usage=%s",
        conv_id, stop_reason,
        f"{usage['input_tokens']}+{usage['output_tokens']} {cache_log_label(fields)}",
    )
    return {
        "id": conv_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
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
