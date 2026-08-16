#!/usr/bin/env python3
"""
Mistral GLM Bridge — OpenAI-compatible -> Mistral /v1/conversations

Menerima request /v1/chat/completions format OpenAI (yang dipakai 9router),
menerjemahkan ke Mistral /v1/conversations, dan mengembalikan response
format OpenAI. Berguna karena /v1/chat/completions Mistral gampang kena 429
untuk model pihak ketiga (GLM), sedangkan /v1/conversations bebas limit.

Env vars (semua opsional):
    MISTRAL_KEY     API key Mistral (WAJIB — tidak ada default)
    BRIDGE_MODEL    model yang dipakai (default: glm-5-2)
    BRIDGE_PORT     port listen (default: 8090)
    BRIDGE_HOST     host listen (default: 0.0.0.0)
"""

import os
import time
import json
import logging

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── Konfigurasi ──────────────────────────────────────────────────────────────
MISTRAL_KEY  = os.environ.get("MISTRAL_KEY", "")  # fallback — client key takes precedence
MODEL        = os.environ.get("BRIDGE_MODEL", "glm-5-2")
PORT         = int(os.environ.get("BRIDGE_PORT", 8090))
HOST         = os.environ.get("BRIDGE_HOST", "0.0.0.0")
MISTRAL_API  = "https://api.mistral.ai/v1"
MISTRAL_BASE = f"{MISTRAL_API}/conversations"

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
        return "".join(parts)
    return str(content)


def openai_tool_call_to_entry(tc: dict):
    """OpenAI tool_calls[] item -> Mistral FunctionCallEntry."""
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name")
    if not name:
        return None
    args = fn.get("arguments", tc.get("arguments", "{}"))
    if args is None:
        args = "{}"
    elif not isinstance(args, (str, dict)):
        args = _as_json_string(args)
    tcid = tc.get("id") or ""
    if not tcid:
        tcid = f"call_{name}"
    return {
        "type": "function.call",
        "tool_call_id": str(tcid),
        "name": str(name),
        "arguments": args,
    }


def _function_result_entry(tool_call_id: str, result) -> dict:
    return {
        "type": "function.result",
        "tool_call_id": str(tool_call_id),
        "result": _as_json_string(result) if not isinstance(result, str) else result,
    }


def normalize_messages(messages: list) -> tuple:
    """OpenAI/Anthropic messages -> (Mistral inputs, instructions).

    assistant.tool_calls / Anthropic tool_use  -> function.call
    role=tool / Anthropic tool_result          -> function.result
    system / developer                         -> instructions
    """
    inputs = []
    instructions_parts = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content")

        if role in ("system", "developer"):
            text = content_to_text(content)
            if text.strip():
                instructions_parts.append(text)
            continue

        if role in ("tool", "function"):
            tcid = m.get("tool_call_id") or m.get("id") or m.get("name") or ""
            result = content_to_text(content)
            if tcid:
                inputs.append(_function_result_entry(tcid, result))
            elif result.strip():
                inputs.append({"role": "user", "content": f"[tool result] {result}"})
            continue

        if role not in ("user", "assistant"):
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
                if btype == "tool_result":
                    tcid = block.get("tool_use_id") or block.get("tool_call_id") or ""
                    res = content_to_text(block.get("content"))
                    if tcid:
                        inputs.append(_function_result_entry(tcid, res))
                    elif res.strip():
                        leftover.append(res)
                elif btype == "tool_use" and not openai_tcs:
                    tcid = block.get("id") or ""
                    name = block.get("name") or ""
                    args = block.get("input", {})
                    if name:
                        inputs.append({
                            "type": "function.call",
                            "tool_call_id": str(tcid or f"call_{name}"),
                            "name": str(name),
                            "arguments": args if isinstance(args, (dict, str)) else _as_json_string(args),
                        })
                elif btype == "thinking":
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
                entry = openai_tool_call_to_entry(tc)
                if entry:
                    inputs.append(entry)
        elif text.strip():
            inputs.append({"role": "user", "content": text})

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
        # Forced function: Conversations only accepts the enum, not a name.
        return "any"
    return None


def resolve_reasoning(body: dict):
    """Client options -> Mistral reasoning_effort ('none'/'high'). Default: thinking ON.
    Upstream only accepts 'none'/'high' — all other effort values are mapped to 'high'."""
    if body.get("reasoning_effort"):
        effort = str(body["reasoning_effort"]).lower()
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


def openai_to_mistral(body: dict) -> dict:
    """Translate request OpenAI -> payload Mistral conversations."""
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    inputs, instructions = normalize_messages(messages)
    if not inputs:
        inputs = [{"role": "user", "content": " "}]

    temperature = max(0.0, min(1.0, _num(body, "temperature", 0.7, float)))
    top_p = max(0.0, min(1.0, _num(body, "top_p", 1.0, float)))
    max_tokens = max(0, _num(body, "max_tokens", 2048, int))

    completion_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    reasoning = resolve_reasoning(body)
    if reasoning:
        completion_args["reasoning_effort"] = reasoning
    tool_choice = map_tool_choice(body.get("tool_choice"))
    if tool_choice:
        completion_args["tool_choice"] = tool_choice

    payload = {
        "model": resolve_model(body.get("model")),
        "inputs": inputs,
        "completion_args": completion_args,
        "store": False,
    }
    if instructions:
        payload["instructions"] = instructions

    tools = normalize_tools(body.get("tools"), body.get("functions"))
    if tools:
        payload["tools"] = tools
    return payload


def extract_content_parts(content) -> tuple:
    """content (str | list of thinking/text blocks) -> (text, reasoning)."""
    text, reasoning = "", ""
    if isinstance(content, str):
        return content, ""
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text += block.get("text", "")
            elif btype == "thinking":
                for t in block.get("thinking", []):
                    if isinstance(t, dict) and t.get("type") == "text":
                        reasoning += t.get("text", "")
    return text, reasoning


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
    """Translate response Mistral conversations -> format OpenAI chat.completion."""
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

    usage = data.get("usage", {})
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
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        },
    }


def _openai_chunk(model: str, delta: dict, finish_reason=None, chunk_id: str = "") -> dict:
    return {
        "id": chunk_id or f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _args_fragment(prev: str, incoming: str) -> str:
    """Emit only the new suffix if upstream sends cumulative arguments."""
    if not incoming:
        return ""
    if prev and incoming.startswith(prev):
        return incoming[len(prev):]
    return incoming


async def stream_response(upstream_key: str, model: str, payload: dict):
    """SSE proxy: upstream Mistral conversations stream -> OpenAI chat.completion.chunk."""
    chunk_id = f"chatcmpl-{int(time.time())}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                MISTRAL_BASE,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {upstream_key}",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.warning("upstream %s: %s", resp.status, text[:200])
                    yield f"data: {json.dumps({'error': {'message': text[:500], 'type': 'upstream_error'}})}\n\n"
                    return
                first = True
                saw_tool_call = False
                # tool_call_id -> {index, args}
                tool_state = {}
                while True:
                    raw = await resp.content.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue

                    etype = data.get("type")
                    if etype == "conversation.response.error":
                        msg = data.get("message") or "upstream stream error"
                        yield f"data: {json.dumps({'error': {'message': msg, 'type': 'upstream_error'}})}\n\n"
                        return

                    delta = None
                    if etype == "message.output.delta":
                        content = data.get("content")
                        if not content:
                            continue
                        if isinstance(content, dict) and content.get("type") == "thinking":
                            thinking = content.get("thinking") or []
                            texts = [t.get("text", "") for t in thinking
                                     if isinstance(t, dict) and t.get("type") == "text"]
                            delta = {"reasoning_content": "".join(texts)}
                        elif isinstance(content, str):
                            delta = {"content": content}
                        elif isinstance(content, dict) and content.get("type") == "text":
                            delta = {"content": content.get("text") or ""}
                        else:
                            continue
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
                        saw_tool_call = True
                        tc_delta = {"index": state["index"], "function": {}}
                        if not state.get("opened"):
                            tc_delta["id"] = tcid
                            tc_delta["type"] = "function"
                            tc_delta["function"]["name"] = name
                            state["opened"] = True
                        if frag:
                            tc_delta["function"]["arguments"] = frag
                        elif "name" not in tc_delta["function"]:
                            continue
                        delta = {"tool_calls": [tc_delta]}
                    else:
                        continue

                    if first:
                        delta["role"] = "assistant"
                        first = False
                    yield f"data: {json.dumps(_openai_chunk(model, delta, chunk_id=chunk_id))}\n\n"

                finish_reason = "tool_calls" if saw_tool_call else "stop"
                yield f"data: {json.dumps(_openai_chunk(model, {}, finish_reason, chunk_id))}\n\n"
                yield "data: [DONE]\n\n"
        except aiohttp.ClientError as e:
            log.error("connection error: %s", e)
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_unreachable'}})}\n\n"


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

    n_tools = len(payload.get("tools") or [])
    log.info("chat → %s (%d msgs, %d tools)", model, len(payload["inputs"]), n_tools)

    if body.get("stream"):
        payload["stream"] = True
        log.info("stream mode → %s", model)
        return StreamingResponse(
            stream_response(upstream_key, model, payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                MISTRAL_BASE,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {upstream_key}",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    log.warning("upstream %s: %s", resp.status, text[:200])
                    return error_response(
                        resp.status if resp.status < 500 else 502,
                        text[:500],
                        "upstream_error",
                    )
                data = json.loads(text)
        except aiohttp.ClientError as e:
            log.error("connection error: %s", e)
            return error_response(502, str(e), "upstream_unreachable")
        except json.JSONDecodeError:
            log.error("bad upstream JSON")
            return error_response(502, "Bad upstream response", "upstream_error")

    return Response(
        status_code=200,
        media_type="application/json",
        content=json.dumps(mistral_to_openai(data, model)),
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
