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
MISTRAL_KEY  = os.environ.get("MISTRAL_KEY", "")  # fallback — client key优先
MODEL        = os.environ.get("BRIDGE_MODEL", "glm-5-2")
PORT         = int(os.environ.get("BRIDGE_PORT", 8090))
HOST         = os.environ.get("BRIDGE_HOST", "0.0.0.0")
MISTRAL_BASE = "https://api.mistral.ai/v1/conversations"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mistral-bridge")

app = FastAPI(title="Mistral GLM Bridge", version="1.0.0")


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_messages(messages: list) -> list:
    """Saring pesan: cuma role user/assistant dengan content string non-empty."""
    out = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
        elif role == "system":
            out.append({"role": "user", "content": f"[system] {content}"})
    return out


def resolve_reasoning(body: dict):
    """Client options -> Mistral reasoning_effort ('none'/'high'). Default: thinking ON.
    Upstream hanya menerima 'none'/'high' — semua nilai effort lain dipetakan ke 'high'."""
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


def openai_to_mistral(body: dict) -> dict:
    """Translate request OpenAI -> payload Mistral conversations."""
    messages = normalize_messages(body.get("messages", []))
    completion_args = {
        "temperature": float(body.get("temperature", 0.7)),
        "max_tokens": int(body.get("max_tokens", 2048)),
        "top_p": float(body.get("top_p", 1)),
    }
    reasoning = resolve_reasoning(body)
    if reasoning:
        completion_args["reasoning_effort"] = reasoning
    return {
        "model": MODEL,
        "inputs": messages,
        "tools": [],  # bisa dipasang dari body.get("tools") kalau perlu
        "completion_args": completion_args,
        "instructions": "",
    }


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


def mistral_to_openai(data: dict, model: str) -> dict:
    """Translate response Mistral conversations -> format OpenAI chat.completion."""
    text, reasoning = "", ""
    for o in data.get("outputs", []):
        if o.get("type") == "message.output" and o.get("role") == "assistant":
            text, reasoning = extract_content_parts(o.get("content", ""))
            if text or reasoning:
                break
    message = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    usage = data.get("usage", {})
    return {
        "id": data.get("conversation_id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        },
    }


async def stream_response(upstream_key: str, model: str, payload: dict):
    """Proksi SSE: upstream Mistral conversations stream -> OpenAI chat.completion.chunk."""
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
                while True:
                    raw = await resp.content.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") != "message.output.delta":
                        continue
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
                    else:
                        continue
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                finish = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(finish)}\n\n"
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

    model = body.get("model", MODEL)
    try:
        payload = openai_to_mistral(body)
    except Exception as e:
        return error_response(422, f"Payload translation error: {e}", "invalid_payload")

    # 网关模式：优先透传客户端 Authorization，否则用环境变量 MISTRAL_KEY
    auth_header = request.headers.get("Authorization", "")
    key = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    upstream_key = key or MISTRAL_KEY
    if not upstream_key:
        return error_response(401, "Missing API key (client Authorization or MISTRAL_KEY)", "missing_api_key")

    log.info("chat → %s (%d msgs)", model, len(payload["inputs"]))

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


@app.get("/v1/models")
async def models():
    """Daftar model — 9router polling endpoint ini."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL,
            "object": "model",
            "owned_by": "mistral",
            "permission": [],
        }],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "port": PORT}


if __name__ == "__main__":
    log.info("Mistral bridge on %s:%d, model=%s", HOST, PORT, MODEL)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")