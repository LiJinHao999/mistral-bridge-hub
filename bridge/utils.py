"""Small, protocol-agnostic helpers used across the bridge."""

import json

from fastapi import Request, Response

from .config import MISTRAL_KEY, MODEL


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_key(request: Request) -> str:
    """Client Authorization takes precedence; fall back to server MISTRAL_KEY.

    Accepts both OpenAI-style ``Authorization: Bearer …`` and Anthropic-style
    ``x-api-key: …`` headers so the same bridge serves both client families.
    """
    auth_header = request.headers.get("Authorization", "")
    key = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    if key:
        return key
    api_key = request.headers.get("x-api-key", "")
    if api_key.strip():
        return api_key.strip()
    return MISTRAL_KEY


def resolve_model(requested) -> str:
    """Use the client model id when present; strip 9router 'provider/model' prefixes."""
    if not isinstance(requested, str):
        return MODEL
    name = requested.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1].strip()
    return name or MODEL


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


def _num(body: dict, key: str, default, cast):
    """Read a numeric field; treat explicit null as default (avoids float(None))."""
    value = body.get(key, default)
    if value is None:
        return default
    return cast(value)


def _as_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


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


def error_response(status: int, message: str, code: str = "upstream_error") -> Response:
    return Response(
        status_code=status,
        media_type="application/json",
        content=json.dumps({"error": {"message": message, "type": code}}),
    )

