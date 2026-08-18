"""Parse Mistral conversations SSE into (event_type, payload) pairs."""

import json


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

