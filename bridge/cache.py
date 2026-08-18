"""In-memory conversation cache used to decide create vs append.

Matching is by tool_call_id and a hash of the plain-text turns, never by a
literal prefix of the full entry list. Insertion-ordered, so the oldest key
is the eviction candidate.
"""

import hashlib
import json
import time

# conv_id -> {head_len, head_hash, settled, pending, busy_until}.
_conv_cache: dict[str, dict] = {}
_conv_prompt_tokens: dict[str, int] = {}
_CONV_CACHE_LIMIT = 64
# Upper bound on how long one append may hold a conversation. A client that
# drops before the stream generator runs must not pin it forever.
_CONV_BUSY_TTL = 300


def _split_entries(entries: list) -> tuple:
    """Normalized entries -> (messages, result_ids, call_ids), each in order."""
    messages, result_ids, call_ids = [], [], []
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "function.result":
            result_ids.append(str(item.get("tool_call_id") or ""))
        elif itype == "function.call":
            call_ids.append(str(item.get("tool_call_id") or ""))
        elif item.get("role"):
            messages.append(item)
    return messages, result_ids, call_ids


def _messages_hash(messages: list) -> str:
    """Identity of the plain-text turns; tool entries are matched by id instead."""
    blob = json.dumps(
        [[m.get("role") or "", m.get("content") or ""] for m in messages],
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _record_busy(rec: dict) -> bool:
    return float(rec.get("busy_until") or 0) > time.time()


def mark_conversation_busy(conv_id: str, busy: bool):
    """Keep parallel client turns off a conversation that has an append in flight.

    Mistral answers a second concurrent append with 409 'Conversation is being
    updated'. Skipping a busy conversation makes the racing turn create its own
    instead of burning four retries and then failing. The TTL means a client
    that disconnects before the generator runs cannot pin a conversation.

    Releasing clears the flag outright rather than holding a cooldown. Mistral
    may still be writing the thread for a moment, so the next sequential turn
    can still draw a 409 — but that retries with backoff and keeps the prompt
    cache, whereas blocking the match would force a create and drop the cache
    for good. Never trade a recoverable error for an unrecoverable cost.
    """
    rec = _conv_cache.get(conv_id)
    if not rec:
        return
    rec["busy_until"] = (time.time() + _CONV_BUSY_TTL) if busy else 0


def find_append_match(entries: list):
    """Find the conversation this request continues. -> (conv_id, inputs) or None.

    Matching is by tool_call_id, never by a literal prefix of the entry list.
    Clients regroup an assistant turn as soon as it issues a second batch of
    calls — [user, a, fc1, fc2, fr1, fr2] becomes [user, a, fc1, fc2, fc3,
    fr1, fr2, fr3] — so the previous window stops being a prefix of the new one
    and a prefix hash misses every other turn.
    """
    messages, result_ids, _ = _split_entries(entries)
    seen_results = set(result_ids)
    best = None
    best_head = -1
    for rec in list(_conv_cache.values()):
        head_len = rec["head_len"]
        if _record_busy(rec) or head_len > len(messages) or head_len <= best_head:
            continue
        if _messages_hash(messages[:head_len]) != rec["head_hash"]:
            continue
        pending = [tcid for tcid in rec["pending"] if tcid not in rec["settled"]]
        # A conversation waiting on calls accepts nothing but those results.
        if pending and not set(pending) <= seen_results:
            continue
        # Results for calls this conversation never issued belong to another thread.
        if any(tcid not in rec["settled"] and tcid not in pending for tcid in result_ids):
            continue
        inputs = [
            item
            for tcid in pending
            for item in entries
            if isinstance(item, dict)
            and item.get("type") == "function.result"
            and str(item.get("tool_call_id") or "") == tcid
        ]
        # Only new user turns are new to Mistral: assistant text and the calls
        # themselves are already stored on the conversation.
        inputs += [m for m in messages[head_len:] if m.get("role") == "user"]
        if not inputs:
            continue
        best = (rec["conv_id"], inputs)
        best_head = head_len
    return best


def pending_call_ids_from_outputs(data) -> list:
    """tool_call_ids a non-stream response left open, in order."""
    ids = []
    if not isinstance(data, dict):
        return ids
    for item in data.get("outputs") or []:
        if isinstance(item, dict) and item.get("type") == "function.call":
            tcid = str(item.get("tool_call_id") or item.get("id") or "")
            if tcid:
                ids.append(tcid)
    return ids


def pending_call_ids_from_tool_state(tool_state: dict) -> list:
    """tool_call_ids a streamed response left open, in emission order."""
    ordered = sorted(
        (tool_state or {}).items(),
        key=lambda kv: (kv[1] or {}).get("index", 0),
    )
    return [str(tcid) for tcid, state in ordered if tcid and (state or {}).get("name")]


def cache_conversation(conv_id: str, entries: list, pending_ids: list = None, prompt_tokens: int = 0):
    """Record what this conversation now holds, for the next append match.

    settled grows with every tool_call_id the client has ever shown us for this
    conversation, so a replayed window of old results does not read as a foreign
    thread. pending is what the model just asked for and has not been answered.
    prompt_tokens is the last prompt size, used to infer cached_tokens on the
    next append when Mistral omits the cache fields.
    """
    if not conv_id or conv_id.startswith("resp_"):
        return
    messages, result_ids, _ = _split_entries(entries)
    rec = _conv_cache.pop(conv_id, None) or {
        "conv_id": conv_id,
        "settled": set(),
        "busy_until": 0,
    }
    rec["head_len"] = len(messages)
    rec["head_hash"] = _messages_hash(messages)
    rec["settled"] |= {tcid for tcid in result_ids if tcid}
    # This turn is done and the conversation's state is known, so it can serve
    # the next append right now. Waiting for the stream generator's finally to
    # clear the flag is too late: it only runs once the client has consumed the
    # whole SSE body, and the client fires its next request the moment it reads
    # response.completed. That race skipped the match on every other turn and
    # fell back to create, throwing away the whole prompt cache each time.
    rec["busy_until"] = 0
    rec["pending"] = [
        tcid for tcid in (pending_ids or []) if tcid and tcid not in rec["settled"]
    ]
    if prompt_tokens:
        _conv_prompt_tokens[conv_id] = int(prompt_tokens)
    _conv_cache[conv_id] = rec
    while len(_conv_cache) > _CONV_CACHE_LIMIT:
        oldest = next(iter(_conv_cache))
        _conv_cache.pop(oldest, None)
        _conv_prompt_tokens.pop(oldest, None)


def evict_conversation(conv_id: str):
    if not conv_id:
        return
    _conv_cache.pop(conv_id, None)
    _conv_prompt_tokens.pop(conv_id, None)


def get_prompt_tokens(conv_id: str) -> int:
    """Last recorded prompt size for this conversation, else 0."""
    if not conv_id:
        return 0
    return _conv_prompt_tokens.get(conv_id, 0)
