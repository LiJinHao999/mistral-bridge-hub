<div align="center">

<img src="assets/logo.png" alt="Mistral GLM Bridge logo" width="120"/>

# Mistral GLM Bridge

**OpenAI-compatible API → Mistral `/v1/conversations`**

[English](README.md) · [中文](README.zh.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

</div>

Fork of [0xgetz/mistral-bridge](https://github.com/0xgetz/mistral-bridge). This repo is a local OpenAI-compatible proxy so Cursor, Claude Code, and similar clients can talk to Mistral-hosted models (default `glm-5-2`) without using Mistral’s rate-limited Chat Completions endpoint.

## Why

Mistral exposes two APIs:

| Endpoint | Format | Notes |
|---|---|---|
| `/v1/chat/completions` | OpenAI Chat Completions | Third-party models often hit **429** |
| `/v1/conversations` | Mistral-native threads | Create + append, no Chat Completions quota |

This bridge accepts OpenAI-shaped requests, forwards them as Conversations, and translates the stream/JSON back.

## Endpoints

| Method | Path | Behavior |
|---|---|---|
| `POST` | `/v1/chat/completions` | Stateless. Always **creates** a new conversation from the full `messages` window. |
| `POST` | `/v1/responses` | Stateful. Matches a previous thread and **appends** when it can; otherwise creates. `response.id` is the Mistral `conversation_id`. |
| `POST` | `/v1/messages` | Anthropic Messages API. Stateless like Chat Completions. Accepts `system`, `thinking`, `tool_use`/`tool_result` content blocks, and `x-api-key` auth. |
| `GET` | `/v1/models` | Passthrough to Mistral; falls back to the configured local model. |
| `GET` | `/v1/models/{id}` | Same, for a single model card. |
| `GET` | `/health` | `{status, model, port}` |

Auth: client `Authorization: Bearer …` (OpenAI) or `x-api-key: …` (Anthropic) wins; otherwise `MISTRAL_KEY`.

Streaming (`"stream": true`) is supported on Chat Completions, Responses, and Messages.

## Quick start

```bash
git clone https://github.com/LiJinHao999/mistral-bridge-hub.git
cd mistral-bridge-hub
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

echo "MISTRAL_KEY=sk-..." > .env
set -a; . ./.env; set +a

./mistral-bridge.sh start
```

Default listen address is `0.0.0.0:8577`.

```bash
curl http://127.0.0.1:8577/v1/chat/completions \
  -H "Authorization: Bearer $MISTRAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'

curl http://127.0.0.1:8577/v1/responses \
  -H "Authorization: Bearer $MISTRAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","input":"hello","max_output_tokens":50}'

curl http://127.0.0.1:8577/v1/messages \
  -H "x-api-key: $MISTRAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

Point any OpenAI- or Anthropic-compatible client at `http://127.0.0.1:8577/v1` with the same key.

## Environment

| Var | Default | Description |
|---|---|---|
| `MISTRAL_KEY` | *(none)* | Fallback API key if the client does not send `Authorization` |
| `BRIDGE_MODEL` | `glm-5-2` | Default model when the client omits one |
| `BRIDGE_PORT` | `8577` | Listen port |
| `BRIDGE_HOST` | `0.0.0.0` | Listen host |

`./mistral-bridge.sh` reads `.env` and is the Linux control script:

```bash
./mistral-bridge.sh start      # background start
./mistral-bridge.sh stop
./mistral-bridge.sh restart
./mistral-bridge.sh status
./mistral-bridge.sh enable     # systemd user unit, starts on login
./mistral-bridge.sh disable
```

On a headless box, after `enable` also run `sudo loginctl enable-linger $USER` so the unit survives reboot without a login. Without systemd:

```bash
crontab -e
# @reboot /path/to/mistral-bridge/mistral-bridge.sh start
```

## Layout

```
mistral-bridge/
├── server.py              # uvicorn entry
├── bridge/
│   ├── config.py          # env, timeouts, logger
│   ├── utils.py           # keys, content flattening, error helper
│   ├── models.py          # local /v1/models fallback
│   ├── tools.py           # function.call / function.result
│   ├── cache.py           # create vs append matching
│   ├── translate.py       # Chat / Responses / Anthropic ↔ Conversations
│   ├── sse.py             # Mistral SSE parser
│   ├── upstream.py        # create / append / GET
│   ├── streaming.py       # SSE → OpenAI / Anthropic events
│   ├── routes.py          # HTTP routes
│   └── app.py             # FastAPI app
├── mistral-bridge.sh      # start/stop/restart/status/enable
├── requirements.txt
└── LICENSE
```

## License

[MIT](LICENSE). Upstream copyright: 0xgetz.
