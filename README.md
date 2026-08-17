<div align="center">

<img src="assets/logo.png" alt="Mistral GLM Bridge logo" width="120"/>

# 🌉 Mistral GLM Bridge

**OpenAI-compatible → Mistral `/v1/conversations` proxy for 9router**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)
[![Release](https://img.shields.io/github/v/release/0xgetz/mistral-bridge.svg)](https://github.com/0xgetz/mistral-bridge/releases)

</div>

---

## 🌍 README Languages

| 🌐 | Language | |
|----|----------|---|
| 🇬🇧 | **English** | [Read](README.md) |
| 🇮🇩 | **Bahasa Indonesia** | [Baca](README.id.md) |
| 🇨🇳 | **中文** | [阅读](README.zh.md) |
| 🇯🇵 | **日本語** | [読む](README.ja.md) |
| 🇰🇷 | **한국어** | [읽기](README.ko.md) |

---

## 📌 Overview

Mistral has **two different endpoints**:

- `/v1/chat/completions` (OpenAI-compatible) — **often rate-limited (429)** for third-party models
- `/v1/conversations` (Mistral-native) — **no rate limit**, smooth

**9router** needs the OpenAI-compatible format. This bridge:

1. Receives OpenAI-format requests from 9router
2. Translates them to `/v1/conversations`
3. Translates the response back to OpenAI format
4. Returns via local port (default `8090`)

Result: **GLM-5.2 via Mistral without 429**, plugged in as a regular 9router provider.

## 🚀 Quick Start

```bash
git clone https://github.com/0xgetz/mistral-bridge.git
cd mistral-bridge
pip install -r requirements.txt

# Setup API key (no default — DO NOT commit your key)
echo "MISTRAL_KEY=sk-..." > .env

# Start bridge + ensure 9router node
./mistral-bridge.sh start
```

## 🛠 Commands

```bash
./mistral-bridge.sh start     # start bridge + ensure 9router node
./mistral-bridge.sh stop      # stop bridge
./mistral-bridge.sh status    # check bridge + node status
./mistral-bridge.sh watch     # watchdog: auto-restart on crash
```

## 📁 Structure

```
mistral-bridge/
├── server.py              # FastAPI bridge server
├── mistral-bridge.sh      # control script (start/stop/status/watch)
├── assets/                # logo (SVG + PNG)
├── requirements.txt       # dependencies
├── LICENSE                # MIT
└── README.md              # this file (+ .id/.zh/.ja/.ko translations)
```

## 🔧 Test

```bash
# Direct bridge test
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'

# Via 9router
curl http://127.0.0.1:20128/v1/chat/completions \
  -H "Authorization: Bearer <9router-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-compatible-chat-xxxx/glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'

# Responses API (conversation_id is returned as response.id)
curl http://127.0.0.1:8090/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","input":"halo","max_output_tokens":50}'
```

## 📦 Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `MISTRAL_KEY` | *(required)* | Mistral API key |
| `BRIDGE_MODEL` | `glm-5-2` | Model to forward |
| `BRIDGE_PORT` | `8090` | Local listen port |
| `BRIDGE_HOST` | `0.0.0.0` | Listen address |

## 🔄 Auto-start on reboot

```bash
crontab -e
# add:
@reboot /root/mistral-bridge/mistral-bridge.sh start
```

## 🤝 Contributing

PRs welcome! Please follow:
- Keep it simple (KISS)
- No secrets in code — always use env vars
- Test before submitting

## 📄 License

[MIT](LICENSE) © 2026 [0xgetz](https://github.com/0xgetz)