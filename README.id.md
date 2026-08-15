<div align="center">

<img src="assets/logo.png" alt="Mistral GLM Bridge logo" width="120"/>

# 🌉 Mistral GLM Bridge

**Proksi OpenAI-compatible → Mistral `/v1/conversations` untuk 9router**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)

</div>

---

## 🌍 Bahasa README

| 🌐 | Bahasa | |
|----|--------|---|
| 🇬🇧 | **English** | [Read](README.md) |
| 🇮🇩 | **Bahasa Indonesia** | [Baca](README.id.md) |
| 🇨🇳 | **中文** | [阅读](README.zh.md) |
| 🇯🇵 | **日本語** | [読む](README.ja.md) |
| 🇰🇷 | **한국어** | [읽기](README.ko.md) |

---

## 📌 Ringkasan

Mistral punya **dua endpoint berbeda**:

- `/v1/chat/completions` (format OpenAI) — **sering kena rate limit (429)** untuk model pihak ketiga
- `/v1/conversations` (format Mistral) — **bebas limit**, mulus

**9router** butuh format OpenAI-compatible. Bridge ini:

1. Terima request format OpenAI dari 9router
2. Terjemahkan ke `/v1/conversations`
3. Terjemahkan balik ke format OpenAI
4. Balikin lewat port lokal (default `8090`)

Hasil: **GLM-5.2 via Mistral tanpa 429**, colok sebagai provider 9router biasa.

## 🚀 Mulai Cepat

```bash
git clone https://github.com/0xgetz/mistral-bridge.git
cd mistral-bridge
pip install -r requirements.txt

# Setup API key (tidak ada default — JANGAN commit key lu)
echo "MISTRAL_KEY=sk-..." > .env

# Start bridge + pastikan node 9router
./mistral-bridge.sh start
```

## 🛠 Perintah

```bash
./mistral-bridge.sh start     # start bridge + pastikan node 9router
./mistral-bridge.sh stop      # stop bridge
./mistral-bridge.sh status    # cek status bridge + node
./mistral-bridge.sh watch     # watchdog: auto-restart kalau crash
```

## 📁 Struktur

```
mistral-bridge/
├── server.py              # FastAPI bridge server
├── mistral-bridge.sh      # control script (start/stop/status/watch)
├── assets/                # logo (SVG + PNG)
├── requirements.txt       # dependencies
├── LICENSE                # MIT
└── README.md              # file ini (+ terjemahan .id/.zh/.ja/.ko)
```

## 🔧 Test

```bash
# Test bridge langsung
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'

# Lewat 9router
curl http://127.0.0.1:20128/v1/chat/completions \
  -H "Authorization: Bearer <9router-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-compatible-chat-xxxx/glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'
```

## 📦 Variabel Lingkungan

| Var | Default | Keterangan |
|-----|---------|------------|
| `MISTRAL_KEY` | *(wajib)* | API key Mistral |
| `BRIDGE_MODEL` | `glm-5-2` | Model yang di-forward |
| `BRIDGE_PORT` | `8090` | Port listen lokal |
| `BRIDGE_HOST` | `0.0.0.0` | Alamat listen |

## 🔄 Auto-start pas reboot

```bash
crontab -e
# tambah:
@reboot /root/mistral-bridge/mistral-bridge.sh start
```

## 🤝 Kontribusi

PR disambut! Mohon ikuti:
- Jaga kesederhanaan (KISS)
- Tanpa secret di kode — selalu pakai env vars
- Test dulu sebelum submit

## 📄 Lisensi

[MIT](LICENSE) © 2026 [0xgetz](https://github.com/0xgetz)