# Mistral→9router GLM Bridge

Bridge server yang menerjemahkan OpenAI-compatible `/v1/chat/completions` menjadi Mistral
`/v1/conversations` API, supaya model **GLM-5.2** (dan model lain yang di-serve Mistral)
bisa dipakai sebagai provider biasa di [9router](https://github.com/...) gateway.

## Kenapa ini ada?

Mistral punya dua endpoint berbeda:

- `/v1/chat/completions` (OpenAI-compatible) — **sering kena rate limit (429)** untuk model non-Mistral
- `/v1/conversations` (format Mistral sendiri) — **bebas limit**, jalan mulus

9router butuh format OpenAI-compatible. Bridge ini:
1. Menerima request OpenAI-format dari 9router
2. Menerjemahkan ke `/v1/conversations`
3. Menerjemahkan balikan ke format OpenAI
4. Mengembalikan lewat port lokal (default `8090`)

Hasil: GLM-5.2 lewat Mistral **tanpa 429**, colok sebagai node 9router biasa.

## Struktur

```
mistral-bridge/
├── server.py              # FastAPI bridge server
├── mistral-bridge.sh      # control script: start/stop/status/watch + auto-colok 9router
├── requirements.txt       # dependencies
└── README.md              # file ini
```

## Install

```bash
git clone https://github.com/xiyyyyz/mistral-bridge.git
cd mistral-bridge
pip install -r requirements.txt
```

## Setup

1. **Isi API key** (tidak ada default — jangan commit key ke repo):

```bash
echo "MISTRAL_KEY=sk-..." > /root/mistral-bridge/.env
```

2. **Cek .env di-gitignore** (sudah otomatis):

```bash
echo ".env" >> /root/mistral-bridge/.gitignore   # kalau belum ada
```

3. Untuk 9router, pastikan node mengarah ke `http://127.0.0.1:8090/v1`
(lihat `mistral-bridge.sh` fungsi `ensure_node`).

## Jalankan

```bash
# start bridge + pastikan node 9router terdaftar
./mistral-bridge.sh start

# cek status
./mistral-bridge.sh status

# stop
./mistral-bridge.sh stop

# watchdog (auto-restart kalau bridge mati)
./mistral-bridge.sh watch
```

## Test manual

```bash
# lewat bridge langsung
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'

# lewat 9router (asumsi node id openai-compatible-chat-xxxx)
curl http://127.0.0.1:20128/v1/chat/completions \
  -H "Authorization: Bearer <9router-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-compatible-chat-xxxx/glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'
```

## Auto-start pas reboot

```bash
crontab -e
# tambah:
@reboot /root/mistral-bridge/mistral-bridge.sh start
```

## Env var (opsional)

| Var             | Default                    | Keterangan                    |
|-----------------|----------------------------|-------------------------------|
| `MISTRAL_KEY`   | (hardcode di server.py)    | API key Mistral               |
| `BRIDGE_MODEL`  | `glm-5-2`                  | model yang di-forward         |
| `BRIDGE_PORT`   | `8090`                     | port lokal                    |

## License

MIT
