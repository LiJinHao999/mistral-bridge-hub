#!/bin/bash
# mistral-bridge.sh — start/stop/watchdog + auto-colok node 9router (opsional)
# Usage:
#   ./mistral-bridge.sh start    # start bridge + pastikan node 9router ada
#   ./mistral-bridge.sh stop     # stop bridge
#   ./mistral-bridge.sh restart  # stop lalu start
#   ./mistral-bridge.sh status   # cek status
#   ./mistral-bridge.sh watch    # watchdog loop (jalan terus, restart kalau mati)

set -u

# Semua path relatif ke lokasi skrip, bukan path absolut yang di-hardcode:
# repo ini di-clone ke direktori yang berbeda-beda.
BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${BRIDGE_DIR}/.env"                # isi: MISTRAL_KEY=sk-...
LOG_FILE="${BRIDGE_DIR}/bridge.log"
PID_FILE="${BRIDGE_DIR}/.bridge.pid"

# Load .env dulu supaya BRIDGE_PORT/MODEL di bawah ikut nilai dari sana.
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
export MISTRAL_KEY="${MISTRAL_KEY:-}"

BRIDGE_PORT="${BRIDGE_PORT:-8577}"           # samakan dengan default server.py
BRIDGE_MODEL="${BRIDGE_MODEL:-glm-5-2}"

# Integrasi 9router: dilewati kalau DB-nya tidak ada di mesin ini.
NODE_ID="${NODE_ID:-openai-compatible-chat-a43c985c}"
DB="${ROUTER_DB:-${HOME}/.9router/db/data.sqlite}"
ROUTER_PORT="${ROUTER_PORT:-20128}"
ROUTER_KEY="${ROUTER_KEY:-}"                 # kunci 9router untuk smoke test

# Prefer venv repo; fallback ke python3 sistem.
if [ -x "${BRIDGE_DIR}/.venv/bin/python" ]; then
  PY="${BRIDGE_DIR}/.venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi

bridge_up() {
  curl -s -m 5 "http://127.0.0.1:${BRIDGE_PORT}/health" >/dev/null 2>&1
}

bridge_pid() {
  # Pidfile dulu; kalau basi, cari lewat path server.py repo ini; terakhir
  # lewat pemilik port — supaya `stop` tetap jalan untuk bridge yang
  # dinyalakan manual, bukan lewat skrip ini.
  local pid=""
  [ -f "$PID_FILE" ] && pid="$(cat "$PID_FILE" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid"; return 0
  fi
  pid="$(pgrep -u "$(id -u)" -f "${BRIDGE_DIR}/server.py" | head -1)"
  if [ -n "$pid" ]; then
    echo "$pid"; return 0
  fi
  pid="$(ss -lntpH "sport = :${BRIDGE_PORT}" 2>/dev/null \
         | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
  [ -n "$pid" ] && echo "$pid"
}

start_bridge() {
  if bridge_up; then
    echo "[bridge] sudah jalan di :${BRIDGE_PORT} (pid $(bridge_pid))"
    return 0
  fi
  if [ -z "$PY" ]; then
    echo "[bridge] python3 tidak ketemu — bikin venv dulu:"
    echo "  python3 -m venv ${BRIDGE_DIR}/.venv && ${BRIDGE_DIR}/.venv/bin/pip install -r ${BRIDGE_DIR}/requirements.txt"
    return 1
  fi

  echo "[bridge] start pakai ${PY}..."
  # setsid: lepas dari terminal, jadi bridge tidak ikut mati waktu shell ditutup.
  # >>: log lama tidak ketimpa.
  setsid nohup "$PY" -u "${BRIDGE_DIR}/server.py" >> "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"

  for _ in $(seq 1 20); do
    bridge_up && break
    sleep 0.5
  done
  if bridge_up; then
    echo "[bridge] OK di :${BRIDGE_PORT} (pid $(cat "$PID_FILE"))"
  else
    echo "[bridge] GAGAL start — cek ${LOG_FILE}"
    tail -5 "$LOG_FILE" 2>/dev/null
    return 1
  fi
}

stop_bridge() {
  local pid
  pid="$(bridge_pid)"
  if [ -z "$pid" ]; then
    echo "[bridge] tidak jalan"
    rm -f "$PID_FILE"
    return 0
  fi
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.3
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[bridge] belum mati, SIGKILL"
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$PID_FILE"
  echo "[bridge] stopped (pid $pid)"
}

ensure_node() {
  if [ ! -f "$DB" ]; then
    echo "[node] 9router tidak terpasang (${DB} tidak ada) — dilewati"
    return 0
  fi
  # Cek node di DB 9router; kalau hilang, bikin ulang (idempotent)
  if "$PY" -c "
import sqlite3
c = sqlite3.connect('$DB')
exists = c.execute(\"SELECT 1 FROM providerNodes WHERE id='$NODE_ID'\").fetchone()
print('yes' if exists else 'no')
c.close()
" 2>/dev/null | grep -q yes; then
    echo "[node] $NODE_ID sudah ada ✅"
  else
    echo "[node] bikin node + connection + kv..."
    "$PY" - <<PYEOF
import sqlite3, json, uuid, datetime
c = sqlite3.connect('$DB')
cur = c.cursor()
now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
node_id = '$NODE_ID'
cur.execute("INSERT INTO providerNodes (id, type, name, data, createdAt, updatedAt) VALUES (?,?,?,?,?,?)",
    (node_id, 'openai-compatible', 'Mistral GLM bridge',
     json.dumps({'baseUrl': 'http://127.0.0.1:$BRIDGE_PORT/v1', 'description': 'GLM-5.2 via Mistral /v1/conversations bridge'}),
     now, now))
conn_id = str(uuid.uuid4())
cur.execute("INSERT INTO providerConnections (id, provider, authType, name, email, priority, isActive, data, createdAt, updatedAt) VALUES (?,?,?,?,?,?,?,?,?,?)",
    (conn_id, node_id, 'apiKey', 'mistral-glm', 'api@mistral.ai', 0, 1,
     json.dumps({'apiKey': 'not-needed',
                 'providerSpecificData': {'prefix': 'mistralglm', 'apiType': 'chat', 'baseUrl': 'http://127.0.0.1:$BRIDGE_PORT/v1', 'nodeName': 'Mistral GLM bridge', 'models': ['$BRIDGE_MODEL']},
                 'testStatus': 'active'}),
     now, now))
cur.execute("INSERT OR REPLACE INTO kv (scope, key, value) VALUES (?,?,?)",
    ('customModels', f"{node_id}|$BRIDGE_MODEL|llm",
     json.dumps({"providerAlias": node_id, "id": "$BRIDGE_MODEL", "type": "llm", "name": "$BRIDGE_MODEL"})))
c.commit(); c.close()
PYEOF
    echo "[node] selesai ✅"
  fi
}

router_check() {
  if ! curl -s -m 5 "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1 \
     && ! curl -s -m 5 -o /dev/null "http://127.0.0.1:${ROUTER_PORT}/" 2>/dev/null; then
    echo "[9router] tidak jalan di :${ROUTER_PORT} — smoke test dilewati"
    return 0
  fi
  if [ -z "$ROUTER_KEY" ]; then
    echo "[9router] ROUTER_KEY kosong — smoke test dilewati"
    echo "  ROUTER_KEY=sk-xxx $0 start"
    return 0
  fi
  echo "── cek via 9router ──"
  curl -s -m 60 "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
    -H "Authorization: Bearer ${ROUTER_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${NODE_ID}/${BRIDGE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"halo\"}],\"max_tokens\":40}" | head -c 250
  echo
}

case "${1:-start}" in
  start)
    start_bridge || exit 1
    ensure_node
    router_check
    ;;
  stop)
    stop_bridge
    ;;
  restart)
    stop_bridge
    start_bridge || exit 1
    ;;
  status)
    if bridge_up; then
      echo "[bridge] UP :${BRIDGE_PORT} (pid $(bridge_pid))"
      curl -s -m 5 "http://127.0.0.1:${BRIDGE_PORT}/health"; echo
    else
      echo "[bridge] DOWN :${BRIDGE_PORT}"
    fi
    if [ -f "$DB" ]; then
      "$PY" -c "
import sqlite3
c = sqlite3.connect('$DB')
print('[node]', 'ADA' if c.execute(\"SELECT 1 FROM providerNodes WHERE id='$NODE_ID'\").fetchone() else 'HILANG')
c.close()" 2>/dev/null
    else
      echo "[node] 9router tidak terpasang"
    fi
    ;;
  watch)
    echo "[watch] loop — Ctrl+C stop. Restart otomatis kalau bridge mati."
    while true; do
      if ! bridge_up; then
        echo "[$(date +%H:%M:%S)] bridge mati → restart"
        start_bridge
      fi
      sleep 15
    done
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|watch}"
    exit 1
    ;;
esac
